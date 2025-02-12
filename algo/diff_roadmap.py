import os
import time
import logging
import hydra
import wandb
import copy
import numpy as np
import torch
import torch.nn as nn

from algo.diffusion.diffusion_actor import DiffusionActor
from algo.diffusion.critic import CriticObsAct
from algo.diffusion.diffusion_ql import DiffusionQL
from algo.util.running_mean_std import RunningMeanStd
from algo.util.replay import ReplayBuffer, BCReplayBuffer
from utils.misc import AverageScalarMeter

logger = logging.getLogger(__name__)

class DiffusionRoadmap:
    def __init__(self, cfg, env, output_dir="debug"):
        self.cfg = cfg
        self.env = env
        self.device = self.cfg.get("rl_device", "cuda:0")

        # Fetch dimension info from env
        self.obs_critic_dim = self.env.cfg.num_states
        self.obs_policy_dim = self.env.cfg.num_observations
        self.act_dim = self.env.cfg.num_actions
        self.num_envs = self.env.num_envs

        # setup batch size
        self.batch_size = self.cfg["policy"]["trainer"]["batch_size"] * self.cfg["policy"]["trainer"]["iterations"]

        # action chunk size
        self.chunk_size = self.cfg["planner"]["rollout_len"]
        self.query_frequency = self.chunk_size

        # ---- Normalization ----
        self.obs_policy_rms = RunningMeanStd((self.obs_policy_dim,)).to(self.device)
        self.obs_critic_rms = RunningMeanStd((self.obs_critic_dim,)).to(self.device)
        self.state_rms = RunningMeanStd((self.env.planning_state_dim + self.env.planner_goal_dim,)).to(self.device)
        self.value_rms = RunningMeanStd((1,)).to(self.device)

        # ---- Replay Buffer ----
        # self.bc_replay_buffer = BCReplayBuffer(
        #     buffer_size=100000,
        #     batch_size=self.cfg["policy"]["trainer"]["batch_size"],
        #     device=self.device,
        # )
        self.replay_buffer = ReplayBuffer(
            buffer_size=100000,
            batch_size=self.cfg["policy"]["trainer"]["batch_size"],
            device=self.device,
        )

        # ---- Models ----
        # Store the modules in a ModuleList for convenience funtions like .train() and .eval()
        self.models = nn.ModuleList()

        # create Diffusion Actor
        self.actor = DiffusionActor(
            obs_dim=self.obs_policy_dim,
            action_dim=self.act_dim,
            action_bound=1.0,
            chunk_size=self.chunk_size,
            device=self.device,
            beta_schedule="cosine",
            num_timesteps=50,
        ).to(self.device)

        # create Diffusion Critic
        self.critic = CriticObsAct(
            mlp_dims=[256, 256, 256],
            obs_critic_dim=self.obs_critic_dim,
            action_dim=self.act_dim,
            action_steps=self.chunk_size,
            activation_type="ReLU",
            use_layernorm=False,
            residual_style=False,
        ).to(self.device)

        # EMA target actor network
        self.actor_target = copy.deepcopy(self.actor)

        # target critic network
        self.critic_target = copy.deepcopy(self.critic)

        self.models.append(self.actor)
        self.models.append(self.actor_target)
        self.models.append(self.critic)
        self.models.append(self.critic_target)

        # ---- Trainer ----
        logger.log(logging.INFO, "Creating trainers")
        model_trainer_cfg = self.cfg["policy"]["trainer"]

        self.trainer = DiffusionQL(
            cfg=model_trainer_cfg,
            env=self.env,
            replay_buffer=self.replay_buffer,
            bc_replay_buffer=None,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            obs_policy_rms=self.obs_policy_rms,
            obs_critic_rms=self.obs_critic_rms,
            value_rms=self.value_rms,
            device=self.device
        )

        # ---- Sampling-based planner ----
        planner_cfg = self.cfg["planner"]
        self.planner = hydra.utils.get_class(planner_cfg["_target_"])(
            cfg=planner_cfg,
            env=env,
            actor_target=self.actor_target,  # use the target task actor for planning
            critic_target=self.critic_target,  # use the target task critic for planning
            obs_policy_rms=self.obs_policy_rms,
            obs_critic_rms=self.obs_critic_rms,
            value_rms=self.value_rms,
            device=self.device,
        )

        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        # dev_output is the temporary output directory for development
        # TODO: configure output dir from hydra config
        logger.log(logging.INFO, "Creating output directory")
        # output_dir = os.path.join(os.path.curdir, output_dir)
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, f"nn")
        self.tb_dif = os.path.join(self.output_dir, f"tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Snapshot
        self.save_freq = self.cfg["save_frequency"]
        self.save_best_after = self.cfg["save_best_after"]

        # ---- Logging ----
        self.eval_freq = self.cfg["eval_frequency"]
        self.extra_info = {}
        # writer = SummaryWriter(self.tb_dif)
        # self.writer = writer
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.eval_episode_rewards = AverageScalarMeter(self.num_envs)
        self.eval_episode_lengths = AverageScalarMeter(self.num_envs)

        # Compute success rate during training
        self.num_train_success = AverageScalarMeter(100)
        self.num_train_episodes = AverageScalarMeter(100)
        # Compute success rate during evaluation
        self.num_eval_success = AverageScalarMeter(100)
        self.num_eval_episodes = AverageScalarMeter(100)

        # ---- Training ----
        self.obs = None
        self.epoch_num = 0
        self.current_rewards = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.eval_current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.eval_current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((self.num_envs,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.cfg["max_agent_steps"]
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0
        self.normalize_value = self.cfg["policy"]["trainer"]["normalize_value"]
        self.gamma = self.cfg["policy"]["trainer"]["discount"]

    def set_eval(self):
        self.models.eval()
        self.obs_policy_rms.eval()
        self.obs_critic_rms.eval()
        self.state_rms.eval()
        if self.normalize_value:
            self.value_rms.eval()

    def set_train(self):
        self.models.train()
        self.obs_policy_rms.train()
        self.obs_critic_rms.train()
        self.state_rms.train()
        if self.normalize_value:
            self.value_rms.train()

    def save(self, name):
        weights = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "obs_policy_rms": self.obs_policy_rms.state_dict(),
            "obs_critic_rms": self.obs_critic_rms.state_dict(),
            "state_rms": self.state_rms.state_dict(),
        }
        if self.normalize_value:
            weights["value_rms"] = self.value_rms.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.obs_policy_rms.load_state_dict(checkpoint["obs_policy_rms"])
        self.obs_critic_rms.load_state_dict(checkpoint["obs_critic_rms"])
        self.state_rms.load_state_dict(checkpoint["state_rms"])
        if self.normalize_value:
            self.value_rms.load_state_dict(checkpoint["value_rms"])

    def test(self):
        """
            Run the model in evaluation mode
        """
        self.env.reset_dist_type = "eval"
        self.set_eval()
        obs, _ = self.env.reset()

        # initialize the time step counter
        time_step = 0
        with torch.no_grad():
            # flag to check if the environment is done
            env_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            while True:
                processed_obs = self.obs_policy_rms(obs["policy"])
                # sample a chunk of actions
                if time_step % self.query_frequency == 0:
                    pred_act_chunk = self.actor_target.sample_action_chunks(processed_obs)

                next_action = pred_act_chunk[:, time_step % self.query_frequency, :]
                actions = torch.clamp(next_action, -1.0, 1.0)

                # step actions in the environment and avoid resetting
                obs, _, dones, timeouts, _ = self.env.step_without_reset(actions)

                # update the environment done flag
                env_done = env_done | dones | timeouts
                # reset the environment after stepping an action chunk
                if time_step % self.query_frequency == self.query_frequency - 1:
                    # fetch the environment idx that are done
                    done_indices = env_done.nonzero(as_tuple=False)
                    if len(done_indices) > 0:
                        self.env.reset_idx(done_indices)
                    # reset the environment done flag
                    env_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

                # increment the time step counter
                time_step += 1


    def write_stats(self, metric):
        bc_loss = metric["bc_loss"]
        ql_loss = metric["ql_loss"]
        actor_loss = metric["actor_loss"]
        critic_loss = metric["critic_loss"]
        wandb.log({"performance/RLTrainFPS": self.agent_steps / self.rl_train_time}, step=self.agent_steps)
        wandb.log({"performance/EnvStepFPS": self.agent_steps / self.data_collect_time}, step=self.agent_steps)
        wandb.log({"losses/bc_loss": torch.mean(torch.stack(bc_loss)).item()}, step=self.agent_steps)
        wandb.log({"losses/ql_loss": torch.mean(torch.stack(ql_loss)).item()}, step=self.agent_steps)
        wandb.log({"losses/actor_loss": torch.mean(torch.stack(actor_loss)).item()}, step=self.agent_steps)
        wandb.log({"losses/critic_loss": torch.mean(torch.stack(critic_loss)).item()}, step=self.agent_steps)
        for k, v in self.extra_info.items():
            wandb.log({f"{k}": v}, step=self.agent_steps)

    def train(self):
        logger.log(logging.INFO, "Starting training")
        _t = time.time()
        _last_t = time.time()
        self.obs, _ = self.env.reset()
        self.agent_steps = self.batch_size

        # Initialize the graph with 5 planning epochs
        for _ in range(5):
            self.planner.run_prm()

        while self.agent_steps < self.max_agent_steps:
            if self.epoch_num == 0 or (self.eval_freq > 0 and self.epoch_num % self.eval_freq == 0):
                # Evaluate the current model
                self.eval_steps()
                eval_mean_rewards = self.eval_episode_rewards.get_mean()
                eval_mean_lengths = self.eval_episode_lengths.get_mean()
                wandb.log({"eval/episode_rewards": eval_mean_rewards}, step=self.agent_steps)
                wandb.log({"eval/episode_lengths": eval_mean_lengths}, step=self.agent_steps)
                print(f"Eval rewards: {eval_mean_rewards:.2f}")
                if self.epoch_num == 0:
                    self.best_rewards = eval_mean_rewards

                # Update evaluation success rate if environment has returned such data
                if self.num_eval_success.current_size > 0:
                    running_mean_success = self.num_eval_success.get_mean()
                    running_mean_term = self.num_eval_episodes.get_mean()
                    mean_success_rate = running_mean_success / running_mean_term
                    wandb.log({"eval_success_rate/step": mean_success_rate}, step=self.agent_steps)
            self.epoch_num += 1

            # policy update
            metric = self._train_diffusion_ql()

            self.agent_steps += self.batch_size
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | "
                f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                f"Best: {self.best_rewards:.2f}"
            )
            print(info_string)

            # Log the training stats
            self.write_stats(metric)

            # Update training metrics and save model
            self.update_training_metric(eval_mean_rewards, eval_mean_lengths)

    def _train_diffusion_ql(self):
        # Execute PRM planning steps
        _t = time.time()
        self.set_eval()

        # Save the episode context
        ep_ctxt = self.env.save_episode_context()

        # Perform PRM planning step
        self.planner.run_prm()

        # Train the task model
        self.set_eval()

        # Extracted walks w.r.t the task critic
        self.env.reset_dist_type = "train"
        # walks, obs_policy_buf, obs_critic_buf, act_buf, state_buf, goal_buf = self.planner.extract_walks(num_walks=100, length=50)
        (
            _,
            obs_policy_buf,
            obs_critic_buf,
            obs_policy_prime_buf,
            obs_critic_prime_buf,
            act_buf,
            *_
        )= self.planner.perform_search(
            critic=self.critic_target,
            num_searches=100,
            length=50,
            search_for_planner=False
        )

        # Compute the reward and done tensor
        reward_sum_buf, env_not_done_buf = self.get_reward_and_done(obs_policy_buf)

        # Remove the rollout_len dimension from the obs buffer
        obs_policy_buf = obs_policy_buf[:, 0, :]
        obs_critic_buf = obs_critic_buf[:, 0, :]
        obs_policy_prime_buf = obs_policy_prime_buf[:, 0, :]
        obs_critic_prime_buf = obs_critic_prime_buf[:, 0, :]

        # Update the replay buffer for behavioral cloning with the extracted walks
        self.replay_buffer.store(
            obs_policy_buf,
            obs_critic_buf,
            act_buf,
            reward_sum_buf,
            env_not_done_buf,
            obs_policy_prime_buf,
            obs_critic_prime_buf
        )

        # self.bc_replay_buffer.store(obs_policy_buf, act_buf)

        # # Collect on-policy data
        # obs_policy, obs_critic, act_chunk, rewsum, env_not_done, obs_policy_prime, obs_critic_prime = self.play_steps()
        # # Add on-policy data to the replay buffer
        # self.replay_buffer.store(
        #     obs_policy,
        #     obs_critic,
        #     act_chunk,
        #     rewsum,
        #     env_not_done,
        #     obs_policy_prime,
        #     obs_critic_prime
        # )

        self.data_collect_time += time.time() - _t

        # Restore the on policy context
        self.obs = self.env.restore_episode_context(ep_ctxt)

        # Train with the extracted walks
        _t = time.time()
        self.set_train()
        metric = self.trainer.train()
        self.rl_train_time += time.time() - _t

        # clear cache
        torch.cuda.empty_cache()

        return metric

    def get_reward_and_done(self, obs_policy_buf):
        # Initialize the reward sum and not done tensor
        reward_sum = torch.zeros((obs_policy_buf.size(0),), device=self.device)
        env_not_dones = torch.ones((obs_policy_buf.size(0),), dtype=torch.int, device=self.device)

        # Compute the reward and done tensor on each time step
        for i in range(self.chunk_size):
            reward = self.env.get_rewards_from_obs(obs_policy_buf[:, i, :])
            env_not_done = self.env.get_not_dones_from_obs(obs_policy_buf[:, i, :])

            # Update the reward sum and not done tensor
            reward_sum += (self.gamma ** i) * env_not_dones * reward
            env_not_dones = torch.logical_and(env_not_dones, env_not_done)

        return reward_sum, env_not_dones

    def update_training_metric(self, eval_mean_rewards, eval_mean_lengths):
        mean_rewards = self.episode_rewards.get_mean()
        mean_lengths = self.episode_lengths.get_mean()
        wandb.log({"train/episode_rewards": mean_rewards}, step=self.agent_steps)
        wandb.log({"train/episode_lengths": mean_lengths}, step=self.agent_steps)
        checkpoint_name = f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}"

        # update training success rate if environment has returned such data
        if self.num_train_success.current_size > 0:
            running_mean_success = self.num_train_success.get_mean()
            running_mean_term = self.num_train_episodes.get_mean()
            mean_success_rate = running_mean_success / running_mean_term
            wandb.log({"train_success_rate/step": mean_success_rate}, step=self.agent_steps)

        mean_rewards = self.episode_rewards.get_mean()
        mean_lengths = self.episode_lengths.get_mean()

        if self.save_freq > 0:
            if self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, "last"))

        if mean_rewards == -np.Inf:  # mean_rewards are -inf if training episodes never end, use eval metrics
            mean_rewards = eval_mean_rewards
            mean_lengths = eval_mean_lengths

        if eval_mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
            print(f"save current best reward: {eval_mean_rewards:.2f}")
            self.best_rewards = eval_mean_rewards
            self.save(os.path.join(self.nn_dir, "best"))

        wandb.log({"agent_steps": self.agent_steps}, step=self.epoch_num)

    def play_steps(self, reset_dist_type="train"):
        self.env.set_reset_dist_type(reset_dist_type)
        self.env.success_rate_mode = "train"

        # randomly pick states in the graph to serve as the reset states
        select_idx = torch.randint(0, self.planner.prm_q.shape[0], (1024,))
        reset_states = self.planner.prm_q[select_idx]
        self.env.set_reset_state_buf(reset_states)

        with (torch.inference_mode()):

            # flag to check if the environment is done
            env_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)  # terminate or timeout
            env_rewsum = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)  # sum of rewards

            for n in range(self.chunk_size):
                obs_policy = self.obs["policy"]
                obs_critic = self.obs["critic"]

                processed_obs = self.obs_policy_rms(obs_policy)
                # sample a chunk of actions
                if n % self.chunk_size == 0:
                    pred_act_chunk = self.actor_target.sample_action_chunks(processed_obs)

                next_action = pred_act_chunk[:, n % self.chunk_size, :]
                next_action = torch.clamp(next_action, -1.0, 1.0)

                self.obs, rewards, self.dones, timeouts, infos = self.env.step_without_reset(next_action)

                # update the environment done flag
                env_done = torch.logical_or(env_done, torch.logical_or(self.dones, timeouts))

                # update the current rewards and lengths
                self.current_rewards += rewards
                self.current_lengths += 1

                # update the sum of rewards
                env_not_done = 1.0 - env_done.float()
                env_rewsum += (self.gamma ** n) * env_not_done * rewards

                # reset the environment after stepping an action chunk
                if n % self.chunk_size == self.chunk_size - 1:
                    # get the next observation after stepping the action chunk
                    obs_policy_prime = self.obs["policy"]
                    obs_critic_prime = self.obs["critic"]

                    # fetch the environment idx that are done
                    done_indices = env_done.nonzero(as_tuple=False).flatten()
                    # reset the environment that are done or timeout
                    if len(done_indices) > 0:
                        self.env.reset_idx(done_indices)
                        self.obs = self.env.get_observations()

                    # update the evaluation metrics
                    self.episode_rewards.update(self.current_rewards[done_indices])
                    self.episode_lengths.update(self.current_lengths[done_indices])
                    self.current_rewards = self.current_rewards * env_not_done
                    self.current_lengths = self.current_lengths * env_not_done

                self.extra_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.extra_info[k] = v

        return obs_policy, obs_critic, pred_act_chunk, env_rewsum, env_not_done, obs_policy_prime, obs_critic_prime

    def eval_steps(self):
        self.set_eval()
        self.env.enable_reset = True
        self.env.reset_dist_type = "eval"

        with torch.inference_mode():
            # save the episode context before evaluation
            ep_ctxt = self.env.save_episode_context()

            obs, _ = self.env.reset()
            eval_current_rewards = torch.zeros(size=(self.num_envs, 1), dtype=torch.float32, device=self.device)
            eval_current_lengths = torch.zeros(size=(self.num_envs,), dtype=torch.float32, device=self.device)
            count = 0  # evaluate once for each environment

            # flag to check if the environment is done
            env_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # terminate or timeout
            env_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # terminate only

            for n in range(self.env.cfg.max_episode_steps):
                processed_obs = self.obs_policy_rms(obs["policy"])
                # sample a chunk of actions
                if n % self.query_frequency == 0:
                    pred_act_chunk = self.actor_target.sample_action_chunks(processed_obs)

                next_action = pred_act_chunk[:, n % self.query_frequency, :]
                next_action = torch.clamp(next_action, -1.0, 1.0)

                # step actions in the environment and avoid resetting
                obs, rewards, dones, timeouts, infos = self.env.step_without_reset(next_action)

                # update the environment done flag
                env_done = env_done | dones | timeouts
                env_terminate = env_terminate | dones

                # update the evaluation metrics
                rewards = rewards.unsqueeze(1)
                eval_current_rewards += rewards
                eval_current_lengths += 1

                # reset the environment after stepping an action chunk
                if n % self.query_frequency == self.query_frequency - 1:
                    # fetch the environment idx that are done
                    done_indices = env_done.nonzero(as_tuple=False).flatten()
                    if len(done_indices) > 0:
                        self.env.reset_idx(done_indices)

                    count += len(done_indices)
                    # update the evaluation metrics
                    self.eval_episode_rewards.update(eval_current_rewards[done_indices])
                    self.eval_episode_lengths.update(eval_current_lengths[done_indices])
                    not_terminate = 1.0 - env_terminate.float()
                    eval_current_rewards = eval_current_rewards * not_terminate.unsqueeze(1)
                    eval_current_lengths = eval_current_lengths * not_terminate

                    # reset the environment done flag
                    env_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

                # if success is in info, then update the evaluation success rate
                if 'success' in infos:
                    num_eval_success = infos['success']
                    self.num_eval_success.update(num_eval_success)
                    num_eval_terminations = dones
                    self.num_eval_episodes.update(num_eval_terminations)

                # Log extra info (success rate)
                self.extra_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (
                            isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.extra_info[k] = v

                if count >= self.env.num_envs:
                    break
            # restore the episode context after evaluation
            self.obs = self.env.restore_episode_context(ep_ctxt)
