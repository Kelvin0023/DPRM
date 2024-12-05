import os
import time
import logging
import hydra
import wandb
import copy
import numpy as np
import torch
import torch.nn as nn

from algo.model.td3 import TD3, Actor, Critic
from trainer.train_td3 import TD3Trainer
from algo.util.running_mean_std import RunningMeanStd
from algo.util.replay import ReplayBuffer, BCReplayBuffer
from utils.misc import AverageScalarMeter

logger = logging.getLogger(__name__)

class Main_TD3:
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
        self.batch_size = self.cfg["policy"]["trainer"]["batch_size"] * self.cfg["policy"]["trainer"]["epochs"]

        # ---- Normalization ----
        self.obs_policy_rms = RunningMeanStd((self.obs_policy_dim,)).to(self.device)
        self.obs_critic_rms = RunningMeanStd((self.obs_critic_dim,)).to(self.device)
        self.state_rms = RunningMeanStd((self.env.planning_state_dim + self.env.planner_goal_dim,)).to(self.device)
        self.value_rms = RunningMeanStd((1,)).to(self.device)

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(
            buffer_size=50000,
            batch_size=self.cfg["policy"]["trainer"]["batch_size"],
            device=self.device,
        )

        # ---- Models ----
        # Store the modules in a ModuleList for convenience funtions like .train() and .eval()
        self.models = nn.ModuleList()
        # Create actor and critic models
        actor = Actor(
            self.obs_policy_dim,
            self.act_dim,
            max_action=1.0,
            units=self.cfg["policy"]["network"]["actor_units"]
        ).to(self.device)
        critic = Critic(
            self.obs_critic_dim,
            self.act_dim,
            units=self.cfg["policy"]["network"]["critic_units"]
        ).to(self.device)

        # Create target networks
        actor_target = copy.deepcopy(actor)
        critic_target = copy.deepcopy(critic)

        # Create TD3 model
        self.model = TD3(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            max_action=1.0,
            policy_noise=self.cfg["policy"]["trainer"]["policy_noise"],
            noise_clip=self.cfg["policy"]["trainer"]["noise_clip"],
        )
        self.models.append(self.model)

        # ---- Trainer ----
        logger.log(logging.INFO, "Creating trainers")
        model_trainer_cfg = self.cfg["policy"]["trainer"]

        # Create the trainer
        self.trainer = TD3Trainer(
            cfg=model_trainer_cfg,
            replay_buffer=self.replay_buffer,
            actor=self.model.actor,
            actor_target=self.model.actor_target,
            critic=self.model.critic,
            critic_target=self.model.critic_target,
            obs_policy_rms=self.obs_policy_rms,
            obs_critic_rms=self.obs_critic_rms,
            value_rms=self.value_rms,
            device=self.device,
        )

        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        # dev_output is the temporary output directory for development
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
        self.dones = torch.ones((self.num_envs,), dtype=torch.uint8, device=self.device)
        self.epoch_num = 0
        self.current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.eval_current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.eval_current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.cfg["max_agent_steps"]
        self.best_rewards = -10000

        # ---- Timing ----
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0
        self.normalize_value = self.cfg["policy"]["trainer"]["normalize_value"]
        self.gamma = self.cfg["policy"]["trainer"]["discount"]
        self.start_timesteps = self.cfg["policy"]["start_timesteps"]

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
            "model": self.model.state_dict(),
            "obs_policy_rms": self.obs_policy_rms.state_dict(),
            "obs_critic_rms": self.obs_critic_rms.state_dict(),
            "state_rms": self.state_rms.state_dict(),
        }
        if self.normalize_value:
            weights["value_rms"] = self.value_rms.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.obs_policy_rms.load_state_dict(checkpoint["obs_policy_rms"])
        self.obs_critic_rms.load_state_dict(checkpoint["obs_critic_rms"])
        self.state_rms.load_state_dict(checkpoint["state_rms"])
        if self.normalize_value:
            self.value_rms.load_state_dict(checkpoint["value_rms"])

    def model_act(self, obs_dict):
        processed_obs = self.obs_policy_rms(obs_dict["policy"])
        actions = self.model.act(processed_obs)
        return {"actions": actions}

    def model_act_inference(self, obs_dict):
        processed_obs = self.obs_policy_rms(obs_dict["policy"])
        actions = self.model.act_inference(processed_obs)
        return {"actions": actions}

    def write_stats(self, metric):
        actor_loss = metric["actor_loss"]
        critic_loss = metric["critic_loss"]
        wandb.log({"performance/RLTrainFPS": self.agent_steps / self.rl_train_time}, step=self.agent_steps)
        wandb.log({"performance/EnvStepFPS": self.agent_steps / self.data_collect_time}, step=self.agent_steps)
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

        while self.agent_steps < self.max_agent_steps:
            if self.epoch_num < self.start_timesteps:
                # Collect enough data to start training
                self.play_steps()
                self.epoch_num += 1
            else:
                # Start training and evaluation
                if ((self.epoch_num - self.start_timesteps) == 0
                        or (self.eval_freq > 0 and (self.epoch_num - self.start_timesteps) % self.eval_freq == 0)):
                    # Evaluate the current model
                    self.eval_steps()
                    eval_mean_rewards = self.eval_episode_rewards.get_mean()
                    eval_mean_lengths = self.eval_episode_lengths.get_mean()
                    wandb.log({"eval/episode_rewards": eval_mean_rewards}, step=self.agent_steps)
                    wandb.log({"eval/episode_lengths": eval_mean_lengths}, step=self.agent_steps)
                    print(f"Eval rewards: {eval_mean_rewards:.2f}")
                    if (self.epoch_num - self.start_timesteps) == 0:
                        self.best_rewards = eval_mean_rewards

                    # Update evaluation success rate if environment has returned such data
                    if self.num_eval_success.current_size > 0:
                        running_mean_success = self.num_eval_success.get_mean()
                        running_mean_term = self.num_eval_episodes.get_mean()
                        mean_success_rate = running_mean_success / running_mean_term
                        wandb.log({"eval_success_rate/step": mean_success_rate}, step=self.agent_steps)

                self.epoch_num += 1

                # policy update
                metric = self._train_td3()

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


    def _train_td3(self):
        _t = time.time()
        self.set_eval()

        # Collect data with the current policy
        self.play_steps()
        self.data_collect_time += time.time() - _t

        # Train with the extracted walks
        _t = time.time()
        self.set_train()
        metric = self.trainer.train()

        self.rl_train_time += time.time() - _t

        # clear cache
        torch.cuda.empty_cache()

        return metric

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

        with (torch.inference_mode()):
            obs_policy = self.obs["policy"]
            obs_critic = self.obs["critic"]

            # get the prediction from the model
            res_dict = self.model_act(self.obs)
            pred_action = torch.clamp(res_dict["actions"], -1.0, 1.0)

            self.obs, rewards, self.dones, timeouts, infos = self.env.step_without_reset(pred_action)

            # update dones and rewards after env step
            self.dones = torch.logical_or(self.dones, timeouts)
            rewards = rewards.unsqueeze(1)
            self.current_rewards += rewards
            self.current_lengths += 1

            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            # get the environment index that are not terminated
            env_not_done = 1.0 - self.dones.float()
            # get the next observation after stepping the action
            obs_policy_prime = self.obs["policy"]
            obs_critic_prime = self.obs["critic"]

            # if success in info, then update success rate
            if 'success' in infos:
                num_train_success = infos['success']
                self.num_train_success.update(num_train_success)
                num_train_terminations = self.dones
                self.num_train_episodes.update(num_train_terminations)
            assert isinstance(infos, dict), "Info Should be a Dict"

            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            # store the experience in the replay buffer
            self.replay_buffer.store(
                obs_policy,
                obs_critic,
                pred_action,
                rewards,
                env_not_done,
                obs_policy_prime,
                obs_critic_prime
            )
            print("---Finished playing steps---")

    def eval_steps(self):
        self.set_eval()
        self.env.set_reset_dist_type("eval")
        self.env.success_rate_mode = "eval"

        with torch.inference_mode():
            # save the episode context before evaluation
            ep_ctxt = self.env.save_episode_context()

            obs_dict, _ = self.env.reset()
            eval_current_rewards = torch.zeros(size=(self.num_envs, 1), dtype=torch.float32, device=self.device)
            eval_current_lengths = torch.zeros(size=(self.num_envs,), dtype=torch.float32, device=self.device)
            count = 0  # evaluate once for each env

            for n in range(self.env.cfg.max_episode_steps):
                res_dict = self.model_act_inference(obs_dict)
                # clip the next action
                pred_action = torch.clamp(res_dict["actions"], -1.0, 1.0)

                # step actions in the environment and avoid resetting
                obs_dict, rewards, dones, timeouts, infos = self.env.step(pred_action)

                # update the evaluation metrics
                rewards = rewards.unsqueeze(1)
                eval_current_rewards += rewards
                eval_current_lengths += 1

                dones = torch.logical_or(dones, timeouts)
                # if success is in info, then update the evaluation success rate
                if 'success' in infos:
                    num_eval_success = infos['success']
                    self.num_eval_success.update(num_eval_success)
                    num_eval_terminations = dones
                    self.num_eval_episodes.update(num_eval_terminations)

                done_indices = dones.nonzero(as_tuple=False)
                count += len(done_indices)
                self.eval_episode_rewards.update(eval_current_rewards[done_indices])
                self.eval_episode_lengths.update(eval_current_lengths[done_indices])
                not_dones = 1.0 - dones.float()
                eval_current_rewards = eval_current_rewards * not_dones.unsqueeze(1)
                eval_current_lengths = eval_current_lengths * not_dones

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

