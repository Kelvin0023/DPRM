import torch
import numpy as np
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from algo.diffusion.auxiliary import EMA


class DiffusionQL(object):
    def __init__(
        self,
        cfg,
        env,
        replay_buffer,
        bc_replay_buffer,
        actor,
        actor_target,
        critic,
        critic_target,
        obs_policy_rms,
        obs_critic_rms,
        value_rms,
        device,
    ):
        # environment settings
        self.obs_dim = actor.obs_dim
        self.state_dim = critic.state_dim
        self.action_dim = actor.action_dim
        self.chunk_size = actor.chunk_size
        self.time_steps = actor.actor.num_timesteps

        self.obs_policy_rms = obs_policy_rms
        self.obs_critic_rms = obs_critic_rms
        self.value_rms = value_rms
        self.normalize_value = cfg["normalize_value"]
        self.device = device

        self.value_scale = cfg.get("value_scale", 1.0)

        # get the simulation environment and networks
        self.env = env
        self.diffusion_actor = actor
        self.diffusion_critic = critic
        self.diffusion_critic_target = critic_target

        # replay buffer
        self.replay_buffer = replay_buffer
        self.bc_replay_buffer = bc_replay_buffer

        self.step = 0
        self.step_start_ema = cfg["step_start_ema"]
        self.ema = EMA(cfg["ema_decay"])
        self.ema_model = actor_target
        self.update_ema_every = cfg["update_ema_every"]

        self.batch_size = cfg["batch_size"]

        self.last_lr = cfg["learning_rate"]
        self.weight_decay = cfg["weight_decay"]

        self.eta = cfg["eta"]  # q_learning weight
        self.tau = cfg["tau"]  # target network update rate
        self.discount = cfg["discount"]

        self.iterations = cfg["iterations"]

        self.grad_norm = cfg["grad_norm"]
        self.truncate_grads = cfg["truncate_grads"]

        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.diffusion_actor.parameters(), self.last_lr, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.diffusion_critic.parameters(), self.last_lr, weight_decay=self.weight_decay
        )

        # learning rate decay
        self.lr_decay = cfg["lr_decay"]
        if self.lr_decay:
            self.lr_max_T = cfg["lr_max_T"]
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=self.lr_max_T, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=self.lr_max_T, eta_min=0.)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_actor)

    def policy_loss(self, action_chunk, obs_policy, weights=1.0):
        t = torch.randint(0, self.time_steps, (self.batch_size,), device=self.device).long()
        flatten_action = action_chunk.view(-1, self.chunk_size * self.action_dim)
        return self.diffusion_actor.actor.p_losses(flatten_action, obs_policy, t, weights)

    def train(self):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        batch = torch.tensor(list(range(self.batch_size)))

        for _ in range(self.iterations):
            # Sample data from replay buffer
            (
                sampled_obs_policy,
                sampled_obs_critic,
                sampled_act_chunk,
                sampled_reward_sum,
                sampled_env_not_done,
                sampled_obs_policy_prime,
                sampled_obs_critic_prime
            ) = self.replay_buffer.sample()

            # (
            #     sampled_obs_policy_demo,
            #     sampled_act_chunk_demo,
            # ) = self.bc_replay_buffer.sample()

            norm_obs_policy = self.obs_policy_rms(sampled_obs_policy)
            norm_obs_critic = self.obs_critic_rms(sampled_obs_critic)

            """ Q Training """
            current_q1, current_q2 = self.diffusion_critic(norm_obs_critic, sampled_act_chunk)

            # predict action chunk for the next time step
            norm_obs_policy_prim = self.obs_policy_rms(sampled_obs_policy_prime)
            norm_obs_critc_prim = self.obs_critic_rms(sampled_obs_critic_prime)

            next_action_chunk = self.ema_model(norm_obs_policy_prim)
            next_q1, next_q2 = self.diffusion_critic_target(norm_obs_critc_prim, next_action_chunk)
            if self.normalize_value:
                self.value_rms.eval()
                unnorm_next_q1 = self.value_rms(next_q1, unnorm=True)
                unnorm_next_q2 = self.value_rms(next_q2, unnorm=True)
                unnorm_target_q = sampled_reward_sum + sampled_env_not_done * (self.discount ** self.chunk_size) * torch.min(unnorm_next_q1, unnorm_next_q2)
                target_q = self.value_rms(unnorm_target_q)
            else:
                target_q = sampled_reward_sum + sampled_env_not_done * (self.discount ** self.chunk_size) * torch.min(next_q1, next_q2)
                target_q *= self.value_scale
                current_q1 *= self.value_scale
                current_q2 *= self.value_scale

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_loss = torch.mean(critic_loss)

            # Step the loss for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_critic.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.policy_loss(sampled_act_chunk, norm_obs_policy)
            bc_loss = torch.mean(bc_loss)
            # norm_obs_policy_demo = self.obs_policy_rms(sampled_obs_policy_demo)
            # bc_loss = self.policy_loss(sampled_act_chunk_demo, norm_obs_policy_demo)
            # bc_loss = torch.mean(bc_loss)

            new_act_chunk = self.diffusion_actor(norm_obs_policy)
            q1_new_action, q2_new_action = self.diffusion_critic(norm_obs_critic, new_act_chunk)

            # Value Unnormalization
            if self.normalize_value:
                q1_new_action = self.value_rms(q1_new_action, unnorm=True)
                q2_new_action = self.value_rms(q2_new_action, unnorm=True)

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            q_loss = torch.mean(q_loss)

            # total actor loss
            actor_loss = bc_loss + self.eta * q_loss
            # actor_loss = bc_loss + self.eta * q_loss + reward_loss

            # Step the loss for actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_actor.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.diffusion_critic.parameters(), self.diffusion_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            metric['actor_loss'].append(actor_loss)
            metric['bc_loss'].append(bc_loss)
            metric['ql_loss'].append(q_loss)
            metric['critic_loss'].append(critic_loss)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric


    def train_planner_policy(self, state_buf, act_buf, goal_buf):
        metric = {'planner_bc_loss': [], 'planner_ql_loss': [], 'planner_actor_loss': [], 'planner_critic_loss': []}
        batch = torch.tensor(list(range(self.batch_size)))

        for _ in range(self.iterations):
            # Sample date from replay buffer
            sampled_idx = torch.randint_like(batch, state_buf.shape[0])
            sampled_state = state_buf[sampled_idx]
            sampled_act_chunk = act_buf[sampled_idx]

            # planner policy try to solve a goal-reaching task
            sampled_goal = goal_buf[sampled_idx]

            # concatenate state and goal
            sampled_state_goal = torch.cat([sampled_state, sampled_goal], dim=1)

            processed_state_goal = self.obs_critic_rms(sampled_state_goal)

            """ Q Training """
            current_q1, current_q2 = self.diffusion_critic(processed_state_goal, sampled_act_chunk)

            with torch.inference_mode():
                self.env.set_env_states(sampled_state, batch.to(self.device))
                self.env.set_planner_goal(sampled_goal, batch)

                # Reset the environment buffer
                self.env.reset_buf[:] = 0
                self.env.episode_length_buf[:] = 0

            # Create target Q value tensor
            target_q = torch.zeros((self.batch_size, 1), device=self.device)
            env_not_dones = torch.ones((self.batch_size, 1), device=self.device)

            with torch.inference_mode():
                for j in range(self.chunk_size):
                    # Step the environment
                    padded_action = torch.zeros((self.env.num_envs, self.action_dim), device=self.device)
                    padded_action[batch, :] = sampled_act_chunk[:, j]
                    new_obs_dict, rewards, dones, _, infos = self.env.planner_step_without_reset(padded_action)
                    rewards = rewards.unsqueeze(1)
                    not_dones = 1.0 - dones.float().unsqueeze(1)
                    env_not_dones = torch.logical_and(env_not_dones, not_dones[batch])
                    target_q += (self.discount ** j) * env_not_dones * rewards[batch, :]

            new_state = self.env.get_env_states()[batch, :]

            # concatenate state and goal
            new_state_goal = torch.cat([new_state, sampled_goal], dim=1)
            processed_new_state_goal = self.obs_critic_rms(new_state_goal)

            assert processed_new_state_goal.shape == processed_state_goal.shape

            # predict the next action chunk
            next_action_chunk = self.ema_model(processed_new_state_goal)
            next_q1, next_q2 = self.diffusion_critic_target(processed_new_state_goal, next_action_chunk)
            target_q += ((self.discount ** self.chunk_size) * env_not_dones * torch.min(next_q1, next_q2)).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_loss = torch.mean(critic_loss)

            # Step the loss for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_critic.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.policy_loss(sampled_act_chunk, processed_state_goal)
            bc_loss = torch.mean(bc_loss)

            new_act_chunk = self.diffusion_actor(processed_state_goal)
            q1_new_action, q2_new_action = self.diffusion_critic(processed_state_goal, new_act_chunk)

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            # total actor loss
            actor_loss = bc_loss + self.eta * q_loss

            # Step the loss for actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_actor.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.diffusion_critic.parameters(),
                                           self.diffusion_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            metric['planner_actor_loss'].append(actor_loss)
            metric['planner_bc_loss'].append(bc_loss)
            metric['planner_ql_loss'].append(q_loss)
            metric['planner_critic_loss'].append(critic_loss)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric