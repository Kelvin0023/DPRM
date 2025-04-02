import torch
import numpy as np
from rl_games.common.common_losses import actor_loss
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from algo.diffusion.auxiliary import EMA


class DiffusionQLTrainer(object):
    def __init__(
        self,
        cfg,
        env,
        replay_buffer,
        bc_replay_buffer,
        model,
        obs_policy_rms,
        obs_critic_rms,
        value_rms,
        device,
    ):
        # dimenstion settings
        self.obs_dim = model.obs_policy_dim
        self.state_dim = model.obs_critic_dim
        self.action_dim = model.action_dim
        self.action_horizon = model.action_horizon
        self.time_steps = model.actor.num_timesteps

        # normalization
        self.obs_policy_rms = obs_policy_rms
        self.obs_critic_rms = obs_critic_rms
        self.value_rms = value_rms
        self.normalize_value = cfg["normalize_value"]
        self.device = device

        self.value_scale = cfg.get("value_scale", 1.0)

        # get the simulation environment and networks
        self.env = env
        self.model = model
        self.diffusion_actor = model.actor
        self.mlp_critic = model.critic
        self.mlp_critic_target = model.critic_target

        # replay buffer
        self.replay_buffer = replay_buffer
        self.bc_replay_buffer = bc_replay_buffer

        # EMA settings
        self.step = 0
        self.step_start_ema = cfg["step_start_ema"]
        self.ema = EMA(cfg["ema_decay"])
        self.ema_model = model.actor_target
        self.update_ema_every = cfg["update_ema_every"]

        # training settings
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
            self.mlp_critic.parameters(), self.last_lr, weight_decay=self.weight_decay
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

    def policy_loss(self, action_sequence, obs_policy, weights=1.0):
        t = torch.randint(0, self.time_steps, (self.batch_size,), device=self.device).long()
        return self.diffusion_actor.p_losses(action_sequence, obs_policy, t, weights)

    def train(self):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}

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

            # """ Critic Q Training """
            current_q1, current_q2 = self.mlp_critic(norm_obs_critic, sampled_act_chunk)

            # predict action chunk for the next time step
            norm_obs_policy_prim = self.obs_policy_rms(sampled_obs_policy_prime)
            norm_obs_critc_prim = self.obs_critic_rms(sampled_obs_critic_prime)

            next_action_chunk = self.ema_model(norm_obs_policy_prim)
            next_q1, next_q2 = self.mlp_critic_target(norm_obs_critc_prim, next_action_chunk)
            if self.normalize_value:
                self.value_rms.eval()
                unnorm_next_q1 = self.value_rms(next_q1, unnorm=True)
                unnorm_next_q2 = self.value_rms(next_q2, unnorm=True)
                unnorm_target_q = sampled_reward_sum + sampled_env_not_done * (self.discount ** self.action_horizon) * torch.min(unnorm_next_q1, unnorm_next_q2)
                target_q = self.value_rms(unnorm_target_q)
            else:
                scaled_next_q1 = next_q1 * self.value_scale
                scaled_next_q2 = next_q2 * self.value_scale
                scaled_target_q = sampled_reward_sum + sampled_env_not_done * (self.discount ** self.action_horizon) * torch.min(scaled_next_q1, scaled_next_q2)
                target_q = scaled_target_q / self.value_scale

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_loss = torch.mean(critic_loss)

            # Step the loss for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.mlp_critic.parameters(),
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

            new_act_chunk = self.model(norm_obs_policy)
            q1_new_action, q2_new_action = self.mlp_critic(norm_obs_critic, new_act_chunk)

            # Value Unnormalization
            if self.normalize_value:
                q1_new_action = self.value_rms(q1_new_action, unnorm=True)
                q2_new_action = self.value_rms(q2_new_action, unnorm=True)
            else:
                q1_new_action *= self.value_scale
                q2_new_action *= self.value_scale

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            q_loss = torch.mean(q_loss)

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

            for param, target_param in zip(self.mlp_critic.parameters(), self.mlp_critic_target.parameters()):
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