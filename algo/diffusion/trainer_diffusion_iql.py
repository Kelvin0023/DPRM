import torch
import numpy as np
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from algo.diffusion.auxiliary import EMA


def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class ImplicitDiffusionQLTrainer(object):
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
        self.chunk_size = model.chunk_size
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
        self.diffusion_actor = model.actor
        self.mlp_critic_v = model.critic_v
        self.mlp_critic_q = model.critic_q
        self.mlp_critic_target_q = model.critic_q_target

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
        self.critic_q_optimizer = torch.optim.Adam(
            self.mlp_critic_q.parameters(), self.last_lr, weight_decay=self.weight_decay
        )
        self.critic_v_optimizer = torch.optim.Adam(
            self.mlp_critic_v.parameters(), self.last_lr, weight_decay=self.weight_decay
        )

        # learning rate decay
        self.lr_decay = cfg["lr_decay"]
        if self.lr_decay:
            self.lr_max_T = cfg["lr_max_T"]
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer,
                T_max=self.lr_max_T,
                eta_min=0.0,
            )
            self.critic_q_lr_scheduler = CosineAnnealingLR(
                self.critic_q_optimizer,
                T_max=self.lr_max_T,
                eta_min=0.0,
            )
            self.critic_v_lr_scheduler = CosineAnnealingLR(
                self.critic_v_optimizer,
                T_max=self.lr_max_T,
                eta_min=0.0,
            )

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_actor)

    def update_target_cirtic(self, tau):
        for target_param, param in zip(self.mlp_critic_target_q.parameters(), self.mlp_critic_q.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def compute_advantages(self, obs, action_chunk):
        # get current Q-function, stop gradient
        with torch.no_grad():
            current_q1, current_q2 = self.mlp_critic_target_q(obs, action_chunk)
        q = torch.min(current_q1, current_q2)

        # get the current V-function
        v = self.mlp_critic_v(obs).reshape(-1)

        # compute advantage
        adv = q - v
        return adv

    def critic_v_loss(self, obs_critic, action_chunk):
        adv = self.compute_advantages(obs_critic, action_chunk)

        # get the value loss
        v_loss = expectile_loss(adv).mean()
        return v_loss

    def critic_q_loss(self, obs_critic, next_obs_critic, action_chunks, rewards, not_terminated, gamma):
        # get current Q-function
        current_q1, current_q2 = self.mlp_critic_q(obs_critic, action_chunks)

        # get the next V-function, stop gradient
        with torch.no_grad():
            next_v = self.mlp_critic_v(next_obs_critic)

        # terminal state mask
        mask = not_terminated

        # flatten
        rewards = rewards.view(-1)
        next_v = next_v.view(-1)
        mask = mask.view(-1)

        # compute target value
        if self.normalize_value:
            # unnorm the q and v values
            self.value_rms.eval()
            next_v = self.value_rms(next_v, unnorm=True)
            # compute target value
            discounted_q = rewards + gamma * next_v * mask
            # normalize the target value
            discounted_q = self.value_rms(discounted_q)
        else:
            # scale up the q and v values
            next_v = next_v * self.value_scale
            # compute target value
            discounted_q = rewards + gamma * next_v * mask
            # scale down the target value
            discounted_q = discounted_q / self.value_scale

        # Update critic
        q_loss = torch.mean((current_q1 - discounted_q) ** 2) + torch.mean(
            (current_q2 - discounted_q) ** 2
        )
        return q_loss

    def policy_loss(self, action_chunk, obs_policy, weights=1.0):
        t = torch.randint(0, self.time_steps, (self.batch_size,), device=self.device).long()
        flatten_action = action_chunk.view(-1, self.chunk_size * self.action_dim)
        return self.diffusion_actor.p_losses(flatten_action, obs_policy, t, weights)

    def train(self):
        metric = {'policy_loss': [], 'critic_v_loss': [], 'critic_q_loss': []}

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

            # Normalize the observations
            norm_obs_policy = self.obs_policy_rms(sampled_obs_policy)
            norm_obs_critic = self.obs_critic_rms(sampled_obs_critic)
            norm_obs_critc_next = self.obs_critic_rms(sampled_obs_critic_prime)

            """ Critic V Training """
            critic_loss_v = self.critic_v_loss(norm_obs_critic, sampled_act_chunk)
            # Step the loss for critic V network
            self.critic_v_optimizer.zero_grad()
            critic_loss_v.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.mlp_critic_v.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.critic_v_optimizer.step()

            # Update metric
            metric['critic_v_loss'].append(critic_loss_v)

            """ Critic Q Training """
            critic_loss_q = self.critic_q_loss(
                norm_obs_critic,
                norm_obs_critc_next,
                sampled_act_chunk,
                sampled_reward_sum,
                sampled_env_not_done,
                self.discount ** self.chunk_size,
            )
            # Step the loss for critic Q network
            self.critic_q_optimizer.zero_grad()
            critic_loss_q.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.mlp_critic_q.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.critic_q_optimizer.step()

            # Update metric
            metric['critic_q_loss'].append(critic_loss_q)

            """ Policy Training """
            policy_loss = self.policy_loss(sampled_act_chunk, norm_obs_policy)
            # Step the loss for policy network
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_actor.parameters(),
                    max_norm=self.grad_norm,
                    norm_type=2
                )
            self.actor_optimizer.step()

            # Update metric
            metric['policy_loss'].append(policy_loss)

            """ Step Target network """
            # Update EMA actor model
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Update target critic q network
            self.update_target_cirtic(self.tau)

            # Increment step count
            self.step += 1

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_q_lr_scheduler.step()
            self.critic_v_lr_scheduler.step()

        return metric