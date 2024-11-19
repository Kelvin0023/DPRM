import torch
import numpy as np
from torch.nn import functional as F

from algo.model.auxiliary import EMA
from algo.util.scheduler import CosineAnnealingWarmupRestarts



class DiffQLTrainer:
    def __init__(
        self,
        cfg,
        replay_buffer,
        model,
        model_target,
        obs_policy_rms,
        obs_critic_rms,
        value_rms,
        device
    ):
        # environment settings
        self.cfg = cfg
        self.device = device
        self.obs_dim = model.obs_policy_dim
        self.action_horizon = model.action_horizon
        self.time_steps = model.denoising_steps

        # model settings
        self.model = model
        self.model_target = model_target
        self.critic_target = model_target.critic

        # EMA for target actor
        self.step = 0
        self.step_start_ema = cfg["step_start_ema"]
        self.ema = EMA(cfg["ema_decay"])
        self.ema_model = model_target.actor
        self.update_ema_every = cfg["update_ema_every"]
        self.tau = cfg["tau"]  # target network update rate

        # normalization
        self.obs_policy_rms = obs_policy_rms
        self.obs_critic_rms = obs_critic_rms
        self.value_rms = value_rms
        self.value_scale = cfg.get("value_scale", 1.0)

        # replay buffer
        self.replay_buffer = replay_buffer

        # whether to apply max q backup
        self.max_q_backup = cfg["max_q_backup"]

        # batch size
        self.batch_size = cfg["batch_size"]
        # training epochs
        self.epochs = cfg["epochs"]

        # Q-learning settings
        self.eta = cfg["eta"]  # ql_loss weight
        self.discount = cfg["discount"]  # discount factor for Q-learning

        # Gradient clipping
        self.max_grad_norm = cfg.get("max_grad_norm", None)

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg["actor_lr"],
            weight_decay=cfg["actor_weight_decay"],
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg["critic_lr"],
            weight_decay=cfg["critic_weight_decay"],
        )

        # Cosine scheduler with linear warmup
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg["actor_lr_scheduler"]["first_cycle_steps"],
            cycle_mult=1.0,
            max_lr=cfg["actor_lr"],
            min_lr=cfg["actor_lr_scheduler"]["min_lr"],
            warmup_steps=cfg["actor_lr_scheduler"]["warmup_steps"],
            gamma=1.0,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg["critic_lr_scheduler"]["first_cycle_steps"],
            cycle_mult=1.0,
            max_lr=cfg["critic_lr"],
            min_lr=cfg["critic_lr_scheduler"]["min_lr"],
            warmup_steps=cfg["critic_lr_scheduler"]["warmup_steps"],
            gamma=1.0,
        )

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.model.actor)

    def train_diffql(self):
        metrics = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}

        for epoch in range(self.epochs):
            (
                sampled_obs_policy,
                sampled_obs_critic,
                sampled_act_chunk,
                sampled_reward_sum,
                sampled_env_not_done,
                sampled_obs_policy_prime,
                sampled_obs_critic_prime
            ) = self.replay_buffer.sample()

            # normalize observations
            norm_obs_policy = self.obs_policy_rms(sampled_obs_policy)
            norm_obs_critic = self.obs_critic_rms(sampled_obs_critic)

            """ Q Training """
            current_q1, current_q2 = self.model.critic(norm_obs_critic, sampled_act_chunk)

            # normalize observations for next state
            norm_obs_policy_prime = self.obs_policy_rms(sampled_obs_policy_prime)
            norm_obs_critic_prime = self.obs_critic_rms(sampled_obs_critic_prime)

            # Max Q Backup
            if self.max_q_backup:
                norm_obs_policy_prime_rpt = torch.repeat_interleave(norm_obs_policy_prime, repeats=10, dim=0)
                norm_obs_critic_prime_rpt = torch.repeat_interleave(norm_obs_critic_prime, repeats=10, dim=0)
                next_action_chunk_rpt = self.model_target.forward_train(
                    cond={"state": norm_obs_policy_prime_rpt},
                    deterministic=False,
                )
                next_q1, next_q2 = self.critic_target(norm_obs_critic_prime_rpt, next_action_chunk_rpt)
                next_q1 = next_q1.view(self.batch_size, 10).max(dim=1, keepdim=True)[0]
                next_q2 = next_q2.view(self.batch_size, 10).max(dim=1, keepdim=True)[0]
            else:
                next_action_chunk = self.model_target.forward_train(
                    cond={"state": norm_obs_policy_prime},
                    deterministic=False,
                )
                next_q1, next_q2 = self.critic_target(norm_obs_critic_prime, next_action_chunk)

            # Normalize values
            if self.value_rms is not None:
                self.value_rms.eval()
                unnorm_next_q1 = self.value_rms(next_q1, unnorm=True)
                unnorm_next_q2 = self.value_rms(next_q2, unnorm=True)
                unnorm_target_q = sampled_reward_sum + sampled_env_not_done * (self.discount ** self.action_horizon) * torch.min(unnorm_next_q1, unnorm_next_q2)
                target_q = self.value_rms(unnorm_target_q)
            else:
                target_q = sampled_reward_sum + sampled_env_not_done * (self.discount ** self.action_horizon) * torch.min(next_q1, next_q2)
                target_q *= self.value_scale
                current_q1 *= self.value_scale
                current_q2 *= self.value_scale

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_loss = torch.mean(critic_loss)

            # step critic optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()


            """ Policy Training """
            bc_loss = self.model.policy_loss(x=sampled_act_chunk, cond={"state": norm_obs_policy})
            bc_loss = torch.mean(bc_loss)

            new_act_chunk = self.model.forward_train(cond={"state": norm_obs_policy}, deterministic=False)
            q1_new_action, q2_new_action = self.model.critic(norm_obs_critic, new_act_chunk)

            # Value Unnormalization
            if self.value_rms is not None:
                q1_new_action = self.value_rms(q1_new_action, unnorm=True)
                q2_new_action = self.value_rms(q2_new_action, unnorm=True)
            else:
                q1_new_action /= self.value_scale
                q2_new_action /= self.value_scale

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
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(),
                    max_norm=self.max_grad_norm,
                    norm_type=2
                )
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.model.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            # Update training metrics
            metrics['actor_loss'].append(actor_loss)
            metrics['bc_loss'].append(bc_loss)
            metrics['ql_loss'].append(q_loss)
            metrics['critic_loss'].append(critic_loss)

        # Update lr, min_sampling_std
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return metrics


