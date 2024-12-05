import torch
from torch.nn import functional as F


class TD3Trainer:
    def __init__(
        self,
        cfg,
        replay_buffer,
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
        self.cfg = cfg
        self.device = device
        self.replay_buffer = replay_buffer

        # model settings
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target

        # normalization settings
        self.obs_policy_rms = obs_policy_rms
        self.obs_critic_rms = obs_critic_rms
        self.value_rms = value_rms

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.batch_size = cfg["batch_size"]
        self.max_action = cfg["max_action"]
        self.discount = cfg["discount"]
        self.tau = cfg["tau"]
        self.policy_noise = cfg["policy_noise"]
        self.noise_clip = cfg["noise_clip"]
        self.policy_freq = cfg["policy_freq"]

        self.total_it = 0
        self.prev_actor_loss = torch.tensor(0.0).to(self.device)

    def train(self):
        metrics = {'actor_loss': [], 'critic_loss': []}
        self.total_it += 1

        # Sample data from replay buffer
        (
            obs_policy,
            obs_critic,
            action,
            reward,
            env_not_done,
            obs_policy_prime,
            obs_critic_prime
        ) = self.replay_buffer.sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(obs_policy_prime) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(obs_critic_prime, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward.squeeze() + env_not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs_critic, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(obs_critic, self.actor(obs_policy)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.prev_actor_loss = actor_loss
        else:
            actor_loss = self.prev_actor_loss

        # Update training metrics
        metrics['actor_loss'].append(actor_loss)
        metrics['critic_loss'].append(critic_loss)

        return metrics
