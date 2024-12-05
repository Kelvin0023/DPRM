import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_policy_dim, action_dim, max_action=1.0, units=[1024, 1024, 1024]):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_policy_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], units[2])
        self.l4 = nn.Linear(units[2], action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.max_action * torch.tanh(self.l4(x))


class Critic(nn.Module):
    def __init__(self, obs_critic_dim, action_dim, units=[256, 256, 256]):
        super(Critic, self).__init__()
        # Q1 network
        self.l1 = nn.Linear(obs_critic_dim + action_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], units[2])
        self.l4 = nn.Linear(units[2], 1)

        # Q2 network
        self.l5 = nn.Linear(obs_critic_dim + action_dim, units[0])
        self.l6 = nn.Linear(units[0], units[1])
        self.l7 = nn.Linear(units[1], units[2])
        self.l8 = nn.Linear(units[2], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        # Q1 network
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        # Q2 network
        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        return self.l4(q1)

class TD3(nn.Module):
    def __init__(
        self,
        actor,
        actor_target,
        critic,
        critic_target,
        max_action=1,
        policy_noise=0.2,
        noise_clip=0.5,
    ):
        super().__init__()
        # model device

        # Actor and critic model
        self.actor = actor
        self.actor_target = actor_target

        self.critic = critic
        self.critic_target = critic_target

        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def act(self, obs_policy):
        pred_action = self.actor_target(obs_policy)
        noise = (torch.randn_like(pred_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        return (pred_action + noise).clamp(-self.max_action, self.max_action)

    @torch.no_grad()
    def act_inference(self, obs_policy):
        return self.actor_target(obs_policy).clamp(-self.max_action, self.max_action)