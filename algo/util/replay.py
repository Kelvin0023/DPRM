import numpy as np
import random
import torch


class BCReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        """
        Replay buffer for Behavioral Cloning

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            batch_size (int): Size of the minibatch for sampling.
            device (torch.device): Device to store and use tensors (CPU/GPU).
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        # Internal buffer to hold experience tuples
        self.memory = []
        self.position = 0

    def store(self, obs_policy, action):
        """
        Store a batch of transitions in the replay buffer.

        Args:
            obs_policy (np.ndarray or torch.Tensor): Batch of current obs.
            actions (np.ndarray or torch.Tensor): Batch of actions chunks.
        """
        batch_size = len(obs_policy)

        # Loop over the batch and add each transition to the buffer
        for i in range(batch_size):
            if len(self.memory) < self.buffer_size:
                self.memory.append(None)

            self.memory[self.position] = (obs_policy[i].cpu(), action[i].cpu())
            self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
            A tuple of batches (states, actions, rewards, next_states, dones), each of which is a tensor.
        """
        batch = random.sample(self.memory, self.batch_size)

        # Unpacking the experience tuples
        obs_policy, action = zip(*batch)

        # Convert them to torch tensors and move to the appropriate device
        obs_policy = torch.stack(obs_policy).to(self.device)
        action = torch.stack(action).to(self.device)

        return obs_policy, action

    def __len__(self):
        """
        Return the current size of the internal memory (buffer).
        """
        return len(self.memory)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        """
        Replay buffer for Q-learning

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            batch_size (int): Size of the minibatch for sampling.
            device (torch.device): Device to store and use tensors (CPU/GPU).
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        # Internal buffer to hold experience tuples
        self.memory = []
        self.position = 0

    def store(self, obs_policy, obs_critic, action, reward_sum, env_not_done, obs_policy_prime, obs_critic_prime):
        """
        Store a batch of transitions in the replay buffer.

        Args:
            states (np.ndarray or torch.Tensor): Batch of current states.
            actions (np.ndarray or torch.Tensor): Batch of actions taken.
            rewards (np.ndarray or torch.Tensor): Batch of rewards received.
            next_states (np.ndarray or torch.Tensor): Batch of next states observed.
            dones (np.ndarray or torch.Tensor): Batch of done flags indicating episode termination.
        """
        batch_size = len(obs_policy)

        # Loop over the batch and add each transition to the buffer
        for i in range(batch_size):
            if len(self.memory) < self.buffer_size:
                self.memory.append(None)

            self.memory[self.position] = (obs_policy[i].cpu(), obs_critic[i].cpu(), action[i].cpu(), reward_sum[i].cpu(), env_not_done[i].cpu(), obs_policy_prime[i].cpu(), obs_critic_prime[i].cpu())
            self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
            A tuple of batches (states, actions, rewards, next_states, dones), each of which is a tensor.
        """
        batch = random.sample(self.memory, self.batch_size)

        # Unpacking the experience tuples
        obs_policy, obs_critic, action, reward_sum, env_not_done, obs_policy_prime, obs_critic_prime = zip(*batch)

        # Convert them to torch tensors and move to the appropriate device
        obs_policy = torch.stack(obs_policy).to(self.device)
        obs_critic = torch.stack(obs_critic).to(self.device)
        action = torch.stack(action).to(self.device)
        reward_sum = torch.stack(reward_sum).squeeze().to(self.device)
        env_not_done = torch.stack(env_not_done).squeeze().to(self.device)
        obs_policy_prime = torch.stack(obs_policy_prime).to(self.device)
        obs_critic_prime = torch.stack(obs_critic_prime).to(self.device)

        return obs_policy, obs_critic, action, reward_sum, env_not_done, obs_policy_prime, obs_critic_prime

    def __len__(self):
        """
        Return the current size of the internal memory (buffer).
        """
        return len(self.memory)
