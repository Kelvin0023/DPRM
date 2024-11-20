import torch
from torch.utils.data import Dataset
from algo.util.experience import transform_op


class ChunkExperienceBuffer(Dataset):
    """
    A class for storing on-policy rollouts which produce action chunks.

    Args:
        num_envs (int): Number of environments.
        horizon_length (int): Length of the horizon.
        obs_critic_dim (int): Dimension of the state.
        obs_policy_dim (int): Dimension of the observation.
        act_dim (int): Dimension of the action.
        action_horizon (int): Size of the action chunk.
        denoising_steps (int): Number of denoising steps in the diffusion model.
        device (str): Device to store the data on.

    Attributes:
        device (str): Device to store the data on.
        num_envs (int): Number of environments.
        transitions_per_env (int): Number of transitions per environment.
        batch_size (int): Batch size.
        data_dict (dict): Dictionary to store the data.
        obs_critic_dim (int): Dimension of the state.
        obs_policy_dim (int): Dimension of the observation.
        act_dim (int): Dimension of the action.
        action_horizon (int): Size of the action chunk.
        denoising_steps (int): Number of denoising steps in the diffusion model.
        storage_dict (dict): Dictionary to store the data.
        minibatch_size (int): Size of the minibatch.
        length (int): Length of the dataset.
        last_range (tuple): Range of the last batch.
    """

    def __init__(
        self,
        num_envs,
        horizon_length,
        obs_critic_dim,
        obs_policy_dim,
        act_dim,
        action_horizon,
        denoising_steps,
        device,
    ):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.batch_size = self.transitions_per_env * self.num_envs

        self.action_horizon = action_horizon
        self.denoising_steps = denoising_steps

        self.data_dict = None
        self.obs_critic_dim = obs_critic_dim
        self.obs_policy_dim = obs_policy_dim
        self.act_dim = act_dim
        self.storage_dict = {
            "obs_critic": torch.zeros((self.transitions_per_env, self.num_envs, self.obs_critic_dim), dtype=torch.float32, device=self.device),
            "obs_policy": torch.zeros((self.transitions_per_env, self.num_envs, self.obs_policy_dim), dtype=torch.float32, device=self.device),
            "rewards": torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            "values": torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            "neglogpacs": torch.zeros((self.transitions_per_env, self.num_envs, self.denoising_steps, self.action_horizon, self.act_dim), dtype=torch.float32, device=self.device),
            "dones": torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.uint8, device=self.device),
            "actions": torch.zeros((self.transitions_per_env, self.num_envs, self.action_horizon, self.act_dim), dtype=torch.float32, device=self.device),
            "returns": torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            "chains": torch.zeros((self.transitions_per_env, self.num_envs, self.denoising_steps + 1, self.action_horizon, self.act_dim), dtype=torch.float32, device=self.device),
        }

    def set_minibatch_size(self, minibatch_size):
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.data_dict.items():
            if isinstance(v, dict):
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        return (
            input_dict["values"],
            input_dict["neglogpacs"],
            input_dict["advantages"],
            input_dict["returns"],
            input_dict["actions"],
            input_dict["obs_policy"],
            input_dict["obs_critic"],
            input_dict["chains"],
        )

    def update_data(self, name, index, val):
        if isinstance(val, dict):
            for k, v in val.items():
                self.storage_dict[name][k][index, :] = v
        else:
            self.storage_dict[name][index, :] = val

    def computer_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict["rewards"])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict["values"][t + 1]
            next_nonterminal = 1.0 - self.storage_dict["dones"].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = self.storage_dict["rewards"][t] + (gamma ** self.action_horizon) * next_values * next_nonterminal - \
                    self.storage_dict["values"][t]
            mb_advs[t] = last_gae_lam = delta + (gamma ** self.action_horizon) * tau * next_nonterminal * last_gae_lam
            self.storage_dict["returns"][t, :] = mb_advs[t] + self.storage_dict["values"][t]

    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        advantages = self.data_dict["returns"] - self.data_dict["values"]
        self.data_dict["advantages"] = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)
        return self.data_dict

    def clear(self):
        self.data_dict = None