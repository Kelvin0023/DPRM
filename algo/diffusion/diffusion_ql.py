import copy
import torch
import torch.nn as nn

from algo.diffusion.critic import CriticQ
from algo.diffusion.diffusion import GaussianDiffusion
from algo.diffusion.mlp import DiffusionMLP

class DiffusionQL(nn.Module):
    def __init__(
            self,
            actor_mlp_cfg,
            critic_mlp_cfg,
            obs_policy_dim,
            obs_critic_dim,
            action_dim,
            action_bound,
            chunk_size,
            device,
            # denoising diffusion parameters
            beta_schedule='linear',
            num_timesteps=100,
            ):
        super().__init__()

        # Parameters for Dimensions
        self.obs_policy_dim = obs_policy_dim
        self.obs_critic_dim = obs_critic_dim
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device

        # MLP for the diffusion actor
        self.mlp_model = DiffusionMLP(
            action_dim=action_dim,
            action_horizon=chunk_size,
            cond_dim=obs_policy_dim,
            time_emb_dim=actor_mlp_cfg["time_dim"],
            mlp_dims=actor_mlp_cfg["mlp_dims"],
            cond_mlp_dims=None,
            activation_type=actor_mlp_cfg["activation_type"],
            out_activation_type="Identity",
            use_layernorm=actor_mlp_cfg["use_layernorm"],
            residual_style=actor_mlp_cfg["residual_style"],
        )

        # Diffusion model actor
        self.actor = GaussianDiffusion(
            model=self.mlp_model,
            input_dim=obs_policy_dim,
            output_dim=action_dim * chunk_size,
            output_bound=action_bound,
            num_timesteps=num_timesteps,
            device=device,
            predict_epsilon=True,
            beta_schedule=beta_schedule,
            loss="Actorl2",
        ).to(self.device)

        # Critic Q model
        self.critic = CriticQ(
            mlp_dims=critic_mlp_cfg["mlp_dims"],
            obs_critic_dim=self.obs_critic_dim,
            action_dim=self.action_dim,
            action_steps=self.chunk_size,
            activation_type=critic_mlp_cfg["activation_type"],
            use_layernorm=critic_mlp_cfg["use_layernorm"],
            residual_style=critic_mlp_cfg["residual_style"],
        ).to(self.device)

        # EMA target actor network
        self.actor_target = copy.deepcopy(self.actor)

        # target critic network
        self.critic_target = copy.deepcopy(self.critic)

    def forward(self, obs):
        action_chunks = self.actor.sample(obs)
        return action_chunks.reshape(-1, self.chunk_size, self.action_dim)

    def sample_action_chunks(self, obs):
        with torch.no_grad():
            action_chunks = self.actor_target.sample(obs)
        return action_chunks.reshape(-1, self.chunk_size, self.action_dim)