import copy
import torch
import torch.nn as nn

from algo.diffusion.critic import CriticQ
from algo.diffusion.diffusion import GaussianDiffusion
from algo.diffusion.unet1d import ConditionalUnet1D


class DiffusionQL(nn.Module):
    def __init__(
            self,
            actor_mlp_cfg,
            critic_mlp_cfg,
            obs_policy_dim,
            obs_critic_dim,
            action_dim,
            action_bound,
            obs_horizon,
            action_horizon,
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
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.device = device

        # U-Net model for the diffusion actor
        self.unet_model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_policy_dim,
        )

        # Diffusion model actor
        self.actor = GaussianDiffusion(
            model=self.unet_model,
            input_dim=obs_policy_dim,
            input_horizon=obs_horizon,
            output_dim=action_dim,
            output_horizon=action_horizon,
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
            action_steps=self.action_horizon,
            activation_type=critic_mlp_cfg["activation_type"],
            use_layernorm=critic_mlp_cfg["use_layernorm"],
            residual_style=critic_mlp_cfg["residual_style"],
        ).to(self.device)

        # EMA target actor network
        self.actor_target = copy.deepcopy(self.actor)

        # target critic network
        self.critic_target = copy.deepcopy(self.critic)

    def forward(self, obs):
        action_seq = self.actor.sample(obs)
        return action_seq

    def sample_action_chunks(self, obs):
        with torch.no_grad():
            action_seq = self.actor_target.sample(obs)
        return action_seq