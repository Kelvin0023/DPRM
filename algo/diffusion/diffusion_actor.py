import torch
import torch.nn as nn

from algo.diffusion.diffusion import GaussianDiffusion
from algo.diffusion.mlp import MLP

class DiffusionActor(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            action_bound,
            chunk_size,
            device,
            beta_schedule='linear',
            num_timesteps=100,
            ):
        super().__init__()

        # Parameters for the Behavior Cloning (BC) model
        self.obs_dim = obs_dim
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device

        # MLP for the diffusion model
        self.mlp_model = MLP(
            input_dim=obs_dim,
            output_dim=action_dim * chunk_size,
            time_emb_dim=16,
            mlp_hidden_dim=512,
            device=device
        )

        # Diffusion model actor
        self.actor = GaussianDiffusion(
            model=self.mlp_model,
            input_dim=obs_dim,
            output_dim=action_dim * chunk_size,
            output_bound=action_bound,
            num_timesteps=num_timesteps,
            device=device,
            predict_epsilon=True,
            beta_schedule=beta_schedule,
            loss="Actorl2",
        ).to(self.device)

    def forward(self, obs):
        action_chunks = self.actor.sample(obs)
        return action_chunks.reshape(-1, self.chunk_size, self.action_dim)

    def sample_action_chunks(self, obs):
        with torch.no_grad():
            action_chunks = self.actor.sample(obs)
        return action_chunks.reshape(-1, self.chunk_size, self.action_dim)