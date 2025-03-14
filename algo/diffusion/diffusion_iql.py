import copy
import torch
import torch.nn as nn

from algo.diffusion.critic import CriticQ, CriticV
from algo.diffusion.diffusion import GaussianDiffusion
from algo.diffusion.mlp import DiffusionMLP

class ImplicitDiffusionQL(nn.Module):
    def __init__(
            self,
            actor_mlp_cfg,
            critic_q_mlp_cfg,
            critic_v_mlp_cfg,
            obs_policy_dim,
            obs_critic_dim,
            action_dim,
            action_bound,
            chunk_size,
            device,
            # denoising diffusion parameters
            beta_schedule='linear',
            num_timesteps=100,
            # expectile exploration parameters
            num_sample=10,
            critic_hyperparam=0.7,
            ):
        super().__init__()

        # Parameters for the Behavior Cloning (BC) model
        self.obs_policy_dim = obs_policy_dim
        self.obs_critic_dim = obs_critic_dim
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device

        # expectile exploration
        self.num_sample = num_sample
        self.critic_hyperparam = critic_hyperparam

        # MLP for the diffusion model
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
        self.critic_q = CriticQ(
            mlp_dims=critic_q_mlp_cfg["mlp_dims"],
            obs_critic_dim=self.obs_critic_dim,
            action_dim=self.action_dim,
            action_steps=self.chunk_size,
            activation_type=critic_q_mlp_cfg["activation_type"],
            use_layernorm=critic_q_mlp_cfg["use_layernorm"],
            residual_style=critic_q_mlp_cfg["residual_style"],
        ).to(self.device)

        # Critic V model
        self.critic_v = CriticV(
            mlp_dims=critic_v_mlp_cfg["mlp_dims"],
            obs_critic_dim=self.obs_critic_dim,
            action_dim=self.action_dim,
            action_steps=self.chunk_size,
            activation_type=critic_v_mlp_cfg["activation_type"],
            use_layernorm=critic_v_mlp_cfg["use_layernorm"],
            residual_style=critic_v_mlp_cfg["residual_style"],
        ).to(self.device)

        # EMA target actor network
        self.actor_target = copy.deepcopy(self.actor)

        # target critic Q network
        self.critic_q_target = copy.deepcopy(self.critic_q)

    def forward(
        self,
        obs,
        use_expectile_exploration=False,
    ):
        stacked_obs = obs.unsqueeze(0).repeat(self.num_sample, 1, 1)  # [S, B, D]
        stacked_obs = stacked_obs.view(-1, self.obs_policy_dim)  # [S * B, D]

        # sample action chunks
        stacked_action_chunks = self.actor.sample(stacked_obs)  # [S * B, T * A]

        # compute Q values
        current_q1, current_q2 = self.critic_q(stacked_obs, stacked_action_chunks)  # [S * B, 1]
        q = torch.min(current_q1, current_q2)
        q = q.view(self.num_sample, -1)  # [S, B]

        if use_expectile_exploration:
            # get the current value function for probabilistic exploration
            current_v = self.critic_v(stacked_obs)  # [S * B, 1]
            v = current_v.view(self.num_sample, -1)  # [S, B]
            # compute advantage value
            adv = q - v  # [S, B]

            # compute weights for sampling
            stacked_action_chunks = stacked_action_chunks.view(
                self.num_sample, -1, self.chunk_size, self.action_dim
            )  # [S, B, T, A]

            # expectile exploration policy
            tau_weights = torch.where(adv > 0, self.critic_hyperparam, 1 - self.critic_hyperparam)
            tau_weights = tau_weights / tau_weights.sum(0)  # normalize

            # select a sample from DP probabilistically -- sample index per batch and compile
            sample_idx = torch.multinomial(tau_weights.T, 1)  # [B, 1]

            # dummy dimension @ dim 0 for batched indexing
            sample_idx = sample_idx[None, :, None]  # [1, B, 1, 1]
            sample_idx = sample_idx.repeat(self.num_sample, 1, self.chunk_size, self.action_dim)

            # Fetch the best action chunks
            best_action_chunks = torch.gather(stacked_action_chunks, 0, sample_idx)  # [B, T, A]

        else:
            # gather the best sample -- filter out suboptimal Q during inference
            best_idx = q.argmax(0)  # [B]
            stacked_action_chunks = stacked_action_chunks.view(
                self.num_sample, -1, self.chunk_size, self.action_dim
            )  # [S, B, T, A]
            best_action_chunks_idx = best_idx[None, :, None, None]  # [1, B, 1, 1]
            best_action_chunks_idx = best_action_chunks_idx.repeat(self.num_sample, 1, self.chunk_size, self.action_dim)

            # Fetch the best action chunks
            best_action_chunks = torch.gather(stacked_action_chunks, 0, best_action_chunks_idx)  # [B, T, A]

        return best_action_chunks[0]

    def sample_action_chunks(
        self,
        obs,
        use_expectile_exploration=False,
    ):
        with torch.no_grad():
            return self(obs, use_expectile_exploration)