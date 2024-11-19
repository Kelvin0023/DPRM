import copy
import torch
import numpy as np

from algo.model.diffusion import DiffusionModel
from algo.model.auxiliary import make_timesteps


class DiffusionQL(DiffusionModel):
    def __init__(
        self,
        actor_model,
        critic_model,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__(network=actor_model, **kwargs)
        assert not self.use_ddim, "DQL does not support DDIM"

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Re-name network to actor
        self.actor = self.network

        # Create target critic network
        self.critic = critic_model.to(self.device)

    # ---------- RL training ----------#

    def policy_loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, cond, t)

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(
            self,
            cond,
            deterministic=False,
    ):
        device = self.beta_t.device
        B = len(cond["state"])

        # Loop
        x = torch.randn((B, self.action_horizon, self.action_dim), device=device)
        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )
            std = torch.exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return x

    def forward_train(
        self,
        cond,
        deterministic=False,
    ):
        """
        Differentiable forward pass used in actor training.
        """
        device = self.beta_t.device
        B = len(cond["state"])

        # Loop
        x = torch.randn((B, self.action_horizon, self.action_dim), device=device)
        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )
            std = torch.exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:  # For DDPM, sample with noise
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value and i == len(t_all) - 1:
                x = torch.clamp(x, -1, 1)
        return x