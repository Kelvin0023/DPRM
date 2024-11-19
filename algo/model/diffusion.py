import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.model.auxiliary import (
    linear_beta_schedule,
    cosine_beta_schedule,
    vp_beta_schedule,
    extract_into_tensor,
    make_timesteps,
)

from collections import namedtuple
Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):
    def __init__(
        self,
        network,
        action_horizon,
        obs_policy_dim,
        action_dim,
        device="cuda:0",
        # Noise schedule
        beta_schedule="cosine",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize='uniform',
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()

        # Model parameters
        self.action_horizon = action_horizon
        self.obs_policy_dim = obs_policy_dim
        self.action_dim = action_dim
        self.device = device

        # DDPM parameters
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon

        # DDIM parameters
        self.use_ddim = use_ddim
        self.ddim_steps = int(ddim_steps) if ddim_steps is not None else None

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Clamp the final sampled action (usually between [-1, 1])
        self.final_action_clip_value = final_action_clip_value

        # Clip sampled randn (from standard deviation) for each denoising step
        # such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # Load network
        self.network = network.to(device)

        """
        DDPM parameters
        """

        """
        βₜ
        """
        # Create noise variance beta_t
        if beta_schedule == 'linear':
            self.beta_t = linear_beta_schedule(denoising_steps).to(device)
        elif beta_schedule == 'cosine':
            self.beta_t = cosine_beta_schedule(denoising_steps).to(device)
        elif beta_schedule == 'vp':
            self.beta_t = vp_beta_schedule(denoising_steps).to(device)
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")
        """
        αₜ = 1 - βₜ
        """
        self.alpha_t = 1.0 - self.beta_t
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alpha_t_cumprod = torch.cumprod(self.alpha_t, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alpha_t_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alpha_t_cumprod[:-1]]
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alpha_t_cumprod = torch.sqrt(self.alpha_t_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alpha_t_cumprod = torch.sqrt(1.0 - self.alpha_t_cumprod)

        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alpha_t_cumprod = torch.sqrt(1.0 / self.alpha_t_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alpha_t_cumprod = torch.sqrt(1.0 / self.alpha_t_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = self.beta_t * (1.0 - self.alpha_t_cumprod_prev) / (1.0 - self.alpha_t_cumprod)
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = self.beta_t * torch.sqrt(self.alpha_t_cumprod_prev) / (1.0 - self.alpha_t_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alpha_t_cumprod_prev) * torch.sqrt(self.alpha_t) / (1. - self.alpha_t_cumprod)

        """
        DDIM parameters

        In DDIM paper https://arxiv.org/pdf/2010.02502, alpha is alpha_cumprod in DDPM https://arxiv.org/pdf/2102.09672
        """
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == 'uniform':  # use the HF "leading" style
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = torch.arange(0, ddim_steps, device=self.device) * step_ratio
            else:
                raise 'Unknown discretization method for DDIM.'
            self.ddim_alpha_t = self.alpha_t_cumprod[self.ddim_t].clone().to(torch.float32)
            self.ddim_alpha_t_sqrt = torch.sqrt(self.ddim_alpha_t)
            self.ddim_alpha_t_prev = torch.cat([
                torch.tensor([1.]).to(torch.float32).to(self.device),
                self.alpha_t_cumprod[self.ddim_t[:-1]]])
            self.ddim_sqrt_one_minus_alpha_t = (1.0 - self.ddim_alpha_t) ** 0.5

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (ddim_eta * \
                                ((1 - self.ddim_alpha_t_prev) / (1 - self.ddim_alpha_t) * \
                                 (1 - self.ddim_alpha_t / self.ddim_alpha_t_prev)) ** .5)

            # Flip all
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alpha_t = torch.flip(self.ddim_alpha_t, [0])
            self.ddim_alpha_t_sqrt = torch.flip(self.ddim_alpha_t_sqrt, [0])
            self.ddim_alpha_t_prev = torch.flip(self.ddim_alpha_t_prev, [0])
            self.ddim_sqrt_one_minus_alpha_t = torch.flip(self.ddim_sqrt_one_minus_alpha_t, [0])
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    # ---------- Sampling ----------#

    def p_mean_var(self, x, t, cond, index=None):
        """
        Compute the mean and variance of the diffusion posterior p(x_t | x_0, \hat{x}_0)
        """
        noise = self.network(x, t, cond=cond)
        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract_into_tensor(self.ddim_alpha_t, index, x.shape)
                alpha_prev = extract_into_tensor(self.ddim_alpha_t_prev, index, x.shape)
                sqrt_one_minus_alpha = extract_into_tensor(self.ddim_sqrt_one_minus_alpha_t, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha ** 0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                        extract_into_tensor(self.sqrt_recip_alpha_t_cumprod, t, x.shape) * x
                        - extract_into_tensor(self.sqrt_recipm1_alpha_t_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε 

            eta=0
            """
            sigma = extract_into_tensor(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma ** 2).sqrt() * noise
            mu = (alpha_prev ** 0.5) * x_recon + dir_xt
            var = sigma ** 2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                    extract_into_tensor(self.ddpm_mu_coef1, t, x.shape) * x_recon
                    + extract_into_tensor(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract_into_tensor(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=False, return_chain=False):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        device = self.beta_t.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Loop
        x = torch.randn((B, self.action_horizon, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = torch.zeros_like(std)
            else:
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

        return Sample(x, None)

    # ---------- Supervised training ----------#

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, cond, t)

    def p_losses(
        self,
        x_start,
        cond: dict,
        t,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, action_horizon, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean")
        else:
            return F.mse_loss(x_recon, x_noisy, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = x_start.device
            noise = torch.randn_like(x_start, device=device)
        return (
                extract_into_tensor(self.sqrt_alpha_t_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alpha_t_cumprod, t, x_start.shape) * noise
        )
