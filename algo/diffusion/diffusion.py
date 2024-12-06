import torch
import torch.nn as nn

from algo.diffusion.mlp import MLP
from algo.diffusion.auxiliary import (linear_beta_schedule, cosine_beta_schedule, vp_beta_schedule,
                                      extract_into_tensor, WeightedL1, WeightedL2, ValueL1, ValueL2)
from utils.utils import Progress, Silent



class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            input_dim,
            output_dim,
            output_bound,
            num_timesteps,
            device,
            clip_denoised=True,
            predict_epsilon=True,
            beta_schedule="cosine",
            loss="l2",

    ):
        super(GaussianDiffusion, self).__init__()

        # MLP model
        self.model = model

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_timesteps = num_timesteps
        self.predict_epsilon = predict_epsilon

        self.clip_denoised = clip_denoised
        self.output_bound = output_bound
        if self.output_bound is None:
            assert not self.clip_denoised, "Cannot clip denoised output if output bound is not set"

        if loss == "Actorl1":
            self.loss_function = WeightedL1()
        elif loss == "Actorl2":
            self.loss_function = WeightedL2()
        elif loss == "Criticl1":
            self.loss_function = ValueL1()
        elif loss == "Criticl2":
            self.loss_function = ValueL2()
        else:
            raise ValueError(f"Invalid loss function: {loss}")

        self.device = device

        # Create noise variance beta_t
        if beta_schedule == 'linear':
            beta_t = linear_beta_schedule(num_timesteps)
        elif beta_schedule == 'cosine':
            beta_t = cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'vp':
            beta_t = vp_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        alpha_t = 1.0 - beta_t
        alpha_t_cumprod = torch.cumprod(alpha_t, axis=0)
        alpha_t_cumprod_prev = torch.cat([torch.ones(1), alpha_t_cumprod[:-1]])

        # Store the alpha_t and beta_t values
        self.register_buffer("alpha_t_cumprod", alpha_t_cumprod)
        self.register_buffer("alpha_t_cumprod_prev", alpha_t_cumprod_prev)
        self.register_buffer('beta_t', beta_t)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alpha_t_cumprod', torch.sqrt(alpha_t_cumprod))
        self.register_buffer('sqrt_recip_alpha_t_cumprod',
                             torch.sqrt(1. / alpha_t_cumprod))
        self.register_buffer('sqrt_recip_alpha_t_cumprod_minus_one',
                             torch.sqrt(1. / alpha_t_cumprod - 1))
        self.register_buffer('sqrt_one_minus_alpha_t_cumprod',
                             torch.sqrt(1. - alpha_t_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # Clipped log variance for numerical stability
        # As the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20))
        )
        self.register_buffer(
            'posterior_mean_coef1',
            beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)
        )


    # ------------------------------------------ sampling ------------------------------------------#


    def predict_start_from_noise(self, x_t: torch.tensor, t: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        Reconstruct x_0 using x_t, t and noise. Uses deterministic process

        If self.predict_epsilon, model output is (scaled) noise.
        Otherwise, model predicts x0 directly.

        """
        if self.predict_epsilon:
            return (
                extract_into_tensor(self.sqrt_recip_alpha_t_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recip_alpha_t_cumprod_minus_one, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean, variance and clipped log variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        # Compute the mean of the posterior (weighted combination of x_0 and x_t)
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        clipped_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, clipped_log_variance

    def p_mean_variance(self, x, t, s):
        """
        Compute the mean and variance of the diffusion posterior p(x_t | x_0, \hat{x}_0)
        """
        x_reconstructed = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_reconstructed.clamp_(-self.output_bound, self.output_bound)

        return self.q_posterior(x_start=x_reconstructed, x_t=x, t=t)

    def p_sample(self, x, t, s):
        """
        Progressively remove noise from a noisy sample x_t to get x_{t-1}
        """
        # Extract the batch size from the input tensor
        b, *_, device = *x.shape, x.device

        mean, _, clipped_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        # Sample noise from a normal distribution
        noise = torch.randn_like(x)
        # Mask noise when t=0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(x.shape) - 1))
        return mean + nonzero_mask * (0.5 * clipped_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):

        b = shape[0]
        x = torch.randn(shape, device=self.device)

        # Initialize the diffusion tensor buffer if required
        if return_diffusion:
            diffusion_buf = [x]

        progress = Progress(self.num_timesteps) if verbose else Silent()
        # Loop over the timesteps in reverse order
        for i in reversed(range(self.num_timesteps)):
            time_steps = torch.full((b,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, time_steps, state)

            # Update the progress bar
            progress.update({'t': i})

            # Store the diffusion tensor if required
            if return_diffusion:
                diffusion_buf.append(x)
        progress.close()

        return x, torch.stack(diffusion_buf, dim=1) if return_diffusion else x

    def sample(self, state, *args, **kwargs):
        """
        Sample from the diffusion model to get the predicted action given certain state
        """
        b = state.shape[0]
        output_shape = (b, self.output_dim)
        pred_output, _ = self.p_sample_loop(state, output_shape, *args, **kwargs)

        if self.clip_denoised:
            return pred_output.clamp_(-self.output_bound, self.output_bound)
        else:
            return pred_output


    # ------------------------------------------ training ------------------------------------------#


    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the diffusion model q(x_t | x_{t-1})
        to add noise to the input tensor and get the noisy sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sampled_q = (
            extract_into_tensor(self.sqrt_alpha_t_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alpha_t_cumprod, t, x_start.shape) * noise
        )
        return sampled_q

    def p_losses(self, x_start, state, t, weights=1.0):
        """
        Compute the loss of the diffusion model
        """
        noise = torch.randn_like(x_start)

        # Sample from the diffusion model
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # Predict the noise using the model
        x_reconstructed = self.model(x_noisy, t, state)

        assert x_reconstructed.shape == noise.shape

        if self.predict_epsilon:
            # Compute the loss between the predicted noise and the actual noise
            loss = self.loss_function(x_reconstructed, noise, weights)
        else:
            # Compute the loss between the predicted x_t and the actual x_t
            loss = self.loss_function(x_reconstructed, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        """
        Compute the loss of the diffusion model in a batch with random timesteps t
        """
        batch_size = len(x)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        """
        Forward pass of the diffusion model
        """
        return self.sample(state, *args, **kwargs)




if __name__ == "__main__":
    input_dim = 64
    output_dim = 32
    time_steps = 100

    model = GaussianDiffusion(
        model=None,
        input_dim=input_dim,
        output_dim=output_dim,
        output_bound=1.0,
        num_timesteps=time_steps,
        device='cuda',
        beta_schedule="cosine"
    )


