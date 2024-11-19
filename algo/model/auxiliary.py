import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_beta_schedule(time_steps, s=0.008, dtype=torch.float32):
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ

    Smoother and more gradual compared to linear schedules.
    Particularly useful in Variational Diffusion Models (VDM).
    """
    steps = time_steps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, a_min=0, a_max=0.999), dtype=dtype)

def linear_beta_schedule(time_steps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    """
    Linear beta schedule

    Increases noise variance linearly across timesteps.
    Might lead to an uneven distribution of noise over the timesteps
    """
    beta_list = np.linspace(beta_start, beta_end, time_steps)
    return torch.tensor(beta_list, dtype=dtype)

def vp_beta_schedule(time_steps, dtype=torch.float32):
    """
    VP (Variance Preserving) Beta Schedule

    Often used in Variance Preserving Diffusion Models,
    where the model aims to preserve the total variance across timesteps
    """
    t = np.linspace(1, time_steps + 1)
    T = time_steps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

def extract_into_tensor(a, t, x_shape):
    # Extract the batch size from time tensor
    b, *_ = t.shape
    # Gather the values from the tensor
    out = a.gather(dim=-1, index=t)
    # Reshape the output tensor to shape (b, 1, 1, ..., 1)
    # so that it can be broadcasted with the input tensor
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


# --------------------------------------------------------------------------- #
# Actor Losses
# --------------------------------------------------------------------------- #


class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights=1.0):
        """
        Compute the weighted loss between the predicted and target tensors

        pred, targ : tensor [ batch_size,  action_dim ]
        """
        loss = self._loss(pred, target)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class WeightedL1(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)

class WeightedL2(WeightedLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')


# --------------------------------------------------------------------------- #
# Critic Losses
# --------------------------------------------------------------------------- #


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, target):
        return self._loss(pred, target).mean()

class ValueL1(ValueLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)

class ValueL2(ValueLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')


# --------------------------------------------------------------------------- #
# Empirical Moving Average (EMA)
# --------------------------------------------------------------------------- #


class EMA():
    """
        Empirical Moving Average (EMA) for networks.
        Used to update the model parameters in a smoother way.
    """
    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta

    def update_model_average(self, target_model, current_model):
        """
            Update the model parameters of the existing model.
        """
        for current_params, target_params in zip(current_model.parameters(), target_model.parameters()):
            old_weight, up_weight = target_params.data, current_params.data
            target_params.data = self.update_weights(old_weight, up_weight)

    def update_weights(self, old_weights, new_weights):
        """
            Update the weights of the model by taking the weighted average of the current and target weights.
        """
        if old_weights is None:
            return new_weights
        return old_weights * self.beta + (1 - self.beta) * new_weights
