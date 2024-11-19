import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, Categorical, MixtureSameFamily


class GMMModel(nn.Module):
    def __init__(
        self,
        network,
        action_horizon,
        device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        
        # Model parameters
        self.action_horizon = action_horizon
        self.device = device
        
        # Load network
        self.network = network.to(device)

    def loss(
        self,
        cond,
        deterministic=False,
    ):
        """
        Calls MLP to compute the mean, stds, and logits of the GMM. 
        Returns the torch.Distribution object.
        """
        means, stds, logits = self.network(cond)
        if deterministic:
            # low-noise for all Gaussian dists
            stds = torch.ones_like(means) * 1e-4

        # mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
        # Each mode has mean vector of dim T*D
        component_distribution = Normal(means, stds)
        component_distribution = Independent(component_distribution, 1)

        component_entropy = component_distribution.entropy()
        approx_entropy = torch.mean(
            torch.sum(logits.softmax(-1) * component_entropy, dim=-1)
        )
        std = torch.mean(torch.sum(logits.softmax(-1) * stds.mean(-1), dim=-1))

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = Categorical(logits=logits)
        distr = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        return distr, approx_entropy, std

    def forward(self, cond, deterministic=False):
        batch_size = len(cond["state"])
        distr, _, _ = self.forward_train(
            cond,
            deterministic=deterministic,
        )
        sampled_action = distr.sample()
        sampled_action = sampled_action.view(batch_size, self.action_horizon, -1)
        return sampled_action
        