import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianModel(nn.Module):
    def __init__(
        self,
        network,
        action_horizon,
        device="cuda:0",
        randn_clip_value=10,
        tanh_output=False,
    ):
        super().__init__()
        # Model parameters
        self.action_horizon = action_horizon
        self.device = device

        # Clip sampled randn (from standard deviation) for each denoising step
        # such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Whether to apply tanh to the **sampled** action --- used in SAC
        self.tanh_output = tanh_output

        # Load network
        self.network = network.to(device)


    def loss(
        self,
        gt_action,
        cond,
        entropy_coef,
    ):
        batch_size = len(gt_action)
        distr = self.forward_train(
            cond,
            deterministic=False,
        )
        gt_action = gt_action.view(batch_size, -1)
        loss = -distr.log_prob(gt_action)
        entropy = distr.entropy().mean()
        loss = loss.mean() - entropy * entropy_coef
        return loss, {"entropy": entropy}

    def forward_train(
        self,
        cond,
        deterministic=False,
        network_override=None,
    ):
        """
        Calls the MLP to compute the mean, stds, and logits of the GMM.
        Returns the torch.Distribution object.
        """
        if network_override is not None:
            means, stds = network_override(cond)
        else:
            means, stds = self.network(cond)
        if deterministic:
            # low-noise for all Gaussian dists
            stds = torch.ones_like(means) * 1e-4
        return Normal(means, stds)

    def forward(
        self,
        cond,
        deterministic=False,
        network_override=None,
        reparameterize=False,
        get_logprob=False,
    ):
        batch_size = len(cond["state"])

        distr = self.forward_train(
            cond,
            deterministic=deterministic,
            network_override=network_override,
        )
        if reparameterize:
            sampled_action = distr.rsample()
        else:
            sampled_action = distr.sample()

        # clamp the sampled action
        sampled_action.clamp_(
            distr.loc - self.randn_clip_value * distr.scale,
            distr.loc + self.randn_clip_value * distr.scale,
        )

        # predict log probability
        if get_logprob:
            log_prob = distr.log_prob(sampled_action)

            # For SAC/RLPD, squash mean after sampling here instead of right after model output as in PPO
            if self.tanh_output:
                sampled_action = torch.tanh(sampled_action)
                log_prob -= torch.log(1 - sampled_action.pow(2) + 1e-6)
            return sampled_action.view(batch_size, self.action_horizon, -1), log_prob.sum(1, keepdim=False)
        else:
            if self.tanh_output:
                sampled_action = torch.tanh(sampled_action)
            return sampled_action.view(batch_size, self.action_horizon, -1)
