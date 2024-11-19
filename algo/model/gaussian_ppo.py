import torch
from typing import Optional
from algo.model.gaussian_pg import GaussianPG


class GaussianPPO(GaussianPG):
    def __init__(
        self,
        clip_ploss_coef: float,
        clip_vloss_coef: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # clip value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        # clip value for value loss
        self.clip_vloss_coef = clip_vloss_coef

    def loss(
        self,
        obs_policy,
        obs_critic,
        actions,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
    ):
        """
        PPO loss

        obs_policy: dict with key obs/rgb; more recent obs at the end
            obs: (B, To, Do)
            rgb: (B, To, C, H, W)
        obs_critic: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        actions: (B, Ta, Da)
        returns: (B, )
        oldvalues: (B, )
        advantages: (B,)
        oldlogprobs: (B, )
        """
        newlogprobs, entropy, std = self.get_logprobs(obs_policy, actions)
        # clip log probs
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
        newlogprobs = newlogprobs.clamp(min=-5, max=2)

        # get entropy loss
        entropy_loss = -entropy

        # compute ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # compute kl divergence
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).nanmean()
            clipfrac = (
                ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()
            )

        # compute policy loss
        pg_loss_1 = -advantages * ratio
        pg_loss_2 = -advantages * torch.clamp(
            ratio,
            1 - self.clip_ploss_coef,
            1 + self.clip_ploss_coef,
        )
        pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

        # compute value loss
        newvalues = self.critic(obs_critic).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        bc_loss = 0.0
        # if use_bc_loss:
        #     # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
        #     # Give a reward for maximizing probability of teacher policy's action with current policy.
        #     # Actions are chosen along trajectory induced by current policy.
        #
        #     # Get counterfactual teacher actions
        #     samples = self.forward(
        #         cond=obs.float()
        #         .unsqueeze(1)
        #         .to(self.device),  # B x horizon=1 x obs_dim
        #         deterministic=False,
        #         use_base_policy=True,
        #     )
        #     # Get logprobs of teacher actions under this policy
        #     bc_logprobs, _, _ = self.get_logprobs(obs, samples, use_base_policy=False)
        #     bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
        #     bc_loss = -bc_logprobs.mean()

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            std.item(),
        )