import torch
from algo.model.gaussian import GaussianModel


class GaussianPG(GaussianModel):
    def __init__(
        self,
        actor_model,
        critic_model,
        **kwargs,
    ):
        super().__init__(network=actor_model, **kwargs)
        # Re-name network to actor
        self.actor = self.network
        self.critic = critic_model.to(self.device)

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
            network_override=None,
        )

        # ---------- RL training ----------#

    def get_logprobs(
            self,
            cond,
            actions,
            use_base_policy=False,
    ):
        batch_size = len(actions)
        distr = self.forward_train(
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )
        log_prob = distr.log_prob(actions.view(batch_size, -1))
        log_prob = log_prob.mean(-1)
        entropy = distr.entropy().mean()
        std = distr.scale.mean()
        return log_prob, entropy, std

    def loss(self, obs, actions, reward):
        raise NotImplementedError