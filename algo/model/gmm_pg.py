import torch
from algo.model.gmm import GMMModel


class GMMPG(GMMModel):
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

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(self, cond, deterministic=False):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
    ):
        batch_size = len(actions)
        distr, entropy, std = self.forward_train(
            cond,
            deterministic=False,
        )
        log_prob = distr.log_prob(actions.view(batch_size, -1))
        return log_prob, entropy, std

    def loss(self, obs, chains, reward):
        raise NotImplementedError