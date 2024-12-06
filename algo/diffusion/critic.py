import torch
import torch.nn as nn
from typing import Union
from copy import deepcopy

from algo.model.mlp import MLP, ResidualMLP
    
    
class CriticObsAct(torch.nn.Module):
    """State-action double critic network."""

    def __init__(
        self,
        mlp_dims,
        obs_critic_dim,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        **kwargs,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_steps = action_steps
        self.state_dim = obs_critic_dim

        mlp_dims = [obs_critic_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            self.Q1 = ResidualMLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        else:
            self.Q1 = MLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        self.Q2 = deepcopy(self.Q1)

        # Initialize model parameters
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases for the layers in time_mlp
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action_chunk):
        x = torch.cat(
            [state, action_chunk.view(-1, self.action_steps * self.action_dim)],
            dim=-1
        )
        return self.Q1(x).squeeze(1), self.Q2(x).squeeze(1)

    def sample_q1(self, state, action_chunk):
        """
        Sample the Q-value from the first critic Q1
        """
        x = torch.cat(
            [state, action_chunk.view(-1, self.action_steps * self.action_dim)],
            dim=-1,
        )
        with torch.no_grad():
            return self.Q1(x).squeeze(1)

    def sample_min_q(self, state, action_chunk):
        """
        Sample the minimum Q-value from the two critics to stabilize training
        """
        x = torch.cat(
            [state, action_chunk.view(-1, self.action_steps * self.action_dim)],
            dim=-1,
        )
        with torch.no_grad():
            return torch.min(self.Q1(x).squeeze(1), self.Q2(x).squeeze(1))


