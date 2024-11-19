import torch
import torch.nn as nn
from typing import Union
from copy import deepcopy

from algo.model.mlp import MLP, ResidualMLP


class CriticV(nn.Module):
    """
    Value function V(s) network.
    """
    def __init__(
        self,
        obs_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [obs_dim] + mlp_dims + [1]
        if residual_style:
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

        # Initialize model parameters
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1


    @torch.no_grad()
    def sample(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1
    
    
class CriticObsAct(torch.nn.Module):
    """State-action double critic network."""

    def __init__(
        self,
        mlp_dims,
        obs_dim,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [obs_dim + action_dim * action_steps] + mlp_dims + [1]
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

    def forward(self, cond: Union[dict, torch.Tensor], action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        if isinstance(cond, dict):
            B = len(cond["state"])
            # flatten history
            state = cond["state"].view(B, -1)
        else:
            B = cond.size(0)
            # flatten history
            state = cond.view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1.squeeze(1), q2.squeeze(1)

    @torch.no_grad()
    def sample(self, cond: Union[dict, torch.Tensor], action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        if isinstance(cond, dict):
            B = len(cond["state"])
            # flatten history
            state = cond["state"].view(B, -1)
        else:
            B = cond.size(0)
            state = cond.view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1.squeeze(1), q2.squeeze(1)

    @torch.no_grad()
    def sample_min(self, cond: Union[dict, torch.Tensor], action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        if isinstance(cond, dict):
            B = len(cond["state"])
            # flatten history
            state = cond["state"].view(B, -1)
        else:
            B = cond.size(0)
            # flatten history
            state = cond.view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return torch.min(q1.squeeze(1), q2.squeeze(1))


