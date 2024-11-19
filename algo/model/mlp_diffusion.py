import math
import numpy as np
import torch
import torch.nn as nn

from algo.model.mlp import MLP, ResidualMLP


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.scale = math.log(10000)

    def forward(self, x):
        device = x.device
        # the final embedding dimension will concatenate sin and cos embeddings
        # which together will restore the original dimension size
        half_dim = self.dim // 2
        # scaling factor for the sinusoidal embeddings
        emb = self.scale / (half_dim - 1)
        # create a tensor where each element is an exponential of a scaled index
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # broadcasted multiplication of the input x with the exponential tensor
        # each row corresponds to a different position in the input 'x'
        # each column corresponds to a different dimension in the 'emb'
        emb = x[:, None] * emb[None, :]
        # concatenate sin and cos embeddings in the last dimension to form the final embedding
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class DiffusionMLP(nn.Module):
    def __init__(
        self,
        action_dim,
        action_horizon,
        cond_dim,
        time_emb_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        output_dim = action_dim * action_horizon

        # positional embedding layer
        self.time_emb_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # conditional embedding layer
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_emb_dim + action_dim * action_horizon + cond_mlp_dims[-1]
        else:
            input_dim = time_emb_dim + action_dim * action_horizon + cond_dim

        # check if we want to use residual style
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        # MLP layer for mean
        self.mean_mlp = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

        self.time_emb_dim = time_emb_dim

        # Initialize model parameters
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, time, cond, **kargs):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        # get batch size, action chunk size and action dimension
        B, Ta, Da = x.shape

        # flatten action chunk
        x = x.view(B, -1)

        # flatten observation history
        state = cond["state"].view(B, -1)

        # encode observation
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # add time embedding
        time = time.view(B, 1)
        time_emb = self.time_emb_mlp(time).view(B, self.time_emb_dim)

        # create the concatenated input tensor
        x = torch.cat([x, time_emb, state], dim=-1)

        # pass the input tensor through the mean MLP
        pred_act = self.mean_mlp(x)
        return pred_act.view(B, Ta, Da)




