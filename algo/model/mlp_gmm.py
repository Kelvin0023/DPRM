import torch
import torch.nn as nn
from algo.model.mlp import MLP, ResidualMLP

class GMMMLP(nn.Module):
    def __init__(
        self,
        action_dim,
        action_horizon,
        cond_dim=None,
        mlp_dims=[256, 256, 256],
        num_modes=5,
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # compute the input and output dimensions
        input_dim = cond_dim
        output_dim = action_dim * action_horizon * num_modes
        # number of modes
        self.num_modes = num_modes
        
        # check if we want to use residual style
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if fixed_std is None:
            self.mlp_logvar = model(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        elif (
                learn_fixed_std
        ):  # initialize to fixed_std, separate for each action and mode
            self.logvar = torch.nn.Parameter(
                torch.log(
                    torch.tensor(
                        [fixed_std ** 2 for _ in range(action_dim * num_modes)]
                    )
                ),
                requires_grad=True,
            )
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min ** 2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max ** 2)), requires_grad=False
        )
        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std

        # mode weights
        self.mlp_weights = model(
            [input_dim] + mlp_dims + [num_modes],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        
    def forward(self, cond):
        batch_size = len(cond["state"])
        device = cond["state"].device
        
        # flatten history
        state = cond["state"].view(batch_size, -1)
        
        # mlp
        out_mean = self.mlp_mean(state)
        out_mean = torch.tanh(out_mean).view(
            batch_size,
            self.num_modes,
            self.action_horizon * self.action_dim,
        ) # tanh squashing in [-1, 1]
        
        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.num_modes, self.action_dim)
            out_scale = out_scale.repeat(batch_size, 1, self.action_horizon)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(state).view(
                batch_size, self.num_modes, self.action_horizon * self.action_dim
            )
            out_logvar = torch.clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)

        out_weights = self.mlp_weights(state)
        out_weights = out_weights.view(batch_size, self.num_modes)

        return out_mean, out_scale, out_weights