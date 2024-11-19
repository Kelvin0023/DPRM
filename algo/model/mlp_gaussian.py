import torch
import torch.nn as nn
from algo.model.mlp import MLP, ResidualMLP


class GaussianMLP(nn.Module):
    def __init__(
        self,
        action_dim,
        action_horizon,
        cond_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        tanh_output=True,  # sometimes we want to apply tanh after sampling instead of here, e.g., in SAC
        residual_style=False,
        use_layernorm=False,
        dropout=0.0,
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
        output_dim = action_dim * action_horizon
        # check if we want to use residual style
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        if fixed_std is None:
            # learning std
            self.mlp_base = model(
                [input_dim] + mlp_dims,
                activation_type=activation_type,
                out_activation_type=activation_type,
                use_layernorm=use_layernorm,
                use_layernorm_final=use_layernorm,
            )
            self.mlp_mean = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
            self.mlp_logvar = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
        else:
            # no separate head for mean and std
            self.mlp_mean = model(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
                dropout=dropout,
            )
            if learn_fixed_std:
                # initialize to fixed_std
                self.logvar = torch.nn.Parameter(
                    torch.log(
                        torch.tensor(
                            [fixed_std**2 for _ in range(action_dim)]
                        )
                    ),
                    requires_grad=True,
                )
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max**2)), requires_grad=False
        )
        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std
        self.tanh_output = tanh_output

    def forward(self, cond):
        batch_size = len(cond["state"])
        device = cond["state"].device

        # flatten history
        state = cond["state"].view(batch_size, -1)

        # mlp
        if hasattr(self, "mlp_base"):
            state = self.mlp_base(state)
        out_mean = self.mlp_mean(state)
        if self.tanh_output:
            out_mean = torch.tanh(out_mean)
        # reshape the output mean to [B, Ta, Da]
        out_mean = out_mean.view(batch_size, self.action_horizon * self.action_dim)

        if self.learn_fixed_std:
            # clip the logvar and expand to [B, Ta, Da]
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.action_dim)
            out_scale = out_scale.repeat(batch_size, self.action_horizon)
        elif self.use_fixed_std:
            # use fixed std
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            # predict std with mlp
            out_logvar = self.mlp_logvar(state).view(
                batch_size, self.action_horizon * self.action_dim
            )
            out_logvar = torch.tanh(out_logvar)
            out_logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (
                    out_logvar + 1
            )  # put back to full range
            out_scale = torch.exp(0.5 * out_logvar)

        return out_mean, out_scale


