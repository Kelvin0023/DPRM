import math
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.utils import spectral_norm


activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
    }
)

class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_spectralnorm=False,
    ):
        super(MLP, self).__init__()

        # Construct module list: if use `Python List`, the modules are not
        # added to computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in append_layers:
                i_dim += append_dim

            linear_layer = nn.Linear(i_dim, o_dim)
            if use_spectralnorm:
                linear_layer = spectral_norm(linear_layer)
            if idx == num_layer - 1:
                module = nn.Sequential(
                    OrderedDict(
                        [
                            ("linear_1", linear_layer),
                            ("act_1", activation_dict[out_activation_type]),
                        ]
                    )
                )
            else:
                if use_layernorm:
                    module = nn.Sequential(
                        OrderedDict(
                            [
                                ("linear_1", linear_layer),
                                ("norm_1", nn.LayerNorm(o_dim)),
                                ("act_1", activation_dict[activation_type]),
                            ]
                        )
                    )
                else:
                    module = nn.Sequential(
                        OrderedDict(
                            [
                                ("linear_1", linear_layer),
                                ("act_1", activation_dict[activation_type]),
                            ]
                        )
                    )
            self.moduleList.append(module)

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class ResidualMLP(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input


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
        mlp_dims=[256, 256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super(DiffusionMLP, self).__init__()
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
            self.mean_mlp = ResidualMLP(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type=out_activation_type,
                use_layernorm=use_layernorm,
            )
        else:
            self.mean_mlp = MLP(
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

    def forward(self, x, time, state):
        # Add positional encoding to the time input
        t_emb = self.time_emb_mlp(time)
        # Encode observation
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)
        # Concatenate the input tensor with the positional encoding
        x = torch.cat([x, t_emb, state], dim=1)
        # Pass the concatenated tensor through the middle MLP
        pred_act_chunk = self.mean_mlp(x)
        # Output the final tensor with the shape of the [Ta, Da]
        return pred_act_chunk