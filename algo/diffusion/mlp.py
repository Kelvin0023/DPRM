import math
import numpy as np
import torch
import torch.nn as nn


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


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            time_emb_dim,
            mlp_hidden_dim,
            device,
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_dim = time_emb_dim
        self.device = device

        # positional embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # middle layer
        input_dim = input_dim + output_dim + time_emb_dim
        self.middle_mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.Mish(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Mish(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Mish(),
        )

        # output layer
        self.output_mlp = nn.Linear(mlp_hidden_dim, output_dim)

        # Initialize model parameters
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases for the layers in time_mlp
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Initialize weights and biases for the layers in middle_mlp
        for m in self.middle_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Initialize weights and biases for the output_mlp layer
        nn.init.xavier_uniform_(self.output_mlp.weight)
        nn.init.constant_(self.output_mlp.bias, 0)

    def forward(self, x, time, state):
        # Add positional encoding to the time input
        t_emb = self.time_mlp(time)
        # Concatenate the input tensor with the positional encoding
        x = torch.cat([x, t_emb, state], dim=1)
        # Pass the concatenated tensor through the middle MLP
        x = self.middle_mlp(x)
        # Output the final tensor with the shape of the action dimension
        return self.output_mlp(x)







if __name__ == "__main__":
    # Initialize the dimension size for the positional embedding
    dim = 32
    # Create a SinusoidalPosEmb object
    pos_emb_layer = SinusoidalPosEmb(dim)

    # Generate test input data: a tensor of positions
    test_input = torch.arange(10, dtype=torch.float32)

    # Pass the test input through the SinusoidalPosEmb layer
    output = pos_emb_layer(test_input)

    print("Test Input:\n", test_input.shape)
    print("Output Embeddings:\n", output.shape)

    # Verify output shape
    assert output.shape == (
    test_input.shape[0], dim), f"Expected output shape {(test_input.shape[0], dim)}, but got {output.shape}"
    print("Test passed!")
