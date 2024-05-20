import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    timestep_embedding,
)

class TimestepEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_hidden_layers=5, timestep_emb_dim=256):
        super().__init__()
        self.layers = nn.ModuleList()
        self.timestep_mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.timestep_emb_dim = timestep_emb_dim

        prev_dim = input_dim
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.timestep_mlps.append(TimestepEmbeddingMLP(timestep_emb_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, timesteps):
        """
        x.shape = bs, 1, input_dim
        timesteps.shape = bs
        """
        x = x[:, 0, :] 
        t_emb = timestep_embedding(timesteps, self.timestep_emb_dim)

        for layer, timestep_mlp, norm in zip(self.layers, self.timestep_mlps, self.norms):
            x = F.relu(layer(x))
            t_emb_transformed = timestep_mlp(t_emb)
            x = norm(x + t_emb_transformed)

        x = self.output_layer(x)
        return x[:, None, :] # return bs, 1, input_dim