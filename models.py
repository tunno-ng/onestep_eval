"""
models.py — Simple MLP for predicting in different spaces.

Input:  xt (B, D) concatenated with time embedding (B, time_emb_dim) -> (B, D + time_emb_dim)
Output: (B, D) — interpreted as pred_space prediction

Time embedding: sinusoidal + linear projection (simple but effective for MLPs).
"""
import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t in [0,1] to a vector of dimension `emb_dim`.
    Uses sinusoidal frequencies, then a small linear layer.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even"
        self.emb_dim = emb_dim
        half = emb_dim // 2
        # Fixed frequency bands
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)  # (half,)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B, 1) — values in [0, 1]
        Returns: (B, emb_dim)
        """
        if t.dim() == 2:
            t = t.squeeze(1)
        # Scale t to [0, 1000] range for sinusoidal embedding
        t_scaled = t * 1000.0  # (B,)
        args = t_scaled.unsqueeze(1) * self.freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, emb_dim)
        return self.proj(emb)


class MLP(nn.Module):
    """
    Simple MLP that takes (xt, t) and outputs a D-dimensional prediction.

    Architecture:
      1. Embed time t -> time_emb
      2. Input = concat(xt, time_emb)  -- shape (B, D + time_emb_dim)
      3. Hidden layers: n_layers - 1 layers of (Linear -> SiLU)
      4. Output: Linear -> (B, D)

    No skip connections, no normalization -- keeps it simple.
    """
    def __init__(self, obs_dim: int, hidden_dim: int, n_layers: int, time_emb_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        input_dim = obs_dim + time_emb_dim
        layers = []
        in_dim = input_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        xt: (B, D)
        t:  (B,) or (B, 1) -- values in [0, 1]
        Returns: (B, D)
        """
        if t.dim() == 1:
            t_1d = t
        else:
            t_1d = t.squeeze(1)
        t_emb = self.time_emb(t_1d)           # (B, time_emb_dim)
        h = torch.cat([xt, t_emb], dim=1)     # (B, D + time_emb_dim)
        return self.net(h)


def build_model(obs_dim: int, hidden_dim: int, n_layers: int,
                time_emb_dim: int) -> MLP:
    return MLP(obs_dim=obs_dim, hidden_dim=hidden_dim,
               n_layers=n_layers, time_emb_dim=time_emb_dim)
