"""
models.py — Model definitions for different prediction spaces.

Two distinct model classes with different input signatures:

  StandardMLP:  input = (z_t, t)      for pred_space in {x, eps, v}
  UModel:       input = (z_t, r, t)   for pred_space = u

The distinction is preserved throughout training and sampling.
Do NOT use UModel for x/eps/v-pred or StandardMLP for u-pred.
"""
import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar t in [0,1] to a vector of dimension emb_dim.

    Uses sinusoidal frequencies (like positional encoding),
    followed by a learned linear projection.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even"
        half = emb_dim // 2
        # Log-spaced frequency bands, fixed (not learned)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)   # (half,)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B, 1) in [0, 1]
        Returns: (B, emb_dim)
        """
        if t.dim() == 2:
            t = t.squeeze(1)                        # (B,)
        t_scaled = t * 1000.0                       # scale to [0, 1000]
        args = t_scaled[:, None] * self.freqs[None] # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, emb_dim)
        return self.proj(emb)                       # (B, emb_dim)


class StandardMLP(nn.Module):
    """
    MLP for pred_space in {x, eps, v}.

    Input:  concat(z_t, embed(t))  -- shape (B, obs_dim + time_emb_dim)
    Output: (B, obs_dim)

    Architecture:
      [z_t || t_emb] -> Linear -> SiLU -> ... -> Linear -> output
    """
    def __init__(self, obs_dim: int, hidden_dim: int, n_layers: int, time_emb_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.t_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = obs_dim + time_emb_dim
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, obs_dim)
        t:   (B, 1) or (B,) -- values in [0, 1]
        Returns: (B, obs_dim)
        """
        t_emb = self.t_emb(t)                    # (B, time_emb_dim)
        h = torch.cat([z_t, t_emb], dim=1)       # (B, obs_dim + time_emb_dim)
        return self.net(h)


class UModel(nn.Module):
    """
    MLP for pred_space = u (MeanFlow).

    Input:  concat(z_t, embed(r), embed(t))  -- shape (B, obs_dim + 2*time_emb_dim)
    Output: (B, obs_dim)

    Both r and t are embedded separately to give the model full information
    about both the current time and the target time.

    This model MUST be called with (z_t, r, t) -- never with just (z_t, t).
    """
    def __init__(self, obs_dim: int, hidden_dim: int, n_layers: int, time_emb_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.r_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.t_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = obs_dim + 2 * time_emb_dim
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, obs_dim)
        r:   (B, 1) or (B,) -- "from" time (target)
        t:   (B, 1) or (B,) -- "current" time
        Returns: (B, obs_dim) -- predicted average velocity u(z_t, r, t)
        """
        r_emb = self.r_emb(r)                    # (B, time_emb_dim)
        t_emb = self.t_emb(t)                    # (B, time_emb_dim)
        h = torch.cat([z_t, r_emb, t_emb], dim=1)
        return self.net(h)


def build_model(
    pred_space: str,
    obs_dim: int,
    hidden_dim: int,
    n_layers: int,
    time_emb_dim: int,
) -> nn.Module:
    """Factory: return the right model for the given pred_space."""
    if pred_space == "u":
        return UModel(obs_dim, hidden_dim, n_layers, time_emb_dim)
    else:
        return StandardMLP(obs_dim, hidden_dim, n_layers, time_emb_dim)
