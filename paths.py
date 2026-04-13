"""
paths.py — Path construction and time sampling.

==============================================================
CONVENTION (used everywhere in this codebase)
==============================================================

    z_t = (1 - t) * x + t * eps       [linear interpolation]
    v   = eps - x                      [instantaneous conditional velocity]

where:
  x   ~ p_data   (clean data in R^D)
  eps ~ N(0, I)  (prior noise in R^D)
  t   ~ Uniform(0, 1)

At t=0: z_t = x   (clean data)
At t=1: z_t = eps (pure noise)

Note on JiT convention: the JiT paper uses reversed indexing where t=0 is noise.
This codebase uses the OPPOSITE convention (t=0 is data, t=1 is noise),
matching the standard flow-matching / stochastic-interpolant literature.

==============================================================
NUMERICAL SAFETY
==============================================================

Several transforms divide by t or (1-t):
  - x -> v  divides by t:     v = (z_t - x) / t
  - eps -> v divides by (1-t): v = (eps - z_t) / (1-t)
  - eps -> x divides by (1-t): x = (z_t - t*eps) / (1-t)
  - x -> eps divides by t:    eps = (z_t - (1-t)*x) / t

We never sample t = 0 or t = 1 exactly.
All time sampling keeps t in [T_EPS, 1 - T_EPS].

For u-pred, we sample pairs (r, t) with 0 <= r < t <= 1.
The gap (t - r) is kept >= T_EPS to avoid division-by-zero in the MeanFlow identity.
"""
import torch

# Numerical safety margin for time sampling
T_EPS: float = 1e-3


def sample_noise(shape: tuple, device: torch.device) -> torch.Tensor:
    """Sample eps ~ N(0, I) of given shape."""
    return torch.randn(shape, device=device)


def interpolate(x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute z_t = (1-t)*x + t*eps.

    x:   (B, D)
    eps: (B, D)
    t:   (B, 1)
    Returns: (B, D)
    """
    return (1.0 - t) * x + t * eps


def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample t ~ Uniform(T_EPS, 1 - T_EPS) for x/eps/v prediction.

    Returns: (B, 1)
    """
    t = torch.rand(batch_size, 1, device=device)
    t = t * (1.0 - 2.0 * T_EPS) + T_EPS   # rescale to [T_EPS, 1-T_EPS]
    return t


def sample_r_t(batch_size: int, device: torch.device) -> tuple:
    """
    Sample (r, t) with 0 <= r < t <= 1 for u prediction (MeanFlow).

    Method:
      t ~ Uniform(2*T_EPS, 1 - T_EPS)   [ensures room for r below t]
      r ~ Uniform(0,        t - T_EPS)   [ensures t - r >= T_EPS]

    This gives:
      t in [2*T_EPS, 1-T_EPS]
      r in [0, t-T_EPS]
      t - r >= T_EPS  (guaranteed)

    Returns: (r: (B, 1), t: (B, 1))
    """
    t = torch.rand(batch_size, 1, device=device)
    t = t * (1.0 - 3.0 * T_EPS) + 2.0 * T_EPS    # t in [2*T_EPS, 1-T_EPS]

    r = torch.rand(batch_size, 1, device=device)
    r = r * (t - T_EPS)                            # r in [0, t - T_EPS]

    return r, t


def get_smoothed_x(x: torch.Tensor, eps: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Compute the smoothed / relaxed target x_eff:
      x_eff = (1 - tau) * x + tau * eps

    tau=0  ->  x_eff = x         (exact clean data)
    tau>0  ->  x_eff blends toward eps (softer target)

    All loss targets are derived from x_eff, not x.
    The interpolation path still uses the original x (path is not changed by tau).
    """
    return (1.0 - tau) * x + tau * eps
