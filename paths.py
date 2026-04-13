"""
paths.py — Linear interpolation path between clean data and noise.

Convention (used throughout this codebase):
  x0 ~ p_data  (clean data in R^D)
  x1 ~ N(0, I) (standard Gaussian noise in R^D)
  t  ~ Uniform(0, 1)

Linear interpolation path:
  xt = (1 - t) * x0 + t * x1

At t=0: xt = x0  (clean data)
At t=1: xt = x1  (pure noise)

This is the standard "flow matching" / "stochastic interpolant" path
(also called "straight-path OT" or "rectified flow" path).

Smoothed endpoint (target relaxation, controlled by tau in [0,1]):
  x_target = (1 - tau) * x0 + tau * x1

When tau=0: x_target = x0  (exact clean endpoint)
When tau>0: x_target is a blended version, making the prediction target
            smoother and potentially easier to learn.
"""
import torch


def sample_noise(shape: tuple, device: torch.device) -> torch.Tensor:
    """Sample x1 ~ N(0, I)."""
    return torch.randn(shape, device=device)


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute xt = (1-t)*x0 + t*x1 along the linear path.

    x0: (B, D)
    x1: (B, D)
    t:  (B,) or (B, 1)  -- will be broadcast correctly
    Returns: (B, D)
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)  # (B, 1) for broadcasting
    return (1.0 - t) * x0 + t * x1


def get_smoothed_target(x0: torch.Tensor, x1: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Compute relaxed/smoothed endpoint target:
      x_target = (1 - tau) * x0 + tau * x1

    tau=0 -> exact clean data x0
    tau>0 -> blend towards noise, making target less sharp

    x0: (B, D)
    x1: (B, D)
    Returns: (B, D)
    """
    return (1.0 - tau) * x0 + tau * x1
