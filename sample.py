"""
sample.py — One-step and multi-step generation.

==============================================================
GENERATION OVERVIEW
==============================================================

For all methods, generation starts from x1 ~ N(0, I) in R^D.

ONE-STEP GENERATION (steps=1):
  1. Sample x1 ~ N(0, I)
  2. Query model at t=1: pred = model(x1, t=1)
  3. Convert pred from pred_space to x0_pred
  4. Output x0_pred as the generated sample

  The conversion from pred_space to x0 at t=1:
    x-space:   x0 = pred          (direct)
    u-space:   x0 = x1 + pred     (since u = x0 - x1)
    v-space:   x0 = x1 - pred     (since v = x1 - x0)
    eps-space: x0 = x1 - pred     (heuristic; see spaces.py)

MULTI-STEP GENERATION (steps=K, K>1):
  Use a simple explicit Euler integration along the learned flow.
  The flow velocity in the model's frame is always v = x1 - x0.
  We convert the model output to v-space, then integrate.

  For K steps, use times t = [1, 1-dt, 1-2*dt, ..., 0] where dt = 1/K.

  At each step from t_curr to t_next = t_curr - dt:
    1. pred = model(x_curr, t_curr)
    2. v_pred = convert_to_v_space(pred, pred_space, x_curr, t_curr)
    3. x_next = x_curr + (t_next - t_curr) * v_pred
             = x_curr - dt * v_pred

  Note: for x-space predictions in multi-step mode, we use the DDIM-style
  deterministic update:
    x_next = x_curr + (t_next - t_curr) * v_est
  where v_est = (x_curr - x0_pred) / t_curr  [since xt = x0 + t*v, so v = (xt-x0)/t]
  Wait, v = x1 - x0, xt = x0 + t*v, so v = (xt - x0) / t for t > 0.

  This is more numerically stable than blindly integrating for small t.
"""
import torch
from spaces import pred_to_x, pred_to_x0_at_t1


@torch.no_grad()
def generate_one_step(
    model: torch.nn.Module,
    n_samples: int,
    obs_dim: int,
    pred_space: str,
    device: torch.device,
) -> torch.Tensor:
    """
    One-step generation.

    Returns: (n_samples, obs_dim) generated samples in R^D.
    """
    model.eval()

    # Start from pure noise
    x1 = torch.randn(n_samples, obs_dim, device=device)  # (N, D)

    # Query model at t=1
    t = torch.ones(n_samples, 1, device=device)          # (N, 1)
    pred = model(x1, t)                                   # (N, D)

    # Convert to x0
    x0_pred = pred_to_x0_at_t1(pred, pred_space, x1)    # (N, D)

    return x0_pred


@torch.no_grad()
def generate_multistep(
    model: torch.nn.Module,
    n_samples: int,
    obs_dim: int,
    pred_space: str,
    steps: int,
    device: torch.device,
) -> tuple:
    """
    Multi-step generation using explicit Euler integration.

    Returns:
      x_final: (n_samples, obs_dim) final generated samples
      trajectory: list of (n_samples, obs_dim) tensors, one per step
                  (useful for visualization)
    """
    model.eval()

    dt = 1.0 / steps
    # Time sequence: [1, 1-dt, 1-2*dt, ..., dt, 0]
    times = [1.0 - k * dt for k in range(steps)]  # start times for each step

    x = torch.randn(n_samples, obs_dim, device=device)  # (N, D)
    trajectory = [x.clone()]

    for t_curr_val in times:
        t_curr = torch.full((n_samples, 1), t_curr_val, device=device)

        pred = model(x, t_curr)  # (N, D) in pred_space

        # Convert prediction to velocity v = x1 - x0
        v_pred = pred_to_v(pred, pred_space, x, t_curr)  # (N, D)

        # Euler step: x_next = x_curr - dt * v_pred
        # (moving backwards in t by dt)
        x = x - dt * v_pred
        trajectory.append(x.clone())

    return x, trajectory


def pred_to_v(
    pred: torch.Tensor,
    pred_space: str,
    xt: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a model prediction in pred_space to velocity v = x1 - x0.

    This is used by the multi-step Euler integrator.
    We route through x0_pred to get v_pred = -u_pred.

    xt: (B, D), t: (B, 1)
    """
    if pred_space == "v":
        return pred

    # Convert to x0, then to v
    # For x0 -> v we need x1_est.
    # Under the path: xt = (1-t)*x0 + t*x1 => x1 = (xt - (1-t)*x0) / t
    x0_pred = pred_to_x(pred, pred_space, xt, t)

    # Recover v from x0_pred: v = (xt - x0_pred) / t
    # (because xt = x0 + t*v => v = (xt - x0)/t for t > 0)
    t_safe = t.clamp(min=1e-5)
    v_pred = (xt - x0_pred) / t_safe
    return v_pred
