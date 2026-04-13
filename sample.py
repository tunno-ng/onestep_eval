"""
sample.py — One-step and multi-step generation.

==============================================================
CONVENTION
==============================================================

    z_t = (1-t)*x + t*eps,   v = eps - x

Generation always starts from z_1 = eps ~ N(0, I) at t=1
and aims to reach z_0 = x ~ p_data at t=0.

==============================================================
x / eps / v PREDICTION: SAMPLING
==============================================================

For pred_space in {x, eps, v}, the model outputs a prediction
in its native space. All sampling proceeds by first converting
the prediction to v-space (velocity), then integrating.

ONE-STEP (t=1 -> t=0, single Euler step):
    pred = net(z_1, t=1)
    v_theta = convert_to_v(pred, pred_space, z_1, t=1)
    x_gen = z_1 - 1.0 * v_theta         [Euler step of size dt=1]

  Equivalently, for each space at t=1:
    x-pred:   v = (z_1 - x_theta) / 1 = z_1 - x_theta  =>  x_gen = x_theta
    eps-pred: v = (eps_theta - z_1) / 0  ... SINGULAR at t=1 for eps->v.
              Instead use: x_gen = z_1 - 1*v; v = (eps_theta-z_1)/(1-t) is singular.
              Safe fallback: use x_gen = z_1 - eps_theta  (direct heuristic).
    v-pred:   v = v_theta                =>  x_gen = z_1 - v_theta

  NOTE: eps->v conversion divides by (1-t), which is 0 at t=1.
  For ONE-STEP sampling with eps-pred, we use the algebraically equivalent
  direct formula:
      x_gen = v_to_x(v_pred, z_1, t=1) where v_pred is via eps_to_v
  Since eps_to_v is singular at t=1, we use a limiting argument:
      At t->1: z_t -> eps, so x = z_t - t*v -> z_t - v (at t=1)
  But eps_theta is predicting eps ~ z_t at t=1, so v = eps - x and
  x = eps - v. With eps_theta ~ eps and x_gen = eps - v_theta... still singular.

  PRACTICAL CHOICE: for one-step, use t = 1 - T_EPS instead of t=1 exactly.
  This avoids all singularities while being numerically identical.

MULTI-STEP (K Euler steps from t=1 to t=0):
    dt = 1.0 / K
    times = [1 - k*dt for k in range(K)]   [1, 1-dt, ..., dt]
    for t_curr in times:
        pred = net(z, t_curr)
        v_pred = convert_to_v(pred, pred_space, z, t_curr)
        z = z - dt * v_pred

==============================================================
u PREDICTION: SAMPLING
==============================================================

Training uses the compound V_theta (via JVP) for the loss.
Inference uses the DIRECT MeanFlow jump:

    z_r = z_t - (t - r) * u_theta(z_t, r, t)

This directly "fast-forwards" from time t to time r.

ONE-STEP (t=1, r=0):
    x_gen = z_1 - 1.0 * u_theta(z_1, r=0, t=1)

MULTI-STEP (K steps, each jumping from t_k to r_k = t_k - 1/K):
    for k in range(K):
        t_k = 1 - k/K
        r_k = t_k - 1/K = 1 - (k+1)/K
        z = z - (t_k - r_k) * u_theta(z, r_k, t_k)
      = z - (1/K) * u_theta(z, r_k, t_k)

This is the principled MeanFlow multi-step scheme.
"""
import torch
import torch.nn as nn

from transforms import to_v
from paths import T_EPS


@torch.no_grad()
def generate(
    model: nn.Module,
    n_samples: int,
    obs_dim: int,
    pred_space: str,
    steps: int,
    device: torch.device,
) -> tuple:
    """
    Generate samples using the trained model.

    Dispatches to the appropriate function based on pred_space.

    Returns:
      x_gen:      (n_samples, obs_dim) final generated samples
      trajectory: list of (n_samples, obs_dim) tensors, length = steps + 1
                  (trajectory[0] = initial noise, trajectory[-1] = generated sample)
    """
    if pred_space == "u":
        return _generate_u(model, n_samples, obs_dim, steps, device)
    else:
        return _generate_standard(model, n_samples, obs_dim, pred_space, steps, device)


def _generate_standard(
    model: nn.Module,
    n_samples: int,
    obs_dim: int,
    pred_space: str,
    steps: int,
    device: torch.device,
) -> tuple:
    """
    Generation for pred_space in {x, eps, v} via Euler integration in v-space.

    Uses t = 1 - T_EPS for the first step to avoid singularity in eps->v at t=1.
    """
    model.eval()
    dt = 1.0 / steps

    z = torch.randn(n_samples, obs_dim, device=device)  # z_1 ~ N(0,I)
    trajectory = [z.clone()]

    # Step times: [1-T_EPS, 1-dt, 1-2dt, ..., dt]
    # We cap the first step at 1-T_EPS to avoid singularities.
    for k in range(steps):
        t_curr_val = 1.0 - k * dt
        t_curr_val = min(t_curr_val, 1.0 - T_EPS)   # safety clamp
        t_curr = torch.full((n_samples, 1), t_curr_val, device=device)

        pred = model(z, t_curr)                       # (N, D)

        # Convert prediction to v-space for Euler update
        v_pred = to_v(pred, pred_space, z, t_curr)   # (N, D)

        # Euler step: z_{t-dt} = z_t - dt * v
        z = z - dt * v_pred
        trajectory.append(z.clone())

    return z, trajectory


def _generate_u(
    model: nn.Module,
    n_samples: int,
    obs_dim: int,
    steps: int,
    device: torch.device,
) -> tuple:
    """
    Generation for pred_space = u using direct MeanFlow jumps.

    At each step from t_k to r_k:
        z_r = z_t - (t_k - r_k) * u_theta(z_t, r_k, t_k)

    This is the INFERENCE-TIME formula, distinct from the training
    compound V_theta. See module docstring for explanation.
    """
    model.eval()
    dt = 1.0 / steps

    z = torch.randn(n_samples, obs_dim, device=device)  # z_1 ~ N(0,I)
    trajectory = [z.clone()]

    for k in range(steps):
        t_val = 1.0 - k * dt
        r_val = t_val - dt                              # = 1 - (k+1)*dt

        t = torch.full((n_samples, 1), t_val, device=device)
        r = torch.full((n_samples, 1), max(r_val, 0.0), device=device)

        u_pred = model(z, r, t)                        # (N, D)

        # Direct MeanFlow jump: z_r = z_t - (t - r) * u_theta
        z = z - (t - r) * u_pred
        trajectory.append(z.clone())

    return z, trajectory
