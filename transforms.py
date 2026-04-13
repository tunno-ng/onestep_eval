"""
transforms.py — AUTHORITATIVE space transforms and u->V JVP conversion.

This is the single source of truth for all algebraic transformations between
prediction spaces. Do NOT duplicate these formulas in other files.

==============================================================
BASE CONVENTION
==============================================================

    z_t = (1 - t) * x + t * eps       t in [0, 1]
    v   = eps - x                      (instantaneous velocity, constant along path)

Consequences:
    z_t = x + t * v                    (useful for recovering x from v)

==============================================================
ONE-STEP ALGEBRAIC TRANSFORMS (x, eps, v)
==============================================================

Given x and (z_t, t):
    eps = (z_t - (1-t)*x) / t          [requires t > 0]
    v   = (z_t - x) / t                [requires t > 0]

Given eps and (z_t, t):
    x   = (z_t - t*eps) / (1-t)        [requires t < 1]
    v   = (eps - z_t) / (1-t)          [requires t < 1]

Given v and (z_t, t):
    x   = z_t - t * v                  [no singularity]
    eps = z_t + (1-t) * v              [no singularity]

These six formulas are exact under the linear path.

==============================================================
MEANFLOW AVERAGE VELOCITY u
==============================================================

The MeanFlow quantity u is defined as:

    u(z_t, r, t) = (1 / (t - r)) * integral_r^t v(z_tau, tau) d_tau

where the integral is along the path z_tau = (1-tau)*x + tau*eps.
Since v = eps - x is CONSTANT along the linear path:

    u(z_t, r, t) = v = eps - x         (exact, for the linear path!)

However, the NETWORK u_theta(z_t, r, t) is trained to approximate this,
and the training objective uses the MeanFlow / iMF identity to form a
COMPOUND velocity prediction V_theta that respects the structure:

    v(z_t, t) = u(z_t, r, t) + (t - r) * [d/dt u(z_t, r, t)]

This is the MeanFlow identity. In training:

    V_theta = u_theta + (t - r) * JVP_sg(u_theta; v_tilde_theta)

where:
- JVP_sg(u_theta; v_tilde_theta) = [∂u_theta/∂z_t * v_tilde + ∂u_theta/∂t]
- v_tilde_theta(z_t, t) := u_theta(sg(z_t), t, t)  [boundary condition, stop-grad]
- sg() means stop-gradient (detach)

V_theta is then used as the "effective v-prediction" for the loss.

NOTE: because v is constant along the linear path, u = v exactly.
But the *network* u_theta(z_t, r, t) is NOT constrained to output a constant.
The MeanFlow identity provides a principled way to convert u_theta -> V_theta
that is consistent with v-space even when u_theta is an imperfect estimate.

==============================================================
INFERENCE-TIME SAMPLING (u-pred)
==============================================================

At inference, we use the direct MeanFlow jump:
    z_r = z_t - (t - r) * u_theta(z_t, r, t)

This directly "integrates" from t to r using the average velocity.
For one-step generation: set t=1, r=0:
    x_gen = z_1 - 1 * u_theta(z_1, 0, 1)

This is DIFFERENT from the V_theta compound used in training.
The two are related by the MeanFlow identity (they agree when u_theta is perfect),
but the direct jump is used at inference for efficiency.
"""
import torch
import torch.nn as nn
from torch.func import jvp as func_jvp


# -----------------------------------------------------------------------
# Algebraic transforms between x, eps, v
# -----------------------------------------------------------------------

def x_to_eps(x_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """eps = (z_t - (1-t)*x) / t.  Requires t > 0."""
    return (z_t - (1.0 - t) * x_pred) / t


def x_to_v(x_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v = (z_t - x) / t.  Requires t > 0."""
    return (z_t - x_pred) / t


def eps_to_x(eps_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x = (z_t - t*eps) / (1-t).  Requires t < 1."""
    return (z_t - t * eps_pred) / (1.0 - t)


def eps_to_v(eps_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v = (eps - z_t) / (1-t).  Requires t < 1."""
    return (eps_pred - z_t) / (1.0 - t)


def v_to_x(v_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x = z_t - t*v.  No singularity."""
    return z_t - t * v_pred


def v_to_eps(v_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """eps = z_t + (1-t)*v.  No singularity."""
    return z_t + (1.0 - t) * v_pred


# -----------------------------------------------------------------------
# Convert any pred_space prediction -> v_pred (for loss/sampling)
# -----------------------------------------------------------------------

def to_v(pred: torch.Tensor, pred_space: str,
         z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Convert a prediction in pred_space {x, eps, v} to v-space.
    For u-pred, use compute_V_theta() instead.

    pred_space in {x, eps, v}.
    """
    if pred_space == "v":
        return pred
    elif pred_space == "x":
        return x_to_v(pred, z_t, t)
    elif pred_space == "eps":
        return eps_to_v(pred, z_t, t)
    else:
        raise ValueError(f"Use compute_V_theta for pred_space='u'. Got: {pred_space}")


def to_x(pred: torch.Tensor, pred_space: str,
         z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Convert a prediction in pred_space {x, eps, v} to x-space.
    For u-pred, first call compute_V_theta() to get V_theta, then v_to_x.

    pred_space in {x, eps, v}.
    """
    if pred_space == "x":
        return pred
    elif pred_space == "eps":
        return eps_to_x(pred, z_t, t)
    elif pred_space == "v":
        return v_to_x(pred, z_t, t)
    else:
        raise ValueError(f"Use compute_V_theta for pred_space='u'. Got: {pred_space}")


# -----------------------------------------------------------------------
# MeanFlow / iMF: u-pred -> V_theta (compound velocity)
# -----------------------------------------------------------------------

def compute_V_theta(
    net: nn.Module,
    z_t: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """
    Compute the compound velocity prediction V_theta for u-pred.

    V_theta = u_theta + (t - r) * JVP_sg

    where:
      u_theta  = net(z_t, r, t)                          [model prediction]
      v_tilde  = u_theta(sg(z_t), t, t)                  [boundary condition; stop-grad]
      JVP_sg   = ∂u_theta/∂z_t * v_tilde + ∂u_theta/∂t  [forward-mode AD; stop-grad]

    The JVP term captures how u_theta changes as we move along the path,
    making V_theta a more accurate local velocity estimate (MeanFlow identity).

    Stop-gradient strategy (following iMF):
      - v_tilde is computed on DETACHED inputs (sg(z_t)) -> no grad through v_tilde
      - JVP is computed on DETACHED inputs via no_grad context -> no grad through JVP
      - Gradient of the loss flows ONLY through u_theta = net(z_t, r, t)
      - This stabilizes training by avoiding second-order through the JVP tangent

    Args:
      net:  UModel -- takes (z_t, r, t) and returns (B, D)
      z_t:  (B, D)
      r:    (B, 1)
      t:    (B, 1)

    Returns:
      u_theta:  (B, D)  -- direct u-network output (carries grad for backprop)
      V_theta:  (B, D)  -- compound velocity (used for loss)
    """
    # Step 1: u_theta with gradient tracking (loss will backprop through this)
    u_theta = net(z_t, r, t)       # (B, D), has grad wrt model params

    # Step 2: boundary condition v_tilde = u_theta(sg(z_t), t, t)
    # Inputs are fully detached -> v_tilde has no grad
    with torch.no_grad():
        v_tilde = net(z_t.detach(), t.detach(), t.detach())   # (B, D)
    # v_tilde is now a plain tensor with no gradient.

    # Step 3: JVP of u_theta(z, r, t) wrt (z, t) in direction (v_tilde, 1)
    # This computes: ∂u/∂z * v_tilde + ∂u/∂t
    # All inputs are detached -> jvp_val has no grad wrt model params.
    # We capture r as a constant in the closure (detached).
    r_d  = r.detach()
    z_d  = z_t.detach()
    t_d  = t.detach()

    def u_fn(z: torch.Tensor, t_: torch.Tensor) -> torch.Tensor:
        # r_d is captured as a constant; only z and t_ are differentiated.
        return net(z, r_d, t_)

    with torch.no_grad():
        _, jvp_val = func_jvp(
            u_fn,
            (z_d, t_d),                          # primals
            (v_tilde, torch.ones_like(t_d)),      # tangents: (v_tilde for z, 1 for t)
        )
    # jvp_val: (B, D), no gradient

    # Step 4: compound velocity
    # V_theta = u_theta + (t - r) * jvp_val
    # Gradient flows only through u_theta (and (t-r) which are just scalars).
    V_theta = u_theta + (t - r) * jvp_val    # (B, D)

    return u_theta, V_theta


# -----------------------------------------------------------------------
# Sanity check helpers (used in tests)
# -----------------------------------------------------------------------

def verify_transforms(x: torch.Tensor, eps: torch.Tensor,
                      z_t: torch.Tensor, t: torch.Tensor,
                      tol: float = 1e-5) -> dict:
    """
    Verify algebraic consistency of transforms.
    Returns dict of {check_name: max_abs_error}.
    """
    results = {}

    # x -> v -> x
    v_from_x = x_to_v(x, z_t, t)
    x_recovered = v_to_x(v_from_x, z_t, t)
    results["x->v->x"] = (x - x_recovered).abs().max().item()

    # eps -> x -> eps
    x_from_eps = eps_to_x(eps, z_t, t)
    eps_recovered = x_to_eps(x_from_eps, z_t, t)
    results["eps->x->eps"] = (eps - eps_recovered).abs().max().item()

    # x -> eps, check consistency with ground truth eps
    eps_from_x = x_to_eps(x, z_t, t)
    results["x->eps error"] = (eps - eps_from_x).abs().max().item()

    # eps -> v, check consistency with ground truth v = eps - x
    v_true = eps - x
    v_from_eps = eps_to_v(eps, z_t, t)
    results["eps->v error"] = (v_true - v_from_eps).abs().max().item()

    # v_to_eps(x_to_v(x)) == eps
    eps_from_v = v_to_eps(x_to_v(x, z_t, t), z_t, t)
    results["x->v->eps error"] = (eps - eps_from_v).abs().max().item()

    return results
