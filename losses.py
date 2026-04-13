"""
losses.py — All 8 training objectives, implemented explicitly.

CONVENTION: z_t = (1-t)*x + t*eps,  v = eps - x

==============================================================
TIER 1: v-loss (loss computed in v-space)
==============================================================

1. x-pred + v-loss:
       x_theta = net(z_t, t)
       v_theta = (z_t - x_theta) / t
       L = E[||v_theta - (eps - x)||^2]

2. eps-pred + v-loss:
       eps_theta = net(z_t, t)
       v_theta = (eps_theta - z_t) / (1 - t)
       L = E[||v_theta - (eps - x)||^2]

3. v-pred + v-loss:
       v_theta = net(z_t, t)
       L = E[||v_theta - (eps - x)||^2]

4. u-pred + v-loss:
       u_theta = net(z_t, r, t)
       V_theta = u_theta + (t-r) * JVP_sg(u_theta; v_tilde_theta)
       L = E[||V_theta - (eps - x)||^2]

==============================================================
TIER 2: x-loss (loss computed in x-space)
==============================================================

5. x-pred + x-loss:
       x_theta = net(z_t, t)
       L = E[||x_theta - x||^2]

6. eps-pred + x-loss:
       eps_theta = net(z_t, t)
       x_theta = (z_t - t * eps_theta) / (1 - t)
       L = E[||x_theta - x||^2]

7. v-pred + x-loss:
       v_theta = net(z_t, t)
       x_theta = z_t - t * v_theta
       L = E[||x_theta - x||^2]

8. u-pred + x-loss:
       u_theta = net(z_t, r, t)
       V_theta = u_theta + (t-r) * JVP_sg(u_theta; v_tilde_theta)
       x_theta = z_t - t * V_theta
       L = E[||x_theta - x||^2]

==============================================================
TARGET SMOOTHING (tau)
==============================================================

When tau > 0, targets are derived from the smoothed clean signal:
    x_eff = (1 - tau) * x + tau * eps

For v-loss: v_eff = eps - x_eff = (1-tau)*(eps - x)
For x-loss: target = x_eff

The interpolation path still uses original x (path is unchanged by tau).

==============================================================
EXTENSION NOTES
==============================================================

eps-loss and u-loss are not implemented in Tier 1/2 but the structure
makes them trivial to add. For eps-loss:
    - Convert pred -> eps using transforms.py formulas
    - Compare to eps target
For u-loss: same idea, but u = v on the linear path, so u-loss == v-loss.
"""
import torch
import torch.nn.functional as F

from transforms import (
    x_to_v, eps_to_v, v_to_x, eps_to_x,
    compute_V_theta,
)
from models import UModel


def _get_targets(x: torch.Tensor, eps: torch.Tensor, tau: float) -> tuple:
    """
    Compute smoothed targets.
    Returns: (x_eff, v_eff) where
      x_eff = (1-tau)*x + tau*eps
      v_eff = eps - x_eff
    """
    x_eff = (1.0 - tau) * x + tau * eps    # smoothed clean target
    v_eff = eps - x_eff                    # = (1-tau)*(eps - x)
    return x_eff, v_eff


# -----------------------------------------------------------------------
# Tier 1: v-loss
# -----------------------------------------------------------------------

def loss_x_vloss(
    net: torch.nn.Module,
    z_t: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    x-pred + v-loss.
    x_theta = net(z_t, t)
    v_theta = (z_t - x_theta) / t
    L = MSE(v_theta, eps - x_eff)
    """
    _, v_eff = _get_targets(x, eps, tau)
    x_theta = net(z_t, t)                   # (B, D)
    v_theta = x_to_v(x_theta, z_t, t)       # (z_t - x_theta) / t
    return F.mse_loss(v_theta, v_eff)


def loss_eps_vloss(
    net: torch.nn.Module,
    z_t: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    eps-pred + v-loss.
    eps_theta = net(z_t, t)
    v_theta = (eps_theta - z_t) / (1 - t)
    L = MSE(v_theta, eps - x_eff)
    """
    _, v_eff = _get_targets(x, eps, tau)
    eps_theta = net(z_t, t)                 # (B, D)
    v_theta = eps_to_v(eps_theta, z_t, t)  # (eps_theta - z_t) / (1-t)
    return F.mse_loss(v_theta, v_eff)


def loss_v_vloss(
    net: torch.nn.Module,
    z_t: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    v-pred + v-loss.
    v_theta = net(z_t, t)
    L = MSE(v_theta, eps - x_eff)
    """
    _, v_eff = _get_targets(x, eps, tau)
    v_theta = net(z_t, t)                  # (B, D)
    return F.mse_loss(v_theta, v_eff)


def loss_u_vloss(
    net: UModel,
    z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    u-pred + v-loss  (MeanFlow / iMF objective).
    u_theta = net(z_t, r, t)
    V_theta = u_theta + (t-r) * JVP_sg(u_theta; v_tilde_theta)
    L = MSE(V_theta, eps - x_eff)
    """
    _, v_eff = _get_targets(x, eps, tau)
    _, V_theta = compute_V_theta(net, z_t, r, t)   # (B, D)
    return F.mse_loss(V_theta, v_eff)


# -----------------------------------------------------------------------
# Tier 2: x-loss
# -----------------------------------------------------------------------

def loss_x_xloss(
    net: torch.nn.Module,
    z_t: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    x-pred + x-loss.
    x_theta = net(z_t, t)
    L = MSE(x_theta, x_eff)
    """
    x_eff, _ = _get_targets(x, eps, tau)
    x_theta = net(z_t, t)                  # (B, D)
    return F.mse_loss(x_theta, x_eff)


def loss_eps_xloss(
    net: torch.nn.Module,
    z_t: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    eps-pred + x-loss.
    eps_theta = net(z_t, t)
    x_theta = (z_t - t * eps_theta) / (1 - t)
    L = MSE(x_theta, x_eff)
    """
    x_eff, _ = _get_targets(x, eps, tau)
    eps_theta = net(z_t, t)                 # (B, D)
    x_theta = eps_to_x(eps_theta, z_t, t)  # (z_t - t*eps_theta) / (1-t)
    return F.mse_loss(x_theta, x_eff)


def loss_v_xloss(
    net: torch.nn.Module,
    z_t: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    v-pred + x-loss.
    v_theta = net(z_t, t)
    x_theta = z_t - t * v_theta
    L = MSE(x_theta, x_eff)
    """
    x_eff, _ = _get_targets(x, eps, tau)
    v_theta = net(z_t, t)                  # (B, D)
    x_theta = v_to_x(v_theta, z_t, t)     # z_t - t * v_theta
    return F.mse_loss(x_theta, x_eff)


def loss_u_xloss(
    net: UModel,
    z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
    x: torch.Tensor, eps: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    u-pred + x-loss  (MeanFlow compound, x-space comparison).
    u_theta = net(z_t, r, t)
    V_theta = u_theta + (t-r) * JVP_sg(u_theta; v_tilde_theta)
    x_theta = z_t - t * V_theta
    L = MSE(x_theta, x_eff)
    """
    x_eff, _ = _get_targets(x, eps, tau)
    _, V_theta = compute_V_theta(net, z_t, r, t)   # (B, D)
    x_theta = v_to_x(V_theta, z_t, t)              # z_t - t * V_theta
    return F.mse_loss(x_theta, x_eff)


# -----------------------------------------------------------------------
# Dispatch table
# -----------------------------------------------------------------------

# Maps (pred_space, loss_space) -> loss function
_LOSS_FN = {
    ("x",   "v"): loss_x_vloss,
    ("eps", "v"): loss_eps_vloss,
    ("v",   "v"): loss_v_vloss,
    ("u",   "v"): loss_u_vloss,
    ("x",   "x"): loss_x_xloss,
    ("eps", "x"): loss_eps_xloss,
    ("v",   "x"): loss_v_xloss,
    ("u",   "x"): loss_u_xloss,
}


def compute_loss(
    net: torch.nn.Module,
    z_t: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    eps: torch.Tensor,
    pred_space: str,
    loss_space: str,
    tau: float,
    r: torch.Tensor = None,   # required when pred_space == 'u'
) -> torch.Tensor:
    """
    Dispatch to the appropriate loss function.

    For pred_space == 'u', r must be provided.
    For all others, r is ignored.
    """
    key = (pred_space, loss_space)
    if key not in _LOSS_FN:
        raise ValueError(
            f"Unsupported (pred_space, loss_space) = {key}. "
            f"Supported: {list(_LOSS_FN.keys())}"
        )

    fn = _LOSS_FN[key]

    if pred_space == "u":
        if r is None:
            raise ValueError("r must be provided for pred_space='u'")
        return fn(net, z_t, r, t, x, eps, tau)
    else:
        return fn(net, z_t, t, x, eps, tau)
