"""
spaces.py — Definitions and transformations between prediction spaces.

All definitions are under the linear interpolation path:
  xt = (1 - t) * x0 + t * x1,   x0 ~ p_data,  x1 ~ N(0, I)

==============================================================
SPACE DEFINITIONS
==============================================================

x  (endpoint / data):
  The clean data sample x0.
  Target: x_target = x0   (or smoothed: (1-tau)*x0 + tau*x1)

  Intuitively: "predict where you came from."
  Used in one-step generation by directly outputting the predicted x0.

eps (noise):
  The noise sample x1 = x0 + ... reparameterized as:
    eps = x1   (standard Gaussian draw)
  Or equivalently: the "noise" direction in diffusion models.
  xt = (1-t)*x0 + t*x1  =>  x1 = (xt - (1-t)*x0) / t   [t > 0]
  Target: eps_target = x1

  Intuitively: "predict what the noise was."

v  (instantaneous velocity / flow):
  The time derivative of xt along the path:
    v = d(xt)/dt = x1 - x0
  Target: v_target = x1 - x0

  Intuitively: "predict the direction of flow."
  Under this path, v is constant in t (straight-line flow).
  This is the standard "flow matching" velocity target.

u  (integrated / mean transport quantity for one-step reasoning):
  For one-step generation from x1 at t=1 to x0 at t=0,
  we want a quantity that captures the *integrated* displacement.

  We define:
    u = x0 - x1   (displacement from noise to data)

  This is the *negative* of v (since v = x1 - x0).
  u_target = x0 - x1

  Rationale: in one-step generation with a single step from t=1 to t=0,
  the ideal update is:  x0_pred = x1 + u_pred
  (or equivalently: x0_pred = x1 - v_pred).

  u is essentially a "denoising direction" or "reverse velocity."
  It differs from x only by the offset x1, but factoring out the
  noise can sometimes be a better-conditioned learning target.

  Alternative view: u can be seen as the "flow" that you'd integrate
  along a straight path from x1 to x0 in one step.

==============================================================
TRANSFORMATIONS
==============================================================

Given (x0, x1, t), the targets in each space are:
  x_target   = x0
  eps_target = x1
  v_target   = x1 - x0
  u_target   = x0 - x1

The targets are related by:
  u = -v
  x = xt - t * v      (data from velocity and current position)
  x = xt + t * u      (data from u and current position)
  eps = xt + (1-t)*u  (noise from u and current position)
       or = xt - (1-t)*v

Given a predicted quantity in one space, we can convert to another.
All conversions are explicit below.

NOTE: During one-step generation we use t=1 (start from pure noise),
so some conversions simplify at t=1:
  x   = x1 + 1*u = x1 + u_pred           (at t=1)
  v   = x1 - x0  => x0 = x1 - v_pred     (at t=1)
  u   = x0 - x1  => x0 = x1 + u_pred     (at t=1)
  eps = x1        => x0 = xt - t*x1/(?) ... (see below)
"""
import torch


def targets(x0: torch.Tensor, x1: torch.Tensor) -> dict:
    """
    Compute all four space targets given clean data x0 and noise x1.

    Returns a dict with keys {x, eps, v, u} each of shape (B, D).
    """
    return {
        "x":   x0,
        "eps": x1,
        "v":   x1 - x0,
        "u":   x0 - x1,
    }


# ---- Converting a prediction from one space into another ----
# We need xt and t to do general conversions.

def pred_to_x(pred: torch.Tensor, space: str,
              xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Convert a prediction in `space` to x-space (clean data prediction).

    pred: (B, D) - model output interpreted in `space`
    xt:   (B, D) - current noisy sample
    t:    (B, 1) - time values in [0, 1]
    Returns: (B, D) - predicted x0
    """
    if space == "x":
        return pred
    elif space == "u":
        # x0 = xt + t * u    [since xt = (1-t)*x0 + t*x1, u = x0-x1]
        # => x0 = xt + t*(x0 - x1)
        # => x0 = xt + t*u   where u = x0-x1
        # Check: xt + t*u = (1-t)*x0 + t*x1 + t*(x0-x1) = (1-t)*x0 + t*x0 = x0 ✓
        return xt + t * pred
    elif space == "v":
        # xt = (1-t)*x0 + t*x1 = x0 + t*(x1-x0) = x0 + t*v
        # => x0 = xt - t*v
        return xt - t * pred
    elif space == "eps":
        # xt = (1-t)*x0 + t*x1 = (1-t)*x0 + t*eps
        # => x0 = (xt - t*eps) / (1-t)    [undefined at t=1]
        # At t=1 this is singular; handle numerically with clamp
        denom = (1.0 - t).clamp(min=1e-5)
        return (xt - t * pred) / denom
    else:
        raise ValueError(f"Unknown space: {space}")


def pred_to_target_space(pred: torch.Tensor, pred_space: str, loss_space: str,
                         xt: torch.Tensor, t: torch.Tensor,
                         x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Convert model prediction (in pred_space) into loss_space for loss computation.

    Strategy: convert pred -> x0_pred, then convert x0_pred -> loss_space target.

    pred:       (B, D)
    pred_space: str in {x, u, v, eps}
    loss_space: str in {x, u, v, eps}
    xt, t, x0, x1: standard interpolation variables
    Returns: (B, D) prediction in loss_space
    """
    if pred_space == loss_space:
        return pred  # No conversion needed

    # Step 1: convert to x-space
    x0_pred = pred_to_x(pred, pred_space, xt, t)

    # Step 2: convert x0_pred to loss_space
    # We need x1 for this. In training, x1 is available.
    # x0_pred is our predicted clean data.
    # The "x1" for converting is just x1 (the actual noise), since
    # in a well-trained model x1 is the same whether we're predicting or not.
    # This is correct: loss is computed as distance to the *target* in loss_space.
    return x0_to_space(x0_pred, loss_space, x1)


def x0_to_space(x0: torch.Tensor, space: str, x1: torch.Tensor) -> torch.Tensor:
    """
    Convert x0 (clean data) to a target in `space`, given x1.
    Used to convert predictions or targets between spaces.
    """
    if space == "x":
        return x0
    elif space == "u":
        return x0 - x1
    elif space == "v":
        return x1 - x0
    elif space == "eps":
        return x1
    else:
        raise ValueError(f"Unknown space: {space}")


def pred_to_x0_at_t1(pred: torch.Tensor, space: str, x1: torch.Tensor) -> torch.Tensor:
    """
    Convert prediction to x0 specifically at t=1 (one-step generation).
    Avoids division by zero in eps-space.

    x1 is the starting noise sample (the model input).
    pred is the model output.
    """
    if space == "x":
        return pred
    elif space == "u":
        # x0 = x1 + u  (at t=1: xt=x1, so x0 = xt + t*u = x1 + u)
        return x1 + pred
    elif space == "v":
        # x0 = x1 - v  (at t=1: xt=x1, so x0 = xt - t*v = x1 - v)
        return x1 - pred
    elif space == "eps":
        # eps = x1, so pred ≈ x1
        # At t=1, eps-space prediction is just the noise itself.
        # We can't recover x0 from eps alone at t=1 without additional info.
        # Common trick: use the score function. Here we use the "DDPM-style" step:
        #   x0 = (x1 - sqrt(t)*eps) / sqrt(1-t) -- but our path is not score-based.
        # Under our linear path, at t=1: xt = x1 = (0)*x0 + 1*x1
        # eps = x1 is completely independent of x0 at t=1.
        # This means eps-space one-step generation is ill-defined under our path!
        # We handle this by treating eps prediction as if it were a denoising step:
        #   x0_pred = xt - pred  (subtract predicted noise, like DDPM at t=1)
        # This is a reasonable heuristic for this toy setup.
        return x1 - pred
    else:
        raise ValueError(f"Unknown space: {space}")
