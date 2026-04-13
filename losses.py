"""
losses.py — Loss computation supporting independent pred_space and loss_space.

The model outputs a prediction in pred_space.
The loss is computed in loss_space.

If pred_space == loss_space: direct MSE between prediction and target.
If pred_space != loss_space: convert prediction into loss_space, then MSE.

Conversion route:
  pred (pred_space) -> x0_pred -> target (loss_space)

This uses the actual x1 (noise sample) and xt (noisy sample) as context.
"""
import torch
import torch.nn.functional as F
from spaces import pred_to_x, x0_to_space, targets


def compute_loss(
    pred: torch.Tensor,
    pred_space: str,
    loss_space: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    tau: float = 0.0,
) -> torch.Tensor:
    """
    Compute MSE loss between model prediction and the loss_space target.

    Args:
      pred:       (B, D) model output in pred_space
      pred_space: space in which pred is defined
      loss_space: space in which to compute the loss
      x0:         (B, D) clean data
      x1:         (B, D) noise
      xt:         (B, D) interpolated point
      t:          (B, 1) time values
      tau:        float, target smoothing parameter

    Returns:
      scalar loss

    Notes on tau (target smoothing):
      When tau > 0, the "effective clean target" used for x-space comparisons is:
        x0_eff = (1-tau)*x0 + tau*x1
      rather than x0. This makes the x/u targets softer.
      For v and eps, smoothing is applied via x0_eff as well.
    """
    # Apply target smoothing: replace x0 with smoothed version
    if tau > 0.0:
        x0_eff = (1.0 - tau) * x0 + tau * x1
    else:
        x0_eff = x0

    # Compute ground-truth target in loss_space using (possibly smoothed) x0_eff
    target = targets(x0_eff, x1)[loss_space]  # (B, D)

    # Compute prediction in loss_space
    if pred_space == loss_space:
        pred_in_loss_space = pred
    else:
        # Convert pred -> x0_pred -> loss_space
        # Use unsmoothed xt, t for the conversion (geometric quantities)
        x0_pred = pred_to_x(pred, pred_space, xt, t)
        pred_in_loss_space = x0_to_space(x0_pred, loss_space, x1)

    loss = F.mse_loss(pred_in_loss_space, target)
    return loss
