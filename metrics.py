"""
metrics.py — Quantitative evaluation metrics.

==============================================================
EVALUATION PHILOSOPHY
==============================================================

Raw training loss is NOT a valid cross-method comparison metric,
because different loss spaces (v-loss, x-loss) have different scales.

All cross-method comparisons use COMMON evaluation spaces:

  1. x-MSE:  E[ ||x_hat - x||^2 ]   — endpoint accuracy
  2. v-MSE:  E[ ||v_hat - v||^2 ]   — velocity-space accuracy
  3. MMD:    distributional quality of generated samples

These are computed for every method regardless of pred_space / loss_space.

==============================================================
FUNCTIONS
==============================================================

  eval_common_metrics(model, data_D, pred_space, device, ...)
    -> x-MSE and v-MSE at fixed t values, for all pred_spaces

  compute_metrics(gen_2d, true_2d)
    -> MMD between generated and true 2D samples

  mmd_2d / mmd_rbf
    -> RBF-kernel MMD implementation
"""
import torch
import torch.nn as nn
import numpy as np

from paths import interpolate, T_EPS
from transforms import to_x, to_v, v_to_x, compute_V_theta


# -----------------------------------------------------------------------
# MMD
# -----------------------------------------------------------------------

def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float = None) -> float:
    """
    Unbiased MMD^2 estimate using RBF kernel.

    X: (n, d), Y: (m, d)
    Returns: MMD^2 (float). Can be slightly negative due to unbiased estimator.
    """
    n, m = X.shape[0], Y.shape[0]

    def sq_dist(A, B):
        A2 = (A ** 2).sum(1, keepdims=True)
        B2 = (B ** 2).sum(1, keepdims=True)
        return A2 + B2.T - 2 * (A @ B.T)

    XX, YY, XY = sq_dist(X, X), sq_dist(Y, Y), sq_dist(X, Y)

    if bandwidth is None:
        all_d = np.concatenate([XX.flatten(), YY.flatten(), XY.flatten()])
        bandwidth = max(float(np.median(all_d[all_d > 0])) ** 0.5, 1e-5)

    h = 2.0 * bandwidth ** 2
    Kxx, Kyy, Kxy = np.exp(-XX / h), np.exp(-YY / h), np.exp(-XY / h)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    return float(Kxx.sum() / (n * (n - 1))
                 + Kyy.sum() / (m * (m - 1))
                 - 2.0 * Kxy.mean())


def mmd_2d(gen_2d: np.ndarray, true_2d: np.ndarray, bandwidth: float = None) -> float:
    return mmd_rbf(gen_2d, true_2d, bandwidth)


def compute_metrics(gen_2d: np.ndarray, true_2d: np.ndarray) -> dict:
    """MMD between generated and true 2D samples."""
    return {
        "mmd_2d": mmd_2d(gen_2d, true_2d),
        "n_generated": len(gen_2d),
        "n_true": len(true_2d),
    }


# -----------------------------------------------------------------------
# Common evaluation metrics (x-MSE, v-MSE) — primary cross-method metrics
# -----------------------------------------------------------------------

_DEFAULT_T_VALUES = [0.25, 0.5, 0.75, 1.0]


def eval_common_metrics(
    model: nn.Module,
    data_D: torch.Tensor,            # (N, D) eval dataset in R^D
    pred_space: str,                  # {x, eps, v, u}
    device: torch.device,
    n_samples: int = 2000,
    t_values: list = None,
) -> dict:
    """
    Compute x-MSE and v-MSE for a model, in common evaluation spaces.

    Works for ALL pred_spaces: x, eps, v, u.

    For each t in t_values:
      1. Sample n_samples pairs (x, eps), form z_t = (1-t)*x + t*eps
      2. Run model, convert output to x_hat and v_hat:
           x/eps/v pred:  use algebraic transforms (to_x, to_v)
           u pred:        V_theta = compute_V_theta(model, z_t, r=0, t)
                          v_hat = V_theta,  x_hat = z_t - t * V_theta
      3. Compute x_mse = E[||x_hat - x||^2], v_mse = E[||v_hat - v||^2]

    For u-pred, r=0 corresponds to the one-step generation scenario
    (predict from current time t all the way to r=0).

    NOTE: compute_V_theta uses torch.func.jvp and requires gradient tracking.
    This function handles grad context per-branch internally.

    Returns dict with keys:
      "x_mse_t{t}"   : x-space MSE at each t
      "v_mse_t{t}"   : v-space MSE at each t
      "x_mse_mean"   : mean x-MSE over t_values
      "v_mse_mean"   : mean v-MSE over t_values
    """
    if t_values is None:
        t_values = _DEFAULT_T_VALUES

    model.eval()
    N = data_D.shape[0]
    results = {}
    x_mse_list, v_mse_list = [], []

    for t_val in t_values:
        idx = torch.randint(0, N, (n_samples,))
        x   = data_D[idx].to(device)
        eps = torch.randn_like(x)

        t_safe   = float(np.clip(t_val, T_EPS, 1.0 - T_EPS))
        t_tensor = torch.full((n_samples, 1), t_safe, device=device)
        z_t      = interpolate(x, eps, t_tensor)
        v_true   = eps - x                              # ground truth v

        if pred_space == "u":
            # u-pred: use DIRECT inference formula (r=0, one-step jump).
            #   x_hat = z_t - t * u_theta(z_t, r=0, t)
            #   v_hat = u_theta             (u ≈ v on the linear path)
            #
            # We do NOT use compute_V_theta (the compound JVP-based prediction)
            # here because V_theta = u + t*JVP can explode for randomly
            # initialized networks, making early-training val curves unreadable.
            # The direct formula is the correct inference-time evaluation.
            with torch.no_grad():
                r_tensor = torch.zeros(n_samples, 1, device=device)
                u_pred   = model(z_t, r_tensor, t_tensor)
            v_hat = u_pred
            x_hat = v_to_x(u_pred, z_t, t_tensor)   # z_t - t * u_pred
        else:
            with torch.no_grad():
                pred  = model(z_t, t_tensor)
                v_hat = to_v(pred, pred_space, z_t, t_tensor)
                x_hat = to_x(pred, pred_space, z_t, t_tensor)

        x_mse = ((x_hat - x) ** 2).mean().item()
        v_mse = ((v_hat - v_true) ** 2).mean().item()

        results[f"x_mse_t{t_val}"] = x_mse
        results[f"v_mse_t{t_val}"] = v_mse
        x_mse_list.append(x_mse)
        v_mse_list.append(v_mse)

    results["x_mse_mean"] = float(np.mean(x_mse_list))
    results["v_mse_mean"] = float(np.mean(v_mse_list))
    return results
