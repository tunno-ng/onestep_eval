"""
metrics.py — Quantitative evaluation metrics.

Implemented:
  - endpoint_mse: MSE between generated and true samples in 2D latent space
                  (only meaningful if we have paired targets, which we don't in general)
  - mmd_2d: Maximum Mean Discrepancy between generated and true 2D samples
            using RBF kernel, bandwidth selected by median heuristic

MMD is the main metric since we don't have paired targets.
It measures distributional similarity, not per-sample accuracy.
"""
import torch
import numpy as np


def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float = None) -> float:
    """
    Unbiased MMD^2 estimate using RBF kernel.

    X: (n, d), Y: (m, d) -- samples from two distributions
    bandwidth: RBF kernel bandwidth (sigma). If None, uses median heuristic.

    Returns: MMD^2 estimate (float). Negative values can occur with unbiased estimator.
    """
    n = X.shape[0]
    m = Y.shape[0]

    # Compute pairwise squared distances
    def sq_dist(A, B):
        # (n, d), (m, d) -> (n, m)
        A2 = (A ** 2).sum(1, keepdims=True)  # (n, 1)
        B2 = (B ** 2).sum(1, keepdims=True)  # (m, 1)
        AB = A @ B.T                           # (n, m)
        return A2 + B2.T - 2 * AB

    XX = sq_dist(X, X)
    YY = sq_dist(Y, Y)
    XY = sq_dist(X, Y)

    # Median heuristic for bandwidth
    if bandwidth is None:
        all_dists = np.concatenate([XX.flatten(), YY.flatten(), XY.flatten()])
        bandwidth = np.median(all_dists[all_dists > 0]) ** 0.5
        bandwidth = max(bandwidth, 1e-5)

    h = 2.0 * bandwidth ** 2

    # RBF kernel
    Kxx = np.exp(-XX / h)
    Kyy = np.exp(-YY / h)
    Kxy = np.exp(-XY / h)

    # Unbiased MMD^2
    # Remove diagonal for XX and YY (unbiased)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())

    return float(mmd2)


def mmd_2d(gen_2d: np.ndarray, true_2d: np.ndarray, bandwidth: float = None) -> float:
    """
    MMD in 2D latent space between generated and true samples.

    gen_2d:  (N, 2) -- generated samples projected to 2D
    true_2d: (M, 2) -- true samples in 2D
    """
    return mmd_rbf(gen_2d, true_2d, bandwidth)


def compute_metrics(gen_2d: np.ndarray, true_2d: np.ndarray) -> dict:
    """
    Compute all metrics and return as dict.

    gen_2d:  (N, 2)
    true_2d: (M, 2)
    """
    mmd = mmd_2d(gen_2d, true_2d)
    return {
        "mmd_2d": mmd,
        "n_generated": len(gen_2d),
        "n_true": len(true_2d),
    }
