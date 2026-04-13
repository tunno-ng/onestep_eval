"""
viz.py — Visualization utilities.

All plots work in 2D latent space (after projecting back via P^T).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_samples(
    true_2d: np.ndarray,
    gen_2d: np.ndarray,
    title: str,
    save_path: str,
    n_plot: int = 1000,
):
    """
    Scatter plot of true vs generated samples in 2D.
    """
    n_true = min(n_plot, len(true_2d))
    n_gen  = min(n_plot, len(gen_2d))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(true_2d[:n_true, 0], true_2d[:n_true, 1],
                    s=5, alpha=0.5, c="steelblue", rasterized=True)
    axes[0].set_title("True samples (2D projection)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(*_auto_lim(true_2d[:, 0]))
    axes[0].set_ylim(*_auto_lim(true_2d[:, 1]))

    axes[1].scatter(gen_2d[:n_gen, 0], gen_2d[:n_gen, 1],
                    s=5, alpha=0.5, c="tomato", rasterized=True)
    axes[1].set_title("Generated samples (2D projection)")
    axes[1].set_aspect("equal")
    # Use true data limits for fair comparison
    axes[1].set_xlim(*_auto_lim(true_2d[:, 0]))
    axes[1].set_ylim(*_auto_lim(true_2d[:, 1]))

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_samples_overlay(
    true_2d: np.ndarray,
    gen_2d: np.ndarray,
    title: str,
    save_path: str,
    n_plot: int = 1000,
):
    """
    Overlay true and generated samples in a single plot.
    """
    n_true = min(n_plot, len(true_2d))
    n_gen  = min(n_plot, len(gen_2d))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_2d[:n_true, 0], true_2d[:n_true, 1],
               s=5, alpha=0.4, c="steelblue", label="True", rasterized=True)
    ax.scatter(gen_2d[:n_gen, 0], gen_2d[:n_gen, 1],
               s=5, alpha=0.4, c="tomato", label="Generated", rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectory(
    trajectory_2d: list,
    true_2d: np.ndarray,
    title: str,
    save_path: str,
    n_lines: int = 50,
    n_plot: int = 300,
):
    """
    Plot multi-step generation trajectories in 2D.

    trajectory_2d: list of (N, 2) arrays, one per step (from t=1 to t=0)
    """
    n_steps = len(trajectory_2d)
    n_show = min(n_lines, trajectory_2d[0].shape[0])
    n_true = min(n_plot, len(true_2d))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot true data background
    ax.scatter(true_2d[:n_true, 0], true_2d[:n_true, 1],
               s=5, alpha=0.2, c="steelblue", rasterized=True, label="True")

    # Color-code steps
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_steps - 1, 1)) for i in range(n_steps)]

    # Plot trajectories for a subset of samples
    for sample_idx in range(n_show):
        xs = [trajectory_2d[step][sample_idx, 0] for step in range(n_steps)]
        ys = [trajectory_2d[step][sample_idx, 1] for step in range(n_steps)]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.7)

    # Plot points at each step
    for step_idx in range(n_steps):
        pts = trajectory_2d[step_idx][:n_show]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6,
                   color=colors[step_idx], zorder=3, rasterized=True)

    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_loss_curve(
    loss_history: list,
    title: str,
    save_path: str,
):
    """
    Plot training loss curve.
    loss_history: list of (iter, loss_val) tuples.
    """
    iters = [x[0] for x in loss_history]
    losses = [x[1] for x in loss_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, losses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _auto_lim(vals: np.ndarray, pad_frac: float = 0.1):
    """Auto range with a bit of padding."""
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad
