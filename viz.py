"""
viz.py — Visualization utilities.

==============================================================
PRESENTATION PRIORITY (follow this order in all outputs)
==============================================================

When presenting results, prioritize:
  1. Side-by-side scatter plots (generated vs real in 2D)
  2. val_x_mse  (primary cross-method metric)
  3. val_v_mse  (primary cross-method metric)
  4. MMD        (distributional quality)
  5. Training loss curves (secondary diagnostic only)

Raw training loss is NOT directly comparable across different loss_spaces.
val_x_mse and val_v_mse use a common evaluation space and ARE comparable.

==============================================================
FUNCTIONS
==============================================================

Per-experiment plots (called by run.py):
  plot_eval_summary       -- scatter + val metrics on one page (PRIMARY)
  plot_training_curves    -- val MSE (primary) + train loss (secondary)
  plot_samples            -- side-by-side scatter
  plot_samples_overlay    -- overlay scatter
  plot_trajectory         -- multi-step trajectory

Cross-experiment plots (called by compare.py):
  plot_comparison_grid    -- multiple methods side-by-side
  plot_metric_vs_x        -- metric vs obs_dim / tau / steps
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Per-experiment: primary plot — eval summary
# -----------------------------------------------------------------------

def plot_eval_summary(
    true_2d: np.ndarray,
    gen_2d: np.ndarray,
    val_x_mse_history: list,     # [(iter, val), ...]
    val_v_mse_history: list,     # [(iter, val), ...]
    final_mmd: float,
    title: str,
    save_path: str,
    n_plot: int = 800,
):
    """
    Primary per-experiment figure.

    Layout (1 row, 3 panels):
      [Scatter: true vs gen] [val_x_mse + val_v_mse vs iter] [metrics text]

    Scatter is first because it is the most important qualitative diagnostic.
    val_x_mse and val_v_mse are shown before train loss because they are
    the primary cross-method comparable signals.
    """
    has_val = len(val_x_mse_history) > 0

    fig = plt.figure(figsize=(14, 4))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.8, 0.6], wspace=0.35)

    # ---- Panel 1: scatter ----
    ax0 = fig.add_subplot(gs[0])
    n_t = min(n_plot, len(true_2d))
    n_g = min(n_plot, len(gen_2d))
    ax0.scatter(true_2d[:n_t, 0], true_2d[:n_t, 1],
                s=4, alpha=0.4, c="steelblue", label="Real", rasterized=True)
    ax0.scatter(gen_2d[:n_g, 0], gen_2d[:n_g, 1],
                s=4, alpha=0.4, c="tomato", label="Generated", rasterized=True)
    ax0.set_aspect("equal")
    ax0.set_xlim(*_auto_lim(true_2d[:, 0]))
    ax0.set_ylim(*_auto_lim(true_2d[:, 1]))
    ax0.legend(markerscale=3, fontsize=8)
    ax0.set_title("Generated vs Real (2D)", fontsize=9)

    # ---- Panel 2: val MSE curves (primary) ----
    ax1 = fig.add_subplot(gs[1])
    if has_val:
        iters_x = [v[0] for v in val_x_mse_history]
        vals_x  = [v[1] for v in val_x_mse_history]
        iters_v = [v[0] for v in val_v_mse_history]
        vals_v  = [v[1] for v in val_v_mse_history]
        ax1.plot(iters_x, vals_x, "-o", ms=3, lw=1.5,
                 color="tomato", label="val x-MSE")
        ax1.plot(iters_v, vals_v, "-s", ms=3, lw=1.5,
                 color="steelblue", label="val v-MSE")
        ax1.set_xlabel("Iteration", fontsize=8)
        ax1.set_ylabel("MSE (common eval space)", fontsize=8)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No val history\n(eval_every=0)",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=9)
        ax1.set_axis_off()
    ax1.set_title("val x-MSE / val v-MSE  [primary metrics]", fontsize=9)

    # ---- Panel 3: summary text ----
    ax2 = fig.add_subplot(gs[2])
    ax2.set_axis_off()
    lines = ["Metrics\n"]
    if has_val:
        final_xmse = val_x_mse_history[-1][1]
        final_vmse = val_v_mse_history[-1][1]
        lines += [
            f"x-MSE: {final_xmse:.4f}",
            f"v-MSE: {final_vmse:.4f}",
        ]
    lines.append(f"MMD:   {final_mmd:.4f}")
    ax2.text(0.05, 0.95, "\n".join(lines),
             va="top", ha="left", transform=ax2.transAxes,
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(title, fontsize=10)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Per-experiment: training curves (secondary diagnostic)
# -----------------------------------------------------------------------

def plot_training_curves(
    train_result: dict,
    title: str,
    save_path: str,
):
    """
    Training diagnostic figure.

    Layout (2 panels):
      Top:    val_x_mse + val_v_mse  [primary — common eval spaces]
      Bottom: training loss          [secondary — diagnostic only, NOT cross-method comparable]

    The top panel is larger to reinforce that val MSE is the primary signal.
    The bottom panel carries a warning label about scale non-comparability.
    """
    loss_hist = train_result.get("loss_history", [])
    val_x     = train_result.get("val_x_mse_history", [])
    val_v     = train_result.get("val_v_mse_history", [])
    has_val   = len(val_x) > 0

    fig, axes = plt.subplots(2, 1, figsize=(9, 6),
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: val MSE (primary)
    ax_top = axes[0]
    if has_val:
        ax_top.plot([v[0] for v in val_x], [v[1] for v in val_x],
                    "-o", ms=4, lw=1.8, color="tomato", label="val x-MSE")
        ax_top.plot([v[0] for v in val_v], [v[1] for v in val_v],
                    "-s", ms=4, lw=1.8, color="steelblue", label="val v-MSE")
        ax_top.legend(fontsize=9)
    else:
        ax_top.text(0.5, 0.5, "val_x_mse / val_v_mse\n(eval_every=0, not computed)",
                    ha="center", va="center", transform=ax_top.transAxes, fontsize=9)
    ax_top.set_ylabel("MSE")
    ax_top.set_title("val x-MSE and val v-MSE  [PRIMARY — comparable across methods]",
                     fontsize=9)
    ax_top.grid(True, alpha=0.3)

    # Bottom: train loss (secondary)
    ax_bot = axes[1]
    if loss_hist:
        ax_bot.plot([v[0] for v in loss_hist], [v[1] for v in loss_hist],
                    lw=1.2, color="gray", alpha=0.8)
        ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylabel("Train loss")
    ax_bot.set_title(
        "Training loss  [DIAGNOSTIC ONLY — NOT comparable across different loss_spaces]",
        fontsize=8, color="dimgray"
    )
    ax_bot.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Per-experiment: scatter plots
# -----------------------------------------------------------------------

def plot_samples(
    true_2d: np.ndarray, gen_2d: np.ndarray,
    title: str, save_path: str, n_plot: int = 1000,
):
    n_t = min(n_plot, len(true_2d))
    n_g = min(n_plot, len(gen_2d))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(true_2d[:n_t, 0], true_2d[:n_t, 1],
                    s=5, alpha=0.5, c="steelblue", rasterized=True)
    axes[0].set_title("Real samples")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(*_auto_lim(true_2d[:, 0]))
    axes[0].set_ylim(*_auto_lim(true_2d[:, 1]))
    axes[1].scatter(gen_2d[:n_g, 0], gen_2d[:n_g, 1],
                    s=5, alpha=0.5, c="tomato", rasterized=True)
    axes[1].set_title("Generated samples")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(*_auto_lim(true_2d[:, 0]))
    axes[1].set_ylim(*_auto_lim(true_2d[:, 1]))
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_samples_overlay(
    true_2d: np.ndarray, gen_2d: np.ndarray,
    title: str, save_path: str, n_plot: int = 1000,
):
    n_t = min(n_plot, len(true_2d))
    n_g = min(n_plot, len(gen_2d))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_2d[:n_t, 0], true_2d[:n_t, 1],
               s=5, alpha=0.4, c="steelblue", label="Real", rasterized=True)
    ax.scatter(gen_2d[:n_g, 0], gen_2d[:n_g, 1],
               s=5, alpha=0.4, c="tomato", label="Generated", rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectory(
    trajectory_2d: list, true_2d: np.ndarray,
    title: str, save_path: str, n_lines: int = 50, n_plot: int = 300,
):
    n_steps = len(trajectory_2d)
    n_show  = min(n_lines, trajectory_2d[0].shape[0])
    n_true  = min(n_plot, len(true_2d))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_2d[:n_true, 0], true_2d[:n_true, 1],
               s=5, alpha=0.2, c="steelblue", rasterized=True, label="Real")
    cmap   = plt.cm.viridis
    colors = [cmap(i / max(n_steps - 1, 1)) for i in range(n_steps)]
    for si in range(n_show):
        xs = [trajectory_2d[k][si, 0] for k in range(n_steps)]
        ys = [trajectory_2d[k][si, 1] for k in range(n_steps)]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.7)
    for k in range(n_steps):
        pts = trajectory_2d[k][:n_show]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6,
                   color=colors[k], zorder=3, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Cross-experiment: comparison grid
# -----------------------------------------------------------------------

def plot_comparison_grid(
    panels: list,        # [(label, gen_2d, true_2d), ...]
    title: str,
    save_path: str,
    n_plot: int = 500,
):
    """
    Side-by-side scatter comparison of multiple methods.

    Each panel shows generated (tomato) overlaid on real (steelblue) in 2D.
    Axes are fixed to the range of the first true_2d for fair visual comparison.

    This is the PRIMARY cross-method qualitative diagnostic.
    """
    n = len(panels)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    # Compute axis limits from the first panel's true_2d
    ref_true = panels[0][2]
    xlim = _auto_lim(ref_true[:, 0])
    ylim = _auto_lim(ref_true[:, 1])

    for i, (label, gen_2d, true_2d) in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        n_t = min(n_plot, len(true_2d))
        n_g = min(n_plot, len(gen_2d))
        ax.scatter(true_2d[:n_t, 0], true_2d[:n_t, 1],
                   s=4, alpha=0.35, c="steelblue", rasterized=True, label="Real")
        ax.scatter(gen_2d[:n_g, 0], gen_2d[:n_g, 1],
                   s=4, alpha=0.35, c="tomato", rasterized=True, label="Gen")
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Hide unused axes
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    # Add legend to first panel only
    axes[0][0].legend(markerscale=3, fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Cross-experiment: metric vs scalar (obs_dim, tau, steps, ...)
# -----------------------------------------------------------------------

def plot_metric_vs_x(
    records: list,
    x_key: str,
    metric_keys: list,
    title: str,
    save_path: str,
    x_label: str = None,
    log_x: bool = False,
    log_y: bool = False,
    group_by: str = None,   # if set, draw one line per unique value of this field
):
    """
    Plot one or more metrics against a scalar variable.

    Typical uses:
      - x_key="obs_dim",   metric_keys=["x_mse_mean","v_mse_mean","mmd_2d"]
      - x_key="tau",       metric_keys=["x_mse_mean","mmd_2d"]
      - x_key="steps",     metric_keys=["x_mse_mean","v_mse_mean","mmd_2d"]

    If group_by is set (e.g. group_by="pred_space"), draws one line per group
    for each metric_key, making comparisons across methods easy.

    Metric ordering in legend follows presentation priority:
      x_mse_mean, v_mse_mean, mmd_2d, then others.
    """
    # Sort records by x_key (numeric if possible, else lexicographic)
    def _sort_key(r):
        v = r.get(x_key, "")
        try:
            return (0, float(v))
        except (ValueError, TypeError):
            return (1, str(v))
    records = sorted(records, key=_sort_key)

    # Priority ordering for legend
    priority = ["x_mse_mean", "v_mse_mean", "mmd_2d"]
    metric_keys = sorted(metric_keys,
                         key=lambda k: priority.index(k) if k in priority else 99)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if group_by is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [r[x_key] for r in records if x_key in r]
        for i, mk in enumerate(metric_keys):
            ys = [r.get(mk, float("nan")) for r in records if x_key in r]
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)],
                    lw=2, ms=6, label=mk)
        ax.set_xlabel(x_label or x_key)
        ax.set_ylabel("Metric value")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if log_x: ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
    else:
        # One subplot per metric_key; within each, one line per group_by value
        n_metrics = len(metric_keys)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4),
                                 squeeze=False)
        groups = sorted(set(r.get(group_by, "?") for r in records))
        for mi, mk in enumerate(metric_keys):
            ax = axes[0][mi]
            for gi, g in enumerate(groups):
                sub = [r for r in records if r.get(group_by) == g and x_key in r]
                sub = sorted(sub, key=_sort_key)
                xs = [r[x_key] for r in sub]
                ys = [r.get(mk, float("nan")) for r in sub]
                ax.plot(xs, ys, "o-", color=colors[gi % len(colors)],
                        lw=2, ms=6, label=str(g))
            ax.set_xlabel(x_label or x_key)
            ax.set_ylabel(mk)
            ax.set_title(mk, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if log_x: ax.set_xscale("log")
            if log_y: ax.set_yscale("log")
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _auto_lim(vals: np.ndarray, pad_frac: float = 0.1):
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad
