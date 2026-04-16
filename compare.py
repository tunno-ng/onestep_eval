"""
compare.py — Cross-experiment comparison: summary table + comparison plots.

==============================================================
USAGE
==============================================================

# Compare a fixed set of experiment dirs:
python compare.py --dirs exp/dir1 exp/dir2 exp/dir3 --out_dir compare_out/

# Scan a parent directory with optional substring filter:
python compare.py --parent exp --filter "D16_lossv_steps1" --out_dir compare_out/

# Group A: Tier 1 comparison at D=16
python compare.py --parent exp --filter "8gaussians_D16" --filter2 "steps1" \\
                  --out_dir compare_out/tierA_D16

# Obs-dim scaling plot, grouped by pred_space:
python compare.py --parent exp --filter "8gaussians" --filter2 "lossv_steps1" \\
                  --x_key obs_dim --group_by pred_space --out_dir compare_out/scaling

# Tau sweep:
python compare.py --parent exp --filter "moons_D64_predx_lossv_steps1" \\
                  --x_key tau --out_dir compare_out/tau_sweep

==============================================================
OUTPUTS (saved to out_dir/)
==============================================================

  summary.csv           — table: pred_space, loss_space, obs_dim, tau, steps,
                          final_val_x_mse, final_val_v_mse, final_mmd,
                          final_train_loss
  comparison_grid.png   — side-by-side scatter (if --plot_grids)
  metric_vs_{x_key}.png — metric vs x_key (one line per group_by value)

==============================================================
EVALUATION PRIORITY REMINDER
==============================================================

Cross-method comparison uses (in priority order):
  1. Scatter plots (generated vs real)
  2. final_val_x_mse
  3. final_val_v_mse
  4. final_mmd
  5. final_train_loss  [DIAGNOSTIC ONLY — not comparable across loss_spaces]
"""
import os
import csv
import json
import argparse
import numpy as np

from viz import plot_comparison_grid, plot_metric_vs_x


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------

def find_exp_dirs(parent: str, filters: list = None) -> list:
    """
    List immediate subdirectories of parent.
    Keep only those whose basename contains ALL strings in filters.
    """
    entries = []
    try:
        for name in sorted(os.listdir(parent)):
            path = os.path.join(parent, name)
            if not os.path.isdir(path):
                continue
            if filters:
                if not all(f in name for f in filters):
                    continue
            entries.append(path)
    except FileNotFoundError:
        print(f"[WARN] parent dir not found: {parent}")
    return entries


def load_exp(exp_dir: str, load_arrays: bool = False) -> dict:
    """
    Load config.json + metrics.json from exp_dir into a flat dict.
    Optionally also loads gen_samples_2d.npy and true_samples_2d.npy.
    Returns {} if either JSON is missing.
    """
    cfg_path = os.path.join(exp_dir, "config.json")
    met_path = os.path.join(exp_dir, "metrics.json")

    if not (os.path.exists(cfg_path) and os.path.exists(met_path)):
        print(f"  [skip] missing config/metrics in {exp_dir}")
        return {}

    with open(cfg_path) as f:
        record = json.load(f)
    with open(met_path) as f:
        record.update(json.load(f))

    record["exp_dir"] = exp_dir
    record["exp_name"] = os.path.basename(exp_dir)

    if load_arrays:
        gen_path  = os.path.join(exp_dir, "gen_samples_2d.npy")
        true_path = os.path.join(exp_dir, "true_samples_2d.npy")
        if os.path.exists(gen_path):
            record["gen_2d"] = np.load(gen_path)
        if os.path.exists(true_path):
            record["true_2d"] = np.load(true_path)

    return record


# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------

# Columns in priority order for cross-method comparison
_SUMMARY_COLS = [
    "dataset", "obs_dim", "pred_space", "loss_space", "steps", "tau", "seed",
    # Primary metrics (comparable across methods)
    "final_val_x_mse", "final_val_v_mse", "final_mmd",
    "x_mse_mean", "v_mse_mean", "mmd_2d",
    # Secondary (diagnostic only)
    "final_train_loss",
]


def build_summary_table(records: list, save_path: str = None) -> list:
    """
    Build a summary table from a list of experiment records.

    Sorts by (dataset, obs_dim, pred_space, loss_space).
    Prints to stdout and optionally writes CSV.

    Column order follows evaluation priority:
      x_mse, v_mse, mmd first; train_loss last.
    """
    if not records:
        print("[compare] No records to summarize.")
        return []

    # Determine which columns are actually present
    cols = [c for c in _SUMMARY_COLS if any(c in r for r in records)]

    def sort_key(r):
        return (r.get("dataset", ""), r.get("obs_dim", 0),
                r.get("pred_space", ""), r.get("loss_space", ""))

    records = sorted(records, key=sort_key)

    # Print table
    header = " | ".join(f"{c:>18}" for c in cols)
    print("\n" + "=" * len(header))
    print("COMPARISON TABLE  (priority: val_x_mse > val_v_mse > mmd > train_loss)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in records:
        row = " | ".join(f"{_fmt(r.get(c, '')):>18}" for c in cols)
        print(row)
    print("=" * len(header) + "\n")

    # Write CSV
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"  Saved: {save_path}")

    return records


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-experiment comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--dirs",   nargs="+", help="Explicit list of exp dirs")
    inp.add_argument("--parent", type=str,  help="Parent dir to scan")

    parser.add_argument("--filter",  type=str, default=None,
                        help="Substring filter for --parent")
    parser.add_argument("--filter2", type=str, default=None,
                        help="Second substring filter (both must match)")

    # Plot controls
    parser.add_argument("--x_key",     type=str, default="obs_dim",
                        help="x-axis variable for metric plot")
    parser.add_argument("--group_by",  type=str, default="pred_space",
                        help="Field to group lines by in metric plot")
    parser.add_argument("--metric_keys", nargs="+",
                        default=["x_mse_mean", "v_mse_mean", "mmd_2d"],
                        help="Metrics to plot (priority order)")
    parser.add_argument("--log_x",     action="store_true")
    parser.add_argument("--log_y",     action="store_true")
    parser.add_argument("--plot_grids", action="store_true",
                        help="Produce comparison grid scatter plots (loads .npy)")

    parser.add_argument("--out_dir", type=str, default="compare_out")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect experiment dirs
    if args.dirs:
        exp_dirs = args.dirs
    else:
        filters = [f for f in [args.filter, args.filter2] if f]
        exp_dirs = find_exp_dirs(args.parent, filters)

    if not exp_dirs:
        print("[compare] No experiment dirs found.")
        return

    print(f"[compare] Loading {len(exp_dirs)} experiments...")

    # Load scalar records
    records = [r for r in (load_exp(d) for d in exp_dirs) if r]
    if not records:
        print("[compare] All experiments failed to load.")
        return

    # 1. Summary table (primary metrics first)
    build_summary_table(records,
                        save_path=os.path.join(args.out_dir, "summary.csv"))

    # 2. Metric vs x_key plot
    # Check that x_key exists in at least some records
    valid = [r for r in records if args.x_key in r]
    if valid:
        plot_metric_vs_x(
            records=valid,
            x_key=args.x_key,
            metric_keys=args.metric_keys,
            title=f"Metrics vs {args.x_key}  [primary: x_mse, v_mse, mmd]",
            save_path=os.path.join(args.out_dir, f"metric_vs_{args.x_key}.png"),
            log_x=args.log_x,
            log_y=args.log_y,
            group_by=args.group_by if len(set(r.get(args.group_by) for r in valid)) > 1
                     else None,
        )
    else:
        print(f"  [skip] x_key='{args.x_key}' not found in records")

    # 3. Comparison grid scatter (PRIMARY qualitative diagnostic)
    if args.plot_grids:
        records_with_arrays = [r for r in
                               (load_exp(d, load_arrays=True) for d in exp_dirs) if r]
        panels = []
        for r in records_with_arrays:
            if "gen_2d" in r and "true_2d" in r:
                label = (f"{r.get('pred_space','?')}-pred\n"
                         f"{r.get('loss_space','?')}-loss")
                panels.append((label, r["gen_2d"], r["true_2d"]))

        if panels:
            # Sort by pred_space for consistent layout
            panels.sort(key=lambda p: p[0])
            group_info = ", ".join(
                f"{r.get('dataset')} D={r.get('obs_dim')} steps={r.get('steps')}"
                for r in records_with_arrays[:1]
            )
            plot_comparison_grid(
                panels=panels,
                title=f"Comparison grid  [PRIMARY: generated vs real]  |  {group_info}",
                save_path=os.path.join(args.out_dir, "comparison_grid.png"),
            )
        else:
            print("  [skip] comparison_grid: no gen_samples_2d.npy / true_samples_2d.npy found")

    print(f"\n[compare] Done. Results in: {args.out_dir}/")


if __name__ == "__main__":
    main()
