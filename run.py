"""
run.py — Main entry point.

Trains a model, then evaluates and saves results in this priority order:
  1. Scatter plots (generated vs real)  <- primary qualitative diagnostic
  2. val_x_mse, val_v_mse              <- primary quantitative metrics
  3. MMD                               <- distributional quality
  4. Training curves                   <- diagnostic / secondary

Training loss is NOT used as the main comparison metric.
See metrics.py for the evaluation philosophy.
"""
import os
import numpy as np
import torch

from configs import get_args, make_exp_dir, save_config
from datasets import get_dataset, TensorDataset2D
from embedding import Embedding
from models import build_model
from train import train
from sample import generate
from metrics import compute_metrics, eval_common_metrics
from viz import (plot_eval_summary, plot_training_curves,
                 plot_samples, plot_samples_overlay, plot_trajectory)
from utils import set_seed, get_device, save_metrics, save_loss_history, save_samples


def main():
    cfg = get_args()
    set_seed(cfg.seed)
    device = get_device()

    exp_path = make_exp_dir(cfg)
    save_config(cfg, exp_path)

    print(f"Device: {device}")
    print(f"Exp dir: {exp_path}")
    print(f"pred={cfg.pred_space}  loss={cfg.loss_space}  "
          f"steps={cfg.steps}  tau={cfg.tau}  D={cfg.obs_dim}")

    # ---- Data ----
    rng           = np.random.default_rng(cfg.seed)
    data_2d_train = get_dataset(cfg.dataset, cfg.n_train, rng)
    data_2d_eval  = get_dataset(cfg.dataset, cfg.n_eval,  rng)

    emb          = Embedding(cfg.obs_dim, cfg.latent_dim, cfg.seed)
    data_D_train = emb.embed_numpy(data_2d_train)
    data_D_eval  = emb.embed_numpy(data_2d_eval)

    train_dataset   = TensorDataset2D(torch.from_numpy(data_D_train))
    val_data_D_t    = torch.from_numpy(data_D_eval)   # kept as tensor for eval

    # ---- Model ----
    model    = build_model(cfg.pred_space, cfg.obs_dim,
                           cfg.hidden_dim, cfg.n_layers, cfg.time_emb_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}  params={n_params:,}")

    # ---- Train ----
    eval_every = max(1, cfg.n_iters // 20)   # ~20 validation checkpoints
    print(f"\nTraining {cfg.n_iters} iters  (val every {eval_every})...")

    train_result = train(
        model=model,
        dataset=train_dataset,
        pred_space=cfg.pred_space,
        loss_space=cfg.loss_space,
        n_iters=cfg.n_iters,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        tau=cfg.tau,
        device=device,
        log_every=max(1, cfg.n_iters // 50),
        eval_every=eval_every,
        val_data_D=val_data_D_t,
        n_val_samples=min(cfg.n_eval, 1000),
    )

    loss_history      = train_result["loss_history"]
    val_x_mse_history = train_result["val_x_mse_history"]
    val_v_mse_history = train_result["val_v_mse_history"]

    save_loss_history(loss_history, os.path.join(exp_path, "loss_history.json"))
    torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))

    # ---- Generate ----
    print(f"\nGenerating {cfg.n_eval} samples (steps={cfg.steps})...")
    gen_D, trajectory = generate(
        model=model,
        n_samples=cfg.n_eval,
        obs_dim=cfg.obs_dim,
        pred_space=cfg.pred_space,
        steps=cfg.steps,
        device=device,
    )
    gen_D_np = gen_D.cpu().numpy()
    gen_2d   = emb.project_numpy(gen_D_np)

    save_samples(gen_D_np, os.path.join(exp_path, "gen_samples_D.npy"))
    save_samples(gen_2d,   os.path.join(exp_path, "gen_samples_2d.npy"))
    save_samples(data_2d_eval, os.path.join(exp_path, "true_samples_2d.npy"))

    # ---- Metrics (priority order) ----
    print("\n--- Evaluation (priority order) ---")

    # 1+2+3: MMD (scatter quality) and common x/v MSE
    dist_metrics = compute_metrics(gen_2d, data_2d_eval)

    common_metrics = eval_common_metrics(
        model=model,
        data_D=val_data_D_t,
        pred_space=cfg.pred_space,
        device=device,
        n_samples=cfg.n_eval,
    )

    # Report in priority order
    if val_x_mse_history:
        print(f"  val_x_mse (final): {val_x_mse_history[-1][1]:.5f}")
        print(f"  val_v_mse (final): {val_v_mse_history[-1][1]:.5f}")
    print(f"  x_mse_mean (post-train, r=0): {common_metrics['x_mse_mean']:.5f}")
    print(f"  v_mse_mean (post-train, r=0): {common_metrics['v_mse_mean']:.5f}")
    print(f"  MMD (2D):  {dist_metrics['mmd_2d']:.6f}")
    if loss_history:
        print(f"  final train_loss: {loss_history[-1][1]:.5f}  "
              f"[diagnostic only, not comparable across loss_spaces]")

    # Merge and save
    metrics = {}
    metrics.update(dist_metrics)
    metrics.update(common_metrics)
    metrics.update({
        "pred_space": cfg.pred_space, "loss_space": cfg.loss_space,
        "steps": cfg.steps, "obs_dim": cfg.obs_dim,
        "tau": cfg.tau, "dataset": cfg.dataset, "seed": cfg.seed,
    })
    if val_x_mse_history:
        metrics["final_val_x_mse"] = val_x_mse_history[-1][1]
        metrics["final_val_v_mse"] = val_v_mse_history[-1][1]
    if loss_history:
        metrics["final_train_loss"] = loss_history[-1][1]
    metrics["final_mmd"] = dist_metrics["mmd_2d"]

    save_metrics(metrics, os.path.join(exp_path, "metrics.json"))

    # Save val histories
    import json
    with open(os.path.join(exp_path, "val_x_mse_history.json"), "w") as f:
        json.dump(val_x_mse_history, f)
    with open(os.path.join(exp_path, "val_v_mse_history.json"), "w") as f:
        json.dump(val_v_mse_history, f)

    # ---- Visualize (priority order) ----
    print("\nGenerating plots (priority order)...")
    title = (f"{cfg.dataset} | D={cfg.obs_dim} | "
             f"pred={cfg.pred_space} loss={cfg.loss_space} | "
             f"steps={cfg.steps} tau={cfg.tau}")

    # 1. Eval summary: scatter + val MSE + MMD  [PRIMARY]
    plot_eval_summary(
        true_2d=data_2d_eval, gen_2d=gen_2d,
        val_x_mse_history=val_x_mse_history,
        val_v_mse_history=val_v_mse_history,
        final_mmd=dist_metrics["mmd_2d"],
        title=title,
        save_path=os.path.join(exp_path, "eval_summary.png"),
    )

    # 2. Scatter side-by-side
    plot_samples(data_2d_eval, gen_2d, title,
                 os.path.join(exp_path, "samples_side_by_side.png"))

    # 3. Training curves: val MSE (top) + train loss (bottom/secondary)
    plot_training_curves(
        train_result,
        title=f"Training curves — {title}",
        save_path=os.path.join(exp_path, "training_curves.png"),
    )

    # 4. Trajectory (multi-step only)
    if cfg.steps > 1:
        traj_2d = [emb.project_numpy(s.cpu().numpy()) for s in trajectory]
        plot_trajectory(traj_2d, data_2d_eval,
                        f"Trajectory ({cfg.steps} steps) — {title}",
                        os.path.join(exp_path, "trajectory.png"))

    print(f"\nDone. Results: {exp_path}")
    print(f"  [1] eval_summary.png     <- scatter + val MSE + MMD")
    print(f"  [2] samples_side_by_side.png")
    print(f"  [3] training_curves.png  <- val MSE (top) + train loss (bottom)")
    if val_x_mse_history:
        print(f"  val_x_mse={val_x_mse_history[-1][1]:.5f}  "
              f"val_v_mse={val_v_mse_history[-1][1]:.5f}  "
              f"MMD={dist_metrics['mmd_2d']:.6f}")


if __name__ == "__main__":
    main()
