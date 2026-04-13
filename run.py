"""
run.py — Main entry point. Train a model, generate samples, evaluate, visualize.

Usage examples:
  python run.py --dataset 8gaussians --obs_dim 16 --pred_space x --loss_space x --steps 1
  python run.py --dataset 8gaussians --obs_dim 16 --pred_space u --loss_space u --steps 1
  python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space x --steps 4
  python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space x --steps 1 --tau 0.2
  python run.py --dataset moons --obs_dim 64 --pred_space v --loss_space v --steps 1
  python run.py --dataset moons --obs_dim 64 --pred_space eps --loss_space eps --steps 1
"""
import os
import json
import numpy as np
import torch

from configs import get_args, make_exp_dir, save_config
from datasets import get_dataset, TensorDataset2D
from embedding import Embedding
from models import build_model
from train import train
from sample import generate_one_step, generate_multistep
from metrics import compute_metrics
from viz import plot_samples, plot_samples_overlay, plot_trajectory, plot_loss_curve
from utils import set_seed, get_device, save_metrics, save_loss_history, save_samples


def main():
    cfg = get_args()

    # Setup
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    # Experiment directory
    exp_path = make_exp_dir(cfg)
    save_config(cfg, exp_path)
    print(f"Experiment dir: {exp_path}")
    print(f"Config: pred_space={cfg.pred_space}, loss_space={cfg.loss_space}, "
          f"steps={cfg.steps}, tau={cfg.tau}, obs_dim={cfg.obs_dim}")

    # ---- DATA ----
    rng = np.random.default_rng(cfg.seed)
    data_2d_train = get_dataset(cfg.dataset, cfg.n_train, rng)
    data_2d_eval  = get_dataset(cfg.dataset, cfg.n_eval, rng)

    # Embedding: R^2 -> R^D
    emb = Embedding(obs_dim=cfg.obs_dim, latent_dim=cfg.latent_dim, seed=cfg.seed)
    data_D_train = emb.embed_numpy(data_2d_train)  # (N_train, D)
    data_D_eval  = emb.embed_numpy(data_2d_eval)   # (N_eval, D)

    # Wrap as tensor dataset
    train_dataset = TensorDataset2D(torch.from_numpy(data_D_train))

    print(f"Dataset: {cfg.dataset}, n_train={cfg.n_train}, obs_dim={cfg.obs_dim}")

    # ---- MODEL ----
    model = build_model(
        obs_dim=cfg.obs_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_emb_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ---- TRAIN ----
    print(f"\nTraining for {cfg.n_iters} iterations...")
    loss_history = train(
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
    )

    # Save loss history and plot
    save_loss_history(loss_history, os.path.join(exp_path, "loss_history.json"))
    plot_loss_curve(
        loss_history,
        title=f"Training Loss — pred={cfg.pred_space} loss={cfg.loss_space}",
        save_path=os.path.join(exp_path, "loss_curve.png"),
    )

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))
    print(f"  Saved model checkpoint.")

    # ---- GENERATE ----
    print(f"\nGenerating {cfg.n_eval} samples with steps={cfg.steps}...")
    n_gen = cfg.n_eval

    if cfg.steps == 1:
        gen_D = generate_one_step(
            model=model,
            n_samples=n_gen,
            obs_dim=cfg.obs_dim,
            pred_space=cfg.pred_space,
            device=device,
        )
        gen_D_np = gen_D.cpu().numpy()
        trajectory_2d = None
    else:
        gen_D, trajectory = generate_multistep(
            model=model,
            n_samples=n_gen,
            obs_dim=cfg.obs_dim,
            pred_space=cfg.pred_space,
            steps=cfg.steps,
            device=device,
        )
        gen_D_np = gen_D.cpu().numpy()
        # Project trajectory to 2D for visualization
        trajectory_2d = [emb.project_numpy(step.cpu().numpy()) for step in trajectory]

    # Project to 2D for evaluation and visualization
    gen_2d = emb.project_numpy(gen_D_np)              # (N, 2)
    true_2d_eval = data_2d_eval                        # (N_eval, 2) -- already in 2D

    # Save generated samples
    save_samples(gen_D_np, os.path.join(exp_path, "gen_samples_D.npy"))
    save_samples(gen_2d, os.path.join(exp_path, "gen_samples_2d.npy"))

    # ---- METRICS ----
    print("\nComputing metrics...")
    metrics = compute_metrics(gen_2d, true_2d_eval)
    metrics["pred_space"] = cfg.pred_space
    metrics["loss_space"] = cfg.loss_space
    metrics["steps"] = cfg.steps
    metrics["obs_dim"] = cfg.obs_dim
    metrics["tau"] = cfg.tau
    metrics["dataset"] = cfg.dataset
    print(f"  MMD (2D): {metrics['mmd_2d']:.6f}")
    save_metrics(metrics, os.path.join(exp_path, "metrics.json"))

    # ---- VISUALIZE ----
    print("\nGenerating visualizations...")
    title_str = (
        f"{cfg.dataset} | D={cfg.obs_dim} | "
        f"pred={cfg.pred_space} loss={cfg.loss_space} | "
        f"steps={cfg.steps} tau={cfg.tau}"
    )

    plot_samples(
        true_2d=true_2d_eval,
        gen_2d=gen_2d,
        title=title_str,
        save_path=os.path.join(exp_path, "samples_side_by_side.png"),
    )

    plot_samples_overlay(
        true_2d=true_2d_eval,
        gen_2d=gen_2d,
        title=title_str,
        save_path=os.path.join(exp_path, "samples_overlay.png"),
    )

    if trajectory_2d is not None and cfg.steps > 1:
        plot_trajectory(
            trajectory_2d=trajectory_2d,
            true_2d=true_2d_eval,
            title=f"Trajectory ({cfg.steps} steps) — {title_str}",
            save_path=os.path.join(exp_path, "trajectory.png"),
        )

    print(f"\nDone. Results in: {exp_path}")
    print(f"  MMD (2D): {metrics['mmd_2d']:.6f}")


if __name__ == "__main__":
    main()
