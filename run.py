"""
run.py — Main entry point.

Trains a model, generates samples, evaluates, and saves results.

Usage:
  python run.py --dataset 8gaussians --obs_dim 16 --pred_space x  --loss_space v --steps 1
  python run.py --dataset 8gaussians --obs_dim 16 --pred_space eps --loss_space v --steps 1
  python run.py --dataset 8gaussians --obs_dim 16 --pred_space v  --loss_space v --steps 1
  python run.py --dataset 8gaussians --obs_dim 16 --pred_space u  --loss_space v --steps 1
  python run.py --dataset moons      --obs_dim 64 --pred_space x  --loss_space x --steps 1
  python run.py --dataset moons      --obs_dim 64 --pred_space u  --loss_space x --steps 4
  python run.py --dataset moons      --obs_dim 64 --pred_space x  --loss_space v --tau 0.2
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
from metrics import compute_metrics
from viz import plot_samples, plot_samples_overlay, plot_trajectory, plot_loss_curve
from utils import set_seed, get_device, save_metrics, save_loss_history, save_samples


def main():
    cfg = get_args()
    set_seed(cfg.seed)
    device = get_device()

    exp_path = make_exp_dir(cfg)
    save_config(cfg, exp_path)

    print(f"Device: {device}")
    print(f"Exp dir: {exp_path}")
    print(f"pred_space={cfg.pred_space}  loss_space={cfg.loss_space}  "
          f"steps={cfg.steps}  tau={cfg.tau}  obs_dim={cfg.obs_dim}")

    # ---- Data ----
    rng = np.random.default_rng(cfg.seed)
    data_2d_train = get_dataset(cfg.dataset, cfg.n_train, rng)
    data_2d_eval  = get_dataset(cfg.dataset, cfg.n_eval,  rng)

    emb = Embedding(obs_dim=cfg.obs_dim, latent_dim=cfg.latent_dim, seed=cfg.seed)
    data_D_train = emb.embed_numpy(data_2d_train)
    data_D_eval  = emb.embed_numpy(data_2d_eval)

    train_dataset = TensorDataset2D(torch.from_numpy(data_D_train))

    # ---- Model ----
    model = build_model(
        pred_space=cfg.pred_space,
        obs_dim=cfg.obs_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_emb_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}  params={n_params:,}")

    # ---- Train ----
    print(f"\nTraining {cfg.n_iters} iters...")
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

    save_loss_history(loss_history, os.path.join(exp_path, "loss_history.json"))
    plot_loss_curve(
        loss_history,
        title=f"Loss — pred={cfg.pred_space} loss={cfg.loss_space}",
        save_path=os.path.join(exp_path, "loss_curve.png"),
    )
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

    # ---- Metrics ----
    metrics = compute_metrics(gen_2d, data_2d_eval)
    metrics.update({
        "pred_space": cfg.pred_space, "loss_space": cfg.loss_space,
        "steps": cfg.steps, "obs_dim": cfg.obs_dim,
        "tau": cfg.tau, "dataset": cfg.dataset,
    })
    print(f"  MMD (2D): {metrics['mmd_2d']:.6f}")
    save_metrics(metrics, os.path.join(exp_path, "metrics.json"))

    # ---- Visualize ----
    title = (f"{cfg.dataset} | D={cfg.obs_dim} | "
             f"pred={cfg.pred_space} loss={cfg.loss_space} | "
             f"steps={cfg.steps} tau={cfg.tau}")

    plot_samples(data_2d_eval, gen_2d, title,
                 os.path.join(exp_path, "samples_side_by_side.png"))
    plot_samples_overlay(data_2d_eval, gen_2d, title,
                         os.path.join(exp_path, "samples_overlay.png"))

    if cfg.steps > 1:
        traj_2d = [emb.project_numpy(s.cpu().numpy()) for s in trajectory]
        plot_trajectory(traj_2d, data_2d_eval, f"Trajectory — {title}",
                        os.path.join(exp_path, "trajectory.png"))

    print(f"\nDone. Results: {exp_path}")
    print(f"  MMD (2D): {metrics['mmd_2d']:.6f}")


if __name__ == "__main__":
    main()
