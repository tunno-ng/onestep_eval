"""
configs.py — Default configuration and argument parsing.

loss_space is restricted to {v, x} in Tier 1/2.
eps-loss and u-loss are not yet implemented (see losses.py for extension notes).
"""
import argparse
import os
import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Dataset
    dataset: str = "8gaussians"       # "8gaussians" or "moons"
    n_train: int = 50000
    n_eval:  int = 2000

    # Embedding
    obs_dim:    int = 16              # ambient observation dimension D
    latent_dim: int = 2               # intrinsic dimension (always 2)

    # Path
    tau: float = 0.0                  # target smoothing: x_eff = (1-tau)*x + tau*eps

    # Spaces
    pred_space: str = "x"             # {x, eps, v, u}
    loss_space: str = "v"             # {v, x}  (eps-loss/u-loss not yet implemented)

    # Sampling
    steps: int = 1                    # number of generation steps (1, 4, etc.)

    # Model
    hidden_dim:   int = 256
    n_layers:     int = 4
    time_emb_dim: int = 32

    # Training
    n_iters:    int = 20000
    batch_size: int = 512
    lr:         float = 1e-3
    seed:       int = 42

    # Output
    exp_dir:    str = "exp"


def get_args() -> Config:
    parser = argparse.ArgumentParser(
        description="One-step generation toy experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    type=str,   default="8gaussians",
                        choices=["8gaussians", "moons"])
    parser.add_argument("--n_train",    type=int,   default=50000)
    parser.add_argument("--n_eval",     type=int,   default=2000)
    parser.add_argument("--obs_dim",    type=int,   default=16)
    parser.add_argument("--tau",        type=float, default=0.0)
    parser.add_argument("--pred_space", type=str,   default="x",
                        choices=["x", "eps", "v", "u"])
    parser.add_argument("--loss_space", type=str,   default="v",
                        choices=["v", "x"])
    parser.add_argument("--steps",      type=int,   default=1)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--time_emb_dim", type=int, default=32)
    parser.add_argument("--n_iters",    type=int,   default=20000)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--exp_dir",    type=str,   default="exp")

    args = parser.parse_args()
    return Config(**vars(args))


def make_exp_dir(cfg: Config) -> str:
    """Create a unique experiment directory derived from the config."""
    name = (
        f"{cfg.dataset}_D{cfg.obs_dim}"
        f"_pred{cfg.pred_space}_loss{cfg.loss_space}"
        f"_steps{cfg.steps}_tau{cfg.tau}"
        f"_seed{cfg.seed}"
    )
    path = os.path.join(cfg.exp_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_config(cfg: Config, exp_path: str):
    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
