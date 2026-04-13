"""
train.py — Training loop.

Dispatches to the right sampling logic (r,t) vs (t) based on pred_space.
"""
import torch
import torch.optim as optim

from datasets import TensorDataset2D
from paths import sample_noise, interpolate, sample_t, sample_r_t
from losses import compute_loss


def train(
    model: torch.nn.Module,
    dataset: TensorDataset2D,
    pred_space: str,
    loss_space: str,
    n_iters: int,
    batch_size: int,
    lr: float,
    tau: float,
    device: torch.device,
    log_every: int = 200,
) -> list:
    """
    Training loop.

    For pred_space in {x, eps, v}: samples (x, eps, t) per batch.
    For pred_space == u: samples (x, eps, r, t) per batch.

    Returns: list of (iter, loss_val) for plotting.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    is_u = (pred_space == "u")

    for it in range(1, n_iters + 1):
        model.train()
        optimizer.zero_grad()

        # --- sample clean data ---
        x = dataset.sample_batch(batch_size, device)   # (B, D)

        # --- sample noise ---
        eps = sample_noise(x.shape, device)             # (B, D)

        # --- sample time(s) ---
        if is_u:
            r, t = sample_r_t(batch_size, device)      # (B,1), (B,1)
        else:
            t = sample_t(batch_size, device)            # (B, 1)
            r = None

        # --- interpolate ---
        z_t = interpolate(x, eps, t)                   # (B, D)

        # --- loss ---
        loss = compute_loss(
            net=model,
            z_t=z_t, t=t, x=x, eps=eps,
            pred_space=pred_space,
            loss_space=loss_space,
            tau=tau,
            r=r,
        )

        loss.backward()
        optimizer.step()

        if it % log_every == 0:
            loss_val = loss.item()
            loss_history.append((it, loss_val))
            print(f"  iter {it:6d}/{n_iters}  loss={loss_val:.5f}")

    return loss_history
