"""
train.py — Training loop.

Given a model, dataset, and config, runs the training loop and returns:
  - trained model
  - list of (iteration, loss) tuples
"""
import torch
import torch.optim as optim

from datasets import TensorDataset2D
from paths import sample_noise, interpolate
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

    Returns: list of (iter, loss_val) for plotting.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for it in range(1, n_iters + 1):
        model.train()
        optimizer.zero_grad()

        # Sample batch of clean data in R^D
        x0 = dataset.sample_batch(batch_size, device)  # (B, D)

        # Sample noise in R^D
        x1 = sample_noise(x0.shape, device)            # (B, D)

        # Sample time
        t = torch.rand(batch_size, 1, device=device)   # (B, 1) in [0, 1]

        # Compute noisy sample xt
        xt = interpolate(x0, x1, t)                    # (B, D)

        # Forward pass: predict in pred_space
        pred = model(xt, t)                             # (B, D)

        # Compute loss in loss_space
        loss = compute_loss(pred, pred_space, loss_space,
                            x0, x1, xt, t, tau)

        loss.backward()
        optimizer.step()

        if it % log_every == 0:
            loss_val = loss.item()
            loss_history.append((it, loss_val))
            print(f"  iter {it:6d}/{n_iters}  loss={loss_val:.5f}")

    return loss_history
