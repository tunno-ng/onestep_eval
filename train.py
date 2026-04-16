"""
train.py — Training loop with periodic common-metric validation.

Returns a dict with:
  loss_history:       [(iter, loss), ...]      -- training loss (diagnostic only)
  val_x_mse_history:  [(iter, x_mse_mean), ...]  -- primary cross-method metric
  val_v_mse_history:  [(iter, v_mse_mean), ...]  -- primary cross-method metric

IMPORTANT: training loss is logged as an optimization diagnostic.
Cross-method comparison must use val_x_mse and val_v_mse, not training loss,
because different loss_spaces have different scales.
"""
import torch
import torch.optim as optim

from datasets import TensorDataset2D
from paths import sample_noise, interpolate, sample_t, sample_r_t
from losses import compute_loss
from metrics import eval_common_metrics


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
    eval_every: int = 0,              # 0 = no periodic validation
    val_data_D: torch.Tensor = None,  # (N, D) validation data; required if eval_every > 0
    n_val_samples: int = 1000,
) -> dict:
    """
    Training loop.

    For pred_space == 'u': samples (x, eps, r, t) per batch.
    For pred_space in {x, eps, v}: samples (x, eps, t) per batch.

    Periodic validation (every eval_every steps) computes x_mse_mean and
    v_mse_mean on val_data_D using common evaluation spaces.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    is_u = (pred_space == "u")

    loss_history      = []
    val_x_mse_history = []
    val_v_mse_history = []

    for it in range(1, n_iters + 1):
        model.train()
        optimizer.zero_grad()

        x   = dataset.sample_batch(batch_size, device)
        eps = sample_noise(x.shape, device)

        if is_u:
            r, t = sample_r_t(batch_size, device)
        else:
            t = sample_t(batch_size, device)
            r = None

        z_t  = interpolate(x, eps, t)
        loss = compute_loss(
            net=model, z_t=z_t, t=t, x=x, eps=eps,
            pred_space=pred_space, loss_space=loss_space,
            tau=tau, r=r,
        )

        loss.backward()
        # Gradient clipping: always applied to keep training stable.
        # Especially important for u-pred, where the JVP term can create
        # large Jacobian-induced gradients during early training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if it % log_every == 0:
            loss_history.append((it, loss.item()))
            print(f"  iter {it:6d}/{n_iters}  train_loss={loss.item():.5f}")

        # Periodic validation: x-MSE and v-MSE in common evaluation spaces
        if eval_every > 0 and val_data_D is not None and it % eval_every == 0:
            val = eval_common_metrics(
                model, val_data_D, pred_space, device,
                n_samples=n_val_samples,
            )
            val_x_mse_history.append((it, val["x_mse_mean"]))
            val_v_mse_history.append((it, val["v_mse_mean"]))
            print(f"             val_x_mse={val['x_mse_mean']:.5f}"
                  f"  val_v_mse={val['v_mse_mean']:.5f}")
            model.train()

    return {
        "loss_history":      loss_history,
        "val_x_mse_history": val_x_mse_history,
        "val_v_mse_history": val_v_mse_history,
    }
