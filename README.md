# onestep_eval — One-Step Generation Toy Experiments

A minimal PyTorch codebase for studying one-step generative modeling.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset & Embedding

**2D datasets** (`latent_dim=2`):
- `8gaussians`: 8 Gaussians on a circle (radius=5, std=0.5)
- `moons`: sklearn two-moons, centered and scaled

**High-dimensional embedding**: the 2D data is embedded into R^D via a fixed column-orthogonal matrix P ∈ R^{D×2} (first 2 columns of a random orthogonal matrix via QR).
- Embed: `x_D = P x_2`  
- Project back: `x_2 = P^T x_D`  (exact, since P^T P = I_2)

The model trains entirely in R^D. For visualization/metrics, samples are projected back to 2D via P^T.

---

## Supported Combinations

| `pred_space` | `loss_space` | Tier |
|---|---|---|
| x, eps, v, u | v | Tier 1 |
| x, eps, v, u | x | Tier 2 |

All 8 combinations are implemented in `losses.py`.

---

## Prediction Space vs Loss Space

- **`pred_space`**: what the model directly outputs (x, eps, v, or u)
- **`loss_space`**: what space the MSE loss is computed in (v or x)

These are independent. When pred_space ≠ loss_space, the prediction is converted through `transforms.py` before computing the loss.

---

## Model Input Signatures

| pred_space | Model class | Input |
|---|---|---|
| x, eps, v | `StandardMLP` | `(z_t, t)` |
| u | `UModel` | `(z_t, r, t)` |

The distinction is enforced throughout training and sampling.

---

## Sampling

**One-step** (`--steps 1`):
- x/eps/v: Euler step of size dt=1 in v-space from z_1
- u: direct MeanFlow jump `x_gen = z_1 - u_theta(z_1, r=0, t=1)`

**Multi-step** (`--steps K`):
- x/eps/v: K Euler steps, each converting prediction to v-space
- u: K MeanFlow jumps, each `z_{r} = z_t - (t-r) * u_theta(z_t, r, t)`

---

## Target Smoothing (`--tau`)

Relaxes the prediction target:
```
x_eff = (1-tau)*x + tau*eps
v_eff = eps - x_eff
```
`tau=0` → exact clean target. `tau>0` → softer target.
The interpolation path itself is unchanged.

---

## CLI

```bash
# Tier 1: v-loss
python run.py --dataset 8gaussians --obs_dim 16 --pred_space x   --loss_space v --steps 1
python run.py --dataset 8gaussians --obs_dim 16 --pred_space eps  --loss_space v --steps 1
python run.py --dataset 8gaussians --obs_dim 16 --pred_space v   --loss_space v --steps 1
python run.py --dataset 8gaussians --obs_dim 16 --pred_space u   --loss_space v --steps 1

# Tier 2: x-loss
python run.py --dataset 8gaussians --obs_dim 16 --pred_space x   --loss_space x --steps 1
python run.py --dataset 8gaussians --obs_dim 16 --pred_space u   --loss_space x --steps 1

# Multi-step
python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space v --steps 4

# Target smoothing
python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space v --tau 0.2 --steps 1

# Dimension sweep
python run.py --dataset 8gaussians --obs_dim 2   --pred_space x --loss_space v --steps 1
python run.py --dataset 8gaussians --obs_dim 256 --pred_space x --loss_space v --steps 1
```

Or run the full first sweep:
```bash
bash sweep.sh
```

---

## Output Files

Each run creates `exp/{name}/`:
```
config.json             full config
model.pt                trained weights
loss_history.json       [(iter, loss), ...]
loss_curve.png
metrics.json            MMD and metadata
samples_side_by_side.png
samples_overlay.png
gen_samples_D.npy       generated samples in R^D
gen_samples_2d.npy      projected to R^2
trajectory.png          (multi-step only)
```

---

## File Structure

```
run.py          Main entry point
configs.py      Config dataclass + argparse
datasets.py     8gaussians, two-moons in R^2
embedding.py    Fixed orthogonal projection R^2 <-> R^D
paths.py        Path z_t = (1-t)x + t*eps, time sampling
transforms.py   AUTHORITATIVE: all space transforms + JVP for u->V
models.py       StandardMLP (z_t,t) and UModel (z_t,r,t)
losses.py       All 8 explicit loss objectives
train.py        Training loop
sample.py       One-step and K-step generation
metrics.py      MMD evaluation
viz.py          Scatter, trajectory, loss curve plots
utils.py        Seeding, device, saving
tests.py        Sanity checks (python tests.py)
sweep.sh        First experiment sweep
```

---

## Sanity Checks

```bash
python tests.py
```

Checks: transform round-trips, u-pred JVP shape and backprop, time sampling bounds, generation output shapes.

---

## How Methods Are Compared

**Raw training loss is NOT directly comparable across different `loss_space` choices.**  
v-loss and x-loss have different scales and meanings. Comparing them directly is misleading.

Instead, all methods are compared in **common evaluation spaces**:

| Metric | Description | Comparable? |
|---|---|---|
| `val_x_mse` | E[‖x_hat − x‖²] in common x-space | **Yes — primary** |
| `val_v_mse` | E[‖v_hat − v‖²] in common v-space | **Yes — primary** |
| MMD (2D) | Distribution distance of generated samples | **Yes — primary** |
| Scatter plots | Visual quality of generated samples | **Yes — primary** |
| train_loss | Optimization diagnostic within one method | No — diagnostic only |

**Presentation priority** (followed by all plots and tables):
1. Side-by-side scatter plots (generated vs real in 2D)
2. `val_x_mse`
3. `val_v_mse`
4. MMD
5. Training loss curves (secondary, labeled as diagnostic)

For u-pred specifically: `val_x_mse` and `val_v_mse` use the direct inference formula `x_hat = z_t − t·u_θ(z_t, r=0, t)`, not the training-time compound V_θ.

---

## Generated Artifacts

**Per run** (in `exp/<name>/`):

| File | Description | Priority |
|---|---|---|
| `eval_summary.png` | Scatter + val MSE curves + MMD text | **Primary** |
| `samples_side_by_side.png` | Real vs generated scatter | Primary |
| `training_curves.png` | val MSE (top panel) + train loss (bottom) | Secondary |
| `trajectory.png` | Multi-step trajectory (if steps > 1) | Optional |
| `metrics.json` | All metrics: val_x_mse, val_v_mse, mmd, train_loss | — |
| `val_x_mse_history.json` | Per-checkpoint val x-MSE | — |
| `val_v_mse_history.json` | Per-checkpoint val v-MSE | — |
| `gen_samples_2d.npy` | Generated samples projected to 2D | — |
| `true_samples_2d.npy` | True eval samples in 2D | — |

**Cross-run** (in `compare_out/<name>/`):

| File | Description | Priority |
|---|---|---|
| `comparison_grid.png` | Side-by-side scatter: all methods in one figure | **Primary** |
| `metric_vs_{x_key}.png` | val_x_mse, val_v_mse, MMD vs obs_dim / tau / steps | Primary |
| `summary.csv` | Table: all methods, metrics in priority order | Primary |

---

## Cross-Run Comparison Commands

```bash
# Group A: Tier 1 (v-loss), all pred_spaces
python compare.py --parent exp --filter "8gaussians_D16" --filter2 "lossv_steps1" \
    --x_key pred_space --plot_grids --out_dir compare_out/groupA

# Group B: Tier 2 (x-loss), all pred_spaces
python compare.py --parent exp --filter "8gaussians_D16" --filter2 "lossx_steps1" \
    --x_key pred_space --plot_grids --out_dir compare_out/groupB

# Obs-dim scaling, grouped by pred_space
python compare.py --parent exp --filter "8gaussians" --filter2 "lossv_steps1_tau0.0" \
    --x_key obs_dim --group_by pred_space --log_x --out_dir compare_out/scaling

# Tau sweep
python compare.py --parent exp --filter "moons_D64_predx_lossv_steps1" \
    --x_key tau --out_dir compare_out/tau_sweep

# 1-step vs 4-step
python compare.py --parent exp --filter "moons_D64_predx_lossv_tau0.0" \
    --x_key steps --out_dir compare_out/steps
```

---

## Known Limitations / Approximations

1. **eps-pred one-step**: `eps_to_v` divides by `(1-t)`, which is 0 at t=1. One-step generation uses `t = 1 - T_EPS` (T_EPS=1e-3) to avoid this. The error is negligible but not zero.

2. **u-pred MeanFlow identity**: the linear path has constant v = eps - x, so u = v exactly in expectation. The value of training with u-pred is not algebraic but structural — the UModel receives (r,t) and learns to output the average velocity for a range, which may generalize differently than v-pred. At small (t-r), u should approach v; at large (t-r), u averages over the path.

3. **JVP stop-gradient**: the JVP term in V_theta is fully stop-gradient'd (computed in `torch.no_grad()` on detached inputs). Gradient flows only through u_theta. This follows the iMF paper.

4. **eps-loss / u-loss**: not implemented in the first version. See `losses.py` extension notes.

5. **No EMA**: training without exponential moving average. Add EMA if sample quality is noisy.

6. **MMD bandwidth**: median heuristic, unbiased estimator. Can be negative for small n. Use `n_eval >= 1000` for reliable comparisons.

---

# 基本設定と変換

## パス定義
$$
z_t = (1 - t)x + t\epsilon
$$

$$
v = \epsilon - x
$$

---

## 空間間の線形変換

### x → ε, v
$$
\epsilon = \frac{z_t - (1 - t)x}{t}
$$

$$
v = \frac{z_t - x}{t}
$$

---

### ε → x, v
$$
x = \frac{z_t - t\epsilon}{1 - t}
$$

$$
v = \frac{\epsilon - z_t}{1 - t}
$$

---

### v → x, ε
$$
x = z_t - t v
$$

$$
\epsilon = z_t + (1 - t)v
$$

---

## 平均速度 (MeanFlow)
$$
u(z_t, r, t) = \frac{1}{t-r} \int_r^t v(z_\tau, \tau) d\tau
$$

---

## MeanFlow identity (iMF)
$$
v(z_t,t) = u(z_t,r,t) + (t-r) \frac{d}{dt}u(z_t,r,t)
$$

---

# Tier 1: v-loss

## x-pred + v-loss
$$
\mathcal L_{x \to v}
=
\mathbb E \left[
\left\|
\frac{z_t - x_\theta}{t} - (\epsilon - x)
\right\|^2
\right]
$$

---

## ε-pred + v-loss
$$
\mathcal L_{\epsilon \to v}
=
\mathbb E \left[
\left\|
\frac{\epsilon_\theta - z_t}{1 - t} - (\epsilon - x)
\right\|^2
\right]
$$

---

## v-pred + v-loss
$$
\mathcal L_{v \to v}
=
\mathbb E \left[
\|v_\theta - (\epsilon - x)\|^2
\right]
$$

---

## u-pred + v-loss
$$
V_\theta = u_\theta + (t-r) \frac{d}{dt}u_\theta
$$

$$
\mathcal L_{u \to v}
=
\mathbb E \left[
\|V_\theta - (\epsilon - x)\|^2
\right]
$$

---

# Tier 2: x-loss

## x-pred + x-loss
$$
\mathcal L_{x \to x}
=
\mathbb E \left[
\|x_\theta - x\|^2
\right]
$$

---

## ε-pred + x-loss
$$
\mathcal L_{\epsilon \to x}
=
\mathbb E \left[
\left\|
\frac{z_t - t\epsilon_\theta}{1 - t} - x
\right\|^2
\right]
$$

---

## v-pred + x-loss
$$
\mathcal L_{v \to x}
=
\mathbb E \left[
\|z_t - t v_\theta - x\|^2
\right]
$$

---

## u-pred + x-loss
$$
x_\theta^{(u)} = z_t - t V_\theta
$$

$$
\mathcal L_{u \to x}
=
\mathbb E \left[
\|z_t - t V_\theta - x\|^2
\right]
$$

---

# ε-loss

## x-pred + ε-loss
$$
\mathcal L_{x \to \epsilon}
=
\mathbb E \left[
\left\|
\frac{z_t - (1 - t)x_\theta}{t} - \epsilon
\right\|^2
\right]
$$

---

## ε-pred + ε-loss
$$
\mathcal L_{\epsilon \to \epsilon}
=
\mathbb E \left[
\|\epsilon_\theta - \epsilon\|^2
\right]
$$

---

## v-pred + ε-loss
$$
\mathcal L_{v \to \epsilon}
=
\mathbb E \left[
\|z_t + (1 - t)v_\theta - \epsilon\|^2
\right]
$$

---

## u-pred + ε-loss
$$
\epsilon_\theta^{(u)} = z_t + (1 - t)V_\theta
$$

$$
\mathcal L_{u \to \epsilon}
=
\mathbb E \left[
\|z_t + (1 - t)V_\theta - \epsilon\|^2
\right]
$$
