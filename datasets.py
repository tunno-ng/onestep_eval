"""
datasets.py — 2D toy datasets (intrinsic space).

All datasets return samples in R^2 (intrinsic/latent space).
Embedding into R^D is handled separately in embedding.py.
"""
import numpy as np
import torch

# -----------------------------------------------------------------------
# Spiral JiT normalization constants (computed once from a fixed reference pool)
# -----------------------------------------------------------------------
# We standardize spiral samples using stats from a large fixed pool so that
# normalization is identical across all calls regardless of n or rng.
#
# Reference pool: 100,000 samples, fixed seed=12345
# Normalization: per-dimension (x1, x2) standardization → zero mean, unit std
# -----------------------------------------------------------------------

def _compute_spiral_norm_stats() -> tuple:
    """Return (mean, std) of shape (2,) computed from a large fixed reference pool."""
    _rng = np.random.default_rng(12345)
    _n   = 100_000
    _theta = _rng.uniform(0.0, 4.0 * np.pi, _n)
    _r     = 0.2 * _theta
    _noise = _rng.standard_normal((_n, 2)) * 0.01
    _x_raw = np.stack([_r * np.cos(_theta), _r * np.sin(_theta)], axis=1) + _noise
    return _x_raw.mean(axis=0).astype(np.float32), _x_raw.std(axis=0).astype(np.float32)


_SPIRAL_MEAN, _SPIRAL_STD = _compute_spiral_norm_stats()


def sample_8gaussians(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    8 Gaussians arranged in a circle of radius 5 in R^2.
    Returns shape (n, 2), values in roughly [-7, 7]^2.
    """
    radius = 5.0
    std = 0.5
    centers = np.array([
        [radius * np.cos(2 * np.pi * k / 8), radius * np.sin(2 * np.pi * k / 8)]
        for k in range(8)
    ])  # (8, 2)

    # Assign each sample to a random Gaussian
    idx = rng.integers(0, 8, size=n)
    samples = centers[idx] + rng.normal(0, std, size=(n, 2))
    return samples.astype(np.float32)


def sample_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Two-moons dataset in R^2 using scikit-learn.
    Returns shape (n, 2), values in roughly [-1.5, 2.5]^2.
    """
    from sklearn.datasets import make_moons
    # make_moons uses its own RNG; we pass a seed derived from our rng
    seed = int(rng.integers(0, 2**31))
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=seed)
    # Center and scale
    X = X - X.mean(axis=0)
    X = X / X.std()
    return X.astype(np.float32)


def sample_spiral_jit(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    JiT-style 2D spiral dataset.

    Underlying distribution:
      θ  ~ Uniform(0, 4π)
      r  = 0.2 * θ
      x_hat_raw = (r cosθ, r sinθ) + σ η,  σ=0.01, η ~ N(0, I_2)

    Normalization (per-dimension standardization):
      x_hat = (x_hat_raw − μ) / σ_ref
    where (μ, σ_ref) are precomputed from a fixed reference pool of 100k samples
    (seed=12345). This ensures identical normalization across all calls.

    Returns shape (n, 2), standardized to ≈ zero mean, unit std.
    """
    theta = rng.uniform(0.0, 4.0 * np.pi, n)
    r     = 0.2 * theta
    noise = rng.standard_normal((n, 2)) * 0.01
    x_raw = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) + noise
    x_norm = (x_raw - _SPIRAL_MEAN) / (_SPIRAL_STD + 1e-8)
    return x_norm.astype(np.float32)


def get_dataset(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n samples from named 2D dataset."""
    if name == "8gaussians":
        return sample_8gaussians(n, rng)
    elif name == "moons":
        return sample_moons(n, rng)
    elif name == "spiral_jit":
        return sample_spiral_jit(n, rng)
    else:
        raise ValueError(f"Unknown dataset: {name}")


class TensorDataset2D:
    """
    Wraps a fixed 2D dataset in R^D (embedded).
    Supports random batch sampling.
    """
    def __init__(self, data_D: torch.Tensor):
        # data_D: (N, D) tensor
        self.data = data_D
        self.N = data_D.shape[0]

    def sample_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(0, self.N, (batch_size,))
        return self.data[idx].to(device)
