"""
datasets.py — 2D toy datasets (intrinsic space).

All datasets return samples in R^2 (intrinsic/latent space).
Embedding into R^D is handled separately in embedding.py.
"""
import numpy as np
import torch


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


def get_dataset(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n samples from named 2D dataset."""
    if name == "8gaussians":
        return sample_8gaussians(n, rng)
    elif name == "moons":
        return sample_moons(n, rng)
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
