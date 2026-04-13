"""
embedding.py — Fixed random orthogonal projection between R^2 and R^D.

Setup:
  - True data lives in R^2 (latent/intrinsic space)
  - We embed it into R^D via a fixed column-orthogonal matrix P of shape (D, 2)
  - P has orthonormal columns: P^T P = I_2
  - Embedding:   x_D = P @ x_2          (shape: D)
  - Projection:  x_2 = P^T @ x_D        (shape: 2, exact inverse when D >= 2)

Because P is column-orthogonal, the projection P^T is an exact left-inverse:
  P^T @ (P @ x_2) = x_2  for any x_2 in R^2.

The model operates entirely in R^D.
For visualization we project generated R^D samples back to R^2 via P^T.
"""
import numpy as np
import torch


def make_projection_matrix(obs_dim: int, latent_dim: int, seed: int) -> np.ndarray:
    """
    Create a fixed (obs_dim, latent_dim) column-orthogonal matrix P.

    Method: take the first `latent_dim` columns of a random orthogonal matrix
    of shape (obs_dim, obs_dim), obtained via QR decomposition.

    Returns P of shape (obs_dim, latent_dim) with P^T P = I_{latent_dim}.
    """
    assert obs_dim >= latent_dim, "obs_dim must be >= latent_dim"
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((obs_dim, obs_dim))
    Q, _ = np.linalg.qr(A)  # Q is orthogonal (obs_dim, obs_dim)
    P = Q[:, :latent_dim]    # take first latent_dim columns -> (obs_dim, latent_dim)
    return P.astype(np.float32)


class Embedding:
    """
    Handles embedding and projection between R^latent and R^obs.

    P: (obs_dim, latent_dim) column-orthogonal matrix.
    """
    def __init__(self, obs_dim: int, latent_dim: int, seed: int):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        P_np = make_projection_matrix(obs_dim, latent_dim, seed)
        self.P = torch.from_numpy(P_np)    # (obs_dim, latent_dim)
        self.Pt = self.P.t()               # (latent_dim, obs_dim)

    def embed(self, x2: torch.Tensor) -> torch.Tensor:
        """
        Embed latent samples into observation space.
        x2: (..., latent_dim)  ->  out: (..., obs_dim)
        """
        P = self.P.to(x2.device)
        return x2 @ P.t()  # (..., latent_dim) @ (latent_dim, obs_dim)

    def project(self, xD: torch.Tensor) -> torch.Tensor:
        """
        Project observation samples back to latent space.
        xD: (..., obs_dim)  ->  out: (..., latent_dim)
        """
        Pt = self.Pt.to(xD.device)
        return xD @ Pt.t()  # (..., obs_dim) @ (obs_dim, latent_dim)

    def embed_numpy(self, x2: np.ndarray) -> np.ndarray:
        P = self.P.numpy()
        return x2 @ P.T  # (N, obs_dim)

    def project_numpy(self, xD: np.ndarray) -> np.ndarray:
        Pt = self.Pt.numpy()
        return xD @ Pt.T  # (N, latent_dim)
