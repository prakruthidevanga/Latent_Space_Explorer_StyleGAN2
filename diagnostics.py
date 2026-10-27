"""Numeric diagnostics and a simple latent-dimension visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from latent_utils import interpolate_latents


def calculate_diagnostics(z_a: torch.Tensor, z_b: torch.Tensor, z_t: torch.Tensor) -> dict[str, float]:
    """Return Euclidean distances between endpoint/current latent vectors."""
    if z_a.shape != z_b.shape or z_a.shape != z_t.shape:
        raise ValueError("All latent vectors must have the same shape")
    return {
        "endpoint_distance": float(torch.linalg.vector_norm((z_b - z_a).reshape(-1)).item()),
        "distance_from_start": float(torch.linalg.vector_norm((z_t - z_a).reshape(-1)).item()),
        "distance_from_end": float(torch.linalg.vector_norm((z_b - z_t).reshape(-1)).item()),
    }


def plot_latent_dimensions(z_a: torch.Tensor, z_b: torch.Tensor, z_t: torch.Tensor, count: int = 16):
    """Plot a subset of vector coordinates; this is not a visualization of the full latent space."""
    a = z_a.detach().cpu().reshape(-1).numpy()
    b = z_b.detach().cpu().reshape(-1).numpy()
    current = z_t.detach().cpu().reshape(-1).numpy()
    n = min(count, len(a))
    dims = list(range(n))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(dims, a[:n], marker="o", linewidth=1.4, label="Seed A")
    ax.plot(dims, current[:n], marker="o", linewidth=1.4, label="Current interpolation")
    ax.plot(dims, b[:n], marker="o", linewidth=1.4, label="Seed B")
    ax.set_title(f"First {n} coordinates of the latent vectors")
    ax.set_xlabel("Latent dimension index")
    ax.set_ylabel("Coordinate value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    return fig
