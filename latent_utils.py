"""Small, testable helpers for deterministic latent vectors and interpolation."""
from __future__ import annotations

import math
import numpy as np
import torch


def create_latent_from_seed(seed: int, z_dim: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Create a reproducible [1, z_dim] standard-normal Z vector from a seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    if isinstance(z_dim, bool) or not isinstance(z_dim, int) or z_dim <= 0:
        raise ValueError("z_dim must be a positive integer")

    # Generate on CPU first for reproducibility across CPU/GPU inference devices.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    vector = torch.randn((1, z_dim), generator=rng, dtype=torch.float32, device="cpu")
    return vector.to(device)


def interpolate_latents(z_a: torch.Tensor, z_b: torch.Tensor, t: float) -> torch.Tensor:
    """Linearly interpolate between equally shaped latent tensors."""
    if not isinstance(z_a, torch.Tensor) or not isinstance(z_b, torch.Tensor):
        raise TypeError("z_a and z_b must be PyTorch tensors")
    if z_a.shape != z_b.shape:
        raise ValueError(f"latent shapes must match; received {tuple(z_a.shape)} and {tuple(z_b.shape)}")
    if z_a.device != z_b.device:
        raise ValueError("z_a and z_b must be on the same device")
    try:
        t_value = float(t)
    except (TypeError, ValueError) as exc:
        raise TypeError("t must be a number between 0 and 1") from exc
    if not math.isfinite(t_value) or not 0.0 <= t_value <= 1.0:
        raise ValueError("t must be between 0 and 1 inclusive")
    return (1.0 - t_value) * z_a + t_value * z_b


def interpolation_parameters(num_steps: int) -> np.ndarray:
    """Return evenly spaced parameters including both endpoints."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)):
        raise TypeError("num_steps must be an integer")
    if not 2 <= int(num_steps) <= 51:
        raise ValueError("num_steps must be between 2 and 51 inclusive")
    return np.linspace(0.0, 1.0, int(num_steps), dtype=np.float32)
