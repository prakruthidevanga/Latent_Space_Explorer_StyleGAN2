"""Load a trusted pretrained StyleGAN2-ADA PyTorch network pickle."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch


class ModelLoadError(RuntimeError):
    """Raised when the StyleGAN2 implementation or checkpoint cannot be loaded."""


@dataclass
class StyleGAN2Bundle:
    generator: Any
    device: torch.device
    z_dim: int
    c_dim: int
    img_resolution: int
    img_channels: int
    model_name: str


def choose_device(device_choice: str = "Auto") -> torch.device:
    choice = device_choice.strip().lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA was selected, but PyTorch cannot detect a CUDA-compatible GPU. Choose Auto or CPU.")
        return torch.device("cuda")
    raise ValueError("device_choice must be Auto, CPU, or CUDA")


def load_stylegan2(
    network_path: str | Path,
    repo_path: str | Path,
    device_choice: str = "Auto",
) -> StyleGAN2Bundle:
    """Load G_ema from official StyleGAN2-ADA code and a local trusted checkpoint."""
    checkpoint = Path(network_path).expanduser().resolve()
    repo = Path(repo_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ModelLoadError(
            f"Pretrained checkpoint not found: {checkpoint}. Run `python download_model.py --model afhqcat` "
            "or select the correct local .pkl file."
        )
    if not (repo / "dnnlib").is_dir() or not (repo / "legacy.py").is_file() or not (repo / "torch_utils").is_dir():
        raise ModelLoadError(
            f"StyleGAN2-ADA source code was not found in {repo}. Clone the official repository with:\n"
            f"git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git \"{repo}\""
        )

    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Import only after the repository path is added; these modules are part of StyleGAN2-ADA.
    try:
        import dnnlib  # type: ignore
        import legacy  # type: ignore
    except Exception as exc:
        raise ModelLoadError(f"Could not import StyleGAN2-ADA modules from {repo}: {exc}") from exc

    device = choose_device(device_choice)
    try:
        # Pickle files can execute code during deserialization. Only load trusted checkpoints.
        with dnnlib.util.open_url(str(checkpoint)) as file_obj:
            data = legacy.load_network_pkl(file_obj)
        generator = data.get("G_ema")
        if generator is None:
            raise ModelLoadError("The checkpoint does not contain the expected 'G_ema' generator.")
        generator = generator.eval().requires_grad_(False).to(device)
        z_dim = int(generator.z_dim)
        c_dim = int(generator.c_dim)
        resolution = int(generator.img_resolution)
        channels = int(generator.img_channels)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            "The checkpoint could not be loaded by this StyleGAN2-ADA implementation. "
            "Check that the .pkl file is compatible and that the repository/dependencies are installed. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    return StyleGAN2Bundle(
        generator=generator,
        device=device,
        z_dim=z_dim,
        c_dim=c_dim,
        img_resolution=resolution,
        img_channels=channels,
        model_name=checkpoint.name,
    )
