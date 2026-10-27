"""Generate real images with the loaded pretrained StyleGAN2-ADA generator."""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from model_loader import StyleGAN2Bundle

# Standard CIFAR-10 class order used by the official class-conditional checkpoint.
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def generate_image(
    bundle: StyleGAN2Bundle,
    latent: torch.Tensor,
    class_index: int | None = None,
) -> Image.Image:
    """Generate one image from a Z-space vector; support unconditional and CIFAR-10 checkpoints."""
    if not isinstance(latent, torch.Tensor):
        raise TypeError("latent must be a PyTorch tensor")
    if latent.ndim != 2 or latent.shape[0] != 1:
        raise ValueError("latent must have shape [1, z_dim]")
    if latent.shape[1] != bundle.z_dim:
        raise ValueError(f"expected latent dimension {bundle.z_dim}, received {latent.shape[1]}")

    z = latent.to(device=bundle.device, dtype=torch.float32)
    if bundle.c_dim == 0:
        c = None
    else:
        # The official CIFAR-10 checkpoint is class-conditional: c is a one-hot vector.
        if class_index is None:
            if bundle.c_dim == 10:
                class_index = 3  # default to CIFAR-10 'cat'
            else:
                raise ValueError(f"This conditional model needs class_index in [0, {bundle.c_dim - 1}]")
        if isinstance(class_index, bool) or not isinstance(class_index, int):
            raise TypeError("class_index must be an integer")
        if not 0 <= class_index < bundle.c_dim:
            raise ValueError(f"class_index must be between 0 and {bundle.c_dim - 1}")
        c = torch.zeros((1, bundle.c_dim), device=bundle.device, dtype=torch.float32)
        c[0, class_index] = 1.0

    with torch.inference_mode():
        output = bundle.generator(z, c=c)

    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise RuntimeError("The generator returned an unexpected output format; expected an NCHW image tensor.")
    if output.shape[0] < 1:
        raise RuntimeError("The generator returned an empty image batch.")

    image = output[0].detach().to(device="cpu", dtype=torch.float32)
    image = (image.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy()
    if image.shape[-1] == 1:
        result = Image.fromarray(image[:, :, 0], mode="L")
    elif image.shape[-1] == 3:
        result = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
    else:
        raise RuntimeError(f"Unsupported generated image channel count: {image.shape[-1]}")

    # CIFAR-10 is natively 32x32; enlarge with nearest-neighbor only for easier viewing.
    # The displayed image is still model-generated; this is display scaling, not new inference.
    if bundle.img_resolution == 32:
        result = result.resize((256, 256), Image.Resampling.NEAREST)
    return result
