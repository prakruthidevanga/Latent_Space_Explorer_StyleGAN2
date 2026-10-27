"""Opt-in smoke test for actual checkpoint inference.

Run after cloning StyleGAN2-ADA and downloading the checkpoint:
LATENT_EXPLORER_RUN_MODEL_TEST=1 pytest -q tests/test_model_integration.py
"""
import os
from pathlib import Path
import pytest

from image_generator import generate_image
from latent_utils import create_latent_from_seed
from model_loader import load_stylegan2

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.getenv("LATENT_EXPLORER_RUN_MODEL_TEST") != "1", reason="requires external pretrained checkpoint")
def test_real_model_generates_rgb_image():
    model = ROOT / "models" / "afhqcat.pkl"
    repo = ROOT / "third_party" / "stylegan2-ada-pytorch"
    bundle = load_stylegan2(model, repo, "CPU")
    z = create_latent_from_seed(123, bundle.z_dim)
    image = generate_image(bundle, z)
    assert image.size == (bundle.img_resolution, bundle.img_resolution)
    assert image.mode in {"RGB", "L"}
