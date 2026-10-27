import time
from pathlib import Path
from model_loader import load_stylegan2
from latent_utils import create_latent_from_seed
from image_generator import generate_image

root = Path(__file__).resolve().parent
checkpoint = root / "models" / "afhqcat.pkl"
repo = root / "third_party" / "stylegan2-ada-pytorch"

print("[1/3] Loading pretrained model...", flush=True)
start = time.perf_counter()

bundle = load_stylegan2(checkpoint, repo, "CPU")

print(
    f"[2/3] Model loaded. Device: {bundle.device}, "
    f"Resolution: {bundle.img_resolution}",
    flush=True
)

print("[3/3] Generating one image. Please wait...", flush=True)
start = time.perf_counter()

latent = create_latent_from_seed(17, bundle.z_dim)
image = generate_image(bundle, latent)

output = root / "one_image_test.png"
image.save(output)

print(f"SUCCESS: Image saved to {output}", flush=True)
print(f"Generation time: {time.perf_counter() - start:.1f} seconds", flush=True)
