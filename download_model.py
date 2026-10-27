"""Download official StyleGAN2-ADA pretrained checkpoints with resume support."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
MODELS = {
    "cifar10": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl",
    "afhqcat": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl",
    "afhqdog": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
    "afhqwild": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl",
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    "metfaces": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
}


def _progress(downloaded: int, total: int | None) -> None:
    if total and total > 0:
        pct = min(downloaded * 100 / total, 100.0)
        print(
            f"\rDownloaded {downloaded / (1024**2):.1f} / "
            f"{total / (1024**2):.1f} MiB ({pct:.1f}%)",
            end="",
            flush=True,
        )
    else:
        print(f"\rDownloaded {downloaded / (1024**2):.1f} MiB", end="", flush=True)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        mode = "ab" if offset else "wb"
        total: int | None = None

        try:
            if offset:
                print(f"Resuming from {offset / (1024**2):.1f} MiB…")
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                # If the server honors Range it replies 206; otherwise restart from byte 0.
                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise requests.RequestException(
                            f"Unexpected Content-Range when resuming: {content_range!r}"
                        )
                    mode = "ab"
                    length = response.headers.get("Content-Length")
                    total = offset + int(length) if length else None
                else:
                    response.raise_for_status()
                    offset = 0
                    mode = "wb"
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else None

                downloaded = offset
                _progress(downloaded, total)
                with partial.open(mode) as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        out.write(chunk)
                        downloaded += len(chunk)
                        _progress(downloaded, total)
                print()

            final_size = partial.stat().st_size
            if total is not None and final_size != total:
                raise requests.RequestException(
                    f"Download incomplete: have {final_size} bytes, expected {total}; will resume."
                )
            partial.replace(destination)
            return

        except KeyboardInterrupt:
            print("\nDownload interrupted. Partial data has been kept; run the same command to resume.")
            raise
        except requests.RequestException as exc:
            print(f"\nDownload attempt {attempt}/5 failed: {exc}", file=sys.stderr)
            if attempt == 5:
                raise
            time.sleep(min(attempt * 2, 8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(MODELS), default="cifar10",
        help="checkpoint to download (default: cifar10, smaller for CPU demos)",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional destination .pkl path")
    args = parser.parse_args()
    destination = (args.output or (ROOT / "models" / f"{args.model}.pkl")).expanduser().resolve()
    try:
        print(f"Downloading {args.model} checkpoint from NVIDIA's official StyleGAN2-ADA host…")
        download(MODELS[args.model], destination)
        print(f"Saved checkpoint: {destination}")
        print("Only load model pickle files from sources you trust.")
        return 0
    except KeyboardInterrupt:
        return 130
    except requests.RequestException as exc:
        print(f"Download failed after retries: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
