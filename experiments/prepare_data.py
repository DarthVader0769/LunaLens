"""Recreate bundled images offline, or explicitly fetch the credited sources.

Run from repository root: python -m experiments.prepare_data
Optional initial acquisition: python -m experiments.prepare_data --download
The manifest records original bytes, crop, derived hash, and synthetic seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

from src.image_loader import PROJECT_ROOT, load_image
from src.synthetic import generate_terrain


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data(download: bool = False, synthetic_only: bool = False) -> list[dict]:
    data_dir = PROJECT_ROOT / "data"
    source_dir = data_dir / "source_images"
    sample_dir = data_dir / "sample_images"
    source_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)
    sources = json.loads((data_dir / "source_catalog.json").read_text())["sources"]
    old_manifest = data_dir / "manifest.json"
    previous = {}
    if old_manifest.exists():
        previous = {x["id"]: x for x in json.loads(old_manifest.read_text())["images"]}
    images = []
    if not synthetic_only:
        for source in sources:
            original = source_dir / source["source_file"]
            retrieved_at = previous.get(source["id"], {}).get("retrieved_at_utc")
            if download:
                request = urllib.request.Request(source["download_url"], headers={"User-Agent": "LunaLens-educational-prototype/0.1"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read(20 * 1024 * 1024 + 1)
                if len(payload) > 20 * 1024 * 1024:
                    raise ValueError("Unexpectedly large sample download.")
                # Decode before replacing a cached original; fail on HTML/error pages.
                load_image(payload)
                original.write_bytes(payload)
                retrieved_at = datetime.now(timezone.utc).isoformat()
            if not original.exists():
                raise FileNotFoundError(
                    f"Missing cached source {original.name}. Use --download with internet, "
                    "or --synthetic-only to explicitly create synthetic fixtures."
                )
            pixels = load_image(original)
            h, w = pixels.shape
            side = min(512, h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            crop = pixels[y0:y0+side, x0:x0+side]
            if side < 512:
                raise ValueError(f"Source {original.name} unexpectedly smaller than 512 pixels.")
            target = sample_dir / f"{source['id']}.png"
            Image.fromarray(np.rint(crop * 255).astype(np.uint8)).save(target)
            entry = dict(source)
            entry.update({
                "path": f"sample_images/{target.name}",
                "source_path": f"source_images/{original.name}",
                "source_sha256": sha256(original),
                "sha256": sha256(target),
                "retrieved_at_utc": retrieved_at,
                "original_shape": [h, w], "shape": [512, 512],
                "crop_xywh": [x0, y0, side, side],
                "processing": "Decode published JPEG; fixed center crop; Rec.709 grayscale if RGB; uint8 PNG. No scientific calibration or georeferencing.",
                "data_level": "Public outreach image derivative, not calibrated PDS product",
            })
            images.append(entry)
            print(f"Prepared {entry['id']}: {h}x{w} -> 512x512, SHA256 {entry['sha256'][:12]}", flush=True)
    fixtures = (("mixed", 11), ("rugged", 29), ("hidden_detail", 47))
    for style, seed in fixtures:
        pixels = generate_terrain(size=512, seed=seed, style=style)
        identifier = f"synthetic_{style}"
        target = sample_dir / f"{identifier}.png"
        Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(target)
        images.append({
            "id": identifier, "name": f"SYNTHETIC — {style.replace('_', ' ')}",
            "kind": "synthetic", "path": f"sample_images/{target.name}",
            "source_url": "", "mission": "None — procedural software fixture",
            "instrument": "None", "credit": "LunaLens procedural generator (AI-assisted foundation)",
            "seed": seed, "style": style, "shape": [512, 512],
            "sha256": sha256(target), "generator": "src/synthetic.py:generate_terrain",
            "processing": "Procedural texture -> fixed [0,1] to uint8 PNG.",
            "description": (
                "Synthetic stress case: high-frequency detail can be invisible to the initial map. Not a realistic lunar terrain model."
                if style == "hidden_detail" else "Procedural lunar-like texture. Not NASA imagery, physical elevation, or calibrated lunar radiometry."
            ),
        })
        print(f"Prepared {identifier}: seed={seed}", flush=True)
    manifest = {
        "schema_version": 1,
        "selection_policy": "Three source pages and fixed center crops selected before algorithm results; synthetic seeds/styles fixed before comparison.",
        "intensity_scale": "Unsigned8 /255; no per-image contrast stretch",
        "images": images,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly download all three published NASA images.")
    parser.add_argument("--synthetic-only", action="store_true", help="Write a manifest containing only clearly labelled synthetic data.")
    args = parser.parse_args()
    prepare_data(download=args.download, synthetic_only=args.synthetic_only)


if __name__ == "__main__":
    main()
