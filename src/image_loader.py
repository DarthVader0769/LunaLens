"""PNG/JPEG/TIFF loading without hidden contrast stretching or remote reads."""

from __future__ import annotations

import io
import json
from numbers import Integral
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tifffile
from skimage.transform import resize

from .preprocessing import validate_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000


def normalise_image(array: np.ndarray) -> np.ndarray:
    """Convert fixed sensor/dtype range to grayscale float32 in [0, 1].

    Unsigned integers use their dtype maximum (255 or 65535, for example).
    Floating images must already be in [0,1]. No min/max is learned from the
    reference image, so a dim region stays dim. RGB uses Rec.709 luminance
    coefficients on stored intensities (not a radiometric linearisation).
    Alpha is composited on black; transparent pixels are not valid lunar data.
    """
    a = np.asarray(array)
    if a.ndim not in (2, 3) or not a.size:
        raise ValueError("Expected grayscale, RGB, or RGBA image pixels.")
    if a.shape[0] * a.shape[1] > MAX_INPUT_PIXELS:
        raise ValueError("Image exceeds the 40-million-pixel input limit; crop it first.")
    if a.dtype == np.bool_:
        scaled = a.astype(np.float32)
    elif np.issubdtype(a.dtype, np.unsignedinteger):
        scaled = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    elif np.issubdtype(a.dtype, np.floating):
        scaled = a.astype(np.float32)
    else:
        raise ValueError(
            "Use unsigned 8/16-bit pixels or floating pixels scaled to [0, 1]; "
            "signed/complex scientific rasters require explicit calibration."
        )
    if not np.all(np.isfinite(scaled)) or np.any(scaled < 0) or np.any(scaled > 1):
        raise ValueError("Floating image pixels must be finite and in [0, 1].")
    if scaled.ndim == 3:
        channels = scaled.shape[2]
        if channels == 1:
            scaled = scaled[..., 0]
        elif channels == 2:  # grayscale plus alpha
            scaled = scaled[..., 0] * scaled[..., 1]
        elif channels in (3, 4):
            gray = scaled[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            scaled = gray * scaled[..., 3] if channels == 4 else gray
        else:
            raise ValueError("Multispectral arrays need an explicitly selected band.")
    return validate_image(np.clip(scaled, 0, 1))


def _read_bytes(source: str | Path | bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("Image exceeds the 100 MiB file limit; crop it first.")
        content = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        content = bytes(source)
    elif hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        if hasattr(source, "seek"):
            source.seek(0)
        content = source.read(MAX_INPUT_BYTES + 1)
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
    else:
        raise TypeError("Image source must be a local path, bytes, or binary file.")
    if not content or len(content) > MAX_INPUT_BYTES:
        raise ValueError("Image is empty or exceeds the 100 MiB file limit.")
    return content


def load_image(
    source: str | Path | bytes | bytearray | BinaryIO,
    max_dimension: int | None = None,
) -> np.ndarray:
    """Load local pixels and optionally reduce the largest dimension.

    PNG, JPEG, and TIFF are supported. TIFF reads page 0 only, preserving
    unsigned 16-bit values; compressed formats may require ``imagecodecs``.
    GeoTIFF coordinates, no-data masks, signed rasters, and multispectral cubes
    are deliberately not inferred. JPEG/PNG EXIF orientation is applied.
    """
    if max_dimension is not None and (
        isinstance(max_dimension, bool)
        or not isinstance(max_dimension, Integral)
        or max_dimension < 1
    ):
        raise ValueError("max_dimension must be a positive integer or None.")
    content = _read_bytes(source)
    stream = io.BytesIO(content)
    try:
        if content[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            with tifffile.TiffFile(stream) as tif:
                page = tif.pages[0]
                if page.imagewidth * page.imagelength > MAX_INPUT_PIXELS:
                    raise ValueError("TIFF exceeds the pixel limit; crop it first.")
                pixels = page.asarray()
                if pixels.ndim == 3 and page.planarconfig == 2:
                    pixels = np.moveaxis(pixels, 0, -1)
        else:
            with Image.open(stream) as im:
                if im.format not in ("PNG", "JPEG"):
                    raise ValueError("Supported image formats are PNG, JPEG, and TIFF.")
                if im.width * im.height > MAX_INPUT_PIXELS:
                    raise ValueError("Image exceeds the pixel limit; crop it first.")
                im = ImageOps.exif_transpose(im)
                if im.mode == "P":
                    im = im.convert("RGBA" if "transparency" in im.info else "RGB")
                elif im.mode == "CMYK":
                    im = im.convert("RGB")
                pixels = np.array(im)
        gray = normalise_image(pixels)
    except (UnidentifiedImageError, OSError, tifffile.TiffFileError) as exc:
        raise ValueError(f"Cannot decode this PNG/JPEG/TIFF image: {exc}") from exc
    if max_dimension is not None and max(gray.shape) > max_dimension:
        ratio = max_dimension / max(gray.shape)
        target = tuple(max(1, int(round(n * ratio))) for n in gray.shape)
        gray = resize(gray, target, order=1, anti_aliasing=True, preserve_range=True)
    return np.clip(gray, 0, 1).astype(np.float32)


def list_sample_images(data_dir: str | Path | None = None) -> list[dict]:
    """Return local manifest entries with absolute, cwd-independent image paths."""
    directory = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return []
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = document["images"] if isinstance(document, dict) else document
    result = []
    for record in entries:
        entry = dict(record)
        relative = Path(entry["path"])
        path = relative if relative.is_absolute() else directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest sample is missing: {path}")
        entry["path"] = str(path.resolve())
        result.append(entry)
    return result

