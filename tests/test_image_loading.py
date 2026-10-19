import io

import numpy as np
import pytest
from PIL import Image
import tifffile

from src.image_loader import load_image, normalise_image
from src.preprocessing import coarse_shape, degrade_image, validate_image
from src.synthetic import generate_terrain


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_rgb_file_upload_is_grayscale_and_position_is_restored(fmt):
    rgb = np.zeros((13, 19, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format=fmt)
    buffer.seek(5)
    gray = load_image(buffer)
    assert gray.shape == (13, 19)
    assert gray.dtype == np.float32
    assert abs(float(gray.mean()) - .7152) < .01
    assert buffer.tell() == 5


def test_uint16_png_retains_fixed_range():
    pixels = np.array([[0, 65535], [16384, 32768]], dtype=np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    np.testing.assert_allclose(load_image(buffer), pixels / 65535, atol=1e-7)


def test_tiff_first_page_16bit_and_resize(tmp_path):
    path = tmp_path / "sample.tiff"
    first = np.full((10, 20), 32768, dtype=np.uint16)
    with tifffile.TiffWriter(path) as writer:
        writer.write(first)
        writer.write(np.zeros_like(first))
    output = load_image(path, max_dimension=10)
    assert output.shape == (5, 10)
    np.testing.assert_allclose(output, 32768 / 65535)


def test_rgba_transparency_is_explicit_black_composite():
    a = np.array([[[255, 255, 255, 0], [255, 255, 255, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(normalise_image(a), [[0, 1]])


def test_no_truth_dependent_contrast_stretch():
    pixels = np.full((8, 8), 20, dtype=np.uint8)
    np.testing.assert_allclose(normalise_image(pixels), 20/255)


@pytest.mark.parametrize("bad", [b"", b"not an image"])
def test_invalid_encoded_images_fail_clearly(bad):
    with pytest.raises(ValueError):
        load_image(bad)


@pytest.mark.parametrize("bad", [np.zeros((2,2,2)), np.array([[np.nan]]), np.array([[2.0]]), np.empty((0, 2))])
def test_bad_core_images_fail_validation(bad):
    with pytest.raises(ValueError):
        validate_image(bad)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_degradation_factors(factor):
    with pytest.raises(ValueError):
        degrade_image(np.ones((8,8), np.float32), factor=factor)


def test_degrade_identity_constant_and_nondivisible_shape():
    image = np.random.default_rng(0).random((17, 23)).astype(np.float32)
    np.testing.assert_array_equal(degrade_image(image, 1, 0), image)
    degraded = degrade_image(image, 4, 1)
    assert degraded.shape == image.shape
    assert degraded.var() < image.var()
    assert coarse_shape(image.shape, 4) == (5, 6)
    np.testing.assert_allclose(degrade_image(np.full((17,23), .4), 4), .4, atol=1e-6)


def test_all_terrain_styles_are_reproducible_and_different():
    first = generate_terrain(64, 3, "mixed")
    np.testing.assert_array_equal(first, generate_terrain(64, 3, "mixed"))
    assert not np.array_equal(first, generate_terrain(64, 4, "mixed"))
    for style in ("rugged", "hidden_detail"):
        sample = generate_terrain(64, 3, style)
        assert sample.shape == (64,64)
        assert sample.min() >= 0 and sample.max() <= 1
        assert not np.array_equal(first, sample)

