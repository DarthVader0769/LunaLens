"""Mathematical and boundary-case checks for observable-image descriptors."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.feature_extraction import FEATURE_NAMES, extract_features, normalise_features, terrain_proxies
from src.segmentation import Region, segment_image


def test_segmentation_covers_each_pixel_once_and_retains_boundary_tiles():
    shape = (9, 13)
    regions = segment_image(shape, 5)
    coverage = np.zeros(shape, dtype=int)
    for region in regions:
        coverage[region.slices] += 1
    np.testing.assert_array_equal(coverage, np.ones(shape))
    assert [r.id for r in regions] == list(range(6))
    assert [(r.row, r.col) for r in regions] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert (regions[-1].y0, regions[-1].x0, regions[-1].height, regions[-1].width) == (5, 10, 4, 3)
    assert sum(region.area for region in regions) == np.prod(shape)
    assert all(region.cost == 1 for region in regions)
    area_regions = segment_image(shape, 5, cost_mode="area")
    assert sum(region.cost for region in area_regions) == pytest.approx(9 * 13 / 25)
    assert area_regions[-1].cost == 12 / 25


def test_region_is_immutable():
    region = Region(0, 0, 0, 0, 0, 2, 3)
    with pytest.raises(FrozenInstanceError):
        region.cost = 0.0


@pytest.mark.parametrize("shape,size,mode", [
    ((0, 4), 2, "uniform"), ((-1, 4), 2, "uniform"), ((4, 4, 3), 2, "uniform"),
    ((4,), 2, "uniform"), ((4.5, 3), 2, "uniform"), ((True, 3), 2, "uniform"),
    ((4, 4), 0, "uniform"), ((4, 4), 1.5, "uniform"), ((4, 4), True, "uniform"),
    ((4, 4), 2, "mystery"),
])
def test_invalid_segmentation_inputs_fail(shape, size, mode):
    with pytest.raises(ValueError):
        segment_image(shape, size, mode)


@pytest.mark.parametrize("cost", [0, -1, np.inf, np.nan, True, "1"])
def test_invalid_region_costs_fail(cost):
    with pytest.raises(ValueError):
        Region(0, 0, 0, 0, 0, 1, 1, cost)


def test_constant_map_has_zero_descriptor_and_zero_normalized_columns():
    image = np.full((13, 17), 0.5, dtype=np.float32)
    features = extract_features(image, segment_image(image.shape, 8))
    assert features.shape == (6, 7)
    np.testing.assert_allclose(features, 0, atol=1e-12)
    np.testing.assert_array_equal(normalise_features(features), np.zeros((6, 7)))


def test_ramp_features_match_independent_intensity_and_entropy_formulas():
    image = np.linspace(0, 1, 32, dtype=np.float32)[None, :].repeat(32, axis=0)
    values = dict(zip(FEATURE_NAMES, extract_features(image, segment_image(image.shape, 32))[0]))
    assert values["variance"] == pytest.approx(np.var(image.astype(np.float64)))
    assert values["entropy"] == pytest.approx(5.0)  # 32 equiprobable histogram bins.
    assert values["gradient"] == pytest.approx(1 / 32)  # Sobel, reflected borders.
    assert values["edge_density"] == 0  # The ramp's gradient stays below 0.04.
    expected_contrast = float(np.percentile(image, 95) - np.percentile(image, 5))
    assert values["contrast"] == pytest.approx(expected_contrast)
    assert values["information"] == pytest.approx(expected_contrast / 2)


def test_high_frequency_pattern_has_more_texture_and_boundaries_than_flat_map():
    flat = np.full((32, 32), 0.5, dtype=np.float32)
    stripes = np.tile(((np.arange(32) // 2) % 2).astype(np.float32), (32, 1))
    regions = segment_image(flat.shape, 32)
    f_flat = dict(zip(FEATURE_NAMES, extract_features(flat, regions)[0]))
    f_stripes = dict(zip(FEATURE_NAMES, extract_features(stripes, regions)[0]))
    for name in ("variance", "edge_density", "gradient", "texture", "entropy", "contrast", "information"):
        assert f_stripes[name] > f_flat[name]


def test_normalization_is_per_column_constant_safe_and_does_not_mutate_input():
    values = np.array([[1, 3, 5, 0, 1, 4, 7], [2, 3, 9, 0, 5, 2, 7], [3, 3, 7, 0, 3, 0, 7]], dtype=float)
    saved = values.copy()
    result = normalise_features(values)
    np.testing.assert_array_equal(values, saved)
    np.testing.assert_allclose(result[:, 0], [0, 0.5, 1])
    np.testing.assert_allclose(result[:, 2], [0, 1, 0.5])
    np.testing.assert_array_equal(result[:, [1, 3, 6]], 0)
    assert np.all((result >= 0) & (result <= 1))
    assert normalise_features(np.zeros((0, 7))).shape == (0, 7)


@pytest.mark.parametrize("features", [np.zeros((2, 6)), np.zeros(7), np.full((2, 7), np.nan),
                                          np.full((2, 7), np.inf), np.ones((2, 7), dtype=complex)])
def test_invalid_feature_matrices_fail(features):
    with pytest.raises(ValueError):
        normalise_features(features)


def test_region_outside_image_and_duplicate_ids_are_rejected():
    image = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        extract_features(image, [Region(0, 0, 0, 7, 0, 2, 2)])
    with pytest.raises(ValueError, match="unique"):
        extract_features(image, [Region(0, 0, 0, 0, 0, 2, 2)] * 2)


def test_tiny_tiles_and_proxy_masks_remain_well_defined():
    image = np.array([[0.5]], dtype=np.float32)
    features = extract_features(image, segment_image(image.shape, 1))
    np.testing.assert_array_equal(features, np.zeros((1, 7)))
    proxies = terrain_proxies(image)
    for name in ("shadow_mask", "boundary_mask", "rough_texture_mask"):
        assert proxies[name].shape == image.shape
        assert proxies[name].dtype == bool
        assert not proxies[name].any()
    assert proxies["candidate_circles"] == []


def test_optional_circle_proxy_finds_an_ideal_ring_without_calling_it_a_verified_crater():
    y, x = np.mgrid[:64, :64]
    image = np.full((64, 64), 0.3, dtype=np.float32)
    image[np.abs(np.hypot(y - 32, x - 32) - 17) < 1.5] = 0.85
    proxies = terrain_proxies(image)
    circles = proxies["candidate_circles"]
    assert 1 <= len(circles) <= 12
    assert any(abs(c["x"] - 32) < 5 and abs(c["y"] - 32) < 5 and abs(c["radius"] - 17) < 5 for c in circles)
    assert np.isfinite(proxies["roughness_proxy"]).all()
    assert "not verified" in proxies["notes"]["candidate_circles"]
