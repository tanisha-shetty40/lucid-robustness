import numpy as np
import pytest

from research.robustness.kruthi_quality_v2.indicators import (
    blockiness,
    edge_density,
    entropy,
    features,
    hist_spread,
    local_contrast,
)
from research.robustness.kruthi_quality_v2.quality_v2 import quality_v2


def test_indicators_return_finite_values_for_synthetic_image():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[16:48, 16:48] = 200

    values = features(image)

    assert values.shape == (5,)
    assert np.all(np.isfinite(values))


def test_flat_image_has_zero_spread_entropy_and_contrast():
    image = np.full((64, 64), 100, dtype=np.uint8)

    assert hist_spread(image) == 0.0
    assert entropy(image) == 0.0
    assert local_contrast(image) == 0.0


def test_indicator_functions_are_repeatable():
    image = np.indices((64, 64)).sum(axis=0)
    image = ((image % 2) * 255).astype(np.uint8)

    first = features(image)
    second = features(image)

    np.testing.assert_array_equal(first, second)


def test_invalid_image_inputs_are_rejected():
    with pytest.raises(ValueError):
        features(np.array([], dtype=np.uint8))

    with pytest.raises(ValueError):
        features(np.zeros((8, 8, 3), dtype=np.uint8))

    with pytest.raises(TypeError):
        features(np.zeros((8, 8), dtype=np.float32))


def test_quality_score_is_between_zero_and_one_hundred():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[16:48, 16:48] = 200
    weights = {
        "mu": [0, 0, 0, 0, 0],
        "sd": [1, 1, 1, 1, 1],
        "coef": [0, 0, 0, 0, 0],
        "intercept": 0,
    }

    original = image.copy()
    result = quality_v2(image, weights)

    assert 0 <= result["score"] <= 100
    assert set(result["indicators"]) == {
        "hist_spread",
        "edge_density",
        "entropy",
        "blockiness",
        "local_contrast",
    }
    np.testing.assert_array_equal(image, original)