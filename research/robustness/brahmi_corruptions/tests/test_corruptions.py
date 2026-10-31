
import numpy as np
import pytest

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


@pytest.mark.parametrize("name", CORRUPTIONS.keys())
@pytest.mark.parametrize("severity", range(6))
def test_output_shape_dtype_and_range(sample_image, name, severity):
    result = CORRUPTIONS[name](sample_image, severity, seed=42)

    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


@pytest.mark.parametrize("name", CORRUPTIONS.keys())
def test_severity_zero_returns_copy(sample_image, name):
    result = CORRUPTIONS[name](sample_image, 0)

    assert np.array_equal(result, sample_image)
    assert result is not sample_image


@pytest.mark.parametrize("name", CORRUPTIONS.keys())
def test_input_is_not_modified(sample_image, name):
    original = sample_image.copy()

    CORRUPTIONS[name](sample_image, 5, seed=42)

    assert np.array_equal(sample_image, original)


@pytest.mark.parametrize("name", CORRUPTIONS.keys())
def test_same_seed_is_reproducible(sample_image, name):
    first = CORRUPTIONS[name](sample_image, 5, seed=42)
    second = CORRUPTIONS[name](sample_image, 5, seed=42)

    assert np.array_equal(first, second)


def test_noise_changes_with_seed(sample_image):
    first = CORRUPTIONS["noise"](sample_image, 5, seed=42)
    second = CORRUPTIONS["noise"](sample_image, 5, seed=43)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "name",
    ["blur", "noise", "contrast", "brightness_up",
     "brightness_down", "jpeg", "downscale"]
)
def test_severe_corruption_changes_image(sample_image, name):
    result = CORRUPTIONS[name](sample_image, 5, seed=42)

    difference = np.mean(
        np.abs(result.astype(np.float32) -
               sample_image.astype(np.float32))
    )

    assert difference > 0


def test_noise_and_brightness_clipping():
    img = np.array([[250, 5], [250, 5]], dtype=np.uint8)

    for name in ["noise", "brightness_up", "brightness_down"]:
        result = CORRUPTIONS[name](img, 5, seed=42)
        assert result.min() >= 0
        assert result.max() <= 255

def test_mad_does_not_decrease_with_severity():
    from research.robustness.brahmi_corruptions.make_demo_data import make_demo_image

    img = make_demo_image()

    for name, corruption in CORRUPTIONS.items():
        differences = []
        for severity in range(6):
            result = corruption(img, severity, seed=42)
            mad = np.abs(
                result.astype(np.float32) - img.astype(np.float32)
            ).mean()
            differences.append(mad)

        assert all(
            later >= earlier
            for earlier, later in zip(differences, differences[1:])
        ), f"MAD decreased with severity for {name}: {differences}"
