import numpy as np


def identity(image, severity, rng):
    """Return an unchanged copy of the image."""
    return image.copy()


def brightness_shift(image, severity, rng):
    """Apply a simple placeholder brightness increase."""
    shift = 10 * severity

    result = image.astype(np.float32) + shift

    return np.clip(result, 0, 255).astype(np.uint8)


STUB_CORRUPTIONS = {
    "identity": identity,
    "brightness_shift": brightness_shift,
}