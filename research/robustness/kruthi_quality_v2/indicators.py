import cv2
import numpy as np


def _check_image(img):
    """Require a non-empty, 2D grayscale uint8 image."""
    arr = np.asarray(img)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("img must be a non-empty 2D grayscale array")
    if arr.dtype != np.uint8:
        raise TypeError("img must have dtype uint8")
    return arr


def hist_spread(img):
    """Return the 99th percentile minus the 1st percentile."""
    arr = _check_image(img)
    p1, p99 = np.percentile(arr, [1, 99])
    return float(p99 - p1)


def edge_density(img):
    """Return the fraction of pixels marked as edges by Canny."""
    arr = _check_image(img)
    edges = cv2.Canny(arr, 100, 200)
    return float(np.mean(edges > 0))


def entropy(img):
    """Return Shannon entropy, in bits, of the 256-level grayscale histogram."""
    arr = _check_image(img)
    counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / arr.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def blockiness(img):
    """Compare jumps across 8-pixel block boundaries with jumps inside blocks."""
    arr = _check_image(img).astype(np.float32)
    differences = []

    for axis in (0, 1):
        # Differences between adjacent pixels along this direction.
        delta = np.abs(np.diff(arr, axis=axis))
        delta = np.moveaxis(delta, axis, 0)

        boundary = np.zeros(delta.shape[0], dtype=bool)
        boundary[7::8] = True
        inside = ~boundary

        # If there aren't samples on both sides, this direction isn't informative.
        if boundary.any() and inside.any():
            differences.append(float(delta[boundary].mean() - delta[inside].mean()))

    return float(np.mean(differences)) if differences else 0.0


def local_contrast(img, tile=16):
    """Return the mean standard deviation within full 16 x 16 tiles."""
    arr = _check_image(img).astype(np.float32)
    height, width = arr.shape
    usable_height = (height // tile) * tile
    usable_width = (width // tile) * tile

    if usable_height == 0 or usable_width == 0:
        return float(arr.std())

    tiles = arr[:usable_height, :usable_width].reshape(
        usable_height // tile, tile, usable_width // tile, tile
    )
    return float(tiles.std(axis=(1, 3)).mean())


def features(img):
    """Return the five indicators in the agreed, fixed order."""
    return np.array(
        [
            hist_spread(img),
            edge_density(img),
            entropy(img),
            blockiness(img),
            local_contrast(img),
        ],
        dtype=np.float64,
    )