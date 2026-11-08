
import cv2
import numpy as np


def make_demo_image(size=512):
    """Create a synthetic grayscale image for corruption testing."""
    img = np.zeros((size, size), dtype=np.uint8)

    # Create a smooth gradient.
    for y in range(size):
        img[y, :] = int(40 + 120 * y / (size - 1))

    # Add geometric shapes and texture.
    cv2.circle(img, (size // 2, size // 2), size // 4, 220, -1)
    cv2.rectangle(img, (40, 40), (140, 180), 80, -1)

    rng = np.random.default_rng(42)
    texture = rng.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float32) + texture, 0, 255)
    return img.astype(np.uint8)
