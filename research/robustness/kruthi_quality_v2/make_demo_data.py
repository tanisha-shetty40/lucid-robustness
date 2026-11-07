import cv2
import numpy as np


def make_demo_data(seed=42, per_class=24, size=128):
    """Create synthetic sharp and blurred images for a software demo.

    The returned error labels are artificial and must not be used as research data.
    """
    rng = np.random.default_rng(seed)
    images = []
    wrong = []

    for label in (0, 1):
        for _ in range(per_class):
            image = np.zeros((size, size), dtype=np.uint8)

            # Draw simple geometric shapes, not X-rays.
            for _ in range(12):
                center = (
                    int(rng.integers(12, size - 12)),
                    int(rng.integers(12, size - 12)),
                )
                radius = int(rng.integers(3, 14))
                intensity = int(rng.integers(80, 256))
                cv2.circle(image, center, radius, intensity, thickness=-1)

            # Add mild synthetic sensor variation.
            noise = rng.normal(0, 4, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            if label == 1:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0)

            images.append(image)
            wrong.append(label)

    return images, np.asarray(wrong, dtype=np.uint8)