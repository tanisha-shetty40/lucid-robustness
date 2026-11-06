
import numpy as np
from scipy.ndimage import laplace


def calculate_quality(image):
    """
    Calculate basic image-quality measurements.

    Returns:
        score: Overall demo quality score (0-100).
        brightness: Mean pixel intensity.
        contrast: Standard deviation of pixel intensity.
        sharpness: Variance of Laplacian.
        noise: Estimated high-frequency variation.
        clipping: Percentage of pixels near intensity limits.
    """

    image = np.asarray(image, dtype=np.float32)

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid pixel values.")

    # Normalize image to 0-255.
    if image.max() <= 1.0:
        image = image * 255.0

    image = np.clip(image, 0, 255)

    # 1. Brightness
    brightness = float(np.mean(image))

    # 2. Contrast
    contrast = float(np.std(image))

    # 3. Sharpness
    sharpness = float(np.var(laplace(image)))

    # 4. Noise estimate using high-frequency residual
    smooth = (
        np.roll(image, 1, axis=0)
        + np.roll(image, -1, axis=0)
        + np.roll(image, 1, axis=1)
        + np.roll(image, -1, axis=1)
    ) / 4.0

    residual = image - smooth
    noise = float(np.std(residual))

    # 5. Clipping percentage
    clipping = float(
        np.mean((image <= 5) | (image >= 250)) * 100
    )

    # Simple demonstration score.
    # This is NOT the team's approved clinical quality score.
    brightness_score = max(
        0, 1 - abs(brightness - 127.5) / 127.5
    )

    contrast_score = min(contrast / 64.0, 1.0)
    sharpness_score = min(sharpness / 500.0, 1.0)
    noise_score = max(0, 1 - noise / 50.0)
    clipping_score = max(0, 1 - clipping / 100.0)

    score = 100 * np.mean([
        brightness_score,
        contrast_score,
        sharpness_score,
        noise_score,
        clipping_score,
    ])

    return {
        "score": float(score),
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise": noise,
        "clipping": clipping,
    }
