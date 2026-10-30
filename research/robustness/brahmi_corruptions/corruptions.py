
import cv2
import numpy as np

BLUR = [0.002, 0.004, 0.007, 0.010, 0.015]
NOISE = [5, 10, 20, 35, 50]
CONTRAST = [0.8, 0.6, 0.45, 0.3, 0.2]
BRIGHTNESS = [15, 30, 50, 70, 90]
JPEG_QUALITY = [60, 40, 25, 15, 8]
DOWNSCALE = [0.75, 0.5, 0.35, 0.25, 0.15]


def _validate_input(img, severity):
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 2:
        raise ValueError("img must be a grayscale image")
    if img.dtype != np.uint8:
        raise TypeError("img must have dtype uint8")
    if not isinstance(severity, (int, np.integer)):
        raise TypeError("severity must be an integer")
    if not 0 <= severity <= 5:
        raise ValueError("severity must be between 0 and 5")


def _to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _prepare(img, severity):
    _validate_input(img, severity)
    if severity == 0:
        return img.copy()
    return None


def blur(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    sigma = BLUR[severity - 1] * min(img.shape)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def noise(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    rng = np.random.default_rng(seed)
    random_noise = rng.normal(0, NOISE[severity - 1], size=img.shape)
    return _to_u8(img.astype(np.float32) + random_noise)


def contrast(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    factor = CONTRAST[severity - 1]
    img_float = img.astype(np.float32)
    mean = np.mean(img_float)
    return _to_u8(mean + factor * (img_float - mean))


def brightness_up(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    return _to_u8(img.astype(np.float32) + BRIGHTNESS[severity - 1])


def brightness_down(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    return _to_u8(img.astype(np.float32) - BRIGHTNESS[severity - 1])


def jpeg(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    success, encoded = cv2.imencode(
        ".jpg", img,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY[severity - 1]]
    )
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded.astype(np.uint8)


def downscale(img, severity, seed=0):
    result = _prepare(img, severity)
    if result is not None:
        return result
    scale = DOWNSCALE[severity - 1]
    height, width = img.shape
    small = cv2.resize(
        img,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA
    )
    return cv2.resize(
        small, (width, height), interpolation=cv2.INTER_LINEAR
    )


CORRUPTIONS = {
    "blur": blur,
    "noise": noise,
    "contrast": contrast,
    "brightness_up": brightness_up,
    "brightness_down": brightness_down,
    "jpeg": jpeg,
    "downscale": downscale,
}
