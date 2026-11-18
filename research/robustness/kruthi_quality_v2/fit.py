import numpy as np
from sklearn.linear_model import LogisticRegression

from .indicators import features


def fit_weights(images, wrong, *, split):
    """Fit weights on validation data or explicitly labelled synthetic demo data."""
    if split not in {"validation", "demo"}:
        raise ValueError("Fit only on validation data or synthetic demo data")

    images = list(images)
    labels = np.asarray(wrong, dtype=np.int64)

    if not images:
        raise ValueError("At least one image is required")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("wrong must contain one label per image")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("wrong labels must be 0 or 1")
    if len(np.unique(labels)) != 2:
        raise ValueError("Data must include both correct and incorrect predictions")

    x = np.stack([features(image) for image in images])
    if not np.all(np.isfinite(x)):
        raise ValueError("Image indicators must all be finite")

    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    standardized = (x - mean) / scale

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(standardized, labels)

    return {
        "mu": mean.tolist(),
        "sd": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "fit_on": split,
        "seed": 42,
    }
    