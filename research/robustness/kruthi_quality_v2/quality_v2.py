import numpy as np

from .indicators import features


_INDICATOR_NAMES = (
    "hist_spread",
    "edge_density",
    "entropy",
    "blockiness",
    "local_contrast",
)


def quality_v2(img, weights):
    """Return a 0–100 quality score and the five measured indicators."""
    values = features(img)

    mean = np.asarray(weights["mu"], dtype=np.float64)
    scale = np.asarray(weights["sd"], dtype=np.float64)
    coefficients = np.asarray(weights["coef"], dtype=np.float64)
    intercept = float(weights["intercept"])

    if mean.shape != (5,) or scale.shape != (5,) or coefficients.shape != (5,):
        raise ValueError("weights must contain five values for mu, sd, and coef")
    if not (
        np.all(np.isfinite(mean))
        and np.all(np.isfinite(scale))
        and np.all(scale > 0)
        and np.all(np.isfinite(coefficients))
        and np.isfinite(intercept)
    ):
        raise ValueError("weights must contain finite values and positive scales")

    standardized = (values - mean) / scale
    logit = float(standardized @ coefficients + intercept)

    # Clipping keeps exp() numerically stable for very large logits.
    probability_of_error = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
    score = int(round(100.0 * (1.0 - probability_of_error)))

    return {
        "score": score,
        "indicators": dict(zip(_INDICATOR_NAMES, map(float, values))),
    }