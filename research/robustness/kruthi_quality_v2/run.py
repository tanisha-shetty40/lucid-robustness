import argparse

import numpy as np

from .indicators import features
from .make_demo_data import make_demo_data
from .quality_v2 import quality_v2


def _make_demo_weights(images, n_sharp):
    """Create illustrative weights from synthetic sharp and blurred shapes only."""
    values = np.stack([features(image) for image in images])
    mean = values.mean(axis=0)
    scale = values.std(axis=0) + 1e-6

    sharp_mean = values[:n_sharp].mean(axis=0)
    blurred_mean = values[n_sharp:].mean(axis=0)
    separation = (sharp_mean - blurred_mean) / scale

    feature_index = int(np.argmax(np.abs(separation)))
    direction = float(np.sign(separation[feature_index]))
    if direction == 0:
        raise ValueError("Synthetic demo images did not produce different indicators")

    coefficients = np.zeros(5, dtype=np.float64)
    coefficients[feature_index] = -20.0 * direction

    midpoint = (sharp_mean + blurred_mean) / 2
    standardized_midpoint = (midpoint - mean) / scale
    intercept = -float(coefficients @ standardized_midpoint)

    return {
        "mu": mean.tolist(),
        "sd": scale.tolist(),
        "coef": coefficients.tolist(),
        "intercept": intercept,
        "fit_on": "demo",
        "seed": 42,
    }


def run_demo():
    images, _ = make_demo_data()
    n_sharp = len(images) // 2
    weights = _make_demo_weights(images, n_sharp)

    scores = np.array([quality_v2(image, weights)["score"] for image in images])
    sharp_scores = scores[:n_sharp]
    blurred_scores = scores[n_sharp:]

    print("DEMO ONLY — synthetic shapes; not research results")
    print(f"Synthetic images: {len(images)}")
    print(f"Mean score, sharp shapes: {sharp_scores.mean():.1f}")
    print(f"Mean score, blurred shapes: {blurred_scores.mean():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a demonstration using synthetic shapes only",
    )
    args = parser.parse_args()

    if not args.demo:
        parser.error("For now, run with --demo. Real evaluation needs approved inputs.")

    run_demo()


if __name__ == "__main__":
    main()