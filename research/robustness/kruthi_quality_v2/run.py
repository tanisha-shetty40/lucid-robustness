import argparse

import numpy as np

from .fit import fit_weights
from .make_demo_data import make_demo_data
from .quality_v2 import quality_v2


def run_demo():
    images, wrong = make_demo_data()
    weights = fit_weights(images, wrong, split="demo")

    scores = np.array([quality_v2(image, weights)["score"] for image in images])
    per_class = len(images) // 2
    sharp_scores = scores[:per_class]
    blurred_scores = scores[per_class:]

    print("DEMO ONLY — synthetic shapes; not research results")
    print(f"Synthetic images: {len(images)}")
    print(f"Mean score, sharp shapes: {sharp_scores.mean():.1f}")
    print(f"Mean score, blurred shapes: {blurred_scores.mean():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a small demonstration using synthetic images only",
    )
    args = parser.parse_args()

    if not args.demo:
        parser.error("For now, run with --demo. Real evaluation needs the approved inputs.")

    run_demo()


if __name__ == "__main__":
    main()