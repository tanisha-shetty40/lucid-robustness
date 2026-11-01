
from pathlib import Path

import numpy as np
from PIL import Image

from research.robustness.dhathri_benchmark_runner.sweep import run_sweep
from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "demo_images"

# Separate output file for the 7-corruption demo
OUTPUT_CSV = BASE_DIR / "brahmi_corruption_demo_results.csv"


def make_adapter(corruption_fn):
    """Adapt Brahmi's seed-based function to the sweep RNG interface."""
    def adapted_corruption(image, severity, rng):
        seed = int(rng.integers(0, 2**32 - 1))
        return corruption_fn(image, severity, seed=seed)

    return adapted_corruption


def predict_demo(image):
    # Placeholder predictor for pipeline testing only
    return 0.7 if np.mean(image) > 100 else 0.3


def quality_demo(image):
    # Placeholder quality score for pipeline testing only
    return float(np.mean(image) / 255 * 100)


def load_demo_images():
    records = []

    for image_path in sorted(IMAGE_DIR.glob("image_*.png")):
        image = np.asarray(
            Image.open(image_path).convert("L"),
            dtype=np.uint8
        )

        records.append({
            "image_id": image_path.stem,
            "image": image,
            "label": 0,
        })

    return records


def main():
    records = load_demo_images()

    if len(records) != 8:
        raise ValueError(
            f"Expected 8 demo images, found {len(records)}"
        )

    adapted_corruptions = {
        name: make_adapter(function)
        for name, function in CORRUPTIONS.items()
    }

    results = run_sweep(
        records=records,
        corruptions=adapted_corruptions,
        predict_fn=predict_demo,
        quality_fn=quality_demo,
        severities=(0, 1, 2, 3, 4, 5),
        output_csv=OUTPUT_CSV,
    )

    print("Seven-corruption demo completed!")
    print("Total rows:", len(results))
    print("Corruptions:", list(adapted_corruptions.keys()))
    print("CSV saved at:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
