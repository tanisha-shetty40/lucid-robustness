
from pathlib import Path

import numpy as np
from PIL import Image

from research.robustness.dhathri_benchmark_runner.sweep import run_sweep


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "demo_images"
OUTPUT_CSV = BASE_DIR / "mini_sweep_results.csv"


def brightness(image, severity, rng):
    result = image.astype(np.float32) + severity * 10
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_noise(image, severity, rng):
    noise = rng.normal(0, severity * 5, image.shape)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def predict_demo(image):
    return 0.7 if np.mean(image) > 100 else 0.3


def quality_demo(image):
    return float(np.mean(image) / 255 * 100)


def load_demo_images():
    records = []

    for image_path in sorted(IMAGE_DIR.glob("image_*.png")):
        image = np.asarray(
            Image.open(image_path).convert("L")
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

    corruptions = {
        "brightness": brightness,
        "gaussian_noise": gaussian_noise,
    }

    results = run_sweep(
        records=records,
        corruptions=corruptions,
        predict_fn=predict_demo,
        quality_fn=quality_demo,
        severities=(0, 1, 2),
        output_csv=OUTPUT_CSV,
    )

    print("Mini sweep completed!")
    print("Total rows:", len(results))
    print("CSV saved at:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
