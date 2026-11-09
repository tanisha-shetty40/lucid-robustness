
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(
    os.environ.get("LUCID_PROJECT_DIR", r"D:\Projects\Lucid")
)

BACKEND_DIR = PROJECT_DIR / "backend"
CHECKPOINT = PROJECT_DIR / "models" / "classifier.pt"

ARCHIVE_DIR = Path(
    os.environ.get(
        "CHEST_XRAY_DIR",
        r"C:\Users\DELL\Downloads\archive\chest_xray"
    )
)

SPLIT_FILE = PROJECT_DIR / "data" / "processed" / "splits.json"

sys.path.insert(0, str(BASE_DIR.parents[2]))
sys.path.insert(0, str(BACKEND_DIR))

from research.robustness.dhathri_benchmark_runner.sweep import run_sweep

# Brahmi's package is optional.
try:
    from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS
except ImportError:
    CORRUPTIONS = None

from research.common.stubs import STUB_CORRUPTIONS

DEMO_DIR = BASE_DIR / "demo_images"
OUTPUT_DIR = BASE_DIR / "cli_results"


def make_adapter(corruption_fn):
    """Adapt Brahmi's seed-based function to the sweep RNG interface."""

    def adapted(image, severity, rng):
        seed = int(rng.integers(0, 2**32 - 1))
        return corruption_fn(image, severity, seed=seed)

    return adapted


def predict_demo(image):
    """Placeholder predictor for demonstration only."""
    return 0.7 if np.mean(image) > 100 else 0.3


def quality_demo(image):
    """Simple placeholder quality measurements for demo only."""
    image = np.asarray(image, dtype=np.float64)

    brightness = float(np.mean(image))
    contrast = float(np.std(image))

    horizontal = np.diff(image, axis=1)
    vertical = np.diff(image, axis=0)

    sharpness = float(
        (
            np.mean(np.abs(horizontal))
            + np.mean(np.abs(vertical))
        ) / 2
    )

    noise = float(
        (np.std(horizontal) + np.std(vertical)) / 2
    )

    clipping = float(
        np.mean((image <= 2) | (image >= 253)) * 100
    )

    return {
        "score": brightness / 255 * 100,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise": noise,
        "clipping": clipping,
    }


def load_demo_records():
    """Load synthetic demonstration images."""
    records = []

    for path in sorted(DEMO_DIR.glob("image_*.png")):
        with Image.open(path) as img:
            image = np.asarray(img.convert("L"), dtype=np.uint8)

        records.append({
            "image_id": path.stem,
            "image": image,
            "label": 0,
        })

    if not records:
        raise FileNotFoundError(
            f"No demo images found in {DEMO_DIR}"
        )

    return records


def load_real_records(limit=None):
    """Load records from the official test split."""
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    entries = split["test"]

    if limit is not None:
        entries = entries[:limit]

    lookup = {}

    for path in ARCHIVE_DIR.rglob("*"):
        if path.is_file():
            lookup[(path.parent.name.upper(), path.name)] = path

    records = []

    for entry in entries:
        relative_path = Path(entry["path"])
        label = int(entry["label"])

        true_class = "PNEUMONIA" if label == 1 else "NORMAL"
        key = (true_class, relative_path.name)

        if key not in lookup:
            raise FileNotFoundError(
                f"Image not found: {relative_path.name}"
            )

        with Image.open(lookup[key]) as img:
            image = np.asarray(img.convert("L"), dtype=np.uint8)

        records.append({
            "image_id": relative_path.stem,
            "image": image,
            "label": label,
        })

    return records


def load_classifier():
    """Load the approved frozen classifier."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    from app.ml.common import build_model, eval_tf

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    if classes != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(
            f"Unexpected class order: {classes}"
        )

    if checkpoint["preprocessing_version"] != "1.0":
        raise ValueError("Unexpected preprocessing version")

    model = build_model(
        checkpoint["arch"],
        len(classes),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["state"])
    model.eval()

    transform = eval_tf(checkpoint["img_size"])

    return model, transform


def make_real_predictor(model, transform):
    """Return a function that predicts pneumonia probability."""

    def predict(image):
        pil_image = Image.fromarray(image)
        tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            probabilities = torch.softmax(
                model(tensor), dim=1
            )[0]

        return float(probabilities[1].item())

    return predict


def real_quality(image):
    """Use the approved Lucid image-quality function."""
    from app.services.imaging import quality

    result = quality(image)
    indicators = result["indicators"]

    return {
        "score": result["score"],
        "brightness": indicators["Brightness"]["value"],
        "contrast": indicators["Contrast"]["value"],
        "sharpness": indicators["Sharpness"]["value"],
        "noise": indicators["Noise"]["value"],
        "clipping": indicators["Clipping %"]["value"],
    }


def save_separate_outputs(results, corruption_names):
    """Save baseline and individual corruption CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "sweep.csv",
        index=False
    )

    baseline = results[
        results["corruption"] == "none"
    ]

    baseline.to_csv(
        OUTPUT_DIR / "sweep_none.csv",
        index=False
    )

    for name in corruption_names:
        selected = results[
            results["corruption"] == name
        ]

        selected.to_csv(
            OUTPUT_DIR / f"sweep_{name}.csv",
            index=False
        )


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lucid Group B robustness benchmark runner"
    )

    parser.add_argument(
        "--corruptions",
        nargs="+",
        help="Names of corruptions to run"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run the synthetic-image demonstration"
    )

    mode.add_argument(
        "--stub",
        action="store_true",
        help="Run with synthetic images and placeholder corruptions"
    )

    return parser.parse_args()


def main():
    """Run the selected benchmark mode."""
    args = parse_arguments()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    if args.demo:
        records = load_demo_records()
        available = STUB_CORRUPTIONS
        predict_fn = predict_demo
        quality_fn = quality_demo
        mode_name = "DEMO"

    elif args.stub:
        records = load_demo_records()
        available = STUB_CORRUPTIONS
        predict_fn = predict_demo
        quality_fn = quality_demo
        mode_name = "STUB"

    else:
        if CORRUPTIONS is None:
            raise RuntimeError(
                "Real mode requires the Brahmi corruption package. "
                "Please coordinate with the package owner."
            )

        records = load_real_records(args.limit)

        if not records:
            raise ValueError("No real test records were loaded")

        model, transform = load_classifier()

        available = CORRUPTIONS
        predict_fn = make_real_predictor(model, transform)
        quality_fn = real_quality
        mode_name = "REAL MODEL"

    if args.limit is not None:
        records = records[:args.limit]

    selected_names = (
        args.corruptions
        if args.corruptions
        else list(available.keys())
    )

    unknown = set(selected_names) - set(available.keys())

    if unknown:
        raise ValueError(
            f"Unknown corruptions: {sorted(unknown)}. "
            f"Available: {list(available.keys())}"
        )

    selected_corruptions = {
        name: available[name]
        for name in selected_names
    }

    # Adapt only Brahmi's seed-based real corruption functions.
    if not args.demo and not args.stub:
        selected_corruptions = {
            name: make_adapter(function)
            for name, function in selected_corruptions.items()
        }

    if args.demo:
        output_file = OUTPUT_DIR / "cli_demo_results.csv"

    elif args.stub:
        output_file = OUTPUT_DIR / "cli_stub_results.csv"

    else:
        output_file = OUTPUT_DIR / "cli_real_results.csv"

    print("Mode:", mode_name)
    print("Images:", len(records))
    print("Corruptions:", selected_names)
    print("Seed:", args.seed)

    results = run_sweep(
        records=records,
        corruptions=selected_corruptions,
        predict_fn=predict_fn,
        quality_fn=quality_fn,
        severities=(0, 1, 2, 3, 4, 5),
        seed=args.seed,
        limit=None,
        output_csv=output_file,
    )

    save_separate_outputs(results, selected_names)

    print("\nBenchmark completed!")
    print("Total rows:", len(results))
    print("Output CSV:", output_file)
    print("Separate CSV files:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
