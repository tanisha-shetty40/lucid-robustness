
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models
from skimage.metrics import structural_similarity

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS
from research.robustness.real_benchmark.real_sweep import stable_seed


ROOT = Path.cwd()

SPLIT_FILE = ROOT / "data" / "processed" / "splits.json"
SWEEP_FILE = ROOT / "results" / "robustness" / "real_benchmark" / "sweep.csv"
CHECKPOINT = ROOT / "models" / "classifier.pt"

OUTPUT = (
    ROOT / "results" / "robustness" / "real_benchmark"
    / "quality_prediction_analysis.csv"
)


def main():
    print("Loading test split and benchmark results...")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8-sig"))
    records = split["test"]

    image_paths = {}
    for record in records:
        path = ROOT / record["path"]
        image_id = record.get("image_id") or path.name
        image_paths[image_id] = path

    sweep = pd.read_csv(SWEEP_FILE)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    classes = checkpoint["classes"]

    if "PNEUMONIA" not in classes:
        raise ValueError("PNEUMONIA class not found in checkpoint.")

    if len(classes) != 2:
        raise ValueError("This script expects a binary classifier.")

    positive_index = classes.index("PNEUMONIA")

    print("Test images:", len(image_paths))
    print("Benchmark rows:", len(sweep))
    print("Calculating image-quality metrics...")

    results = []
    current_image_id = None
    original = None

    for index, row in enumerate(sweep.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        corruption = row.corruption
        severity = int(row.severity)

        if image_id not in image_paths:
            raise ValueError(f"Image not found: {image_id}")

        if image_id != current_image_id:
            with Image.open(image_paths[image_id]) as source:
                original = np.array(
                    source.convert("L"), dtype=np.uint8
                )
            current_image_id = image_id

        if corruption == "none":
            corrupted = original.copy()
        else:
            if corruption not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {corruption}")

            seed = stable_seed(image_id, corruption, severity)

            corrupted = CORRUPTIONS[corruption](
                original.copy(), severity, seed=seed
            )

        if corrupted.shape != original.shape:
            raise ValueError(f"Image shape mismatch: {image_id}")

        difference = (
            original.astype(np.float32)
            - corrupted.astype(np.float32)
        )

        mse = float(np.mean(difference ** 2))

        psnr = (
            float("inf")
            if mse == 0
            else 10.0 * math.log10((255.0 ** 2) / mse)
        )

        ssim = float(
            structural_similarity(
                original,
                corrupted,
                data_range=255
            )
        )

        probability = float(row.prob_pneumonia)
        true_label = int(row.label)

        predicted_label = (
            positive_index
            if probability >= 0.5
            else 1 - positive_index
        )

        correct = int(predicted_label == true_label)

        results.append({
            "image_id": image_id,
            "label": true_label,
            "corruption": corruption,
            "severity": severity,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "prob_pneumonia": probability,
            "predicted_label": predicted_label,
            "correct": correct,
            "prediction_error": 1 - correct,
        })

        if index % 1000 == 0 or index == len(sweep):
            print(f"Processed {index}/{len(sweep)} rows")

    output_df = pd.DataFrame(results)

    if len(output_df) != len(sweep):
        raise ValueError("Output row count does not match benchmark.")

    keys = ["image_id", "corruption", "severity"]

    if output_df.duplicated(keys).any():
        raise ValueError("Duplicate image-condition combinations.")

    if output_df[["ssim", "mse", "psnr"]].isna().any().any():
        raise ValueError("Missing image-quality values.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT, index=False)

    print("\nImage-quality analysis completed.")
    print("Output:", OUTPUT)
    print("Total rows:", len(output_df))
    print("Unique images:", output_df["image_id"].nunique())
    print("Corruptions:", output_df["corruption"].nunique())
    print("Conditions:", output_df.groupby(
        ["corruption", "severity"]
    ).ngroups)
    print("Prediction errors:", int(output_df["prediction_error"].sum()))
    print("Correct predictions:", int(output_df["correct"].sum()))


if __name__ == "__main__":
    main()
