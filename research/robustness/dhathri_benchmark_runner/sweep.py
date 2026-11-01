
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_COLUMNS = [
    "image_id",
    "label",
    "corruption",
    "severity",
    "probability",
    "prediction",
    "correct",
    "quality_score",
]


def run_sweep(
    records,
    corruptions,
    predict_fn,
    quality_fn=None,
    severities=(0, 1, 2, 3, 4, 5),
    seed=42,
    limit=None,
    output_csv=None,
):
    """Run a benchmark sweep and optionally resume from a CSV."""

    if limit is not None:
        records = records[:limit]

    output_path = Path(output_csv) if output_csv is not None else None

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load previous results if the CSV already exists.
    if output_path is not None and output_path.exists():
        df = pd.read_csv(output_path)

        if not set(RESULT_COLUMNS).issubset(df.columns):
            raise ValueError("Existing CSV has an invalid format.")

        completed = set(
            zip(df["image_id"], df["corruption"], df["severity"])
        )
    else:
        df = pd.DataFrame(columns=RESULT_COLUMNS)
        completed = set()

    new_results = []

    for record in records:
        image_id = record["image_id"]
        image = np.asarray(record["image"])
        label = int(record["label"])

        if label not in (0, 1):
            raise ValueError(f"Invalid label for {image_id}")

        # Baseline condition: original image without corruption.
        conditions = [("baseline", 0, image)]

        # Generate corrupted images.
        for corruption_name, corruption_fn in corruptions.items():

            for severity in severities:

                if not isinstance(severity, (int, float)):
                    raise ValueError("Severity must be numeric")

                if severity == 0:
                    continue

                seed_text = (
                    f"{seed}|{image_id}|{corruption_name}|{severity}"
                )

                condition_seed = int.from_bytes(
                    hashlib.sha256(seed_text.encode()).digest()[:8],
                    "big",
                )

                rng = np.random.default_rng(condition_seed)

                corrupted_image = corruption_fn(
                    image.copy(), severity, rng
                )

                conditions.append(
                    (corruption_name, severity, corrupted_image)
                )

        # Run prediction for each condition.
        for corruption_name, severity, test_image in conditions:

            key = (image_id, corruption_name, severity)

            # Skip conditions already saved in the CSV.
            if key in completed:
                continue

            probability = float(predict_fn(test_image))

            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    "Prediction probability must be between 0 and 1"
                )

            prediction = int(probability >= 0.5)

            if quality_fn is not None:
                quality_score = float(quality_fn(test_image))
            else:
                quality_score = np.nan

            new_results.append({
                "image_id": image_id,
                "label": label,
                "corruption": corruption_name,
                "severity": severity,
                "probability": probability,
                "prediction": prediction,
                "correct": int(prediction == label),
                "quality_score": quality_score,
            })

            completed.add(key)

        # Save progress after each image.
        if output_path is not None and new_results:
            df = pd.concat(
                [df, pd.DataFrame(new_results)],
                ignore_index=True,
            )

            df.to_csv(output_path, index=False)
            new_results = []

    # Add any remaining results.
    if new_results:
        df = pd.concat(
            [df, pd.DataFrame(new_results)],
            ignore_index=True,
        )

    # Save final results.
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
