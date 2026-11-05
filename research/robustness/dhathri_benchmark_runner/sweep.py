
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_COLUMNS = [
    "image_id",
    "label",
    "corruption",
    "severity",
    "prob_pneumonia",
    "quality",
    "q_brightness",
    "q_contrast",
    "q_sharpness",
    "q_noise",
    "q_clipping",
]


def validate_record(record):
    required = {"image_id", "image", "label"}

    if not required.issubset(record):
        raise ValueError(
            f"Each record must contain: {required}"
        )

    if record["label"] not in (0, 1):
        raise ValueError("Label must be 0 or 1.")

    if not isinstance(record["image_id"], str):
        raise ValueError("image_id must be a string.")


def validate_probability(probability):
    probability = float(probability)

    if not np.isfinite(probability):
        raise ValueError("Prediction probability must be finite.")

    if not 0 <= probability <= 1:
        raise ValueError(
            "Prediction probability must be between 0 and 1."
        )

    return probability


def get_condition_seed(seed, image_id, corruption, severity):
    text = f"{seed}|{image_id}|{corruption}|{severity}"

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


def get_quality_values(image, quality_fn):
    default_values = {
        "quality": np.nan,
        "q_brightness": np.nan,
        "q_contrast": np.nan,
        "q_sharpness": np.nan,
        "q_noise": np.nan,
        "q_clipping": np.nan,
    }

    if quality_fn is None:
        return default_values

    result = quality_fn(image)

    # Supports the older quality function that returns one number.
    if isinstance(result, (int, float, np.integer, np.floating)):
        default_values["quality"] = float(result)
        return default_values

    # Supports a quality function returning a dictionary.
    if isinstance(result, dict):
        key_mapping = {
            "score": "quality",
            "quality": "quality",
            "brightness": "q_brightness",
            "q_brightness": "q_brightness",
            "contrast": "q_contrast",
            "q_contrast": "q_contrast",
            "sharpness": "q_sharpness",
            "q_sharpness": "q_sharpness",
            "noise": "q_noise",
            "q_noise": "q_noise",
            "clipping": "q_clipping",
            "q_clipping": "q_clipping",
        }

        for source_key, target_key in key_mapping.items():
            if source_key in result:
                default_values[target_key] = float(
                    result[source_key]
                )

        return default_values

    raise ValueError(
        "quality_fn must return a number, a dictionary, or None."
    )


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
    records = list(records)

    if limit is not None:
        records = records[:limit]

    for record in records:
        validate_record(record)

    if not callable(predict_fn):
        raise ValueError("predict_fn must be a function.")

    for name, function in corruptions.items():
        if not callable(function):
            raise ValueError(
                f"Corruption '{name}' must be a function."
            )

    results = []
    completed = set()

    # Resume from an existing CSV file.
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        if output_path.exists():
            existing = pd.read_csv(output_path)

            if not existing.empty:
                required_keys = {
                    "image_id", "corruption", "severity"
                }

                if not required_keys.issubset(existing.columns):
                    raise ValueError(
                        "Existing CSV does not contain the "
                        "required resume columns."
                    )

                completed = set(
                    zip(
                        existing["image_id"].astype(str),
                        existing["corruption"].astype(str),
                        existing["severity"].astype(int),
                    )
                )

                results = existing.to_dict("records")

    def save_results():
        if output_csv is not None:
            pd.DataFrame(
                results,
                columns=RESULT_COLUMNS
            ).to_csv(output_csv, index=False)

    # Run baseline and corrupted conditions.
    for record in records:
        image_id = str(record["image_id"])
        label = int(record["label"])
        original_image = np.asarray(record["image"])

        conditions = [("none", 0, original_image)]

        for corruption_name, corruption_fn in corruptions.items():
            for severity in severities:
                if severity == 0:
                    continue

                condition_seed = get_condition_seed(
                    seed,
                    image_id,
                    corruption_name,
                    severity,
                )

                rng = np.random.default_rng(condition_seed)

                corrupted_image = corruption_fn(
                    original_image.copy(),
                    severity,
                    rng,
                )

                conditions.append(
                    (
                        corruption_name,
                        severity,
                        corrupted_image,
                    )
                )

        for corruption_name, severity, image in conditions:
            key = (image_id, corruption_name, int(severity))

            if key in completed:
                continue

            probability = validate_probability(
                predict_fn(image)
            )

            quality_values = get_quality_values(
                image,
                quality_fn,
            )

            row = {
                "image_id": image_id,
                "label": label,
                "corruption": corruption_name,
                "severity": int(severity),
                "prob_pneumonia": probability,
                **quality_values,
            }

            results.append(row)
            completed.add(key)

            # Save after each completed condition.
            save_results()

    final_results = pd.DataFrame(
        results,
        columns=RESULT_COLUMNS,
    )

    save_results()

    return final_results
