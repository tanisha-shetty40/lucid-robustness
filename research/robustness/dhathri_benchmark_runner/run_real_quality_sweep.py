import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(r"D:\Projects\Lucid")
BACKEND_DIR = PROJECT_DIR / "backend"
ARCHIVE_DIR = Path(r"C:\Users\DELL\Downloads\archive\chest_xray")
SPLIT_FILE = PROJECT_DIR / "data" / "processed" / "splits.json"

sys.path.insert(0, str(BASE_DIR.parents[2]))
sys.path.insert(0, str(BACKEND_DIR))

from app.services.imaging import quality
from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS

OUTPUT_CSV = BASE_DIR / "sweep_quality_provisional.csv"

COLUMNS = [
    "image_id", "label", "corruption", "severity",
    "prob_pneumonia", "quality", "q_brightness",
    "q_contrast", "q_sharpness", "q_noise", "q_clipping"
]


def load_test_records():
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    test_entries = split["test"][:50]

    lookup = {}
    for path in ARCHIVE_DIR.rglob("*"):
        if path.is_file():
            lookup[(path.parent.name.upper(), path.name)] = path

    records = []

    for entry in test_entries:
        relative_path = Path(entry["path"])
        label = int(entry["label"])
        class_name = "PNEUMONIA" if label == 1 else "NORMAL"
        key = (class_name, relative_path.name)

        if key not in lookup:
            raise FileNotFoundError(
                f"Test image not found: {relative_path.name}"
            )

        image_path = lookup[key]

        with Image.open(image_path) as img:
            image = np.asarray(img.convert("L"), dtype=np.uint8)

        records.append({
            "image_id": relative_path.stem,
            "label": label,
            "image": image,
        })

    return records


def adapt_corruption(function):
    def apply(image, severity, seed):
        return function(image.copy(), severity, seed=seed)
    return apply


def get_quality_fields(image):
    result = quality(image)
    indicators = result["indicators"]

    return {
        "quality": result["score"],
        "q_brightness": indicators["Brightness"]["value"],
        "q_contrast": indicators["Contrast"]["value"],
        "q_sharpness": indicators["Sharpness"]["value"],
        "q_noise": indicators["Noise"]["value"],
        "q_clipping": indicators["Clipping %"]["value"],
    }


def condition_seed(image_id, corruption, severity):
    value = f"42|{image_id}|{corruption}|{severity}"
    digest = hashlib.sha256(value.encode()).hexdigest()
    return int(digest[:8], 16)


def main():
    records = load_test_records()

    if len(records) != 50:
        raise ValueError(f"Expected 50 images, found {len(records)}")

    corruptions = {
        name: adapt_corruption(function)
        for name, function in CORRUPTIONS.items()
    }

    if len(corruptions) != 7:
        raise ValueError(f"Expected 7 corruptions, found {len(corruptions)}")

    completed = set()
    rows = []

    if OUTPUT_CSV.exists():
        old = pd.read_csv(OUTPUT_CSV)
        if not old.empty:
            rows = old.to_dict("records")
            completed = set(zip(
                old["image_id"].astype(str),
                old["corruption"].astype(str),
                old["severity"].astype(int),
            ))

    for record in records:
        image_id = record["image_id"]
        original = record["image"]
        label = record["label"]

        conditions = [("none", 0, original)]

        for name, function in corruptions.items():
            for severity in range(1, 6):
                seed = condition_seed(image_id, name, severity)
                corrupted = function(original, severity, seed)
                conditions.append((name, severity, corrupted))

        for name, severity, image in conditions:
            key = (image_id, name, severity)

            if key in completed:
                continue

            quality_fields = get_quality_fields(image)

            row = {
                "image_id": image_id,
                "label": label,
                "corruption": name,
                "severity": severity,
                "prob_pneumonia": np.nan,
                **quality_fields,
            }

            rows.append(row)
            completed.add(key)

            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                OUTPUT_CSV, index=False
            )

    results = pd.DataFrame(rows, columns=COLUMNS)
    results.to_csv(OUTPUT_CSV, index=False)

    print("Provisional quality benchmark completed.")
    print("Images:", len(records))
    print("Corruptions:", list(corruptions))
    print("Rows:", len(results))
    print("Prediction values: pending approved checkpoint")
    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
