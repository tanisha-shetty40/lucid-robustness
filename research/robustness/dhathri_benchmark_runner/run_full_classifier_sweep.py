
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("LUCID_PROJECT_DIR", r"D:\Projects\Lucid"))
BACKEND_DIR = PROJECT_DIR / "backend"
CHECKPOINT = PROJECT_DIR / "models" / "classifier.pt"
ARCHIVE_DIR = Path(os.environ.get("CHEST_XRAY_DIR", r"C:\Users\DELL\Downloads\archive\chest_xray"))
SPLIT_FILE = PROJECT_DIR / "data" / "processed" / "splits.json"

sys.path.insert(0, str(BASE_DIR.parents[2]))
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.common import build_model, eval_tf
from app.services.imaging import quality
from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS

OUTPUT_CSV = BASE_DIR / "sweep_full_classifier.csv"

COLUMNS = [
    "image_id", "label", "true_class",
    "corruption", "severity",
    "predicted_label", "predicted_class",
    "prob_normal", "prob_pneumonia",
    "confidence", "prediction_correct",
    "quality", "q_brightness", "q_contrast",
    "q_sharpness", "q_noise", "q_clipping"
]


def load_test_records():
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    entries = split["test"]

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
            "label": label,
            "true_class": true_class,
            "image": image
        })

    return records


def condition_seed(image_id, corruption, severity):
    value = f"42|{image_id}|{corruption}|{severity}"
    digest = hashlib.sha256(value.encode()).hexdigest()
    return int(digest[:8], 16)


def load_classifier():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    classes = checkpoint["classes"]

    if classes != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(f"Unexpected class order: {classes}")

    if checkpoint["preprocessing_version"] != "1.0":
        raise ValueError("Unexpected preprocessing version")

    model = build_model(
        checkpoint["arch"],
        len(classes),
        pretrained=False
    )

    model.load_state_dict(checkpoint["state"])
    model.eval()

    transform = eval_tf(checkpoint["img_size"])

    return model, transform, classes


def classify(image, model, transform, classes):
    pil_image = Image.fromarray(image)
    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    probabilities = probabilities.cpu().numpy()
    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_label": predicted_index,
        "predicted_class": classes[predicted_index],
        "prob_normal": float(probabilities[0]),
        "prob_pneumonia": float(probabilities[1]),
        "confidence": float(probabilities[predicted_index])
    }


def get_quality_fields(image):
    result = quality(image)
    indicators = result["indicators"]

    return {
        "quality": result["score"],
        "q_brightness": indicators["Brightness"]["value"],
        "q_contrast": indicators["Contrast"]["value"],
        "q_sharpness": indicators["Sharpness"]["value"],
        "q_noise": indicators["Noise"]["value"],
        "q_clipping": indicators["Clipping %"]["value"]
    }


def main():
    records = load_test_records()

    if len(records) != 874:
        raise ValueError(f"Expected 874 images, got {len(records)}")

    model, transform, classes = load_classifier()

    rows = []

    for record_number, record in enumerate(records, start=1):
        image_id = record["image_id"]
        original = record["image"]
        label = record["label"]
        true_class = record["true_class"]

        conditions = [("none", 0, original)]

        for name, function in CORRUPTIONS.items():
            for severity in range(1, 6):
                seed = condition_seed(image_id, name, severity)
                corrupted = function(
                    original.copy(), severity, seed=seed
                )

                if corrupted.dtype != np.uint8:
                    raise ValueError("Corruption returned non-uint8 image")

                if corrupted.shape != original.shape:
                    raise ValueError("Corruption changed image dimensions")

                conditions.append((name, severity, corrupted))

        for name, severity, image in conditions:
            prediction = classify(image, model, transform, classes)
            quality_fields = get_quality_fields(image)

            row = {
                "image_id": image_id,
                "label": label,
                "true_class": true_class,
                "corruption": name,
                "severity": severity,
                **prediction,
                "prediction_correct": (
                    prediction["predicted_label"] == label
                ),
                **quality_fields
            }

            rows.append(row)

        print(
            f"Processed image {record_number}/874: {image_id}"
        )

    results = pd.DataFrame(rows, columns=COLUMNS)

    if len(results) != 31464:
        raise ValueError(f"Expected 31464 rows, got {len(results)}")

    if results.isnull().any().any():
        raise ValueError("Missing values found in benchmark results")

    results.to_csv(OUTPUT_CSV, index=False)

    print("\nFull 874-image classifier benchmark completed.")
    print("Images:", len(records))
    print("Rows:", len(results))
    print("Corruptions:", list(CORRUPTIONS))
    print("Model:", "ResNet-18")
    print("Classes:", classes)
    print("Prediction mode: REAL MODEL")
    print("Correct predictions:", int(results["prediction_correct"].sum()))
    print(
        "Accuracy:",
        round(results["prediction_correct"].mean() * 100, 2),
        "%"
    )
    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
