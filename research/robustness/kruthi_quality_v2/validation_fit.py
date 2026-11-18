import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS
from research.robustness.kruthi_quality_v2.indicators import features


INDICATOR_NAMES = (
    "hist_spread",
    "edge_density",
    "entropy",
    "blockiness",
    "local_contrast",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucid-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    lucid_root = Path(args.lucid_root).resolve()
    role_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(lucid_root / "backend"))
    sys.path.insert(0, str(role_root))

    from app.ml.common import build_model, eval_tf

    split_path = lucid_root / "data" / "processed" / "splits.json"
    checkpoint_path = lucid_root / "models" / "classifier.pt"

    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = split["val"]
    if not records:
        raise ValueError("The validation split is empty")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = [str(name).upper() for name in checkpoint["classes"]]
    split_classes = [str(name).upper() for name in split.get("classes", [])]
    if split_classes and classes != split_classes:
        raise ValueError("Checkpoint class order does not match splits.json")
    if "PNEUMONIA" not in classes:
        raise ValueError("Checkpoint classes do not include PNEUMONIA")

    pneumonia_index = classes.index("PNEUMONIA")
    classifier = build_model(checkpoint["arch"], len(classes), pretrained=False)
    classifier.load_state_dict(checkpoint["state"])
    classifier.eval()
    transform = eval_tf(checkpoint["img_size"])

    feature_rows = []
    error_labels = []
    summary = {}
    prediction_count = 0
    started = time.perf_counter()

    for image_index, record in enumerate(records):
        if record["label"] not in (0, 1):
            raise ValueError("Validation labels must be 0 or 1")

        image_path = Path(str(record["path"]).replace("\\", "/"))
        if not image_path.is_absolute():
            image_path = lucid_root / image_path
        with Image.open(image_path) as source:
            original = np.asarray(source.convert("L"), dtype=np.uint8)

        cases = [("baseline", 0, original)]
        seed = 1000 + image_index

        for name, corruption_fn in CORRUPTIONS.items():
            for severity in range(1, 6):
                corrupted = corruption_fn(original.copy(), severity, seed=seed)
                cases.append((name, severity, corrupted))

        tensors = []
        case_features = []
        for _, _, image in cases:
            values = features(image)
            if not np.all(np.isfinite(values)):
                raise ValueError("An indicator value was not finite")
            case_features.append(values)
            tensors.append(transform(Image.fromarray(image)))

        for start in range(0, len(tensors), args.batch_size):
            end = start + args.batch_size
            batch = torch.stack(tensors[start:end])

            with torch.inference_mode():
                probabilities = torch.softmax(classifier(batch), dim=1).cpu().numpy()

            if not np.all(np.isfinite(probabilities)):
                raise ValueError("Model returned a non-finite probability")

            for offset, probability in enumerate(probabilities):
                case_index = start + offset
                condition, severity, _ = cases[case_index]
                predicted_pneumonia = float(probability[pneumonia_index]) >= 0.5
                actual_pneumonia = int(record["label"]) == pneumonia_index
                wrong = int(predicted_pneumonia != actual_pneumonia)

                feature_rows.append(case_features[case_index])
                error_labels.append(wrong)

                key = (condition, severity)
                item = summary.setdefault(
                    key, {"images": 0, "wrong": 0}
                )
                item["images"] += 1
                item["wrong"] += wrong
                prediction_count += 1

        completed = image_index + 1
        if completed % 25 == 0 or completed == len(records):
            print(
                f"Validation progress: {completed}/{len(records)} images "
                f"({prediction_count} predictions)",
                flush=True,
            )

    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(error_labels, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != len(INDICATOR_NAMES):
        raise ValueError("Unexpected indicator matrix shape")
    if not np.all(np.isfinite(x)):
        raise ValueError("Indicator matrix contains non-finite values")
    if len(np.unique(y)) != 2:
        raise ValueError(
            "Validation conditions did not include both correct and wrong "
            "predictions, so logistic regression cannot be fitted."
        )

    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    standardized = (x - mean) / scale

    quality_model = LogisticRegression(max_iter=1000, random_state=42)
    quality_model.fit(standardized, y)

    elapsed = time.perf_counter() - started
    output_dir = (
        role_root / "results" / "robustness" / "kruthi"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        "mu": mean.tolist(),
        "sd": scale.tolist(),
        "coef": quality_model.coef_[0].tolist(),
        "intercept": float(quality_model.intercept_[0]),
        "fit_on": "validation",
        "seed": 42,
        "indicators": list(INDICATOR_NAMES),
        "validation_images": len(records),
        "conditions_per_image": 1 + len(CORRUPTIONS) * 5,
        "predictions": prediction_count,
        "checkpoint_architecture": checkpoint["arch"],
        "note": "Validation fit only; not final test performance.",
    }

    weights_path = output_dir / "quality_v2_validation_weights.json"
    weights_path.write_text(
        json.dumps(weights, indent=2) + "\n", encoding="utf-8"
    )

    summary_path = output_dir / "quality_v2_validation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["condition", "severity", "images", "wrong", "wrong_rate"]
        )
        for (condition, severity), item in sorted(summary.items()):
            writer.writerow(
                [
                    condition,
                    severity,
                    item["images"],
                    item["wrong"],
                    round(item["wrong"] / item["images"], 6),
                ]
            )

    print(f"Validation images processed: {len(records)}")
    print(f"Conditions per image: {1 + len(CORRUPTIONS) * 5}")
    print(f"Predictions completed: {prediction_count}")
    print(f"Validation predictions marked wrong: {int(y.sum())}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print(f"Weights saved: {weights_path}")
    print(f"Condition summary saved: {summary_path}")
    print("The test split was not loaded; these are validation outputs only.")


if __name__ == "__main__":
    main()
