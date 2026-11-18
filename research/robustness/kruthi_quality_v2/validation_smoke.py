import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS
from research.robustness.kruthi_quality_v2.indicators import features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucid-root", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.limit < 1 or args.batch_size < 1:
        parser.error("--limit and --batch-size must be positive")

    lucid_root = Path(args.lucid_root).resolve()
    role_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(lucid_root / "backend"))
    sys.path.insert(0, str(role_root))

    from app.ml.common import build_model, eval_tf

    split_path = lucid_root / "data" / "processed" / "splits.json"
    checkpoint_path = lucid_root / "models" / "classifier.pt"

    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = split["val"][:args.limit]
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
    model = build_model(checkpoint["arch"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    transform = eval_tf(checkpoint["img_size"])

    conditions_per_image = 1 + len(CORRUPTIONS) * 5
    prediction_count = 0
    wrong_count = 0
    started = time.perf_counter()

    for image_index, record in enumerate(records):
        if record["label"] not in (0, 1):
            raise ValueError("Validation labels must be 0 or 1")

        image_path = Path(str(record["path"]).replace("\\", "/"))
        if not image_path.is_absolute():
            image_path = lucid_root / image_path

        with Image.open(image_path) as source:
            original = np.asarray(source.convert("L"), dtype=np.uint8)

        cases = [("none", 0, original)]
        seed = 1000 + image_index

        for name, corruption_fn in CORRUPTIONS.items():
            for severity in range(1, 6):
                corrupted = corruption_fn(
                    original.copy(), severity, seed=seed
                )
                cases.append((name, severity, corrupted))

        tensors = []
        labels = []

        for _, _, image in cases:
            values = features(image)
            if not np.all(np.isfinite(values)):
                raise ValueError("An indicator value was not finite")
            tensors.append(transform(Image.fromarray(image)))
            labels.append(int(record["label"]))

        for start in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[start:start + args.batch_size])
            batch_labels = labels[start:start + args.batch_size]

            with torch.no_grad():
                probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()

            if not np.all(np.isfinite(probabilities)):
                raise ValueError("Model returned a non-finite probability")

            for probability, label in zip(
                probabilities[:, pneumonia_index], batch_labels
            ):
                predicted_pneumonia = float(probability) >= 0.5
                actual_pneumonia = int(label) == pneumonia_index
                wrong_count += int(predicted_pneumonia != actual_pneumonia)
                prediction_count += 1

    elapsed = time.perf_counter() - started
    print(f"Validation images processed: {len(records)}")
    print(f"Conditions per image: {conditions_per_image}")
    print(f"Predictions completed: {prediction_count}")
    print(f"Validation conditions marked wrong: {wrong_count} (smoke run only)")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print("The test split was not loaded; this is not test performance.")


if __name__ == "__main__":
    main()
