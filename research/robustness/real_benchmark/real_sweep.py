import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms as T

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS


ROOT = Path.cwd()
CHECKPOINT = ROOT / "models" / "classifier.pt"
SPLIT_FILE = ROOT / "data" / "processed" / "splits.json"
OUTPUT = ROOT / "results" / "robustness" / "real_benchmark" / "sweep.csv"

COLUMNS = [
    "image_id",
    "label",
    "corruption",
    "severity",
    "prob_pneumonia",
]


def stable_seed(image_id, corruption, severity):
    value = f"{image_id}|{corruption}|{severity}|42"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def load_model():
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    classes = checkpoint["classes"]
    arch = checkpoint["arch"]
    size = checkpoint["img_size"]

    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()

    if "PNEUMONIA" not in classes:
        raise ValueError(f"PNEUMONIA class not found: {classes}")

    positive_index = classes.index("PNEUMONIA")

    transform = T.Compose([
        T.Grayscale(3),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return model, transform, positive_index, classes


def get_label(record, classes):
    label = record["label"]

    if isinstance(label, str):
        if label in classes:
            return classes.index(label)
        return int(label)

    return int(label)


def run_sweep(limit=None, output=OUTPUT):
    torch.set_num_threads(4)

    model, transform, positive_index, classes = load_model()

    import json

    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8-sig")
    )

    records = split["test"]

    if limit is not None:
        records = records[:limit]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()

    if output.exists() and output.stat().st_size > 0:
        with output.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add((
                    row["image_id"],
                    row["corruption"],
                    int(row["severity"])
                ))

        print("Existing results found:", len(completed))

    write_header = not output.exists() or output.stat().st_size == 0

    conditions = [("none", 0, None)]

    for name, function in CORRUPTIONS.items():
        for severity in range(1, 6):
            conditions.append((name, severity, function))

    print("Images selected:", len(records))
    print("Conditions per image:", len(conditions))
    print("Total possible results:", len(records) * len(conditions))
    print("Starting real model benchmark...")

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)

        if write_header:
            writer.writeheader()

        for index, record in enumerate(records, start=1):
            image_path = ROOT / record["path"]
            image_id = record.get("image_id") or image_path.name
            label = get_label(record, classes)

            if not 0 <= label < len(classes):
                raise ValueError(
                    f"Invalid label {label} for {image_id}"
                )

            missing = [
                (name, severity, function)
                for name, severity, function in conditions
                if (image_id, name, severity) not in completed
            ]

            if not missing:
                continue

            with Image.open(image_path) as source:
                original = np.array(
                    source.convert("L"), dtype=np.uint8
                )

            tensors = []

            for name, severity, function in missing:
                if function is None:
                    corrupted = original.copy()
                else:
                    seed = stable_seed(image_id, name, severity)
                    corrupted = function(
                        original.copy(), severity, seed=seed
                    )

                if (
                    corrupted.shape != original.shape
                    or corrupted.dtype != np.uint8
                ):
                    raise ValueError(
                        f"Invalid output from {name}, severity {severity}"
                    )

                tensors.append(
                    transform(Image.fromarray(corrupted))
                )

            batch = torch.stack(tensors)

            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(batch), dim=1
                )[:, positive_index].tolist()

            for (name, severity, _), probability in zip(
                missing, probabilities
            ):
                writer.writerow({
                    "image_id": image_id,
                    "label": label,
                    "corruption": name,
                    "severity": severity,
                    "prob_pneumonia": probability,
                })

                completed.add((image_id, name, severity))

            f.flush()

            if index % 10 == 0 or index == len(records):
                print(
                    f"Processed {index}/{len(records)} images "
                    f"| Results: {len(completed)}"
                )

    print("\nBenchmark sweep completed.")
    print("Output:", output)
    print("Total result rows:", len(completed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    args = parser.parse_args()

    run_sweep(limit=args.limit, output=args.output)