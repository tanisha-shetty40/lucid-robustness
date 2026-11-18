import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from .make_demo_data import make_demo_data
from .indicators import features
from .quality_v2 import quality_v2


def _make_demo_weights(images, n_sharp):
    """Create illustrative weights from synthetic shapes only."""
    values = np.stack([features(image) for image in images])
    mean = values.mean(axis=0)
    scale = values.std(axis=0) + 1e-6

    sharp_mean = values[:n_sharp].mean(axis=0)
    blurred_mean = values[n_sharp:].mean(axis=0)
    separation = (sharp_mean - blurred_mean) / scale

    feature_index = int(np.argmax(np.abs(separation)))
    direction = float(np.sign(separation[feature_index]))
    if direction == 0:
        raise ValueError("Synthetic demo images did not produce different indicators")

    coefficients = np.zeros(5, dtype=np.float64)
    coefficients[feature_index] = -20.0 * direction

    midpoint = (sharp_mean + blurred_mean) / 2
    standardized_midpoint = (midpoint - mean) / scale
    intercept = -float(coefficients @ standardized_midpoint)

    return {
        "mu": mean.tolist(),
        "sd": scale.tolist(),
        "coef": coefficients.tolist(),
        "intercept": intercept,
        "fit_on": "demo",
        "seed": 42,
    }


def run_demo():
    images, _ = make_demo_data()
    n_sharp = len(images) // 2
    weights = _make_demo_weights(images, n_sharp)

    scores = np.array(
        [quality_v2(image, weights)["score"] for image in images]
    )

    print("DEMO ONLY — synthetic shapes; not research results")
    print(f"Synthetic images: {len(images)}")
    print(f"Mean score, sharp shapes: {scores[:n_sharp].mean():.1f}")
    print(f"Mean score, blurred shapes: {scores[n_sharp:].mean():.1f}")


def get_condition_seed(seed, image_id, corruption, severity):
    """Match the deterministic seed calculation in the sweep script."""
    text = f"{seed}|{image_id}|{corruption}|{severity}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_sweep(path):
    required = {
        "image_id",
        "label",
        "corruption",
        "severity",
        "prob_pneumonia",
        "quality",
    }

    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Sweep CSV must contain columns: {sorted(required)}")

        rows = list(reader)

    if not rows:
        raise ValueError("Sweep CSV is empty")

    for row in rows:
        row["image_id"] = str(row["image_id"])
        row["label"] = int(row["label"])
        row["corruption"] = str(row["corruption"])
        row["severity"] = int(row["severity"])
        row["prob_pneumonia"] = float(row["prob_pneumonia"])
        row["quality"] = float(row["quality"])

        if row["label"] not in (0, 1):
            raise ValueError(f"Invalid label for {row['image_id']}")
        if not 0 <= row["prob_pneumonia"] <= 1:
            raise ValueError(f"Invalid probability for {row['image_id']}")
        if not np.isfinite(row["quality"]):
            raise ValueError(f"Non-finite quality score for {row['image_id']}")

    return rows


def auc_comparison(rows, bootstrap_count=500, seed=42):
    labels = np.asarray([row["wrong"] for row in rows], dtype=np.int64)
    old_risk = np.asarray([100 - row["quality"] for row in rows])
    new_risk = np.asarray([100 - row["quality_v2"] for row in rows])

    if len(np.unique(labels)) != 2:
        raise ValueError("AUC requires both correct and incorrect predictions")

    auc_old = float(roc_auc_score(labels, old_risk))
    auc_new = float(roc_auc_score(labels, new_risk))

    rng = np.random.default_rng(seed)
    boot_old = []
    boot_new = []
    boot_difference = []

    for _ in range(bootstrap_count):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]

        if len(np.unique(sampled_labels)) != 2:
            continue

        old_value = roc_auc_score(sampled_labels, old_risk[indices])
        new_value = roc_auc_score(sampled_labels, new_risk[indices])

        boot_old.append(old_value)
        boot_new.append(new_value)
        boot_difference.append(new_value - old_value)

    if not boot_old:
        raise ValueError("Could not calculate bootstrap intervals")

    old_low, old_high = np.percentile(boot_old, [2.5, 97.5])
    new_low, new_high = np.percentile(boot_new, [2.5, 97.5])
    diff_low, diff_high = np.percentile(boot_difference, [2.5, 97.5])

    return {
        "n_rows": len(rows),
        "n_images": len({row["image_id"] for row in rows}),
        "auc_old": auc_old,
        "auc_old_ci_low": float(old_low),
        "auc_old_ci_high": float(old_high),
        "auc_new": auc_new,
        "auc_new_ci_low": float(new_low),
        "auc_new_ci_high": float(new_high),
        "auc_difference_new_minus_old": auc_new - auc_old,
        "difference_ci_low": float(diff_low),
        "difference_ci_high": float(diff_high),
    }


def run_sweep_analysis(sweep_path, weights_path, lucid_root, seed=42):
    repo_root = Path(__file__).resolve().parents[3]
    lucid_root = Path(lucid_root).resolve()

    sys.path.insert(0, str(lucid_root / "backend"))
    from app.services.imaging import quality as old_quality
    from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS

    split_path = lucid_root / "data" / "processed" / "splits.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_records = split["test"]

    weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    if weights.get("fit_on") != "validation":
        raise ValueError("Quality Score V2 weights must be fitted on validation data")

    records_by_id = {}
    for record in test_records:
        image_path_text = str(record["path"]).replace("\\", "/")
        image_id = Path(image_path_text).stem
        if image_id in records_by_id:
            raise ValueError(f"Duplicate image ID in test split: {image_id}")
        records_by_id[image_id] = record

    rows = load_sweep(sweep_path)
    rows_by_id = defaultdict(list)
    seen_conditions = set()

    for row in rows:
        image_id = row["image_id"]
        if image_id not in records_by_id:
            raise ValueError(f"Image is not in the approved test split: {image_id}")

        key = (image_id, row["corruption"], row["severity"])
        if key in seen_conditions:
            raise ValueError(f"Duplicate sweep condition: {key}")
        seen_conditions.add(key)
        rows_by_id[image_id].append(row)

        if row["label"] != int(records_by_id[image_id]["label"]):
            raise ValueError(f"CSV label disagrees with test split for {image_id}")

    if set(rows_by_id) != set(records_by_id):
        raise ValueError("CSV image IDs do not exactly match the approved test split")

    expected_conditions = {("none", 0)}
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            expected_conditions.add((corruption, severity))

    for image_id, image_rows in rows_by_id.items():
        conditions = {(row["corruption"], row["severity"]) for row in image_rows}
        if conditions != expected_conditions:
            raise ValueError(f"Missing or unexpected conditions for {image_id}")

    old_score_mismatches = []
    scored_rows = []

    for image_id, record in records_by_id.items():
        image_path_text = str(record["path"]).replace("\\", "/")
        image_path = Path(image_path_text)
        if not image_path.is_absolute():
            image_path = lucid_root / image_path

        with Image.open(image_path) as source:
            original = np.asarray(source.convert("L"), dtype=np.uint8)

        for row in rows_by_id[image_id]:
            corruption = row["corruption"]
            severity = row["severity"]

            if corruption == "none":
                image = original
            else:
                condition_seed = get_condition_seed(
                    seed, image_id, corruption, severity
                )
                image = CORRUPTIONS[corruption](
                    original.copy(), severity, seed=condition_seed
                )
               
            old_result = old_quality(image)
            recomputed_old_score = float(old_result["score"])
            if recomputed_old_score != row["quality"]:
                if len(old_score_mismatches) < 5:
                    old_score_mismatches.append(
                        (image_id, corruption, severity,
                         row["quality"], recomputed_old_score)
                    )
                continue

            new_score = quality_v2(image, weights)["score"]
            wrong = int(
                (row["prob_pneumonia"] >= 0.5) != bool(row["label"])
            )

            scored_rows.append({
                **row,
                "quality_v2": float(new_score),
                "wrong": wrong,
            })

    if old_score_mismatches:
        details = "\n".join(str(item) for item in old_score_mismatches)
        raise ValueError(
            "Recreated old scores do not match sweep.csv. "
            "Stop and report this before comparing.\n" + details
        )

    if len(scored_rows) != len(rows):
        raise ValueError("Not every sweep row received a Quality Score V2 score")

    comparisons = []
    overall = auc_comparison(scored_rows, seed=42)
    comparisons.append({"corruption": "overall", **overall})

    for corruption in CORRUPTIONS:
        group = [row for row in scored_rows if row["corruption"] == corruption]
        comparisons.append({
            "corruption": corruption,
            **auc_comparison(group, seed=42),
        })

    output_dir = repo_root / "results" / "robustness" / "kruthi"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "comparison.csv"
    columns = list(comparisons[0].keys())
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparisons)

    metadata = {
        "project": "lucid",
        "group": "B",
        "author": "kruthi",
        "date": date.today().isoformat(),
        "checkpoint": "classifier.pt v1.0.0 (predictions from supplied sweep)",
        "dataset": "approved Kermany test split",
        "n_images": len(records_by_id),
        "n_rows": len(scored_rows),
        "base_seed": seed,
        "corruption_seed_method": (
            "SHA256 of base seed, image ID, corruption, and severity; "
            "matches the script used to create the supplied sweep"
        ),
        "protocol_note": (
            "This seed method differs from the manual's 42 + image index rule. "
            "Analysis uses the supplied sweep's actual method for reproduction."
        ),
        "weights_fit_on": "validation",
        "weights_seed": weights.get("seed"),
        "classifier_rerun": False,
        "bootstrap_method": "Hithesh row bootstrap, 500 resamples, seed 42",
        "results": comparisons,
    }

    metadata_path = output_dir / "quality_v2.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    summary_path = output_dir / "quality_v2_summary.md"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("# Quality Score V2 test comparison\n\n")
        file.write(
            "Research and educational use only; not a diagnostic result.\n\n"
        )
        file.write(
            f"Approved test images: {len(records_by_id)}. "
            f"Rows analyzed: {len(scored_rows)}. "
            "The classifier was not rerun.\n\n"
        )
        file.write(
            "The supplied sweep used condition-specific hash seeds, which "
            "differs from the manual's `42 + image index` rule. "
            "This deviation is recorded for review.\n\n"
        )
        file.write(
            "| Corruption | Rows | Old AUC (95% CI) | "
            "V2 AUC (95% CI) | V2 - old |\n"
        )
        file.write("|---|---:|---:|---:|---:|\n")
        for result in comparisons:
            file.write(
                f"| {result['corruption']} | {result['n_rows']} | "
                f"{result['auc_old']:.3f} "
                f"({result['auc_old_ci_low']:.3f}–{result['auc_old_ci_high']:.3f}) | "
                f"{result['auc_new']:.3f} "
                f"({result['auc_new_ci_low']:.3f}–{result['auc_new_ci_high']:.3f}) | "
                f"{result['auc_difference_new_minus_old']:.3f} |\n"
            )

    print(f"Rows analyzed: {len(scored_rows)}")
    print(f"Comparison saved: {comparison_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--sweep")
    parser.add_argument("--weights")
    parser.add_argument("--lucid-root")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.sweep or not args.weights:
        parser.error("Use --demo, or provide both --sweep and --weights")

    repo_root = Path(__file__).resolve().parents[3]
    lucid_root = args.lucid_root or (repo_root.parent / "Lucid")
    run_sweep_analysis(args.sweep, args.weights, lucid_root, seed=args.seed)


if __name__ == "__main__":
    main()