import numpy as np
import pandas as pd

from research.robustness.dhathri_benchmark_runner.sweep import run_sweep
from research.robustness.dhathri_benchmark_runner.summarize import summarize_results


def test_sweep_runs_with_two_corruptions():
    records = [
        {
            "image_id": f"demo_{i}",
            "label": i % 2,
            "image": np.zeros((8, 8), dtype=np.uint8),
        }
        for i in range(8)
    ]

    def identity(image, severity, rng):
        return image

    def brightness(image, severity, rng):
        return np.clip(
            image.astype(float) + severity * 10, 0, 255
        ).astype(np.uint8)

    def predict(image):
        return 0.7 if image.mean() > 0 else 0.3

    df = run_sweep(
        records,
        {"identity": identity, "brightness": brightness},
        predict,
        severities=(0, 1, 2),
    )

    assert len(df) == 40
    assert "image_id" in df.columns
    assert "label" in df.columns
    assert "prob_pneumonia" in df.columns
    assert "quality" in df.columns


def test_quality_is_calculated_on_corrupted_image():
    records = [{
        "image_id": "demo_1",
        "label": 1,
        "image": np.zeros((8, 8), dtype=np.uint8),
    }]

    def add_brightness(image, severity, rng):
        return np.clip(
            image.astype(float) + severity * 50, 0, 255
        ).astype(np.uint8)

    def predict(image):
        return 0.8

    def quality(image):
        return float(image.mean())

    df = run_sweep(
        records=records,
        corruptions={"brightness": add_brightness},
        predict_fn=predict,
        quality_fn=quality,
        severities=(0, 1),
    )

    baseline = df[df["corruption"] == "none"].iloc[0]
    corrupted = df[df["corruption"] == "brightness"].iloc[0]

    assert baseline["quality"] == 0.0
    assert corrupted["quality"] == 50.0


def test_sweep_resumes_without_repeating_completed_work(tmp_path):
    records = [{
        "image_id": "demo_1",
        "label": 1,
        "image": np.ones((8, 8), dtype=np.uint8),
    }]

    call_count = {"count": 0}

    def predict(image):
        call_count["count"] += 1
        return 0.8

    output_file = tmp_path / "resume_results.csv"

    first_run = run_sweep(
        records, {}, predict,
        severities=(0,),
        output_csv=output_file,
    )

    first_call_count = call_count["count"]

    second_run = run_sweep(
        records, {}, predict,
        severities=(0,),
        output_csv=output_file,
    )

    assert len(first_run) == 1
    assert len(second_run) == 1
    assert call_count["count"] == first_call_count


def test_same_seed_produces_identical_results():
    records = [{
        "image_id": "demo_1",
        "label": 1,
        "image": np.zeros((8, 8), dtype=np.uint8),
    }]

    def random_noise(image, severity, rng):
        noise = rng.integers(
            0, 50, size=image.shape, dtype=np.uint8
        )
        return np.clip(
            image.astype(int) + noise.astype(int) * severity,
            0, 255
        ).astype(np.uint8)

    def predict(image):
        return float(image.mean() / 255)

    first = run_sweep(
        records,
        {"noise": random_noise},
        predict,
        severities=(0, 1, 2),
        seed=42,
    )

    second = run_sweep(
        records,
        {"noise": random_noise},
        predict,
        severities=(0, 1, 2),
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_summary_calculates_correct_accuracy_and_auc(tmp_path):
    data = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "label": [0, 0, 1, 1],
        "corruption": ["blur"] * 4,
        "severity": [1] * 4,
        "prob_pneumonia": [0.1, 0.4, 0.6, 0.9],
        "quality": [90, 85, 80, 75],
        "q_brightness": [1, 1, 1, 1],
        "q_contrast": [1, 1, 1, 1],
        "q_sharpness": [1, 1, 1, 1],
        "q_noise": [1, 1, 1, 1],
        "q_clipping": [1, 1, 1, 1],
    })

    input_file = tmp_path / "small_sweep.csv"
    output_file = tmp_path / "small_summary.csv"

    data.to_csv(input_file, index=False)

    summary = summarize_results(input_file, output_file)

    assert len(summary) == 1
    assert summary.iloc[0]["n"] == 4
    assert summary.iloc[0]["accuracy"] == 1.0
    assert summary.iloc[0]["auc"] == 1.0
    assert summary.iloc[0]["sensitivity"] == 1.0
    assert summary.iloc[0]["specificity"] == 1.0
    assert output_file.exists()
