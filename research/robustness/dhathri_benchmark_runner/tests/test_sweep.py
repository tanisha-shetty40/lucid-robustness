
import numpy as np
import pandas as pd

from research.robustness.dhathri_benchmark_runner.sweep import run_sweep


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

    corruptions = {
        "identity": identity,
        "brightness": brightness,
    }

    df = run_sweep(
        records,
        corruptions,
        predict,
        severities=(0, 1, 2),
    )

    assert len(df) == 40
    assert "image_id" in df.columns
    assert "probability" in df.columns
    assert "correct" in df.columns


def test_sweep_saves_csv_and_quality_score(tmp_path):
    records = [
        {
            "image_id": "demo_1",
            "label": 1,
            "image": np.ones((8, 8), dtype=np.uint8),
        }
    ]

    def predict(image):
        return 0.8

    def quality(image):
        return 75.0

    output_file = tmp_path / "results.csv"

    df = run_sweep(
        records=records,
        corruptions={},
        predict_fn=predict,
        quality_fn=quality,
        severities=(0,),
        output_csv=output_file,
    )

    assert len(df) == 1
    assert df.iloc[0]["quality_score"] == 75.0
    assert output_file.exists()

    saved_df = pd.read_csv(output_file)
    assert len(saved_df) == 1
    assert saved_df.iloc[0]["quality_score"] == 75.0



def test_sweep_resumes_without_repeating_completed_work(tmp_path):
    import numpy as np

    from research.robustness.dhathri_benchmark_runner.sweep import run_sweep

    records = [
        {
            "image_id": "demo_1",
            "label": 1,
            "image": np.ones((8, 8), dtype=np.uint8),
        }
    ]

    prediction_calls = []

    def brightness(image, severity, rng):
        return image + severity

    def predict(image):
        prediction_calls.append(1)
        return 0.8

    output_file = tmp_path / "resume_results.csv"

    first_run = run_sweep(
        records=records,
        corruptions={"brightness": brightness},
        predict_fn=predict,
        severities=(0, 1),
        output_csv=output_file,
    )

    assert len(first_run) == 2
    assert len(prediction_calls) == 2

    second_run = run_sweep(
        records=records,
        corruptions={"brightness": brightness},
        predict_fn=predict,
        severities=(0, 1),
        output_csv=output_file,
    )

    assert len(second_run) == 2
    assert len(prediction_calls) == 2
