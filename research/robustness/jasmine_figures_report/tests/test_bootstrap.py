
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from research.robustness.jasmine_figures_report.bootstrap import (
    bootstrap_metric,
)


def accuracy_metric(df):
    predictions = (df["prob_pneumonia"] >= 0.5).astype(int)
    return accuracy_score(df["label"], predictions)


def auc_metric(df):
    if df["label"].nunique() < 2:
        return np.nan

    return roc_auc_score(
        df["label"],
        df["prob_pneumonia"]
    )


def sample_data():
    return pd.DataFrame({
        "corruption": ["blur"] * 6 + ["noise"] * 6,
        "severity": [0] * 6 + [1] * 6,
        "label": [0, 1, 0, 1, 1, 0] * 2,
        "prob_pneumonia": [
            0.1, 0.9, 0.3, 0.8, 0.7, 0.2,
            0.2, 0.8, 0.4, 0.9, 0.6, 0.1
        ],
    })


def test_bootstrap_contains_point_estimate():
    df = sample_data()

    result = bootstrap_metric(
        df,
        accuracy_metric,
        ["corruption", "severity"],
        n_boot=500,
        seed=42
    )

    assert len(result) == 2
    assert result["value"].notna().all()

    for _, row in result.iterrows():
        assert row["low"] <= row["value"] <= row["high"]


def test_bootstrap_is_reproducible():
    df = sample_data()

    first = bootstrap_metric(
        df, accuracy_metric,
        ["corruption", "severity"],
        n_boot=500, seed=42
    )

    second = bootstrap_metric(
        df, accuracy_metric,
        ["corruption", "severity"],
        n_boot=500, seed=42
    )

    pd.testing.assert_frame_equal(first, second)


def test_single_class_auc_does_not_crash():
    df = pd.DataFrame({
        "corruption": ["blur"] * 4,
        "severity": [1] * 4,
        "label": [1, 1, 1, 1],
        "prob_pneumonia": [0.8, 0.7, 0.9, 0.6],
    })

    result = bootstrap_metric(
        df,
        auc_metric,
        ["corruption", "severity"],
        n_boot=500,
        seed=42
    )

    assert len(result) == 1
    assert np.isnan(result.iloc[0]["value"])

