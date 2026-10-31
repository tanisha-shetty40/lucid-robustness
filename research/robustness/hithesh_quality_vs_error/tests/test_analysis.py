import numpy as np
import pandas as pd
from research.robustness.hithesh_quality_vs_error.analysis import (
    add_wrong,
    quality_error_auc,
    spearman_quality_confidence,
    rejection_curve,
)
from research.robustness.hithesh_quality_vs_error.make_demo_data import make_demo_sweep

def test_add_wrong():
    df = pd.DataFrame({
        "prob_pneumonia": [0.8, 0.2, 0.6],
        "label": [1, 1, 0]
    })
    res = add_wrong(df)
    assert "wrong" in res.columns
    assert "confidence" in res.columns
    # Row 0: p=0.8, label=1 -> correct (wrong=0), conf=0.8
    # Row 1: p=0.2, label=1 -> wrong (wrong=1), conf=0.8
    # Row 2: p=0.6, label=0 -> wrong (wrong=1), conf=0.6
    assert list(res["wrong"]) == [0, 1, 1]
    assert list(res["confidence"]) == [0.8, 0.8, 0.6]

def test_quality_error_auc_demo():
    df = make_demo_sweep(n_samples=300, seed=42)
    auc, low, high = quality_error_auc(df, n_boot=100, seed=42)
    # Demo data has a strong relationship, AUC should be well above 0.5
    assert auc > 0.7
    assert low <= auc <= high

def test_spearman_quality_confidence():
    df = make_demo_sweep(n_samples=100, seed=42)
    rho, p_value = spearman_quality_confidence(df)
    assert -1.0 <= rho <= 1.0
    assert 0.0 <= p_value <= 1.0

def test_rejection_curve():
    df = make_demo_sweep(n_samples=300, seed=42)
    curve = rejection_curve(df, score_col="quality", coverages=(1.0, 0.8, 0.5), seed=42)
    assert len(curve) == 3
    # Coverage 1.0 should equal overall accuracy (1 - wrong.mean())
    df_added = add_wrong(df)
    overall_acc = 1.0 - df_added["wrong"].mean()
    assert abs(curve.loc[curve["coverage"] == 1.0, "accuracy_by_quality"].values[0] - overall_acc) < 1e-4