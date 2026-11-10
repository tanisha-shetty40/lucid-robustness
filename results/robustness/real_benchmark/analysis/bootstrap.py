

import numpy as np
import pandas as pd


def bootstrap_metric(df, metric_fn, group_cols, n_boot=500, seed=42):
    """Calculate grouped metrics with 95% bootstrap confidence intervals."""

    required = list(group_cols)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")

    rng = np.random.default_rng(seed)
    results = []

    for group_key, group_df in df.groupby(
        required, dropna=False, sort=True
    ):
        if len(required) == 1:
            group_key = (group_key,)

        group_df = group_df.reset_index(drop=True)

        try:
            point_estimate = metric_fn(group_df)
        except (ValueError, ZeroDivisionError):
            point_estimate = np.nan

        bootstrap_values = []

        for _ in range(n_boot):
            indices = rng.integers(
                0, len(group_df), size=len(group_df)
            )
            sample = group_df.iloc[indices]

            try:
                score = metric_fn(sample)
                if np.isfinite(score):
                    bootstrap_values.append(float(score))
            except (ValueError, ZeroDivisionError):
                continue

        if np.isfinite(point_estimate) and bootstrap_values:
            low, high = np.percentile(
                bootstrap_values, [2.5, 97.5]
            )
        else:
            low, high = np.nan, np.nan

        row = dict(zip(required, group_key))
        row.update({
            "value": point_estimate,
            "low": low,
            "high": high
        })
        results.append(row)

    return pd.DataFrame(
        results,
        columns=required + ["value", "low", "high"]
    )
