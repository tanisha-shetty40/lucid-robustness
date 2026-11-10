
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

INPUT = (
    ROOT / "results" / "robustness" / "real_benchmark"
    / "quality_error_correlations.csv"
)

OUTPUT = (
    ROOT / "results" / "robustness" / "real_benchmark"
    / "quality_error_correlations_corrected.csv"
)


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)

    count = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = sorted_p * count / np.arange(1, count + 1)

    # Enforce monotonic adjusted p-values.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(count)
    result[order] = adjusted

    return result


def main():
    print("Loading correlation results...")

    df = pd.read_csv(INPUT)

    if df["p_value"].isna().any():
        raise ValueError("Missing p-values detected.")

    print("Applying Benjamini-Hochberg correction...")

    df["q_value"] = benjamini_hochberg(df["p_value"])
    df["significant_fdr_05"] = df["q_value"] < 0.05

    df = df.sort_values("q_value")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    significant = df[df["significant_fdr_05"]]

    print("\nCorrection completed.")
    print("Total tests:", len(df))
    print("Significant after FDR correction:", len(significant))
    print("Corrected results saved:", OUTPUT)

    print("\nSignificant results:")
    if significant.empty:
        print("No correlations passed the FDR threshold.")
    else:
        print(
            significant[
                [
                    "corruption",
                    "severity",
                    "metric",
                    "spearman_rho",
                    "p_value",
                    "q_value",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
