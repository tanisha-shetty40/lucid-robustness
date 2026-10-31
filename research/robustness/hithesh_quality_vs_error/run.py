import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from research.robustness.hithesh_quality_vs_error.make_demo_data import make_demo_sweep
from research.robustness.hithesh_quality_vs_error.analysis import (
    add_wrong,
    quality_error_auc,
    spearman_quality_confidence,
    rejection_curve,
)

def main():
    parser = argparse.ArgumentParser(description="Run Quality vs Error Analysis")
    parser.add_argument("--sweep", type=str, default="demo", help="Sweep mode (e.g. demo)")
    args = parser.parse_args()

    out_dir = "results/robustness/hithesh"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running sweep: {args.sweep}")
    df = make_demo_sweep(n_samples=500, seed=42)
    df_added = add_wrong(df)

    # 1. AUC calculation
    auc, low, high = quality_error_auc(df, n_boot=200, seed=42)
    auc_data = {"auc": float(auc), "ci_low": float(low), "ci_high": float(high)}
    with open(os.path.join(out_dir, "quality_error_auc.json"), "w") as f:
        json.dump(auc_data, f, indent=4)

    # 2. Spearman correlation
    rho, p_val = spearman_quality_confidence(df)

    # 3. Rejection curve & tables
    coverages = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    rc = rejection_curve(df, score_col="quality", coverages=coverages, seed=42)
    rc.to_csv(os.path.join(out_dir, "tables.csv"), index=False)

    # 4. Plots
    plt.figure(figsize=(6, 4))
    if "severity" in df.columns:
        df.boxplot(column="quality", by="severity", ax=plt.gca())
        plt.title("Quality vs Severity")
        plt.suptitle("")
    else:
        plt.scatter(df["confidence"], df["quality"], alpha=0.5)
        plt.title("Quality vs Confidence")
    plt.savefig(os.path.join(out_dir, "fig_quality_vs_severity.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(rc["coverage"], rc["accuracy_by_quality"], marker="o", label="Quality-based")
    plt.title("Error AUC / Accuracy by Coverage")
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "fig_error_auc.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(rc["coverage"], rc["accuracy_by_quality"], color="purple", marker="s", label="Rejection Curve")
    plt.title("Rejection Curve")
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "fig_rejection.png"))
    plt.close()

    # One page summary markdown
    summary_md = f"""# One-Page Summary: Quality vs Error Analysis
- **AUC**: {auc:.4f} (95% CI: [{low:.4f}, {high:.4f}])
- **Spearman Rho**: {rho:.4f} (p-value: {p_val:.4e})
- **Outputs generated successfully in `{out_dir}`**.
"""
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_md)

    print("Successfully generated all outputs in", out_dir)

if __name__ == "__main__":
    main()