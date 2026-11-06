from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CSV = BASE / "classification_summary_full.csv"
OUT = BASE / "classifier_graphs_full"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)

for col in ["severity", "accuracy_percent", "average_confidence", "average_quality"]:
    df[col] = pd.to_numeric(df[col])

corruptions = sorted(df[df["corruption"] != "none"]["corruption"].unique())
baseline = df[df["corruption"] == "none"].iloc[0]

# Graph 1: Accuracy vs severity
plt.figure(figsize=(10, 6))
for corruption in corruptions:
    part = df[df["corruption"] == corruption].sort_values("severity")
    plt.plot(part["severity"], part["accuracy_percent"],
             marker="o", label=corruption)

plt.axhline(baseline["accuracy_percent"], linestyle="--",
            label=f"Clean baseline ({baseline['accuracy_percent']:.2f}%)")
plt.xlabel("Corruption severity")
plt.ylabel("Accuracy (%)")
plt.title("Classifier Accuracy under Image Corruptions")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 105)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "accuracy_vs_severity.png", dpi=300)
plt.close()

# Graph 2: Average confidence vs severity
plt.figure(figsize=(10, 6))
for corruption in corruptions:
    part = df[df["corruption"] == corruption].sort_values("severity")
    plt.plot(part["severity"], part["average_confidence"],
             marker="o", label=corruption)

plt.axhline(baseline["average_confidence"], linestyle="--",
            label="Clean baseline")
plt.xlabel("Corruption severity")
plt.ylabel("Average model confidence")
plt.title("Model Confidence under Image Corruptions")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "confidence_vs_severity.png", dpi=300)
plt.close()

# Graph 3: Average image quality vs severity
plt.figure(figsize=(10, 6))
for corruption in corruptions:
    part = df[df["corruption"] == corruption].sort_values("severity")
    plt.plot(part["severity"], part["average_quality"],
             marker="o", label=corruption)

plt.axhline(baseline["average_quality"], linestyle="--",
            label="Clean baseline")
plt.xlabel("Corruption severity")
plt.ylabel("Average quality score")
plt.title("Image Quality under Different Corruptions")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "quality_vs_severity.png", dpi=300)
plt.close()

print("Graphs created successfully:")
for path in sorted(OUT.glob("*.png")):
    print(path.name)
