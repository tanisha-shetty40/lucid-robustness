import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "quality_summary.csv")

metrics = [
    ("avg_quality", "Average Quality Score"),
    ("avg_brightness", "Average Brightness"),
    ("avg_contrast", "Average Contrast"),
    ("avg_sharpness", "Average Sharpness"),
    ("avg_noise", "Average Noise"),
    ("avg_clipping", "Average Clipping (%)"),
]

output = folder / "quality_graphs"
output.mkdir(exist_ok=True)

for column, title in metrics:
    plt.figure(figsize=(10, 6))

    for name, group in data[data.corruption != "none"].groupby("corruption"):
        plt.plot(
            group["severity"],
            group[column],
            marker="o",
            label=name
        )

    plt.xlabel("Corruption Severity")
    plt.ylabel(title)
    plt.title(title + " vs Corruption Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / f"{column}.png")
    plt.close()

print("Six quality graphs generated.")
print("Saved in:", output)
