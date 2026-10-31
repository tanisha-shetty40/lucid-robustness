from pathlib import Path

import matplotlib.pyplot as plt

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS
from research.robustness.brahmi_corruptions.make_demo_data import make_demo_image


def create_gallery(img, out_path):
    """Save all seven corruptions at severities 0 to 5."""
    names = list(CORRUPTIONS.keys())

    fig, axes = plt.subplots(
        len(names),
        6,
        figsize=(18, 3 * len(names)),
        squeeze=False,
    )

    for col, severity in enumerate(range(6)):
        for row, name in enumerate(names):
            result = CORRUPTIONS[name](img, severity, seed=42)

            axes[row, col].imshow(
                result,
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            axes[row, col].set_title(f"{name}\nSeverity {severity}")
            axes[row, col].axis("off")

    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    img = make_demo_image()
    output = "results/robustness/brahmi/gallery_synthetic.png"
    create_gallery(img, output)
    print(f"Synthetic gallery saved to: {output}")


if __name__ == "__main__":
    main()
