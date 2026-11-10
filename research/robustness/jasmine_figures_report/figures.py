
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_vs_severity(summary_ci, metric, out_path):
    """
    Plot a metric against corruption severity.

    Each corruption type gets a separate panel.
    The shaded region represents the 95% bootstrap confidence interval.

    Expected columns:
    corruption, severity, value, low, high
    """

    required = {"corruption", "severity", "value", "low", "high"}
    missing = required - set(summary_ci.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    corruptions = sorted(summary_ci["corruption"].dropna().unique())

    if not corruptions:
        raise ValueError("No corruption groups found in summary data.")

    n_panels = len(corruptions)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4 * n_panels, 3.5),
        squeeze=False
    )

    axes = axes.ravel()

    for ax, corruption in zip(axes, corruptions):
        group = summary_ci[
            summary_ci["corruption"] == corruption
        ].sort_values("severity")

        x = group["severity"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        low = group["low"].to_numpy(dtype=float)
        high = group["high"].to_numpy(dtype=float)

        valid = pd.notna(y)

        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=metric
        )

        ci_valid = pd.notna(low) & pd.notna(high)

        ax.fill_between(
            x[ci_valid],
            low[ci_valid],
            high[ci_valid],
            alpha=0.20
        )

        ax.set_title(str(corruption), fontsize=10)
        ax.set_xlabel("Severity", fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        f"{metric} vs Corruption Severity",
        fontsize=11
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def create_synthetic_gallery(out_path):
    """
    Create an illustrative gallery of synthetic grayscale images.
    These are NOT real X-ray images.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    conditions = [
        "Original",
        "Blur",
        "Noise",
        "Low contrast",
        "Brightness down",
        "Downscaled"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for ax, condition in zip(axes.ravel(), conditions):
        image = np.zeros((100, 100), dtype=float)

        y, x = np.mgrid[-1:1:100j, -1:1:100j]
        image = 0.35 + 0.25 * np.exp(
            -((x / 0.55) ** 2 + (y / 0.9) ** 2)
        )

        image += 0.12 * np.exp(
            -(((x - 0.25) / 0.18) ** 2
              + ((y + 0.1) / 0.3) ** 2)
        )

        if condition == "Blur":
            image = (
                image
                + np.roll(image, 1, axis=0)
                + np.roll(image, -1, axis=0)
            ) / 3

        elif condition == "Noise":
            image += rng.normal(0, 0.12, image.shape)

        elif condition == "Low contrast":
            image = 0.5 + 0.2 * (image - 0.5)

        elif condition == "Brightness down":
            image -= 0.2

        elif condition == "Downscaled":
            small = image[::4, ::4]
            image = np.repeat(
                np.repeat(small, 4, axis=0), 4, axis=1
            )

        image = np.clip(image, 0, 1)

        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(condition, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "Illustrative Synthetic Image Quality Examples",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def create_synthetic_gallery(out_path):
    """
    Create an illustrative gallery of synthetic grayscale images.
    These are NOT real X-ray images.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    conditions = [
        "Original",
        "Blur",
        "Noise",
        "Low contrast",
        "Brightness down",
        "Downscaled"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for ax, condition in zip(axes.ravel(), conditions):
        image = np.zeros((100, 100), dtype=float)

        y, x = np.mgrid[-1:1:100j, -1:1:100j]
        image = 0.35 + 0.25 * np.exp(
            -((x / 0.55) ** 2 + (y / 0.9) ** 2)
        )

        image += 0.12 * np.exp(
            -(((x - 0.25) / 0.18) ** 2
              + ((y + 0.1) / 0.3) ** 2)
        )

        if condition == "Blur":
            image = (
                image
                + np.roll(image, 1, axis=0)
                + np.roll(image, -1, axis=0)
            ) / 3

        elif condition == "Noise":
            image += rng.normal(0, 0.12, image.shape)

        elif condition == "Low contrast":
            image = 0.5 + 0.2 * (image - 0.5)

        elif condition == "Brightness down":
            image -= 0.2

        elif condition == "Downscaled":
            small = image[::4, ::4]
            image = np.repeat(
                np.repeat(small, 4, axis=0), 4, axis=1
            )

        image = np.clip(image, 0, 1)

        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(condition, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "Illustrative Synthetic Image Quality Examples",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
