
import numpy as np
import pandas as pd


def generate_demo_data(seed=42):
    rng = np.random.default_rng(seed)

    corruptions = [
        "blur",
        "noise",
        "contrast",
        "brightness_up",
        "brightness_down",
        "jpeg",
        "downscale",
    ]

    severity_levels = [0, 1, 2, 3, 4, 5]
    n_images = 200

    labels = rng.integers(0, 2, size=n_images)
    rows = []

    for image_index in range(n_images):
        label = int(labels[image_index])
        base_probability = 0.85 if label == 1 else 0.15

        for corruption in corruptions:
            for severity in severity_levels:
                degradation = severity * 0.045

                probability = base_probability + rng.normal(0, 0.04) - degradation
                probability = float(np.clip(probability, 0.01, 0.99))

                rows.append({
                    "image_id": f"image_{image_index:04d}",
                    "label": label,
                    "corruption": corruption,
                    "severity": severity,
                    "prob_pneumonia": probability,
                    "quality": max(0.0, 1.0 - severity * 0.12),
                    "q_brightness": rng.uniform(0.5, 1.0),
                    "q_contrast": rng.uniform(0.5, 1.0),
                    "q_sharpness": rng.uniform(0.5, 1.0),
                    "q_noise": rng.uniform(0.5, 1.0),
                    "q_clipping": rng.uniform(0.5, 1.0),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = generate_demo_data()
    print("Synthetic rows:", len(data))
    print(data.head())
