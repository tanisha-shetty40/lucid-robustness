import numpy as np
import pandas as pd

def make_demo_sweep(n_samples=600, seed=42):
    """Generates a synthetic sweep dataframe where low quality correlates with wrong predictions."""
    rng = np.random.default_rng(seed)
    
    image_ids = np.repeat(np.arange(n_samples // 6), 6)
    labels = rng.integers(0, 2, size=len(image_ids))
    
    corruptions = ["blur", "noise", "contrast", "brightness_up", "brightness_down", "jpeg"]
    severities = np.tile(np.arange(6), len(image_ids) // 6)
    
    rows = []
    for i in range(len(image_ids)):
        sev = severities[i]
        # Quality decreases as severity increases
        quality = float(np.clip(100 - (sev * 15) + rng.normal(0, 5), 5, 100))
        
        # Error probability increases as quality drops
        err_prob = 1.0 / (1.0 + np.exp((quality - 50) / 10.0))
        is_wrong = rng.random() < err_prob
        
        if is_wrong:
            prob_pneumonia = 0.2 if labels[i] == 1 else 0.8
        else:
            prob_pneumonia = 0.8 if labels[i] == 1 else 0.2
            
        row = {
            "image_id": int(image_ids[i]),
            "label": int(labels[i]),
            "corruption": corruptions[i % len(corruptions)],
            "severity": int(sev),
            "prob_pneumonia": float(np.clip(prob_pneumonia + rng.normal(0, 0.05), 0, 1)),
            "quality": quality,
            "q_brightness": float(quality / 100.0),
            "q_contrast": float(quality / 100.0),
            "q_sharpness": float(quality / 100.0),
            "q_noise": float(quality / 100.0),
            "q_clipping": 1.0
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = make_demo_sweep()
    df.to_csv("results/robustness/hithesh/demo_sweep.csv", index=False)
    print("Demo sweep generated successfully with", len(df), "rows.")