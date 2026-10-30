
from pathlib import Path

import cv2
import numpy as np

from research.robustness.brahmi_corruptions.corruptions import CORRUPTIONS
from research.robustness.brahmi_corruptions.make_demo_data import make_demo_image


def create_gallery(img, out_path):
    """Save a gallery of all corruptions at severities 0 to 5."""
    tile_size = 160
    label_height = 28
    margin = 10

    names = list(CORRUPTIONS.keys())
    columns = 6
    rows = len(names)

    gallery = np.full(
        (
            rows * (tile_size + label_height + margin) + margin,
            columns * (tile_size + margin) + margin,
        ),
        255,
        dtype=np.uint8,
    )

    for col, severity in enumerate(range(6)):
        for row, name in enumerate(names):
            result = CORRUPTIONS[name](img, severity, seed=42)
            tile = cv2.resize(
                result, (tile_size, tile_size),
                interpolation=cv2.INTER_AREA
            )

            x = margin + col * (tile_size + margin)
            y = margin + row * (tile_size + label_height + margin)

            gallery[y:y + tile_size, x:x + tile_size] = tile

            label = f"{name} S{severity}"
            cv2.putText(
                gallery,
                label,
                (x, y + tile_size + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                0,
                1,
                cv2.LINE_AA,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(out_path), gallery):
        raise RuntimeError(f"Could not save gallery to {out_path}")


def main():
    img = make_demo_image()
    output = "results/robustness/brahmi/gallery_synthetic.png"
    create_gallery(img, output)
    print(f"Synthetic gallery saved to: {output}")


if __name__ == "__main__":
    main()

