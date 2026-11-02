from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_DIR / "demo_images"


def create_demo_images():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for i in range(8):
        image_data = np.zeros((128, 128), dtype=np.uint8)
        image_data[30:90, 30:90] = 100 + (i * 15)

        image = Image.fromarray(image_data)
        image.save(OUTPUT_FOLDER / f"image_{i+1}.png")

    print("8 demo images created successfully!")
    print("Saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    create_demo_images()
