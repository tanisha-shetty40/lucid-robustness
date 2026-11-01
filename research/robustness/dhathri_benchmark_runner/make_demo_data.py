
from pathlib import Path
import numpy as np
from PIL import Image

def create_demo_images():
    output_folder = Path("demo_images")
    output_folder.mkdir(exist_ok=True)

    for i in range(8):
        image_data = np.zeros((128, 128), dtype=np.uint8)

        image_data[30:90, 30:90] = 100 + (i * 15)

        image = Image.fromarray(image_data)
        image.save(output_folder / f"image_{i+1}.png")

    print("8 demo images created successfully!")

if __name__ == "__main__":
    create_demo_images()

