import os  # noqa: I001
from pathlib import Path
from PIL import Image
import math


def concat_images_grid(input_folder, output_path, grid_shape, frame_width=4):
    # Get list of image files
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]
    image_files.sort()
    if not image_files:
        print("No images found in the folder.")
        return

    rows, cols = grid_shape
    total_needed = rows * cols

    # Open images and resize to the size of the first image
    images = [Image.open(f) for f in image_files[:total_needed]]
    w, h = images[0].size

    # If not enough images, add black images
    while len(images) < total_needed:
        black_img = Image.new("RGB", (w, h), color=(0, 0, 0))
        images.append(black_img)

    images = [img.resize((w, h)) for img in images]

    # Add black frame to each image
    framed_images = []
    fw = frame_width
    for img in images:
        framed = Image.new("RGB", (w + 2 * fw, h + 2 * fw), color=(0, 0, 0))
        framed.paste(img, (fw, fw))
        framed_images.append(framed)

    fw_w, fw_h = w + 2 * fw, h + 2 * fw

    # Create a blank canvas
    grid_img = Image.new("RGB", (cols * fw_w, rows * fw_h), color=(255, 255, 255))

    # Paste images into grid
    for idx, img in enumerate(framed_images):
        r = idx // cols
        c = idx % cols
        grid_img.paste(img, (c * fw_w, r * fw_h))

    # Remove outer frame
    grid_img = grid_img.crop(
        (
            frame_width,
            frame_width,
            grid_img.width - frame_width,
            grid_img.height - frame_width,
        )
    )

    # Resize image
    scale = 0.25
    if scale != 1:
        new_size = (
            math.ceil(grid_img.width * scale),
            math.ceil(grid_img.height * scale),
        )
        grid_img = grid_img.resize(new_size)

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_img.save(output_path)
    print(f"Saved grid image to {output_path}")


if __name__ == "__main__":
    rows = 6
    cols = 6
    concat_images_grid(
        input_folder=Path("figures/fig04-cameraCallib/calibPics/M3Tz"),
        output_path=Path("figures/fig04-cameraCallib/fig04-cameraCallib.png"),
        grid_shape=(rows, cols),
        frame_width=5,  # You can change the frame width here
    )
