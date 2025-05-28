import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def points_in_mask(mask_path: Path, diameter, sample_feq=5) -> list[tuple[int, int]]:
    mask_image = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)
    mask_np = np.array(mask_image)

    radius = diameter // 2
    height, width = mask_np.shape

    ys, xs = np.where(mask_np == 255)
    points = list(zip(xs, ys))
    fitting_points = []

    yy, xx = np.ogrid[:diameter, :diameter]
    circle_mask = (xx - radius) ** 2 + (yy - radius) ** 2 <= radius**2

    mask_copy = mask_np.copy()

    for x, y in points:
        x_start = x - radius
        y_start = y - radius
        x_end = x + radius
        y_end = y + radius

        if x_start < 0 or y_start < 0 or x_end >= width or y_end >= height:
            continue

        region = mask_copy[y_start : y_end + 1, x_start : x_end + 1]
        region_circle = region[circle_mask]

        if np.all(region_circle == 255):
            fitting_points.append((x, y))
            region[circle_mask] = 0

            # plt.imshow(mask_copy, cmap='gray')
            # plt.show()

    return fitting_points[::sample_feq]


def points_in_mask_wrapper(mask_path):
    image_id = int(mask_path.stem)
    points = points_in_mask(mask_path, diameter=5, sample_feq=5)
    points_py = [[int(coord) for coord in p] for p in points]
    return image_id, points_py


def generate_points():
    masks_folder = Path("dataset/train/masks")
    save_path = Path("dataset/train/points.pkl")

    mask_paths = [p for p in masks_folder.iterdir()]
    all_points = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(points_in_mask_wrapper, path) for path in mask_paths]
        for future in tqdm(futures, total=len(futures)):
            image_id, points = future.result()
            all_points[image_id] = [tuple(p) for p in points]

    with open(save_path, "wb") as f:
        pickle.dump(all_points, f)

if __name__ == "__main__":
    mask_path = Path("dataset/test/masks/1/2.png")

    points = points_in_mask(mask_path, diameter=5, sample_feq=5)
    mask_image = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)
    plt.imshow(mask_image, cmap="gray")
    xs, ys = zip(*points)
    plt.axis("off")
    plt.scatter(xs, ys, c="red", s=55)
    plt.savefig("report/figures/promt_points.png", bbox_inches="tight", pad_inches=0)
    plt.close()