from pathlib import Path
import json
import matplotlib.pyplot as plt
from src.classic.main import GaborResults
import cv2 as cv
import numpy as np

plt.rcParams["font.family"] = "DejaVu Serif"


def sample_figure(gabor: GaborResults, texture_id, image_id):
    image_path = gabor.test_dataset_images_path / texture_id / f"{image_id}.png"
    mask_path = gabor.test_dataset_masks_path / texture_id / f"{image_id}.png"

    gabor._compure_winkle(image_path, mask_path)

    fig, ax = plt.subplots()
    im = ax.imshow(gabor.magnitude_mask)
    ax.axis("off")
    fig.savefig(
        f"report/figures/magnitude_gabor_{texture_id}_{image_id}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    del ax

    fig, ax = plt.subplots()
    im = ax.imshow(gabor.winkles_mask, cmap="gray")
    ax.axis("off")
    fig.savefig(
        f"report/figures/mask_gabor_{texture_id}_{image_id}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    del ax


def calculate_average(data_iou):
    averages = {}
    for texture_id, image_iou_dict in data_iou.items():
        ious = list(image_iou_dict.values())
        print(ious)
        mean_iou = sum(ious) / len(ious) if ious else 0
        averages[texture_id] = mean_iou
    return averages

def main():
    gabor = GaborResults()
    k_size = 31
    gamma = 0.1  # Aspect ration
    lambd = 3 / np.pi  # wavelength
    sigma = 5  # Spread/width
    psi = 0  # Phase offset
    theta = 0
    kernel = cv.getGaborKernel(
                (k_size, k_size),
                sigma,
                theta,
                lambd,
                gamma,
                psi,
                ktype=cv.CV_32F,
            )
    plt.imsave("report/figures/gabor_kernel.png", kernel)

    data_iou_path = Path("data/gabor_iou.json")
    with data_iou_path.open("r") as fp:
        data_iou = json.load(fp)

    sample_figure(gabor, texture_id="1", image_id="2")
    print(calculate_average(data_iou))


if __name__ == "__main__":
    main()
