from pathlib import Path
from tqdm import tqdm
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class GaborResults:
    def __init__(self):
        self.test_dataset_path = Path("dataset/test")
        self.test_dataset_images_path = self.test_dataset_path / "images"
        self.test_dataset_masks_path = self.test_dataset_path / "masks"
        self.block_size = 10

        k_size = 31  # kernel size (odd) [3,5,7,..., 100]
        gamma = 0.1  # Aspect ration
        lambd = 3 / np.pi  # wavelength
        sigma = 5  # Spread/width
        psi = 0  # Phase offset
        rotations = 8

        self.args = (k_size, sigma, lambd, gamma, psi, rotations)

    def binary_mask(self, heatmap):
        nonzero_vals = heatmap[heatmap != 0]
        mean = np.mean(nonzero_vals)
        std = np.std(nonzero_vals)
        threshold_value = 2_000

        winkles_chekc = heatmap > threshold_value
        if not np.any(winkles_chekc):
            return np.zeros_like(heatmap)

        winkels = np.zeros_like(heatmap, dtype=np.uint8)
        strong_deviation = np.abs(heatmap - mean) > 2 * std
        winkels[np.logical_and(strong_deviation, heatmap != 0)] = 1
        return winkels

    def binary_cloth_mask(self, rgb_image, r_border_pixels: int = 120):
        height, width = rgb_image.shape[:2]
        roi_image = np.zeros((height, width), dtype=np.uint8)
        # All possible background colors
        lower_rgb = np.array([178, 178, 178])
        upper_rgb = np.array([230, 230, 230])

        background = cv.inRange(rgb_image, lower_rgb, upper_rgb)
        foreground = np.bitwise_not(background)

        contours, _ = cv.findContours(
            foreground,
            cv.RETR_EXTERNAL,  # Retries only extreme outer contours
            cv.CHAIN_APPROX_SIMPLE,  # Only end points compressed
        )
        # Convert the contours into a 1d list with points
        points = np.vstack(contours).squeeze()

        # Now wrap the the cloth (points)
        hull = cv.convexHull(points)
        cv.drawContours(roi_image, [hull], -1, 255, thickness=cv.FILLED)

        # Shrink the roi cloth by 20 pixels about its border
        # (Morph Erosion)
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (r_border_pixels, r_border_pixels)
        )
        cloth_mask = cv.erode(roi_image, kernel, iterations=1)

        return cloth_mask

    def find_roi_bounding_box(self, cloth_mask):
        contours, _ = cv.findContours(
            cloth_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        # There is only one BLOB, the cloth itself
        contour = contours[0]
        x, y, w, h = cv.boundingRect(contour)

        return x, y, w, h

    def _calculate_winkle_block(self, block):
        k_size, sigma, lambd, gamma, psi, rotations = self.args
        magnitudes = []
        for rotation in range(rotations):
            theta = rotation * np.pi / rotations
            kernel = cv.getGaborKernel(
                (k_size, k_size),
                sigma,
                theta,
                lambd,
                gamma,
                psi,
                ktype=cv.CV_32F,
            )
            filtered_img = cv.filter2D(block, cv.CV_32F, kernel)
            magnitude = np.sum(np.abs(filtered_img))
            magnitudes.append(magnitude)
        return max(magnitudes)

    def _calculate_winkle_block_bgr(self, block):
        magnitudes = []
        for channel in range(3):  # BGR
            channel = block[:, :, channel]
            mag = self._calculate_winkle_block(channel)
            magnitudes.append(mag)
        return np.average(magnitudes)

    def _calculate_winkle(self, image, cloth_mask):
        blocks_x = image.shape[1] // self.block_size
        blocks_y = image.shape[0] // self.block_size

        winkle_magnitudes = np.zeros((blocks_y, blocks_x))

        is_rgb = True if image.ndim == 3 else False
        winkle_func = (
            self._calculate_winkle_block_bgr if is_rgb else self._calculate_winkle_block
        )

        for y in range(blocks_y):
            for x in range(blocks_x):
                y_start, y_end = y * self.block_size, (y + 1) * self.block_size
                x_start, x_end = x * self.block_size, (x + 1) * self.block_size
                mask_block = cloth_mask[y_start:y_end, x_start:x_end]

                # Skip empty blocks
                if not np.any(mask_block):
                    continue

                block = image[y_start:y_end, x_start:x_end]
                winkle_magnitudes[y, x] = winkle_func(block)

        return winkle_magnitudes

    def calculate_iou(self, mask, truth):
        intersection = np.logical_and(mask, truth)
        union = np.logical_or(mask, truth)
        return np.sum(intersection) / np.sum(union)

    def _compure_winkle(self, image_path, mask_path):
        bgr_img = cv.imread(image_path)
        ground_truth = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        ground_truth = (ground_truth > 0).astype(np.uint8)  # 255 -> 1
        assert bgr_img is not None
        assert ground_truth is not None

        cloth_mask = self.binary_cloth_mask(bgr_img)
        x, y, w, h = self.find_roi_bounding_box(cloth_mask)

        bgr_img = bgr_img[y : y + h, x : x + w]
        cloth_mask = cloth_mask[y : y + h, x : x + w]
        ground_truth = ground_truth[y : y + h, x : x + w]
        self.cloth_mask = cloth_mask

        winkles = self._calculate_winkle(bgr_img, cloth_mask)
        self.magnitude_mask = winkles

        self.winkles_mask = self.binary_mask(winkles)

        ground_truth = cv.resize(
            ground_truth,
            (winkles.shape[1], winkles.shape[0]),
            interpolation=cv.INTER_NEAREST,
        )
        return self.calculate_iou(self.winkles_mask, ground_truth)

    def compute_winkles(self, save_path: Path):
        results = {}
        for texture_id_path in tqdm(sorted(self.test_dataset_images_path.iterdir())):
            texture_id = texture_id_path.name
            results[texture_id] = {}

            for image_path in tqdm(sorted(texture_id_path.iterdir()), leave=False):
                image_id = image_path.stem
                mask_path = (
                    self.test_dataset_masks_path / texture_id / f"{image_id}.png"
                )

                results[texture_id][image_id] = self._compure_winkle(
                    image_path, mask_path
                )

        with save_path.open("w") as fp:
            json.dump(results, fp)


def main():
    save_path = Path("data/gabor_iou.json")
    gabor_results = GaborResults()    
    gabor_results.compute_winkles(save_path)


if __name__ == "__main__":
    main()
