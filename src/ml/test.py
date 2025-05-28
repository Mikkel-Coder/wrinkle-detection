import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from transformers import SamModel, SamConfig, SamProcessor
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class SAMResults:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.test_dataset_path = Path("dataset/test")
        self.test_dataset_images_path = self.test_dataset_path / "images"
        self.test_dataset_masks_path = self.test_dataset_path / "masks"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SamModel(config=self.model_config)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.grid_size = 5  # 5x5 points are used as prompt
        self.image = None
        self.prob_mask = None
        self.pred_mask = None
        self.true_mask = None

    def compute_every_iou(self, save_path: Path):
        results = {}
        for texture_id_path in tqdm(sorted(self.test_dataset_images_path.iterdir())):
            texture_id = texture_id_path.name
            results[texture_id] = {}

            for image_path in tqdm(sorted(texture_id_path.iterdir()), leave=False):
                image_id = image_path.stem
                mask_path = (
                    self.test_dataset_masks_path / texture_id / f"{image_id}.png"
                )

                iou = self._prompt_model(image_path, mask_path)
                results[texture_id][image_id] = iou

        with save_path.open("w") as fp:
            json.dump(results, fp)

    def _prompt_model(self, image_path: Path, mask_path: Path):
        self.image = Image.open(image_path).convert("RGB")
        self.image = self.image.resize(
            (256, 256)
        )  # Same size the model has been trained on
        self.ground_truth_mask = Image.open(mask_path).convert("L")
        width, height = self.image.size  # 256

        x = (np.linspace(0, width - 1, self.grid_size),)
        y = np.linspace(0, height - 1, self.grid_size)
        xv, yv = np.meshgrid(x, y)
        x_flat = xv.ravel()
        y_flat = yv.ravel()
        input_points = [[float(x), float(y)] for x, y in zip(x_flat, y_flat)]

        # Process inputs
        inputs = self.processor(
            self.image,
            input_points=[input_points],
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.device)
        input_points = inputs["input_points"].to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_points=input_points,
                multimask_output=False,
            )

        # Get predicted mask
        self.prob_mask = (
            torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
        )
        self.pred_mask = (self.prob_mask > 0.5).astype(np.uint8)

        # Iou
        self.ground_truth_mask = self.ground_truth_mask.resize((256, 256))
        intersection = np.logical_and(self.pred_mask, self.ground_truth_mask)
        union = np.logical_or(self.pred_mask, self.ground_truth_mask)
        return np.sum(intersection) / np.sum(union)


def main():
    model_path = Path("src/ml/models/images_2000_epochs_200.pth")
    dataset_path = Path("dataset/test")
    output_json_path = Path("data/iou_results.json")
    sam_results = SAMResults(model_path)
    sam_results.compute_every_iou(output_json_path)


if __name__ == "__main__":
    main()
