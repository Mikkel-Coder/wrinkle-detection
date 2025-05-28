import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from PIL import Image
import pickle

class SAMDataset(Dataset):
    def __init__(self, dataset_path, processor, num_images=None):
        self.processor = processor
        self.dataset_path = Path(dataset_path)
        self.image_size = 256

        self.labels = self.dataset_path / "points.pkl"
        self.images_path = self.dataset_path / "images"
        self.masks_path = self.dataset_path / "masks"

        self.use_dataset_points = self.labels.is_file()

        self.images = defaultdict(dict)
        self.masks = defaultdict(dict)
        self.points = defaultdict(list)

        for image_path in self.images_path.iterdir():
            image_id = int(image_path.stem)
            self.images[image_id] = image_path

        for mask_path in self.masks_path.iterdir():
            mask_id = int(mask_path.stem)
            self.masks[mask_id] = mask_path
        
        if self.use_dataset_points:
            with self.labels.open("rb") as fp:
                self.points = pickle.load(fp)
        
        if self.use_dataset_points:
            assert len(self.images) == len(self.masks) == len(self.points)
        else:
            assert len(self.images) == len(self.masks)

        if num_images is not None:
            sorted_keys = sorted(self.images.keys())
            slice_keys = sorted_keys[:num_images]
            self.images = {key: self.images[key] for key in slice_keys}
            self.masks = {key: self.masks[key] for key in slice_keys}
            self.points = {key: self.points[key] for key in slice_keys}
        

    def _generate_grid_points(self, grid_size=10):
        x = np.linspace(0, self.image_size-1, grid_size)
        y = np.linspace(0, self.image_size-1, grid_size)
        xv, yv = np.meshgrid(x, y)
        points = np.stack([xv.flatten(), yv.flatten()], axis=1)
        return points / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(str(self.images[idx])).convert("RGB")
        mask = Image.open(str(self.masks[idx])).convert("L")
        if self.use_dataset_points:
            points = self.points[idx]
        else:
            points = self._generate_grid_points(grid_size=10).tolist()


        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        inputs = self.processor(image, input_points=[points], return_tensors="pt")
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        mask = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0
        mask = (mask > 0.5).float()
        inputs["ground_truth_mask"] = mask
                
        return inputs


