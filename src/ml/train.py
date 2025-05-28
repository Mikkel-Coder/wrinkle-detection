import re
from pathlib import Path
from statistics import mean
from transformers import SamModel, SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch
import monai
from src.ml.dataset_model import SAMDataset

# https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb


class TrainML:
    def __init__(self, num_epochs: int = 20, num_images: int = None):
        self.num_epochs = num_epochs
        self.num_images = num_images


        # Path to dataset to train on
        self.dataset_path = Path("dataset")
        self.dataset_train_path = self.dataset_path / "train"

        # Initialize the processor
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # Create an instance of the SAMDataset
        self.training_dataset = SAMDataset(
            self.dataset_train_path, self.processor, num_images=num_images
        )

        # Create a DataLoader used to fetch samples from the dataset
        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=1, shuffle=True
        )

        # Load the model
        self.training_model = SamModel.from_pretrained("facebook/sam-vit-base")

        # Make sure that we ONLY train the mask decoder
        for name, param in self.training_model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        # Initialize the optimizer and the loss function
        self.optimizer = Adam(self.training_model.mask_decoder.parameters(), lr=1e-5)

        # Our loss function used in the training process.
        # Different loss functions to try: DiceFocalLoss, FocalLoss, DiceCELoss
        self.seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models_path = Path("src/ml/models/")
        self.snapshots_path = self.models_path / "snapshots"
        self.epoch_mean_losses = []
        self.start_epoch = 0

        # Check if we can continue from a snapshot
        pattern = f"images_{self.num_images}_epoch_*.pth"
        snapshot_paths = list(self.snapshots_path.glob(pattern=pattern))
        sorted_snapshot_paths = sorted(
            snapshot_paths,
            key=lambda p: int(re.search(r"epoch_(\d+)$", p.stem).group(1)),
        )

        if sorted_snapshot_paths:
            snapshot_path = sorted_snapshot_paths[-1]
            print(f"Snapshot found! Continuing train with {snapshot_path}")
            self._load_snapshot(snapshot_path)

    def _load_snapshot(self, snapshot_path: Path):
        # Model only train on a device like CUDA can only continue on that device
        self.training_model.to(self.device)

        checkpoint = torch.load(snapshot_path, map_location=self.device)

        # Restore model and optimizer states
        self.training_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore training state
        self.start_epoch = checkpoint["epoch"] + 1
        self.epoch_mean_losses = checkpoint.get("epoch_mean_losses", [])

    def _snapshot(self, epoch, mean_loss, number_of_images):
        snapshot_path = (
            self.snapshots_path / f"images_{number_of_images}_epoch_{epoch}.pth"
        )

        snapshot_data = {
            "epoch": epoch,
            "model_state_dict": self.training_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "mean_loss": mean_loss,
            "epoch_mean_losses": self.epoch_mean_losses,
            "number_of_images": number_of_images,
        }

        torch.save(snapshot_data, snapshot_path)

    def train(self):
        self.training_model.to(self.device)
        self.training_model.train()

        self.epochs = []

        for epoch in tqdm(
            range(self.start_epoch, self.num_epochs),
            desc="Epochs",
            initial=self.start_epoch,
            total=self.num_epochs,
        ):
            epoch_losses = []

            for batch in tqdm(
                self.training_dataloader, desc=f"Epoch {epoch}", leave=False
            ):
                outputs = self.training_model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    input_points=batch["input_points"].to(self.device),
                    multimask_output=False,
                )

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
                loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            self.epochs.append(epoch)
            mean_loss = mean(epoch_losses)
            self.epoch_mean_losses.append(mean_loss)

            print(f"EPOCH: {epoch}")
            print(f"Mean loss: {mean_loss:.4f}")

            # Save snapshot
            self._snapshot(epoch, mean_loss, self.num_images)

        # Save final model
        final_model_path = (
            self.models_path / f"images_{self.num_images}_epochs_{self.num_epochs}.pth"
        )
        torch.save(self.training_model.state_dict(), final_model_path)


if __name__ == "__main__":
    train_ml = TrainML(num_epochs=200, num_images=2000)
    train_ml.train()
