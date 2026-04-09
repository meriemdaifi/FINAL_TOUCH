"""
dataset_loader.py
=================
Dataset pipeline for the DMS CNN training.

Loads images from folder structure:
  data/{train,val}/{open,closed,yawning}/

Applies data augmentation and normalization.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 48
NUM_CLASSES: int = 3
CLASS_NAMES: Tuple[str, ...] = ("open", "closed", "yawning")
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_WORKERS: int = 2


class EyeStateDataset(Dataset):
    """PyTorch Dataset for eye/face state images.

    Expects directory layout:
        root_dir/{open, closed, yawning}/*.{png,jpg,...}

    Args:
        root_dir: Path to split directory (e.g. data/train).
        transform: Torchvision transforms to apply.
    """

    def __init__(
        self, root_dir: str, transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning("Class directory not found: %s", class_dir)
                continue
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".bmp"):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name],
                    ))

        logger.info(
            "Dataset loaded: %d samples from %s", len(self.samples), root_dir,
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, class_index).
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


def get_train_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Build training data augmentation pipeline.

    Args:
        input_size: Target image size.

    Returns:
        Composed transform.
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Build validation transform pipeline (no augmentation).

    Args:
        input_size: Target image size.

    Returns:
        Composed transform.
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    input_size: int = INPUT_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Args:
        data_dir: Root data directory containing train/ and val/ subdirs.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        input_size: Image size for transforms.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_ds = EyeStateDataset(
        os.path.join(data_dir, "train"), transform=get_train_transforms(input_size),
    )
    val_ds = EyeStateDataset(
        os.path.join(data_dir, "val"), transform=get_val_transforms(input_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    logger.info(
        "DataLoaders created: train=%d batches, val=%d batches",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader
