"""
cnn_model.py
============
CNN architecture for eye/face state classification.

Classes: OPEN (0), CLOSED (1), YAWNING (2)
Input: Grayscale 48x48 image

Architecture:
  Conv2D(1,32,3) + BN + ReLU + MaxPool(2)
  Conv2D(32,64,3) + BN + ReLU + MaxPool(2)
  Conv2D(64,128,3) + BN + ReLU + MaxPool(2)
  FC(128*feat, 256) + ReLU + Dropout(0.5)
  FC(256, 3)

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 48
NUM_CLASSES: int = 3
CLASS_NAMES: Tuple[str, ...] = ("OPEN", "CLOSED", "YAWNING")


class DrowsinessCNN(nn.Module):
    """CNN for eye/face state classification.

    Args:
        input_size: Height/width of input grayscale image.
        num_classes: Number of output classes.
    """

    def __init__(self, input_size: int = INPUT_SIZE, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        feat_dim = 128 * (input_size // 8) * (input_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, image: np.ndarray) -> Tuple[int, float, str]:
        """Run inference on a single image.

        Args:
            image: Grayscale image (H, W) or (H, W, 1) as uint8 numpy array.

        Returns:
            Tuple of (class_index, confidence, class_name).
        """
        import cv2
        self.eval()

        if len(image.shape) == 3:
            image = image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(image, (self.input_size, self.input_size))
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

        device = next(self.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)

        idx = int(class_idx.item())
        conf = float(confidence.item())
        return idx, conf, CLASS_NAMES[idx]

    def summary(self) -> str:
        """Return a string summary of the model architecture.

        Returns:
            Model summary string.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines = [
            f"DrowsinessCNN(input={self.input_size}x{self.input_size}, classes={self.num_classes})",
            f"  Total parameters: {total:,}",
            f"  Trainable parameters: {trainable:,}",
            "",
            str(self),
        ]
        return "\n".join(lines)


def load_model(
    path: str, device: Optional[torch.device] = None,
    input_size: int = INPUT_SIZE, num_classes: int = NUM_CLASSES,
) -> DrowsinessCNN:
    """Load a trained model from disk.

    Args:
        path: Path to saved state dict (.pth file).
        device: Target device (defaults to CPU).
        input_size: Model input size.
        num_classes: Number of classes.

    Returns:
        Loaded DrowsinessCNN instance.
    """
    device = device or torch.device("cpu")
    model = DrowsinessCNN(input_size=input_size, num_classes=num_classes)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s on %s", path, device)
    return model
