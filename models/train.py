"""
train.py
========
Training script for the DMS CNN model.

Handles training loop with cross-entropy loss, Adam optimizer,
learning rate scheduling, early stopping, and model saving.

Usage:
    python -m models.train --data_path data/ --epochs 50 --batch_size 32

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import DrowsinessCNN, INPUT_SIZE, NUM_CLASSES
from models.dataset_loader import create_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 1e-3
DEFAULT_PATIENCE: int = 7
SAVE_DIR: str = os.path.join(os.path.dirname(__file__), "saved_model")


def train_one_epoch(
    model: DrowsinessCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: CNN model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def validate(
    model: DrowsinessCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model.

    Args:
        model: CNN model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def plot_history(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot and save training history.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Directory to save plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_path, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Training plots saved to %s", plot_path)


def train(
    data_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
) -> DrowsinessCNN:
    """Full training pipeline.

    Args:
        data_path: Root data directory.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Initial learning rate.
        patience: Early stopping patience.

    Returns:
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    train_loader, val_loader = create_dataloaders(data_path, batch_size)
    model = DrowsinessCNN(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
    logger.info("Model:\n%s", model.summary())

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info("Best model saved (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    plot_history(history, SAVE_DIR)

    final_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    return model


def main() -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description="Train DMS CNN model")
    parser.add_argument("--data_path", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
