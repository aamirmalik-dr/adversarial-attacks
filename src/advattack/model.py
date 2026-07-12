"""A small convolutional victim classifier for 1x28x28 images."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Two convolutional blocks and a linear head, for MNIST-sized inputs.

    Kept deliberately small so it trains in a minute on a CPU and so the attack
    experiments run quickly.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
