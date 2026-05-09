"""Model definition for the effect + intensity 2D CNN experiment."""

from __future__ import annotations

import torch
from torch import nn


class EffectIntensityCNN(nn.Module):
    """Shared 2D CNN encoder with separate effect and intensity heads."""

    def __init__(
        self,
        num_effect_classes: int,
        num_intensity_classes: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.20),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
        )
        self.effect_head = nn.Linear(128, num_effect_classes)
        self.intensity_head = nn.Linear(128, num_intensity_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs)
        pooled = self.pool(encoded)
        features = self.shared_head(pooled)
        return self.effect_head(features), self.intensity_head(features)
