from __future__ import annotations

import torch
from torch import nn


class SequenceGRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.input_norm = nn.BatchNorm1d(input_channels)
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(inputs)
        sequence = normalized.transpose(1, 2)
        _, hidden = self.gru(sequence)
        final_hidden = hidden[-1]
        return self.classifier(final_hidden)

