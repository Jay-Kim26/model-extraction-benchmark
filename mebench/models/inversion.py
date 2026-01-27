"""Inversion model for InverseNet."""

from typing import Optional
import torch
import torch.nn as nn


class InversionGenerator(nn.Module):
    """Simple inversion generator mapping soft labels to images."""

    def __init__(
        self,
        num_classes: int,
        output_channels: int = 3,
        base_channels: int = 64,
        output_activation: str = "sigmoid",
        output_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.output_activation = output_activation
        self.output_size = output_size

        self.fc = nn.Linear(num_classes, 4 * 4 * base_channels * 8)
        self.main = nn.Sequential(
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, output_channels, 4, 2, 1, bias=False),
        )

        if output_activation == "tanh":
            self.out = nn.Tanh()
        else:
            self.out = nn.Sigmoid()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.fc(y)
        x = x.view(-1, self.base_channels * 8, 4, 4)
        x = self.main(x)
        x = self.out(x)
        if x.shape[-1] != self.output_size or x.shape[-2] != self.output_size:
            x = torch.nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return x
