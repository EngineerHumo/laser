"""Encoder architectures for spot and global images."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        return self.block(x)


class SpotEncoder(nn.Module):
    """Encoder focusing on 64x64 spot crops."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            nn.MaxPool2d(2),
            ConvBlock(192, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.head(x)
        return x


class GlobalEncoder(nn.Module):
    """Encoder extracting contextual information from 128x128 images."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 192) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 96),
            nn.MaxPool2d(2),
            ConvBlock(96, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
