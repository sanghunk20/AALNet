"""Heatmap prediction head."""

import torch
import torch.nn as nn

from ...modules.heatmap_utils import heatmap_to_coords_soft_argmax


class HeatmapHead(nn.Module):
    """Convolutional head that produces one heatmap per landmark.

    Args:
        in_channels: Input feature channels from the decoder.
        num_landmarks: Number of landmarks to predict.
    """

    def __init__(self, in_channels: int = 64, num_landmarks: int = 33):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_landmarks, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] decoder features -> [B, num_landmarks, H, W] heatmaps."""
        return self.head(x)

    def get_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """[B, N, H, W] heatmaps -> [B, N, 2] coordinates (x, y) by soft-argmax."""
        return heatmap_to_coords_soft_argmax(heatmaps)
