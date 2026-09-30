"""AALNet: HRNet-W48 backbone + FPN-style fusion decoder + heatmap head."""

import torch
import torch.nn as nn

from .backbones.hrnet import HRNetBackbone
from .decoder.fpn_decoder import FPNDecoder
from .heatmap.heatmap_layers import HeatmapLayers


class AALNet(nn.Module):
    """Heatmap-regression landmark detector.

    Args:
        num_landmarks: Number of landmarks (33).
        pretrained: Initialise the backbone with ImageNet weights (timm).
        decoder_drop_path: Stochastic-depth rate of the decoder blocks.
    """

    def __init__(
        self,
        num_landmarks: int = 33,
        pretrained: bool = False,
        decoder_drop_path: float = 0.275,
    ):
        super().__init__()
        self.backbone = HRNetBackbone(model_name='hrnet_w48', pretrained=pretrained)
        self.neck = FPNDecoder(
            in_channels=self.backbone.out_channels,
            out_channels=256,
            decoder_drop_path=decoder_drop_path,
            residual_drop_path=0.2,
        )
        self.head = HeatmapLayers(
            in_channels=self.neck.out_channels,
            num_landmarks=num_landmarks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] images -> [B, num_landmarks, H, W] heatmaps."""
        return self.head(self.neck(self.backbone(x)))

    def get_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """[B, N, H, W] heatmaps -> [B, N, 2] pixel coordinates (x, y) in input space."""
        return self.head.get_coordinates(heatmaps)
