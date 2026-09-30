"""FPN-style fusion decoder.

Based on Wyatt 2024 'Optimising for the Unknown' (MICCAI 2024 CL-Detection Challenge).
Top-down FPN with ConvNeXt V2 blocks and residual channel projection, followed by
learned upsampling of the feature pyramid to full input resolution (Wyatt 2024 §2.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.convnextv2_blocks import ConvNeXtV2Block, LayerNorm2d


class ResidualConvBlock(nn.Module):
    """Residual conv block for channel projection in FPN decoder.

    Conv(3x3) + BN + ReLU + Conv(3x3) + BN + skip.
    Uses 2D dropout for regularization (Wyatt 2024).
    """

    def __init__(self, channels: int, drop_path: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=drop_path) if drop_path > 0.0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.act(out + residual)


class MLPFPNNeck(nn.Module):
    """Top-down FPN decoder with ConvNeXt V2 blocks (Wyatt 2024).

    Supersamples the feature pyramid to full input resolution following
    Wyatt 2024: "we utilise an MLP feature pyramid with subsequent learned
    upsampling layers to supersample the feature pyramid to its original
    resolution."

    Architecture (for an 800x800 input, backbone strides 4/8/16/32):
        === FPN stages (stride 32 -> stride 4) ===
        p3 [B, C3, 25, 25]
          -> lateral + residual_block + 3x ConvNeXtV2Block
          -> Upsample 2x + lateral_p2 -> 2x ConvNeXtV2Block
          -> Upsample 2x + lateral_p1 -> 2x ConvNeXtV2Block
          -> Upsample 2x + lateral_p0 -> [B, 256, 200, 200]  (stride 4)

        === Super-resolution stages (stride 4 -> stride 1) ===
          -> Upsample 2x + Conv(256->128) + 2x ConvNeXtV2Block  (stride 2)
          -> Upsample 2x + Conv(128->64)                        (stride 1)
        -> output: [B, 64, H, W]  (full input resolution)

    Args:
        in_channels: [C0, C1, C2, C3] backbone output channels at strides 4/8/16/32.
        out_channels: FPN internal channels (default 256).
        decoder_drop_path: Drop path rate for ConvNeXt V2 blocks in decoder.
        residual_drop_path: Dropout rate for residual conv block.
    """

    # Final output channels after super-resolution (fpn_channels // 4)
    SR_CHANNEL_RATIO = 4

    def __init__(
        self,
        in_channels: list,
        out_channels: int = 256,
        decoder_drop_path: float = 0.275,
        residual_drop_path: float = 0.2,
    ):
        super().__init__()
        C0, C1, C2, C3 = in_channels
        fpn_ch = out_channels  # internal FPN channels (256)
        sr_mid = fpn_ch // 2   # 128
        sr_out = fpn_ch // self.SR_CHANNEL_RATIO  # 64

        # Lateral projections (1x1 conv to fpn_ch)
        self.lateral_p3 = nn.Conv2d(C3, fpn_ch, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(C2, fpn_ch, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(C1, fpn_ch, kernel_size=1)
        self.lateral_p0 = nn.Conv2d(C0, fpn_ch, kernel_size=1)

        # Layer norms for lateral outputs (before addition)
        self.norm_p3 = LayerNorm2d(fpn_ch)
        self.norm_p2 = LayerNorm2d(fpn_ch)
        self.norm_p1 = LayerNorm2d(fpn_ch)
        self.norm_p0 = LayerNorm2d(fpn_ch)

        # Residual block for initial p3 processing
        self.residual = ResidualConvBlock(fpn_ch, drop_path=residual_drop_path)

        # p3 stage: 3x ConvNeXtV2Block
        self.p3_blocks = nn.Sequential(
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
        )

        # p2 stage: 2x ConvNeXtV2Block
        self.p2_blocks = nn.Sequential(
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
        )

        # p1 stage: 2x ConvNeXtV2Block
        self.p1_blocks = nn.Sequential(
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
            ConvNeXtV2Block(fpn_ch, drop_path=decoder_drop_path),
        )

        # FPN output norm (stride 4)
        self.fpn_norm = LayerNorm2d(fpn_ch)

        # === Super-resolution stages: stride 4 → stride 1 ===
        # Stage 1: stride 4 → stride 2 (256 → 128 channels)
        self.sr_conv1 = nn.Conv2d(fpn_ch, sr_mid, kernel_size=3, padding=1, bias=False)
        self.sr_norm1 = LayerNorm2d(sr_mid)
        self.sr_blocks1 = nn.Sequential(
            ConvNeXtV2Block(sr_mid, drop_path=decoder_drop_path),
            ConvNeXtV2Block(sr_mid, drop_path=decoder_drop_path),
        )

        # Stage 2: stride 2 → stride 1 (128 → 64 channels)
        self.sr_conv2 = nn.Conv2d(sr_mid, sr_out, kernel_size=3, padding=1, bias=False)
        self.sr_norm2 = LayerNorm2d(sr_out)

        self._out_channels = sr_out

    @property
    def out_channels(self):
        """Final output channels after super-resolution."""
        return self._out_channels

    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: {'p0': [B,C0,H/4,W/4], 'p1': [B,C1,H/8,W/8],
                       'p2': [B,C2,H/16,W/16], 'p3': [B,C3,H/32,W/32]}

        Returns:
            [B, 64, H, W]  (full input resolution)
        """
        p0 = features['p0']
        p1 = features['p1']
        p2 = features['p2']
        p3 = features['p3']

        # === FPN stages (stride 32 → stride 4) ===

        # p3: lateral + residual + 3x ConvNeXt blocks
        x = self.norm_p3(self.lateral_p3(p3))
        x = self.residual(x)
        x = self.p3_blocks(x)

        # Upsample + add p2 lateral
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.norm_p2(self.lateral_p2(p2))
        x = self.p2_blocks(x)

        # Upsample + add p1 lateral
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.norm_p1(self.lateral_p1(p1))
        x = self.p1_blocks(x)

        # Upsample + add p0 lateral
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.norm_p0(self.lateral_p0(p0))
        x = self.fpn_norm(x)  # [B, 256, H/4, W/4]

        # === Super-resolution stages (stride 4 → stride 1) ===

        # Stride 4 → stride 2: upsample + channel reduction (256 → 128)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.sr_norm1(self.sr_conv1(x))
        x = self.sr_blocks1(x)  # [B, 128, H/2, W/2]

        # Stride 2 → stride 1: upsample + channel reduction (128 → 64)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.sr_norm2(self.sr_conv2(x))  # [B, 64, H, W]

        return x
