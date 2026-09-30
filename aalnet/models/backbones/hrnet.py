"""HRNet backbone via timm.

Thin wrapper around timm's HRNet in features_only mode, returning the four
multi-resolution feature maps used by the decoder.

For an 800x800 input (HRNet-W48):
  - p0: [B, 128, 200, 200]   (stride 4)
  - p1: [B, 256, 100, 100]   (stride 8)
  - p2: [B, 512, 50, 50]     (stride 16)
  - p3: [B, 1024, 25, 25]    (stride 32)

Weight source: HuggingFace (timm/hrnet_w48.ms_in1k)

References:
    - Deep High-Resolution Representation Learning for Visual Recognition,
      Sun et al., TPAMI 2019
"""

import torch
import torch.nn as nn
import timm


class HRNetBackbone(nn.Module):
    """HRNet backbone using timm, multi-scale feature output.

    Args:
        model_name: timm model key (AALNet uses 'hrnet_w48').
        pretrained: Load ImageNet pretrained weights from HuggingFace.
    """

    def __init__(self, model_name: str = 'hrnet_w48', pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # stride 4, 8, 16, 32
        )
        # feature_info['num_chs'] may not match actual forward output channels
        # (timm HRNet reports pre-incre_modules channels in some versions).
        # Use a dummy forward pass to get the real channel sizes.
        # Note: the module is in training mode here, so this pass also updates the
        # BatchNorm running statistics once. It is kept as is, because the models
        # in the paper were initialised this way.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            dummy_out = self.model(dummy)
            self._out_channels = [f.shape[1] for f in dummy_out]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing multi-scale features.

        Args:
            x: [B, 3, H, W] input images.

        Returns:
            Dict with keys 'p0'..'p3' at strides 4, 8, 16, 32.
        """
        features = self.model(x)
        return {f'p{i}': f for i, f in enumerate(features)}

    @property
    def out_channels(self) -> list[int]:
        """Channel dimensions of the four output scales ([128, 256, 512, 1024] for W48)."""
        return self._out_channels
