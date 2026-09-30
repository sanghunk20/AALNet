"""ConvNeXt V2 building blocks used by the decoder.

- GRN (Global Response Normalization)
- LayerNorm2d (channel-first LayerNorm)
- ConvNeXtV2Block (inverted bottleneck with GRN)
- DropPath (stochastic depth)

References:
    - ConvNeXt V2: Woo et al., "ConvNeXt V2: Co-designing and Scaling
      ConvNets with Masked Autoencoders", CVPR 2023
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (for [B, C, H, W] tensors)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    """Global Response Normalization layer.

    Key innovation of ConvNeXt V2. Normalizes feature responses based on
    global spatial aggregation, providing implicit feature competition.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)    # [B, C, 1, 1]
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: depthwise conv -> LayerNorm -> pointwise -> GELU -> GRN -> pointwise.

    Uses inverted bottleneck design with 4x expansion ratio.
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.grn(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = residual + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor
