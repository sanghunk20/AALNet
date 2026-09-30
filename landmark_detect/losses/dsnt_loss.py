"""DSNT loss for heatmap-based landmark detection.

References:
    - Nibali et al., "Numerical Coordinate Regression with Convolutional Neural Networks", 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.heatmap_utils import heatmap_to_coords_soft_argmax, generate_gaussian_heatmap_at_scale


class DSNTLoss(nn.Module):
    """Base loss: soft-argmax coordinate L2 + JS divergence regularization.

    pred: [B, N, H, W] heatmap logits
    target: [B, N, 2] GT coords in heatmap space (pixels)

    Coordinates are normalized to [-1, 1] before computing L2 (Nibali et al., 2018),
    so that the coordinate term and the JS divergence operate at comparable scales.

    Loss = L2(pred_coords, target) + lambda_js * JS(pred_prob || target_gaussian)

    Reference:
        Nibali et al., "Numerical Coordinate Regression with Convolutional Neural Networks",
        arXiv 2018.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        lambda_js: float = 1.0,
        sigma: float = 5.0,
    ):
        """
        Args:
            temperature: Temperature for spatial softmax (lower = sharper).
            lambda_js: Weight for JS divergence regularization term.
            sigma: Gaussian sigma for target heatmap in heatmap-space pixels.
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_js = lambda_js
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N, H, W] heatmap logits.
            target: [B, N, 2] GT coordinates in heatmap space.

        Returns:
            Scalar loss.
        """
        B, N, H, W = pred.shape

        # 1. soft-argmax → pred_coords [B, N, 2]
        pred_coords = heatmap_to_coords_soft_argmax(pred, temperature=self.temperature)

        # 2. Coord loss on coordinates normalized to [-1, 1] (Nibali et al., 2018)
        half_size = H / 2.0
        pred_norm = pred_coords / half_size - 1.0
        target_norm = target.float() / half_size - 1.0
        coord_loss = F.mse_loss(pred_norm, target_norm)

        # 3. pred_prob = softmax over spatial dimension
        pred_prob = F.softmax(
            pred.view(B, N, -1) / self.temperature, dim=-1
        ).view(B, N, H, W)

        # 4. Build target Gaussian heatmaps (vectorized batch)
        # target is already in heatmap space → pass input_size=H so scale=1
        target_gauss = generate_gaussian_heatmap_at_scale(
            target.float(),   # [B, N, 2] in heatmap space
            output_size=H,
            sigma=self.sigma,
            input_size=H,
        )  # [B, N, H, W]

        # Normalize target_gauss to probability distribution
        target_sum = target_gauss.view(B, N, -1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
        target_prob = target_gauss / target_sum  # [B, N, H, W]

        # 5. JS divergence: M=(pred+target)/2; js = 0.5*KL(p||M)+0.5*KL(t||M)
        eps = 1e-8
        M = 0.5 * (pred_prob + target_prob)

        # KL(p||M) = sum(p * log(p / (M + eps)))
        kl_pm = (pred_prob * torch.log((pred_prob + eps) / (M + eps))).sum(dim=(-2, -1))
        # KL(t||M) = sum(t * log(t / (M + eps)))
        kl_tm = (target_prob * torch.log((target_prob + eps) / (M + eps))).sum(dim=(-2, -1))

        js = 0.5 * kl_pm + 0.5 * kl_tm  # [B, N]
        js_loss = js.mean()

        # 6. Total loss
        return coord_loss + self.lambda_js * js_loss
