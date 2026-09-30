"""Heatmap utilities: soft-argmax coordinate extraction and Gaussian target maps."""

import torch
import torch.nn.functional as F


def heatmap_to_coords_soft_argmax(
    heatmap: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Extract coordinates from heatmaps using differentiable soft-argmax.

    Args:
        heatmap: [B, N, H, W] predicted heatmaps.
        temperature: Temperature for spatial softmax (lower = sharper).

    Returns:
        coords: [B, N, 2] coordinates (x, y) in heatmap space.
    """
    B, N, H, W = heatmap.shape

    # Spatial softmax
    flat = heatmap.view(B, N, -1) / temperature
    weights = F.softmax(flat, dim=-1).view(B, N, H, W)

    # Coordinate grids
    device = heatmap.device
    y_grid = torch.arange(H, dtype=torch.float32, device=device)
    x_grid = torch.arange(W, dtype=torch.float32, device=device)

    # Weighted sum
    x_coords = (weights.sum(dim=2) * x_grid.view(1, 1, -1)).sum(dim=-1)  # [B, N]
    y_coords = (weights.sum(dim=3) * y_grid.view(1, 1, -1)).sum(dim=-1)  # [B, N]

    return torch.stack([x_coords, y_coords], dim=-1)


def generate_gaussian_heatmap_at_scale(
    coords: torch.Tensor,
    output_size: int,
    sigma: float,
    input_size: int = 800,
) -> torch.Tensor:
    """Generate Gaussian heatmaps for a batch at a given output scale.

    Args:
        coords: [B, N, 2] landmark coordinates in input_size pixel space (x, y).
        output_size: Output heatmap spatial size.
        sigma: Gaussian sigma in output_size pixel space.
        input_size: Input image size.

    Returns:
        heatmaps: [B, N, output_size, output_size] Gaussian heatmaps.
    """
    scale = output_size / input_size
    scaled = coords * scale  # [B, N, 2]
    B, N, _ = scaled.shape
    device = coords.device

    y_grid = torch.arange(output_size, dtype=torch.float32, device=device)
    x_grid = torch.arange(output_size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')  # [H, W]

    # [B, N, 1, 1]
    cx = scaled[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    cy = scaled[:, :, 1].unsqueeze(-1).unsqueeze(-1)

    # [B, N, H, W]
    heatmaps = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmaps
