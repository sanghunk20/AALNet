"""Training and validation loops.

- bfloat16 automatic mixed precision
- gradient accumulation
- total loss = base loss + lambda * asymmetry loss
"""

import math
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp

from ..utils.metrics import compute_mre
from ..utils.heatmap_utils import heatmap_to_coords_soft_argmax


def asymmetry_weight(epoch: int, lambda_max: float) -> float:
    """Weight of the asymmetry loss at ``epoch``.

    The first epoch (epoch 0, learning rate 0 under the per-epoch warm-up) runs
    with the base loss alone; from the second epoch on the weight is ``lambda_max``.
    """
    return lambda_max if epoch > 0 else 0.0


def _normalized_coords(coords: torch.Tensor, size: int) -> torch.Tensor:
    """Pixel coordinates in [0, size] -> [-1, 1]."""
    half_size = size / 2.0
    return coords / half_size - 1.0


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0.0,
    accum_iter: int = 1,
    print_freq: int = 20,
    asym_criterion=None,
    asym_lambda_max: float = 0.0,
):
    """Train one epoch.

    Args:
        model: AALNet instance (optionally wrapped in DistributedDataParallel).
        data_loader: Training data loader yielding (images, coords, filenames).
        criterion: Base loss (DSNTLoss).
        optimizer: Optimizer.
        device: CUDA device.
        epoch: Current epoch number.
        loss_scaler: torch GradScaler.
        max_norm: Max gradient norm for clipping (0 = no clipping).
        accum_iter: Gradient accumulation steps.
        print_freq: Print frequency.
        asym_criterion: Asymmetry loss (AsymmetryLoss) or None.
        asym_lambda_max: Weight of the asymmetry loss.

    Returns:
        Dict of average metrics for this epoch.
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_asymmetry_loss = 0.0
    total_fused_loss = 0.0
    total_asym_details = defaultdict(float)
    num_batches = 0

    # Asymmetry-loss weight for this epoch (constant within the epoch)
    asym_lam = 0.0
    if asym_criterion is not None:
        asym_lam = asymmetry_weight(epoch, asym_lambda_max)

    for batch_idx, (images, coords, _) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        gt_coords = coords.to(device, non_blocking=True)

        with amp.autocast(dtype=torch.bfloat16):
            heatmaps = model(images)

            # Base loss on all landmarks
            loss = criterion(heatmaps, gt_coords)
            loss_value = loss.item()

            # Asymmetry loss on the predicted coordinates (with gradient flow)
            asym_loss_value = 0.0
            if asym_criterion is not None and asym_lam > 0:
                pred_coords = heatmap_to_coords_soft_argmax(heatmaps)
                heatmap_size = heatmaps.shape[-1]

                asym_loss_raw, asym_details = asym_criterion(
                    _normalized_coords(pred_coords, heatmap_size),
                    _normalized_coords(gt_coords, heatmap_size),
                    return_details=True,
                )
                asym_loss_value = asym_loss_raw.item()
                for k, v in asym_details.items():
                    total_asym_details[k] += v

                if not math.isfinite(asym_loss_value):
                    print(f"Asymmetry loss is {asym_loss_value}, stopping training")
                    sys.exit(1)

                loss = loss + asym_lam * asym_loss_raw

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Total loss (base + asymmetry) before accum_iter scaling
        total_loss_value = loss.item()

        loss = loss / accum_iter
        update_grad = (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(data_loader)

        loss_scaler.scale(loss).backward()
        if update_grad:
            loss_scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_loss += loss_value
        total_asymmetry_loss += asym_loss_value
        total_fused_loss += total_loss_value
        num_batches += 1

        if batch_idx % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            msg = (f'Epoch [{epoch}][{batch_idx}/{len(data_loader)}] '
                   f'loss: {loss_value:.4f} lr: {lr:.6f}')
            if asym_lam > 0:
                msg += (f' asym: {asym_loss_value:.4f} λ: {asym_lam:.3f}'
                        f' total: {total_loss_value:.4f}')
            print(msg)

    avg_loss = total_loss / max(num_batches, 1)
    result = {'train_loss': avg_loss}
    if asym_criterion is not None:
        result['train_asym_loss'] = total_asymmetry_loss / max(num_batches, 1)
        result['train_total_loss'] = total_fused_loss / max(num_batches, 1)
        result['asym_lambda'] = asym_lam
        for k, v in total_asym_details.items():
            result[f'asym_{k}'] = v / max(num_batches, 1)
    print(f'Epoch [{epoch}] avg_loss: {avg_loss:.4f}')
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device,
    asym_criterion=None,
    asym_lambda: float = 0.0,
):
    """Evaluate the model on a validation set (errors in pixels of the input image).

    Args:
        model: AALNet instance (not wrapped).
        data_loader: Validation data loader yielding (images, coords, filenames).
        criterion: Base loss (DSNTLoss).
        device: CUDA device.
        asym_criterion: Asymmetry loss or None (monitoring only).
        asym_lambda: Asymmetry-loss weight of the current epoch.

    Returns:
        Dict of evaluation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_asymmetry_loss = 0.0
    total_fused_loss = 0.0
    num_batches = 0
    all_pred_coords = []
    all_gt_coords = []

    for images, coords, _ in data_loader:
        images = images.to(device, non_blocking=True)
        gt_coords = coords.to(device, non_blocking=True)

        with amp.autocast(dtype=torch.bfloat16):
            heatmaps = model(images)
            loss = criterion(heatmaps, gt_coords)

        base_loss_value = loss.item()

        # Asymmetry loss, for monitoring only
        asym_loss_value = 0.0
        if asym_criterion is not None and asym_lambda > 0:
            pred_coords = heatmap_to_coords_soft_argmax(heatmaps)
            heatmap_size = heatmaps.shape[-1]
            asym_loss_value = asym_criterion(
                _normalized_coords(pred_coords, heatmap_size),
                _normalized_coords(gt_coords, heatmap_size),
            ).item()
            fused_loss_value = base_loss_value + asym_lambda * asym_loss_value
        else:
            fused_loss_value = base_loss_value

        total_loss += base_loss_value
        total_asymmetry_loss += asym_loss_value
        total_fused_loss += fused_loss_value
        num_batches += 1

        pred_coords = model.get_coordinates(heatmaps)  # [B, N, 2] in input pixels

        all_pred_coords.append(pred_coords.float().cpu().numpy())
        all_gt_coords.append(gt_coords.float().cpu().numpy())

    all_pred = np.concatenate(all_pred_coords, axis=0)
    all_gt = np.concatenate(all_gt_coords, axis=0)

    avg_loss = total_loss / max(num_batches, 1)

    # pixel_spacing = 1.0: MRE in pixels (the mm conversion is done by eval_checkpoint.py)
    metrics = compute_mre(all_pred, all_gt, 1.0)
    metrics['mre_unit'] = 'px'

    metrics['val_loss'] = avg_loss
    if asym_criterion is not None:
        avg_asym = total_asymmetry_loss / max(num_batches, 1)
        avg_fused = total_fused_loss / max(num_batches, 1)
        metrics['val_asym_loss'] = avg_asym
        metrics['val_total_loss'] = avg_fused

    msg = (f'Eval: loss={avg_loss:.4f}, MRE={metrics["mre_mean"]:.2f}px '
           f'(±{metrics["mre_std"]:.2f})')
    if asym_criterion is not None and asym_lambda > 0:
        msg += f', asym={avg_asym:.4f}, total={avg_fused:.4f}'
    print(msg)

    return metrics
