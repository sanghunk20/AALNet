"""Evaluation metrics for landmark detection.

Provides Mean Radial Error (MRE) in mm and Successful Detection Rate (SDR)
at various thresholds.
"""

import numpy as np


def compute_mre(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_spacing: float | np.ndarray,
) -> dict:
    """Compute Mean Radial Error in mm.

    Args:
        pred_coords: [N, L, 2] predicted coordinates in pixel space.
        gt_coords: [N, L, 2] ground truth coordinates in pixel space.
        pixel_spacing: mm/pixel scalar or [N] array per sample.

    Returns:
        dict with 'mre_mean', 'mre_std', 'mre_per_landmark' (shape [L]).
    """
    # Radial error per landmark per sample: sqrt((dx)^2 + (dy)^2)
    diff = pred_coords - gt_coords  # [N, L, 2]
    radial_error_px = np.sqrt((diff ** 2).sum(axis=-1))  # [N, L]

    # Convert to mm
    if isinstance(pixel_spacing, (int, float)):
        radial_error_mm = radial_error_px * pixel_spacing
    else:
        radial_error_mm = radial_error_px * pixel_spacing[:, np.newaxis]

    # Use nanmean/nanstd to handle samples with missing pixel spacing (NaN)
    mre_per_landmark = np.nanmean(radial_error_mm, axis=0)  # [L]
    mre_mean = np.nanmean(radial_error_mm)
    mre_std = np.nanstd(radial_error_mm)

    return {
        'mre_mean': float(mre_mean),
        'mre_std': float(mre_std),
        'mre_per_landmark': mre_per_landmark,
        'radial_errors_mm': radial_error_mm,
    }


def compute_per_axis_mae(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_spacing: float | np.ndarray,
) -> dict:
    """Compute per-axis Mean Absolute Error in mm.

    Separates the x-axis and y-axis errors.

    Args:
        pred_coords: [N, L, 2] predicted coordinates in pixel space.
        gt_coords: [N, L, 2] ground truth coordinates in pixel space.
        pixel_spacing: mm/pixel scalar or [N] array per sample.

    Returns:
        dict with:
            'x_mae_per_landmark': [L] mean |dx| per landmark (mm).
            'y_mae_per_landmark': [L] mean |dy| per landmark (mm).
            'x_errors_mm': [N, L] per-sample x errors (mm).
            'y_errors_mm': [N, L] per-sample y errors (mm).
    """
    diff = pred_coords - gt_coords  # [N, L, 2]
    abs_diff = np.abs(diff)  # [N, L, 2]

    x_error_px = abs_diff[:, :, 0]  # [N, L]
    y_error_px = abs_diff[:, :, 1]  # [N, L]

    # Convert to mm
    if isinstance(pixel_spacing, (int, float)):
        x_errors_mm = x_error_px * pixel_spacing
        y_errors_mm = y_error_px * pixel_spacing
    else:
        x_errors_mm = x_error_px * pixel_spacing[:, np.newaxis]
        y_errors_mm = y_error_px * pixel_spacing[:, np.newaxis]

    return {
        'x_mae_per_landmark': np.nanmean(x_errors_mm, axis=0),  # [L]
        'y_mae_per_landmark': np.nanmean(y_errors_mm, axis=0),  # [L]
        'x_errors_mm': x_errors_mm,
        'y_errors_mm': y_errors_mm,
    }


def compute_sdr(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_spacing: float | np.ndarray,
    thresholds: list[float] = [2.0, 2.5, 3.0, 4.0],
) -> dict:
    """Compute Successful Detection Rate at given thresholds (mm).

    Args:
        pred_coords: [N, L, 2] predicted coordinates in pixel space.
        gt_coords: [N, L, 2] ground truth coordinates in pixel space.
        pixel_spacing: mm/pixel scalar or [N] array per sample.
        thresholds: List of distance thresholds in mm.

    Returns:
        dict mapping threshold -> SDR percentage (0~100).
    """
    diff = pred_coords - gt_coords
    radial_error_px = np.sqrt((diff ** 2).sum(axis=-1))  # [N, L]

    if isinstance(pixel_spacing, (int, float)):
        radial_error_mm = radial_error_px * pixel_spacing
    else:
        radial_error_mm = radial_error_px * pixel_spacing[:, np.newaxis]

    # Filter out NaN samples (missing pixel spacing)
    valid_mask = ~np.isnan(radial_error_mm)
    sdr = {}
    for t in thresholds:
        hits = np.where(valid_mask, radial_error_mm < t, False)
        n_valid = valid_mask.sum()
        sdr[f'sdr_{t}mm'] = float(hits.sum() / n_valid * 100) if n_valid > 0 else 0.0

    return sdr
