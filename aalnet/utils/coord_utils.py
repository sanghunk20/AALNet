"""Coordinate conversion utilities for landmark detection.

Provides functions to convert coordinates between the letterboxed (network input) space
and original image space, using per-image transform parameters saved
during preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_transform_params(params_dir: Path | str, stem: str) -> dict:
    """Load transform parameters for a single image.

    Args:
        params_dir: Directory containing {stem}.json files.
        stem: Image filename stem (no extension).

    Returns:
        Dict with keys: original_size, bbox, scale, pad_top, pad_left.
    """
    params_path = Path(params_dir) / f'{stem}.json'
    with open(params_path, 'r') as f:
        return json.load(f)


def coords_to_original(coords: np.ndarray, params: dict) -> np.ndarray:
    """Convert letterbox-space coordinates to original image space for one sample.

    Args:
        coords: [L, 2] coordinates in letterbox space (x, y).
        params: Transform parameters dict with scale, pad_left, pad_top, bbox.

    Returns:
        [L, 2] coordinates in original image space.
    """
    scale = params['scale']
    pad_left = params['pad_left']
    pad_top = params['pad_top']
    bbox = params['bbox']

    # Reverse letterbox: undo padding, undo scale
    x = (coords[:, 0] - pad_left) / scale
    y = (coords[:, 1] - pad_top) / scale

    # Reverse crop: add bbox offset
    if bbox is not None:
        x += bbox[0]
        y += bbox[1]

    return np.stack([x, y], axis=-1)


def batch_coords_to_original(
    coords: np.ndarray,
    filenames: list[str],
    params_dir: Path | str,
) -> np.ndarray:
    """Convert a batch of letterbox-space coordinates to original image space.

    Args:
        coords: [N, L, 2] predicted or GT coordinates in letterbox space.
        filenames: [N] list of image filename stems.
        params_dir: Directory containing transform_params/{stem}.json.

    Returns:
        [N, L, 2] coordinates in original image space.
    """
    params_dir = Path(params_dir)
    N, L, _ = coords.shape
    result = np.empty_like(coords)

    for i in range(N):
        params = load_transform_params(params_dir, filenames[i])
        result[i] = coords_to_original(coords[i], params)

    return result
