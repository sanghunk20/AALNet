"""Pixel spacing loader for mm-unit MRE computation.

Loads per-image pixel spacing (mm/pixel) from CSV or JSON files
and provides array conversion for batch evaluation.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np


def load_pixel_spacing(path: str | Path) -> dict[str, float]:
    """Load per-image pixel spacing from CSV or JSON.

    CSV format (header required):
        filename,pixel_spacing
        P0001_pod1y,0.100
        P0001_preop,0.100

    JSON format:
        {"P0001_pod1y": 0.100, "P0001_preop": 0.100}

    Args:
        path: Path to CSV or JSON file. Filename column should be
              stem only (no extension).

    Returns:
        Dict mapping filename stem to pixel spacing (mm/pixel).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}

    elif suffix == '.csv':
        spacing_map = {}
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = row['filename'].strip()
                spacing = float(row['pixel_spacing'])
                spacing_map[stem] = spacing
        return spacing_map

    else:
        raise ValueError(
            f'Unsupported pixel spacing file format: {suffix}. '
            f'Use .csv or .json.'
        )


def get_pixel_spacing_array(
    filenames: list[str],
    spacing_map: dict[str, float],
    allow_missing: bool = False,
) -> np.ndarray:
    """Build per-sample pixel spacing array matching filename order.

    Args:
        filenames: [N] list of image filename stems.
        spacing_map: Dict mapping stem to mm/pixel.
        allow_missing: If True, missing filenames get NaN instead of
            raising KeyError. Downstream metric functions should
            exclude NaN samples.

    Returns:
        [N] numpy array of pixel spacing values (NaN for missing if
        allow_missing=True).

    Raises:
        KeyError: If any filename is missing and allow_missing=False.
    """
    missing = [f for f in filenames if f not in spacing_map]
    if missing:
        if not allow_missing:
            raise KeyError(
                f'{len(missing)} filename(s) missing from pixel spacing file. '
                f'First 5: {missing[:5]}'
            )
        warnings.warn(
            f'{len(missing)} filename(s) missing pixel spacing, '
            f'will be excluded from mm-unit metrics. '
            f'First 5: {missing[:5]}'
        )

    return np.array(
        [spacing_map.get(f, np.nan) for f in filenames],
        dtype=np.float64,
    )
