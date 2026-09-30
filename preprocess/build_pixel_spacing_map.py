"""Build per-image pixel spacing map from resolution groups.

Reads original image sizes from the raw image directory, maps each image
to its resolution group in resolution_pixel_spacing.json, and outputs
a per-image pixel spacing JSON file.

Special cases:
- A resolution group may give the pixel spacing per patient instead of one value
  (the patient ID is taken from the file name)
- Images with null pixel_spacing are logged as warnings

Output format (JSON):
    {"P0001_pod1y": 0.135, "P0001_preop": 0.135, ...}

Usage:
    python preprocess/build_pixel_spacing_map.py \
        --raw_img_dir /path/to/landmark_detect/images/raw \
        --resolution_json /path/to/resolution_pixel_spacing.json \
        --output /path/to/pixel_spacing_per_image.json
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def build_resolution_lookup(resolution_json: dict) -> dict:
    """Build {(width, height): pixel_spacing_mm} lookup from resolution groups.

    For per_patient groups (pixel_spacing_mm is null), stores per_patient dict.

    Returns:
        Dict mapping (width, height) -> float or dict.
    """
    lookup = {}
    for key, info in resolution_json['resolutions'].items():
        w, h = info['width'], info['height']
        ps = info.get('pixel_spacing_mm')
        per_patient = info.get('per_patient')

        if ps is not None:
            lookup[(w, h)] = ps
        elif per_patient is not None:
            # Store per_patient dict for special handling
            lookup[(w, h)] = per_patient
        else:
            # null and no per_patient → unmappable
            lookup[(w, h)] = None

    return lookup


def extract_patient_id(filename_stem: str) -> str:
    """Extract patient ID from filename stem (e.g. 'P0001_preop' -> 'P0001')."""
    return filename_stem.split('_')[0]


def main():
    parser = argparse.ArgumentParser(
        description='Build per-image pixel spacing map from resolution groups'
    )
    parser.add_argument('--raw_img_dir', type=str, required=True,
                        help='Directory containing original raw images (*.jpg)')
    parser.add_argument('--resolution_json', type=str, required=True,
                        help='Path to resolution_pixel_spacing.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for per-image pixel spacing JSON')
    args = parser.parse_args()

    raw_dir = Path(args.raw_img_dir)
    if not raw_dir.exists():
        print(f"[ERROR] Raw image directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.resolution_json, 'r') as f:
        resolution_data = json.load(f)

    lookup = build_resolution_lookup(resolution_data)

    image_files = sorted(raw_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} images in {raw_dir}")

    spacing_map = {}
    warnings = []
    unmatched = []

    for img_path in image_files:
        stem = img_path.stem
        try:
            with Image.open(img_path) as pil_img:
                w, h = pil_img.size  # PIL returns (width, height)
        except Exception:
            warnings.append(f"Cannot read: {img_path.name}")
            continue

        key = (w, h)

        if key not in lookup:
            unmatched.append(f"{stem} ({w}x{h})")
            continue

        ps_value = lookup[key]

        if ps_value is None:
            warnings.append(f"{stem} ({w}x{h}): pixel_spacing is null (unmappable)")
            continue

        if isinstance(ps_value, dict):
            # per_patient handling
            patient_id = extract_patient_id(stem)
            if patient_id in ps_value:
                spacing_map[stem] = ps_value[patient_id]['pixel_spacing_mm']
            else:
                warnings.append(
                    f"{stem} ({w}x{h}): per_patient group but "
                    f"patient '{patient_id}' not found in mapping"
                )
                continue
        else:
            spacing_map[stem] = ps_value

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(spacing_map, f, indent=2)

    # Report
    print(f"\nResults:")
    print(f"  Mapped: {len(spacing_map)} / {len(image_files)}")
    if unmatched:
        print(f"  Unmatched resolutions: {len(unmatched)}")
        for u in unmatched[:10]:
            print(f"    - {u}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    - {w}")

    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
