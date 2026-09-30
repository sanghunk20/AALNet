"""ROI Crop + Letterbox Resize for Landmark Detection dataset.

Applies ROI crop (using Otsu bbox CSV) and letterbox resize to all images,
and transforms landmark coordinates accordingly.

Source structure (flat):
    src_root/images/raw/   *.jpg
    src_root/coords/       *.csv  (3-row format: names, X, Y)

Output structure (flat):
    dst_root/images/           *.jpg  (target_size × target_size, letterboxed)
    dst_root/coords/           *.csv  (transformed coordinates)
    dst_root/transform_params/ *.json (scale, pad_top, pad_left, bbox)

fold_splits.json is NOT generated here — run create_fold_splits.py separately.

Usage:
    python preprocess/crop_and_letterbox_landmark.py \
        --bbox_csv /path/to/bboxes.csv \
        --src_root /path/to/landmark_detect \
        --dst_root /path/to/landmark_detect_800 \
        --target_size 800
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def letterbox_resize_with_params(image: np.ndarray, target_size: int) -> tuple:
    """Aspect-ratio preserving resize with zero-padding (letterbox).

    Returns both the resized image and the transformation parameters
    needed to transform landmark coordinates.

    Args:
        image: Input image (H, W) or (H, W, C).
        target_size: Output square size (e.g. 800).

    Returns:
        Tuple of (letterboxed_image, params_dict).
        params_dict contains: scale, pad_top, pad_left, new_h, new_w.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    if image.ndim == 3:
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((target_size, target_size), dtype=image.dtype)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    params = {
        'scale': scale,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'new_h': new_h,
        'new_w': new_w,
    }
    return canvas, params


def transform_coordinates(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    bbox: tuple | None,
    letterbox_params: dict,
) -> tuple:
    """Transform landmark coordinates through crop + letterbox pipeline.

    Args:
        coords_x: [L] x coordinates in original image space.
        coords_y: [L] y coordinates in original image space.
        bbox: (x1, y1, x2, y2) crop box or None.
        letterbox_params: Dict from letterbox_resize_with_params.

    Returns:
        Tuple of (new_x, new_y) arrays in letterbox space.
    """
    x = coords_x.astype(np.float64)
    y = coords_y.astype(np.float64)

    # Step 1: Crop offset
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        x = x - bx1
        y = y - by1

    # Step 2: Scale + pad (letterbox)
    scale = letterbox_params['scale']
    pad_left = letterbox_params['pad_left']
    pad_top = letterbox_params['pad_top']

    x = x * scale + pad_left
    y = y * scale + pad_top

    return x, y


def load_bboxes(csv_path: str) -> dict:
    """Load bboxes.csv into a dict mapping filename -> (x1, y1, x2, y2)."""
    bboxes = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            bboxes[fname] = (x1, y1, x2, y2)
    return bboxes


def load_landmark_csv(csv_path: Path) -> tuple:
    """Load landmark CSV file (3-row format: names, X, Y).

    Args:
        csv_path: Path to landmark CSV file.

    Returns:
        Tuple of (landmark_names, x_coords, y_coords).
        landmark_names: list of str.
        x_coords, y_coords: np.ndarray of float.
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Row 0: header (first cell empty, rest are landmark names)
    landmark_names = rows[0][1:]

    # Row 1: X coordinates
    x_coords = np.array([float(v) for v in rows[1][1:]])

    # Row 2: Y coordinates
    y_coords = np.array([float(v) for v in rows[2][1:]])

    return landmark_names, x_coords, y_coords


def save_landmark_csv(csv_path: Path, landmark_names: list, x_coords: np.ndarray, y_coords: np.ndarray):
    """Save landmark coordinates in the same 3-row CSV format."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + landmark_names)
        writer.writerow(['X'] + [f'{v:.2f}' for v in x_coords])
        writer.writerow(['Y'] + [f'{v:.2f}' for v in y_coords])


def main():
    parser = argparse.ArgumentParser(description='ROI Crop + Letterbox for Landmark Detection')
    parser.add_argument('--bbox_csv', type=str, required=True,
                        help='Path to bboxes.csv from ROI detection')
    parser.add_argument('--src_root', type=str, required=True,
                        help='Source dataset root (images/raw/, coords/)')
    parser.add_argument('--dst_root', type=str, required=True,
                        help='Destination root of the preprocessed dataset')
    parser.add_argument('--target_size', type=int, default=800)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    src_img_dir = src_root / 'images' / 'raw'
    src_coord_dir = src_root / 'coords'

    if not src_img_dir.exists():
        print(f"[ERROR] Image directory not found: {src_img_dir}", file=sys.stderr)
        sys.exit(1)
    if not src_coord_dir.exists():
        print(f"[ERROR] Coord directory not found: {src_coord_dir}", file=sys.stderr)
        sys.exit(1)

    dst_img_dir = dst_root / 'images'
    dst_coord_dir = dst_root / 'coords'
    dst_params_dir = dst_root / 'transform_params'
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_coord_dir.mkdir(parents=True, exist_ok=True)
    dst_params_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bboxes from {args.bbox_csv}")
    bboxes = load_bboxes(args.bbox_csv)
    print(f"  Loaded {len(bboxes)} bbox entries")

    image_files = sorted(src_img_dir.glob('*.jpg'))
    if not image_files:
        print(f"[ERROR] No .jpg files found in {src_img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing {len(image_files)} images → {dst_root}")
    stats = defaultdict(int)

    for img_path in tqdm(image_files, unit='img'):
        fname_stem = img_path.stem
        coord_path = src_coord_dir / f'{fname_stem}.csv'

        if not coord_path.exists():
            print(f"  [WARNING] No coordinate file: {coord_path}", file=sys.stderr)
            stats['no_coord'] += 1
            continue

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARNING] Failed to read: {img_path}", file=sys.stderr)
            stats['skipped'] += 1
            continue

        # Load coordinates
        landmark_names, x_coords, y_coords = load_landmark_csv(coord_path)

        # ROI crop
        bbox = bboxes.get(img_path.name, None)
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            h, w = img.shape[:2]
            bx1 = max(0, min(bx1, w - 1))
            by1 = max(0, min(by1, h - 1))
            bx2 = max(bx1 + 1, min(bx2, w))
            by2 = max(by1 + 1, min(by2, h))
            bbox = (bx1, by1, bx2, by2)
            img = img[by1:by2, bx1:bx2]
        else:
            stats['no_bbox'] += 1

        # Record size after crop, before letterbox (for inverse mapping)
        cropped_h, cropped_w = img.shape[:2]

        # Letterbox resize
        result, params = letterbox_resize_with_params(img, args.target_size)

        # Transform coordinates
        new_x, new_y = transform_coordinates(x_coords, y_coords, bbox, params)

        # Save image
        cv2.imwrite(str(dst_img_dir / img_path.name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save transformed coordinates
        save_landmark_csv(dst_coord_dir / f'{fname_stem}.csv', landmark_names, new_x, new_y)

        # Save transform params for inverse mapping during evaluation
        save_params = {
            'original_size': [cropped_h, cropped_w],
            'bbox': list(bbox) if bbox is not None else None,
            'scale': params['scale'],
            'pad_top': params['pad_top'],
            'pad_left': params['pad_left'],
        }
        with open(dst_params_dir / f'{fname_stem}.json', 'w') as f:
            json.dump(save_params, f)

        stats['processed'] += 1

    print(f"\n{'='*60}")
    print(f"processed={stats['processed']}, "
          f"no_bbox={stats['no_bbox']}, "
          f"no_coord={stats['no_coord']}, "
          f"skipped={stats['skipped']}")
    print(f"Output saved to: {dst_root}")
    print(f"\nNext step: run create_fold_splits.py to generate fold_splits.json")


if __name__ == '__main__':
    main()
