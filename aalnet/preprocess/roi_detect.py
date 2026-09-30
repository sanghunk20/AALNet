"""Skull ROI detection by Otsu thresholding.

For every radiograph, a foreground mask is computed by Otsu thresholding and the
skull is selected among its connected components; the bounding box (with a
margin) is written to bboxes.csv, which crop_and_letterbox.py consumes.

Source structure:
    src_root/images/raw/*.jpg

Usage:
    python -m aalnet.preprocess.roi_detect \
        --src_root /path/to/dataset \
        --output_dir /path/to/roi_otsu
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# Foreground mask by thresholding.
def build_threshold_mask(
    gray: np.ndarray,
    thresh_mode: str,
    thresh_value: int,
    thresh_invert: bool,
    border_percent: float,
    thresh_delta: int,
    thresh_scale: float,
    thresh_band: int,
) -> Tuple[np.ndarray, int]:
    # Image size.
    h, w = gray.shape
    # Blur to reduce noise and stabilise the foreground/background separation.
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Estimate the threshold on the central region, so that markers, text and frame edges do not bias it.
    by = int(h * border_percent)
    bx = int(w * border_percent)
    if (h - 2 * by) > 20 and (w - 2 * bx) > 20:
        core = blurred[by : h - by, bx : w - bx]
    else:
        core = blurred

    # Binary or inverted-binary mode.
    binary_type = cv2.THRESH_BINARY_INV if thresh_invert else cv2.THRESH_BINARY

    # Otsu mode: estimate the threshold automatically on the central region.
    if thresh_mode == "otsu":
        otsu_base = cv2.THRESH_BINARY_INV if thresh_invert else cv2.THRESH_BINARY
        threshold_used, _ = cv2.threshold(core, 0, 255, otsu_base + cv2.THRESH_OTSU)
        threshold_used = int(threshold_used)
    else:
        # Fixed mode: use the given threshold.
        threshold_used = int(np.clip(thresh_value, 0, 255))

    # Apply scale and delta to the threshold.
    threshold_center = int(np.clip((threshold_used * thresh_scale) + thresh_delta, 0, 255))
    band = max(0, int(thresh_band))

    # band == 0: single threshold; otherwise a range mask.
    if band == 0:
        _, mask = cv2.threshold(blurred, threshold_center, 255, binary_type)
    else:
        low = int(np.clip(threshold_center - band, 0, 255))
        high = int(np.clip(threshold_center + band, 0, 255))
        # inRange sets [low, high] to 255.
        band_mask = cv2.inRange(blurred, low, high)
        # When inverted, the region outside the band is the foreground.
        if thresh_invert:
            mask = cv2.bitwise_not(band_mask)
        else:
            mask = band_mask

    # Morphology: remove small specks and connect broken foreground.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return mask, threshold_used


# ROI bounding box from the threshold mask.
def extract_bbox_from_mask(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    min_area_ratio: float,
    margin_ratio: float,
    border_percent: float,
    center_weight: float,
    min_bbox_width_ratio: float,
    min_bbox_height_ratio: float,
    reject_ruler_like: bool,
    reject_top_border_ratio: float,
    reject_bottom_border_ratio: float,
    reject_left_border_ratio: float,
    reject_right_border_ratio: float,
) -> Tuple[int, int, int, int, float]:
    # Connected components.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Image size, area and centre.
    h, w = image_shape
    image_area = float(h * w)
    cx_img, cy_img = w / 2.0, h / 2.0
    diag_half = (w**2 + h**2) ** 0.5 / 2.0
    # Margin in pixels used to decide whether a component touches the border.
    # Border exclusion widths (top/bottom/left/right) from their ratios.
    top_border = int(h * reject_top_border_ratio)
    bottom_border = int(h * reject_bottom_border_ratio)
    left_border = int(w * reject_left_border_ratio)
    right_border = int(w * reject_right_border_ratio)

    # Valid component candidates.
    candidates = []
    # Label 0 is the background, so start from 1.
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = float(stats[label, cv2.CC_STAT_AREA])

        # Skip components that are too small.
        if area < min_area_ratio * image_area:
            continue

        # Skull ROI prior: a candidate must reach the minimum width and height ratios.
        if (bw / w) < min_bbox_width_ratio or (bh / h) < min_bbox_height_ratio:
            continue

        # Components touching the image border are likely background or frame; skip them.
        touches_border = (
            x <= left_border
            or y <= top_border
            or (x + bw) >= (w - right_border)
            or (y + bh) >= (h - bottom_border)
        )
        if touches_border:
            continue

        # Skip very long and thin vertical objects such as rulers.
        if reject_ruler_like:
            is_tall_narrow = (bh / max(bw, 1)) >= 2.8 and (bw / w) <= 0.10
            if is_tall_narrow:
                continue

        # Weight the score by the distance to the image centre, preferring central foreground.
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        dist_norm = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5 / max(diag_half, 1e-6)
        area_ratio = area / image_area
        score = area_ratio - center_weight * dist_norm
        candidates.append((score, x, y, x + bw - 1, y + bh - 1))

    # No candidate: retry with relaxed border conditions.
    if not candidates:
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = float(stats[label, cv2.CC_STAT_AREA])

            if area < min_area_ratio * image_area:
                continue
            if (bw / w) < min_bbox_width_ratio or (bh / h) < min_bbox_height_ratio:
                continue
            if reject_ruler_like:
                is_tall_narrow = (bh / max(bw, 1)) >= 2.8 and (bw / w) <= 0.10
                if is_tall_narrow:
                    continue

            cx, cy = float(centroids[label][0]), float(centroids[label][1])
            dist_norm = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5 / max(diag_half, 1e-6)
            area_ratio = area / image_area
            score = area_ratio - center_weight * dist_norm
            candidates.append((score, x, y, x + bw - 1, y + bh - 1))

    # Still no candidate: return the whole image.
    if not candidates:
        return 0, 0, w - 1, h - 1, 0.0

    # Select the candidate with the highest score.
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x1, y1, x2, y2 = candidates[0]

    # Asymmetric margin (smaller at the top, larger on the left and right).
    margin_lr = int((x2 - x1) * margin_ratio * 1.5)
    margin_top = int((y2 - y1) * margin_ratio * 0.5)
    margin_bottom = int((y2 - y1) * margin_ratio * 1.0)
    x1 = max(0, x1 - margin_lr)
    y1 = max(0, y1 - margin_top)
    x2 = min(w - 1, x2 + margin_lr)
    y2 = min(h - 1, y2 + margin_bottom)

    # Record the bbox area ratio as the score.
    score = float((x2 - x1 + 1) * (y2 - y1 + 1) / image_area)
    return x1, y1, x2, y2, score



def detect_roi_single(image_path: Path, args) -> tuple | None:
    """Detect the skull ROI bbox of a single image.

    Returns:
        Tuple of (x1, y1, x2, y2, score), or None if the image cannot be read.
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    mask, _ = build_threshold_mask(
        gray=gray,
        thresh_mode='otsu',
        thresh_value=20,
        thresh_invert=False,
        border_percent=args.border_percent,
        thresh_delta=args.thresh_delta,
        thresh_scale=args.thresh_scale,
        thresh_band=0,
    )
    x1, y1, x2, y2, score = extract_bbox_from_mask(
        mask=mask,
        image_shape=(h, w),
        min_area_ratio=args.min_area_ratio,
        margin_ratio=args.margin_ratio,
        border_percent=args.border_percent,
        center_weight=args.center_weight,
        min_bbox_width_ratio=args.min_bbox_width_ratio,
        min_bbox_height_ratio=args.min_bbox_height_ratio,
        reject_ruler_like=True,
        reject_top_border_ratio=0.0,
        reject_bottom_border_ratio=0.02,
        reject_left_border_ratio=0.03,
        reject_right_border_ratio=0.03,
    )

    # Recompute score as bbox area ratio
    score = float((x2 - x1 + 1) * (y2 - y1 + 1) / float(h * w))
    return x1, y1, x2, y2, score


def main():
    parser = argparse.ArgumentParser(description='Skull ROI detection (Otsu thresholding)')
    parser.add_argument('--src_root', type=str, required=True,
                        help='Source dataset root containing images/raw/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for bboxes.csv')

    # Otsu parameters
    parser.add_argument('--border_percent', type=float, default=0.04)
    parser.add_argument('--thresh_delta', type=int, default=0)
    parser.add_argument('--thresh_scale', type=float, default=2.0 / 3.0)
    parser.add_argument('--margin_ratio', type=float, default=0.12)
    parser.add_argument('--min_area_ratio', type=float, default=0.003)
    parser.add_argument('--center_weight', type=float, default=0.35)
    parser.add_argument('--min_bbox_width_ratio', type=float, default=0.18)
    parser.add_argument('--min_bbox_height_ratio', type=float, default=0.25)

    parser.add_argument('--max_images', type=int, default=0,
                        help='Max number of images to process (0 = all)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images with bbox overlay')

    args = parser.parse_args()

    src_root = Path(args.src_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if args.save_vis:
        vis_dir = output_dir / 'vis'
        vis_dir.mkdir(parents=True, exist_ok=True)

    img_dir = src_root / 'images' / 'raw'
    all_images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if args.max_images > 0:
        all_images = all_images[:args.max_images]
    print(f"Found {len(all_images)} images in {img_dir}")

    csv_path = output_dir / 'bboxes.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'score', 'method'])

        for img_path in tqdm(all_images, desc='ROI detect (otsu)'):
            result = detect_roi_single(img_path, args)
            if result is None:
                print(f"[WARN] Failed to read: {img_path}")
                continue

            x1, y1, x2, y2, score = result
            writer.writerow([
                img_path.name, x1, y1, x2, y2,
                x2 - x1 + 1, y2 - y1 + 1,
                f'{score:.6f}', 'otsu',
            ])

            if vis_dir is not None:
                image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                vis = image_bgr.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.imwrite(str(vis_dir / img_path.name), vis)

    print(f"[DONE] Saved bboxes to: {csv_path}")


if __name__ == '__main__':
    main()
