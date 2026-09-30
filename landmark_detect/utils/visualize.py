"""Overlay of predicted and ground-truth landmarks on the original radiographs."""

from pathlib import Path

import cv2
import numpy as np


# Color scheme (BGR for OpenCV)
COLOR_GT = (0, 255, 0)       # Green
COLOR_PRED = (0, 0, 255)     # Red
COLOR_LINE = (255, 200, 0)   # Cyan - line connecting pred to GT
POINT_RADIUS = 3
LINE_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.3
FONT_COLOR = (255, 255, 255)


def draw_landmarks(
    image: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    landmark_names: list[str],
    draw_lines: bool = True,
    draw_names: bool = False,
    point_radius: int | None = None,
    line_thickness: int | None = None,
    font_scale: float | None = None,
) -> np.ndarray:
    """Draw predicted and GT landmarks on image.

    Args:
        image: [H, W, 3] BGR image (uint8).
        pred_coords: [N, 2] predicted (x, y) in image space.
        gt_coords: [N, 2] ground truth (x, y) in image space.
        landmark_names: List of landmark names.
        draw_lines: Draw lines connecting pred to GT.
        draw_names: Draw landmark name labels.
        point_radius: Override default POINT_RADIUS.
        line_thickness: Override default LINE_THICKNESS.
        font_scale: Override default FONT_SCALE.

    Returns:
        Annotated image.
    """
    pr = point_radius if point_radius is not None else POINT_RADIUS
    lt = line_thickness if line_thickness is not None else LINE_THICKNESS
    fs = font_scale if font_scale is not None else FONT_SCALE

    vis = image.copy()

    for i in range(len(pred_coords)):
        px, py = int(round(pred_coords[i, 0])), int(round(pred_coords[i, 1]))
        gx, gy = int(round(gt_coords[i, 0])), int(round(gt_coords[i, 1]))

        # Draw line from pred to GT
        if draw_lines:
            cv2.line(vis, (px, py), (gx, gy), COLOR_LINE, lt)

        # Draw GT point (green, filled)
        cv2.circle(vis, (gx, gy), pr, COLOR_GT, -1)

        # Draw pred point (red, filled)
        cv2.circle(vis, (px, py), pr, COLOR_PRED, -1)

        # Draw landmark name
        if draw_names:
            cv2.putText(vis, landmark_names[i], (gx + 5, gy - 5),
                        FONT, fs, FONT_COLOR, 1)

    return vis


def save_vis_batch_original(
    pred_orig: np.ndarray,
    gt_orig: np.ndarray,
    filenames: list[str],
    landmark_names: list[str],
    original_images_dir: str,
    output_dir: str | Path,
    spacing_array: np.ndarray | None = None,
    draw_lines: bool = True,
    draw_names: bool = False,
    max_images: int = 0,
) -> None:
    """Save visualization images at original resolution.

    Draws predicted (red) and GT (green) landmarks on original-resolution
    images with MRE overlay and legend.

    Args:
        pred_orig: [N, L, 2] predicted coordinates in original pixel space.
        gt_orig: [N, L, 2] ground truth coordinates in original pixel space.
        filenames: [N] image filename stems.
        landmark_names: List of landmark names.
        original_images_dir: Directory containing original-resolution JPGs.
        output_dir: Directory to save visualization images.
        spacing_array: [N] pixel spacing (mm/px) per image for mm MRE overlay.
            If None, MRE is shown in pixel units.
        draw_lines: Draw lines connecting pred to GT.
        draw_names: Draw landmark name labels.
        max_images: Maximum images to save. 0 = all.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(filenames)
    if max_images > 0:
        n_images = min(n_images, max_images)

    saved = 0
    for idx in range(n_images):
        filename = filenames[idx]
        img_path = Path(original_images_dir) / f'{filename}.jpg'
        if not img_path.exists():
            print(f"  [SKIP VIS] Image not found: {img_path}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = vis.shape[:2]
        scale = max(h, w) / 608.0
        pr = max(3, int(POINT_RADIUS * scale))
        lt = max(1, int(LINE_THICKNESS * scale))
        fs = FONT_SCALE * scale

        vis = draw_landmarks(
            vis, pred_orig[idx], gt_orig[idx], landmark_names,
            draw_lines=draw_lines, draw_names=draw_names,
            point_radius=pr, line_thickness=lt, font_scale=fs,
        )

        # MRE overlay
        errors_px = np.sqrt(
            ((pred_orig[idx] - gt_orig[idx]) ** 2).sum(axis=-1)
        )
        if spacing_array is not None:
            mre_mm = float(errors_px.mean() * spacing_array[idx])
            mre_text = f'MRE: {mre_mm:.2f}mm'
        else:
            mre_text = f'MRE: {errors_px.mean():.2f}px'

        text_fs = max(0.5, 0.7 * scale)
        text_thick = max(1, int(2 * scale))
        cv2.putText(vis, mre_text, (int(10 * scale), int(25 * scale)),
                    FONT, text_fs, (0, 255, 255), text_thick)
        cv2.putText(vis, filename, (int(10 * scale), int(50 * scale)),
                    FONT, max(0.4, 0.5 * scale), (200, 200, 200),
                    max(1, int(scale)))

        # Legend at bottom-left
        legend_y = h - int(30 * scale)
        legend_x = int(10 * scale)
        legend_fs = max(0.3, 0.4 * scale)
        legend_thick = max(1, int(scale))
        gap = int(60 * scale)

        cv2.circle(vis, (legend_x, legend_y), pr, COLOR_GT, -1)
        cv2.putText(vis, 'GT', (legend_x + pr + 5, legend_y + 4),
                    FONT, legend_fs, COLOR_GT, legend_thick)

        cv2.circle(vis, (legend_x + gap, legend_y), pr, COLOR_PRED, -1)
        cv2.putText(vis, 'Pred', (legend_x + gap + pr + 5, legend_y + 4),
                    FONT, legend_fs, COLOR_PRED, legend_thick)

        cv2.line(vis, (legend_x + gap * 2, legend_y),
                 (legend_x + gap * 2 + int(20 * scale), legend_y),
                 COLOR_LINE, lt)
        cv2.putText(vis, 'Error',
                    (legend_x + gap * 2 + int(25 * scale), legend_y + 4),
                    FONT, legend_fs, COLOR_LINE, legend_thick)

        out_path = output_dir / f'{filename}_landmarks.jpg'
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 20 == 0:
            print(f"  [VIS] {saved}/{n_images} images saved")

    print(f"  Visualization: {saved} images saved to {output_dir}")
