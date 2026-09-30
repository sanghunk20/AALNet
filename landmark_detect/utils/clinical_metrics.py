"""Clinical midline and deviation metrics for PA cephalometric analysis.

Computes:
    - Midline (standard: latero-orbital perpendicular bisector, old: CG→ANS)
    - Midline comparison: GT vs Predicted (angle difference, x displacement)
    - Deviation metrics: lowerface, midface, dental, canting

All distance outputs are in pixel units. Multiply by pixel_spacing for mm.
Direction convention: patient Right (+), patient Left (-).
In PA ceph image coordinates: image left = patient right.

Naming: midline_std = MSR (perpendicular bisector of the latero-orbitale pair),
midline_old = crista galli-ANS (Cg-ANS) line.
"""

from __future__ import annotations

import numpy as np


# ── Landmark index constants ────────────────────────────────────────

LM_LATERO_ORBITAL_R = 29
LM_LATERO_ORBITAL_L = 30
LM_CRISTA_GALLI = 20
LM_ANS = 2             # Anterior nasal spine
LM_MENTON = 3
LM_ZYGOMA_R = 25
LM_ZYGOMA_L = 26
LM_UPPER_MIDLINE = 31
LM_MAXILLARY_1_CROWN_R = 6
LM_MAXILLARY_1_CROWN_L = 8
LM_MAXILLARY_6_CROWN_R = 4
LM_MAXILLARY_6_CROWN_L = 10


# ── Helper: ensure batch dimension ──────────────────────────────────

def _ensure_batch(coords: np.ndarray) -> np.ndarray:
    """Ensure coords is [N, L, 2]. If [L, 2], add batch dim."""
    if coords.ndim == 2:
        return coords[np.newaxis]  # [1, L, 2]
    return coords


# ── Midline computation ─────────────────────────────────────────────

def compute_midline_std(
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standard midline: perpendicular bisector of latero-orbital R/L.

    Args:
        coords: [N, L, 2] or [L, 2] landmark coordinates.

    Returns:
        (points, directions) where:
            points: [N, 2] midpoint of latero-orbital R and L.
            directions: [N, 2] unit direction vector of the perpendicular bisector.
                The bisector is perpendicular to the R-L line segment.
    """
    coords = _ensure_batch(coords)
    r = coords[:, LM_LATERO_ORBITAL_R]  # [N, 2]
    l = coords[:, LM_LATERO_ORBITAL_L]  # [N, 2]

    midpoint = (r + l) / 2.0  # [N, 2]

    # Direction from R to L
    rl = l - r  # [N, 2]
    # Perpendicular: rotate 90° CCW → (-dy, dx)
    perp = np.stack([-rl[:, 1], rl[:, 0]], axis=-1)  # [N, 2]
    norm = np.linalg.norm(perp, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    direction = perp / norm  # [N, 2] unit vector

    return midpoint, direction


def compute_midline_old(
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute old midline: crista galli → ANS line.

    Args:
        coords: [N, L, 2] or [L, 2] landmark coordinates.

    Returns:
        (points, directions) where:
            points: [N, 2] crista galli point (line origin).
            directions: [N, 2] unit direction vector from CG to ANS.
    """
    coords = _ensure_batch(coords)
    cg = coords[:, LM_CRISTA_GALLI]  # [N, 2]
    ans = coords[:, LM_ANS]          # [N, 2]

    direction = ans - cg  # [N, 2]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    direction = direction / norm

    # Use midpoint of CG-ANS as line point
    midpoint = (cg + ans) / 2.0

    return midpoint, direction


# ── Midline comparison metrics ───────────────────────────────────────

def compute_midline_angle_diff(
    gt_dir: np.ndarray,
    pred_dir: np.ndarray,
) -> np.ndarray:
    """Compute angle difference between GT and predicted midline directions.

    Args:
        gt_dir: [N, 2] GT midline direction unit vectors.
        pred_dir: [N, 2] predicted midline direction unit vectors.

    Returns:
        [N] angle differences in degrees (always >= 0).
    """
    # Clamp dot product to [-1, 1] for numerical stability
    dot = np.sum(gt_dir * pred_dir, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(dot)))  # abs: direction sign irrelevant
    return angles


def compute_midline_cosine_sim(
    gt_dir: np.ndarray,
    pred_dir: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between GT and predicted midline directions.

    Uses abs(dot) since direction sign is arbitrary.

    Args:
        gt_dir: [N, 2] GT midline direction unit vectors.
        pred_dir: [N, 2] predicted midline direction unit vectors.

    Returns:
        [N] cosine similarity values in [0, 1].
    """
    dot = np.sum(gt_dir * pred_dir, axis=-1)
    return np.abs(dot)


def compute_midline_x_displacement(
    gt_point: np.ndarray,
    gt_dir: np.ndarray,
    pred_point: np.ndarray,
    pred_dir: np.ndarray,
    ref_y: np.ndarray,
) -> np.ndarray:
    """Compute x-axis displacement between two midlines at a reference y.

    For each midline defined by (point, direction), finds the x coordinate
    at ref_y, then computes the difference.

    Args:
        gt_point: [N, 2] GT midline point.
        gt_dir: [N, 2] GT midline direction.
        pred_point: [N, 2] predicted midline point.
        pred_dir: [N, 2] predicted midline direction.
        ref_y: [N] reference y coordinate for comparison.

    Returns:
        [N] absolute x displacement in pixels.
    """
    # Parametric line: p(t) = point + t * dir
    # At y = ref_y: ref_y = point_y + t * dir_y → t = (ref_y - point_y) / dir_y
    # x = point_x + t * dir_x

    gt_dir_y = gt_dir[:, 1]
    pred_dir_y = pred_dir[:, 1]

    # Avoid division by zero (horizontal midline — extremely unlikely in PA ceph)
    gt_dir_y = np.where(np.abs(gt_dir_y) < 1e-8, 1e-8, gt_dir_y)
    pred_dir_y = np.where(np.abs(pred_dir_y) < 1e-8, 1e-8, pred_dir_y)

    gt_t = (ref_y - gt_point[:, 1]) / gt_dir_y
    gt_x = gt_point[:, 0] + gt_t * gt_dir[:, 0]

    pred_t = (ref_y - pred_point[:, 1]) / pred_dir_y
    pred_x = pred_point[:, 0] + pred_t * pred_dir[:, 0]

    return np.abs(gt_x - pred_x)


# ── Signed distance to line ─────────────────────────────────────────

def _ensure_downward(direction: np.ndarray) -> np.ndarray:
    """Ensure direction vectors point downward (dy > 0).

    This normalizes direction sign so that signed_distance_to_line
    produces consistent results regardless of whether the midline was
    computed as CG→ANS or ANS→CG.

    Args:
        direction: [..., 2] direction vectors.

    Returns:
        [..., 2] direction vectors with dy >= 0 guaranteed.
    """
    # Flip vectors where dy < 0
    flip_mask = direction[..., 1:2] < 0
    return np.where(flip_mask, -direction, direction)


def signed_distance_to_line(
    point: np.ndarray,
    line_point: np.ndarray,
    line_dir: np.ndarray,
) -> np.ndarray:
    """Compute signed perpendicular distance from point to line.

    Sign convention: positive = patient Right (image left in PA ceph).

    The line direction is normalized to point downward (dy > 0) before
    computing the cross product, ensuring consistent sign regardless
    of direction construction order.

    With line_dir = (0, 1) (pointing down):
        - point to the left of line in image space → v_x < 0 → cross < 0
        - image left = patient right → we want positive
        - Therefore: signed_dist = -cross

    Args:
        point: [N, 2] or [2] point coordinates.
        line_point: [N, 2] or [2] point on the line.
        line_dir: [N, 2] or [2] unit direction of the line.

    Returns:
        [N] signed distances. Positive = patient Right, Negative = patient Left.
    """
    # Ensure consistent direction for sign convention
    line_dir = _ensure_downward(line_dir)

    # Vector from line_point to point
    v = point - line_point  # [N, 2]

    # Cross product in 2D: v × dir = vx*dy - vy*dx
    cross = v[..., 0] * line_dir[..., 1] - v[..., 1] * line_dir[..., 0]

    # Negate: with downward dir, cross < 0 means image-left (= patient-right)
    return -cross


# ── Deviation metrics ────────────────────────────────────────────────

def _direction_label(signed_dist: np.ndarray) -> np.ndarray:
    """Convert signed distances to direction labels.

    Returns:
        Array of strings: 'R' (positive, patient right), 'L' (negative), 'C' (zero).
    """
    labels = np.where(signed_dist > 0, 'R', np.where(signed_dist < 0, 'L', 'C'))
    return labels


def compute_lowerface_deviation(
    coords: np.ndarray,
    midline_point: np.ndarray,
    midline_dir: np.ndarray,
) -> dict:
    """Compute lowerface deviation: menton distance from midline_std.

    Args:
        coords: [N, L, 2] landmark coordinates.
        midline_point: [N, 2] midline point.
        midline_dir: [N, 2] midline direction.

    Returns:
        dict with keys:
            'distance': [N] signed distance (mm-ready, multiply by spacing).
            'direction': [N] 'R' or 'L'.
    """
    coords = _ensure_batch(coords)
    menton = coords[:, LM_MENTON]  # [N, 2]
    dist = signed_distance_to_line(menton, midline_point, midline_dir)

    return {
        'distance': dist,
        'direction': _direction_label(dist),
    }


def compute_midface_deviation(
    coords: np.ndarray,
    midline_point: np.ndarray,
    midline_dir: np.ndarray,
) -> dict:
    """Compute midface deviation: zygoma R/L asymmetry from midline_std.

    Asymmetry = signed_dist(R) + signed_dist(L).
    Sign convention: R → +, L → −, so sum ≈ 0 when symmetric.
    Positive = R side further from midline, Negative = L side further.

    Args:
        coords: [N, L, 2] landmark coordinates.
        midline_point: [N, 2] midline point.
        midline_dir: [N, 2] midline direction.

    Returns:
        dict with keys:
            'dist_R': [N] signed distance of zygoma R.
            'dist_L': [N] signed distance of zygoma L.
            'asymmetry': [N] signed_dist(R) + signed_dist(L). ≈0 when symmetric.
            'direction': [N] 'R' or 'L' indicating larger side.
    """
    coords = _ensure_batch(coords)
    zyg_r = coords[:, LM_ZYGOMA_R]  # [N, 2]
    zyg_l = coords[:, LM_ZYGOMA_L]  # [N, 2]

    dist_r = signed_distance_to_line(zyg_r, midline_point, midline_dir)
    dist_l = signed_distance_to_line(zyg_l, midline_point, midline_dir)

    asymmetry = dist_r + dist_l

    return {
        'dist_R': dist_r,
        'dist_L': dist_l,
        'asymmetry': asymmetry,
        'direction': _direction_label(asymmetry),
    }


def compute_dental_deviation(
    coords: np.ndarray,
    midline_point: np.ndarray,
    midline_dir: np.ndarray,
) -> dict:
    """Compute dental deviation from midline.

    Measures:
    1. Upper midline point distance from midline_std.
    2. Maxillary 1 crown R/L midpoint distance from midline_std.

    Args:
        coords: [N, L, 2] landmark coordinates.
        midline_point: [N, 2] midline point.
        midline_dir: [N, 2] midline direction.

    Returns:
        dict with keys:
            'upper_midline_dist': [N] signed distance of upper midline.
            'upper_midline_dir': [N] direction label.
            'crown_midpoint_dist': [N] signed distance of crown midpoint.
            'crown_midpoint_dir': [N] direction label.
    """
    coords = _ensure_batch(coords)

    # Upper midline
    upper_mid = coords[:, LM_UPPER_MIDLINE]  # [N, 2]
    um_dist = signed_distance_to_line(upper_mid, midline_point, midline_dir)

    # Maxillary 1 crown midpoint
    crown_r = coords[:, LM_MAXILLARY_1_CROWN_R]  # [N, 2]
    crown_l = coords[:, LM_MAXILLARY_1_CROWN_L]  # [N, 2]
    crown_mid = (crown_r + crown_l) / 2.0  # [N, 2]
    cm_dist = signed_distance_to_line(crown_mid, midline_point, midline_dir)

    return {
        'upper_midline_dist': um_dist,
        'upper_midline_dir': _direction_label(um_dist),
        'crown_midpoint_dist': cm_dist,
        'crown_midpoint_dir': _direction_label(cm_dist),
    }


def compute_canting(
    coords: np.ndarray,
    midline_point: np.ndarray,
    midline_dir: np.ndarray,
) -> dict:
    """Compute canting: signed distance along midline between R/L projections.

    Scalar projection t = (q - anchor) · dir gives distance along midline.
    canting = t_R - t_L: positive if R projects further along midline direction.

    Args:
        coords: [N, L, 2] landmark coordinates.
        midline_point: [N, 2] midline point (anchor).
        midline_dir: [N, 2] midline direction (unit vector).

    Returns:
        dict with keys:
            'proj_R': [N] scalar projection of maxillary 6 crown R onto midline.
            'proj_L': [N] scalar projection of maxillary 6 crown L onto midline.
            'diff': [N] proj_R - proj_L (signed).
            'direction': [N] 'R' or 'L' indicating which side projects further.
    """
    coords = _ensure_batch(coords)
    m6_r = coords[:, LM_MAXILLARY_6_CROWN_R]  # [N, 2]
    m6_l = coords[:, LM_MAXILLARY_6_CROWN_L]  # [N, 2]

    proj_r = np.sum((m6_r - midline_point) * midline_dir, axis=-1)  # [N]
    proj_l = np.sum((m6_l - midline_point) * midline_dir, axis=-1)  # [N]

    diff = proj_r - proj_l

    return {
        'proj_R': proj_r,
        'proj_L': proj_l,
        'diff': diff,
        'direction': _direction_label(diff),
    }


# ── Errors between predicted and ground-truth measurements ─────────

def compute_clinical_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    pixel_spacing: np.ndarray | float,
) -> dict:
    """Compare GT vs Predicted clinical metrics.

    Computes midline comparison (angle + x displacement) and
    deviation error (GT deviation - Pred deviation).

    Args:
        pred: [N, L, 2] predicted coordinates in original pixel space.
        gt: [N, L, 2] ground truth coordinates in original pixel space.
        pixel_spacing: [N] per-image mm/pixel, or scalar.

    Returns:
        dict with all clinical error metrics.
    """
    pred = _ensure_batch(pred)
    gt = _ensure_batch(gt)
    N = pred.shape[0]

    scale = _get_scale(pixel_spacing, N)

    # ── Midline comparison ───────────────────────────────────────
    gt_std_pt, gt_std_dir = compute_midline_std(gt)
    pred_std_pt, pred_std_dir = compute_midline_std(pred)

    gt_old_pt, gt_old_dir = compute_midline_old(gt)
    pred_old_pt, pred_old_dir = compute_midline_old(pred)

    # Angle differences
    std_angle_diff = compute_midline_angle_diff(gt_std_dir, pred_std_dir)
    old_angle_diff = compute_midline_angle_diff(gt_old_dir, pred_old_dir)

    # Cosine similarity (reference)
    std_cos_sim = compute_midline_cosine_sim(gt_std_dir, pred_std_dir)
    old_cos_sim = compute_midline_cosine_sim(gt_old_dir, pred_old_dir)

    # X displacement
    # ref_y for std: midpoint of latero-orbital R/L (y coord)
    ref_y_std = gt_std_pt[:, 1]
    std_x_disp = compute_midline_x_displacement(
        gt_std_pt, gt_std_dir, pred_std_pt, pred_std_dir, ref_y_std,
    ) * scale

    # ref_y for old: midpoint of CG-ANS (y coord)
    ref_y_old = gt_old_pt[:, 1]
    old_x_disp = compute_midline_x_displacement(
        gt_old_pt, gt_old_dir, pred_old_pt, pred_old_dir, ref_y_old,
    ) * scale

    # ── Deviation comparison ─────────────────────────────────────
    # Compute deviations using respective midlines
    gt_lf = compute_lowerface_deviation(gt, gt_std_pt, gt_std_dir)
    pred_lf = compute_lowerface_deviation(pred, pred_std_pt, pred_std_dir)

    gt_mf = compute_midface_deviation(gt, gt_std_pt, gt_std_dir)
    pred_mf = compute_midface_deviation(pred, pred_std_pt, pred_std_dir)

    gt_dental = compute_dental_deviation(gt, gt_std_pt, gt_std_dir)
    pred_dental = compute_dental_deviation(pred, pred_std_pt, pred_std_dir)

    gt_cant = compute_canting(gt, gt_std_pt, gt_std_dir)
    pred_cant = compute_canting(pred, pred_std_pt, pred_std_dir)

    return {
        # Midline angle differences (degrees)
        'midline_std_angle_diff_deg': std_angle_diff,
        'midline_std_x_disp_mm': std_x_disp,
        'midline_std_cos_sim': std_cos_sim,
        'midline_old_angle_diff_deg': old_angle_diff,
        'midline_old_x_disp_mm': old_x_disp,
        'midline_old_cos_sim': old_cos_sim,
        # Lowerface deviation
        'gt_lowerface_dev': gt_lf['distance'] * scale,
        'pred_lowerface_dev': pred_lf['distance'] * scale,
        'err_lowerface_dev': np.abs(
            gt_lf['distance'] * scale - pred_lf['distance'] * scale
        ),
        'gt_lowerface_dir': gt_lf['direction'],
        'pred_lowerface_dir': pred_lf['direction'],
        # Midface deviation
        'gt_midface_dist_R': gt_mf['dist_R'] * scale,
        'gt_midface_dist_L': gt_mf['dist_L'] * scale,
        'gt_midface_asymmetry': gt_mf['asymmetry'] * scale,
        'pred_midface_dist_R': pred_mf['dist_R'] * scale,
        'pred_midface_dist_L': pred_mf['dist_L'] * scale,
        'pred_midface_asymmetry': pred_mf['asymmetry'] * scale,
        'err_midface_asymmetry': np.abs(
            gt_mf['asymmetry'] * scale - pred_mf['asymmetry'] * scale
        ),
        # Dental deviation
        'gt_dental_upper_midline': gt_dental['upper_midline_dist'] * scale,
        'pred_dental_upper_midline': pred_dental['upper_midline_dist'] * scale,
        'err_dental_upper_midline': np.abs(
            gt_dental['upper_midline_dist'] * scale
            - pred_dental['upper_midline_dist'] * scale
        ),
        'gt_dental_crown_midpoint': gt_dental['crown_midpoint_dist'] * scale,
        'pred_dental_crown_midpoint': pred_dental['crown_midpoint_dist'] * scale,
        'err_dental_crown_midpoint': np.abs(
            gt_dental['crown_midpoint_dist'] * scale
            - pred_dental['crown_midpoint_dist'] * scale
        ),
        # Canting
        'gt_canting_diff': gt_cant['diff'] * scale,
        'pred_canting_diff': pred_cant['diff'] * scale,
        'err_canting_diff': np.abs(
            gt_cant['diff'] * scale - pred_cant['diff'] * scale
        ),
        'gt_canting_dir': gt_cant['direction'],
        'pred_canting_dir': pred_cant['direction'],
    }


def _get_scale(
    pixel_spacing: np.ndarray | float | None,
    n: int,
) -> np.ndarray:
    """Build per-sample scale factor from pixel_spacing."""
    if pixel_spacing is None:
        return np.ones(n)
    if isinstance(pixel_spacing, (int, float)):
        return np.full(n, pixel_spacing)
    return np.asarray(pixel_spacing)
