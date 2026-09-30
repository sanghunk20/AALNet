"""Asymmetry loss (L_asym in the paper).

Supervises, on the predicted coordinates, the direction of the two reference
midlines and the five deviation measurements used in facial-asymmetry assessment.

Naming used in the code:
    midline_std  = MSR, the perpendicular bisector of the latero-orbitale pair
    midline_old  = the crista galli-ANS (Cg-ANS) line
    lowerface    = lower-face deviation (menton to MSR)
    midface      = midface asymmetry (right vs. left zygoma to MSR)
    dental       = upper dental midline, and maxillary central-incisor crown midpoint, to MSR
    canting      = occlusal canting (maxillary first-molar crowns projected along the MSR)
"""

import torch
import torch.nn as nn

from ..utils.clinical_metrics import (
    LM_LATERO_ORBITAL_R,
    LM_LATERO_ORBITAL_L,
    LM_CRISTA_GALLI,
    LM_ANS,
    LM_MENTON,
    LM_ZYGOMA_R,
    LM_ZYGOMA_L,
    LM_UPPER_MIDLINE,
    LM_MAXILLARY_1_CROWN_R,
    LM_MAXILLARY_1_CROWN_L,
    LM_MAXILLARY_6_CROWN_R,
    LM_MAXILLARY_6_CROWN_L,
)


# ── Geometry helpers (differentiable) ──────────────────────────────


def _compute_midline_std(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute standard midline: perpendicular bisector of latero-orbital R/L.

    Origin: midpoint of Lo R and Lo L.
    Direction: perpendicular to R→L vector (90° CCW rotation).

    Args:
        coords: [B, L, 2] landmark coordinates.

    Returns:
        (midpoint [B, 2], direction [B, 2]) unit direction vector.
    """
    lo_r = coords[:, LM_LATERO_ORBITAL_R]  # [B, 2]
    lo_l = coords[:, LM_LATERO_ORBITAL_L]  # [B, 2]

    midpoint = (lo_r + lo_l) / 2.0

    # R → L vector, then 90° CCW rotation: (dx, dy) → (-dy, dx)
    rl = lo_l - lo_r  # [B, 2]
    perp = torch.stack([-rl[:, 1], rl[:, 0]], dim=-1)  # [B, 2]
    norm = torch.clamp(torch.norm(perp, dim=-1, keepdim=True), min=1e-8)
    direction = perp / norm

    return midpoint, direction


def _compute_midline_old(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute old midline: crista galli → ANS direction.

    Anchor point: midpoint of crista galli and ANS.
    Direction: CG → ANS (normalized). Reversing to ANS→CG produces
    opposite direction, which should yield higher midline loss.

    Args:
        coords: [B, L, 2] landmark coordinates.

    Returns:
        (midpoint [B, 2], direction [B, 2]) unit direction vector.
    """
    cg = coords[:, LM_CRISTA_GALLI]  # [B, 2]
    ans = coords[:, LM_ANS]          # [B, 2]

    direction = ans - cg
    norm = torch.clamp(torch.norm(direction, dim=-1, keepdim=True), min=1e-8)
    direction = direction / norm

    midpoint = (cg + ans) / 2.0

    return midpoint, direction


def _ensure_downward(direction: torch.Tensor) -> torch.Tensor:
    """Ensure direction vectors point downward (dy >= 0).

    Used only inside _signed_distance for consistent sign convention
    (patient Right = positive). NOT used for midline loss computation
    where direction itself carries meaning.

    Args:
        direction: [B, 2] direction vectors.

    Returns:
        [B, 2] direction vectors with dy >= 0.
    """
    flip_mask = direction[:, 1:2] < 0  # [B, 1]
    return torch.where(flip_mask, -direction, direction)


def _signed_distance(
    point: torch.Tensor,
    anchor: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Compute signed perpendicular distance from point to line.

    Sign convention: positive = patient Right (image left in PA ceph).
    Direction is normalized to point downward before computing cross product.

    Formula: signed_dist = -(v × d) where v = point - anchor
    With downward d: image-left (patient-right) gives negative cross → negated to positive.

    Args:
        point: [B, 2] point coordinates.
        anchor: [B, 2] point on the line.
        direction: [B, 2] unit direction of the line.

    Returns:
        [B] signed distances.
    """
    d = _ensure_downward(direction)
    v = point - anchor
    cross = v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0]
    return -cross



# ── AsymmetryLoss ─────────────────────────────────────────────────────


class AsymmetryLoss(nn.Module):
    """Asymmetry loss between predicted and GT landmark coordinates.

        L_asym = alpha_midline_std * (1 - cos theta_MSR)
               + alpha_midline_old * (1 - cos theta_Cg-ANS)
               + sum_k alpha_k * |d_k(pred) - d_k(gt)|

    where theta is the angle between the predicted and GT midline directions and
    d_k are the five deviation measurements. Predicted deviations are measured
    against the MSR built from the predicted landmarks, GT deviations against the
    GT MSR. Coordinates are expected in the normalized [-1, 1] space of the base
    loss. The loss is returned unweighted; the trainer multiplies it by lambda.

    Args:
        alpha_midline_std: Weight of the MSR direction term.
        alpha_midline_old: Weight of the Cg-ANS direction term.
        alpha_midface: Weight of the midface asymmetry term.
        alpha_lowerface: Weight of the lower-face deviation term.
        alpha_dental: Weight of the dental deviation terms (2 terms).
        alpha_canting: Weight of the occlusal canting term.
    """

    def __init__(
        self,
        alpha_midline_std: float = 0.6,
        alpha_midline_old: float = 0.4,
        alpha_midface: float = 1.0,
        alpha_lowerface: float = 1.0,
        alpha_dental: float = 1.0,
        alpha_canting: float = 1.0,
    ):
        super().__init__()
        self.alpha_midline_std = alpha_midline_std
        self.alpha_midline_old = alpha_midline_old
        self.alpha_midface = alpha_midface
        self.alpha_lowerface = alpha_lowerface
        self.alpha_dental = alpha_dental
        self.alpha_canting = alpha_canting

    def _midline_loss(
        self,
        dir_gt: torch.Tensor,
        dir_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute midline direction loss.

        Direction vectors are used as-is (no sign normalization).
        Origin is fixed (Lo midpoint for std, CG for old), so direction
        is consistent. Opposite direction → large loss (intended).

        Args:
            dir_gt: [B, 2] GT midline unit direction.
            dir_pred: [B, 2] predicted midline unit direction.

        Returns:
            Scalar loss (mean over batch).
        """
        # 1 - cos(θ): 0 when aligned, 2 when opposite
        dot = (dir_gt * dir_pred).sum(dim=-1)  # [B]
        return (1.0 - dot).mean()

    def _deviation_losses(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_point_std: torch.Tensor,
        gt_dir_std: torch.Tensor,
        pred_point_std: torch.Tensor,
        pred_dir_std: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all deviation sub-losses.

        Args:
            pred_coords: [B, 33, 2] predicted coords.
            gt_coords: [B, 33, 2] GT coords.
            gt_point_std: [B, 2] GT midline_std point (precomputed).
            gt_dir_std: [B, 2] GT midline_std direction (precomputed).
            pred_point_std: [B, 2] pred midline_std point (precomputed).
            pred_dir_std: [B, 2] pred midline_std direction (precomputed).

        Returns:
            Dict of scalar losses: midface, lowerface, dental, canting.
        """
        # --- Midface deviation: asymmetry = signed_dist(zyR) + signed_dist(zyL) ---
        # Sign convention: zyR → +, zyL → −. Sum ≈ 0 when symmetric.
        gt_zy_r = _signed_distance(gt_coords[:, LM_ZYGOMA_R], gt_point_std, gt_dir_std)
        gt_zy_l = _signed_distance(gt_coords[:, LM_ZYGOMA_L], gt_point_std, gt_dir_std)
        gt_midface_diff = gt_zy_r + gt_zy_l

        pred_zy_r = _signed_distance(pred_coords[:, LM_ZYGOMA_R], pred_point_std, pred_dir_std)
        pred_zy_l = _signed_distance(pred_coords[:, LM_ZYGOMA_L], pred_point_std, pred_dir_std)
        pred_midface_diff = pred_zy_r + pred_zy_l

        loss_midface = (gt_midface_diff - pred_midface_diff).abs().mean()

        # --- Lowerface deviation: signed_dist(menton) ---
        gt_me = _signed_distance(gt_coords[:, LM_MENTON], gt_point_std, gt_dir_std)
        pred_me = _signed_distance(pred_coords[:, LM_MENTON], pred_point_std, pred_dir_std)
        loss_lowerface = (gt_me - pred_me).abs().mean()

        # --- Dental deviation (2 terms) ---
        # Term 1: upper midline
        gt_um = _signed_distance(gt_coords[:, LM_UPPER_MIDLINE], gt_point_std, gt_dir_std)
        pred_um = _signed_distance(pred_coords[:, LM_UPPER_MIDLINE], pred_point_std, pred_dir_std)
        loss_dental_1 = (gt_um - pred_um).abs().mean()

        # Term 2: maxillary 1 crown midpoint
        gt_crown_mid = (gt_coords[:, LM_MAXILLARY_1_CROWN_R] + gt_coords[:, LM_MAXILLARY_1_CROWN_L]) / 2.0
        pred_crown_mid = (pred_coords[:, LM_MAXILLARY_1_CROWN_R] + pred_coords[:, LM_MAXILLARY_1_CROWN_L]) / 2.0
        gt_cm = _signed_distance(gt_crown_mid, gt_point_std, gt_dir_std)
        pred_cm = _signed_distance(pred_crown_mid, pred_point_std, pred_dir_std)
        loss_dental_2 = (gt_cm - pred_cm).abs().mean()

        loss_dental = loss_dental_1 + loss_dental_2

        # --- Canting: signed distance along midline between R/L projections ---
        # t = (q - a) · d gives scalar projection onto midline direction.
        # Positive when R lies further along the midline than L, negative otherwise (sign kept).
        gt_m6r = gt_coords[:, LM_MAXILLARY_6_CROWN_R]
        gt_m6l = gt_coords[:, LM_MAXILLARY_6_CROWN_L]
        gt_cant = ((gt_m6r - gt_point_std) * gt_dir_std).sum(dim=-1) \
                - ((gt_m6l - gt_point_std) * gt_dir_std).sum(dim=-1)

        pred_m6r = pred_coords[:, LM_MAXILLARY_6_CROWN_R]
        pred_m6l = pred_coords[:, LM_MAXILLARY_6_CROWN_L]
        pred_cant = ((pred_m6r - pred_point_std) * pred_dir_std).sum(dim=-1) \
                  - ((pred_m6l - pred_point_std) * pred_dir_std).sum(dim=-1)

        loss_canting = (gt_cant - pred_cant).abs().mean()

        return {
            'midface': loss_midface,
            'lowerface': loss_lowerface,
            'dental': loss_dental,
            'canting': loss_canting,
        }

    def forward(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute the asymmetry loss.

        Args:
            pred_coords: [B, 33, 2] predicted coordinates.
                Must have gradient (from differentiable soft-argmax).
            gt_coords: [B, 33, 2] GT coordinates.
            return_details: If True, also return dict of per-term loss values.

        Returns:
            Scalar loss (raw, not weighted by lambda).
            If return_details: (total_loss, details_dict).
        """
        # Compute midlines once (reused for both midline loss and deviation loss)
        gt_point_std, gt_dir_std = _compute_midline_std(gt_coords)
        pred_point_std, pred_dir_std = _compute_midline_std(pred_coords)
        _, gt_dir_old = _compute_midline_old(gt_coords)
        _, pred_dir_old = _compute_midline_old(pred_coords)

        # Midline losses
        loss_midline_std = self._midline_loss(gt_dir_std, pred_dir_std)
        loss_midline_old = self._midline_loss(gt_dir_old, pred_dir_old)

        # Deviation losses (pass precomputed midlines)
        dev_losses = self._deviation_losses(
            pred_coords, gt_coords,
            gt_point_std, gt_dir_std,
            pred_point_std, pred_dir_std,
        )

        # Weighted sum
        total = (
            self.alpha_midline_std * loss_midline_std
            + self.alpha_midline_old * loss_midline_old
            + self.alpha_midface * dev_losses['midface']
            + self.alpha_lowerface * dev_losses['lowerface']
            + self.alpha_dental * dev_losses['dental']
            + self.alpha_canting * dev_losses['canting']
        )

        if return_details:
            details = {
                'midline_std': loss_midline_std.item(),
                'midline_old': loss_midline_old.item(),
                'midface': dev_losses['midface'].item(),
                'lowerface': dev_losses['lowerface'].item(),
                'dental': dev_losses['dental'].item(),
                'canting': dev_losses['canting'].item(),
            }
            return total, details

        return total
