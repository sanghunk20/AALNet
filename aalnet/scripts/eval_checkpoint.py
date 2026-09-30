"""Evaluate best checkpoints and save per-sample MRE (mm) to CSV.

Iterates over all model×fold combinations, loads checkpoint_best.pth,
runs inference on the specified split (val or test), converts predictions
to original pixel space, and computes per-sample MRE in mm.

Output per model/fold ({split} = test or val):
    - {split}_per_sample_mre_mm.csv: filename, mre_mm, landmark1_mm, ...
    - {split}_per_sample_axis_mae_mm.csv: filename, x_mae_mm, y_mae_mm, lm1_x, lm1_y, ...
    - {split}_per_sample_clinical_mm.csv: asymmetry measurements and their errors

Output aggregated across folds (in {output_root}/eval/, or --eval_root):
    - summary_mre_mm_{split}.csv: config, fold, mre_mean_mm, mre_std_mm, SDR@2/2.5/3/4mm
    - fold_aggregated_summary_{split}.csv: config, mean_mre, std_mre, SDR averages
    - fold_aggregated_per_landmark_{split}.csv: config, landmark, mre_mean_mm
    - fold_aggregated_per_axis_{split}.csv: config, landmark, x_mae_mm, y_mae_mm
    - fold_aggregated_clinical_{split}.csv: config, measurement, error mean and std

Expected layout of --output_root (as written by scripts/train.py):
    {output_root}/{config_name}/fold{N}/checkpoint_best.pth
    {output_root}/{config_name}/fold{N}/config.json

Usage (run from the repository root):
    # Evaluate on the fixed test set (default)
    python -m aalnet.scripts.eval_checkpoint \
        --output_root outputs \
        --data_root /path/to/landmark_detect_800 \
        --pixel_spacing_file /path/to/pixel_spacing_per_image.json

    # Evaluate on the validation set of each fold
    python -m aalnet.scripts.eval_checkpoint \
        --output_root outputs \
        --data_root /path/to/landmark_detect_800 \
        --pixel_spacing_file /path/to/pixel_spacing_per_image.json \
        --split val

    # Specific models and folds
    python -m aalnet.scripts.eval_checkpoint \
        --output_root outputs \
        --data_root /path/to/landmark_detect_800 \
        --pixel_spacing_file /path/to/pixel_spacing_per_image.json \
        --only lambda02 withoutAsyLoss --folds 0 1 2 3 4
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from contextlib import nullcontext
from torch.amp import autocast
from torch.utils.data import DataLoader

from ..datasets.landmark_dataset import LandmarkDataset
from ..models.aalnet import AALNet
from ..utils.coord_utils import batch_coords_to_original
from ..utils.pixel_spacing import load_pixel_spacing, get_pixel_spacing_array
from ..utils.metrics import compute_mre, compute_sdr, compute_per_axis_mae
from ..utils.clinical_metrics import compute_clinical_errors
from ..utils.visualize import save_vis_batch_original


def load_checkpoint_config(checkpoint_dir: Path) -> dict:
    """Load saved config.json from a training run directory."""
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
    with open(config_path, 'r') as f:
        return json.load(f)


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run inference and collect predictions.

    Returns:
        Tuple of (pred_coords, gt_coords, filenames).
        pred_coords: [N, L, 2] in pixels of the preprocessed (letterboxed) image.
        gt_coords: [N, L, 2] in the same space.
        filenames: [N] list of filename stems.
    """
    model.eval()
    all_pred = []
    all_gt = []
    all_filenames = []

    amp_ctx = autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

    with torch.no_grad():
        for images, gt_coords, filenames in data_loader:
            images = images.to(device, non_blocking=True)

            with amp_ctx:
                output = model(images)

            pred_coords = model.get_coordinates(output)  # [B, N, 2] in input pixels

            all_pred.append(pred_coords.float().cpu().numpy())
            all_gt.append(gt_coords.float().cpu().numpy())
            all_filenames.extend(filenames)

    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_gt, axis=0),
        all_filenames,
    )


def save_per_sample_csv(
    output_path: Path,
    filenames: list[str],
    radial_errors_mm: np.ndarray,
    landmark_names: list[str],
):
    """Save per-sample MRE (mm) as CSV.

    Columns: filename, mre_mm, landmark1_mm, landmark2_mm, ...
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'mre_mm'] + landmark_names)
        for i, fname in enumerate(filenames):
            sample_errors = radial_errors_mm[i]  # [L]
            mre = float(sample_errors.mean())
            row = [fname, f'{mre:.4f}'] + [f'{e:.4f}' for e in sample_errors]
            writer.writerow(row)

    print(f"  Saved {len(filenames)} samples to {output_path}")


def save_per_sample_axis_csv(
    output_path: Path,
    filenames: list[str],
    x_errors_mm: np.ndarray,
    y_errors_mm: np.ndarray,
    landmark_names: list[str],
):
    """Save per-sample per-axis MAE (mm) as CSV.

    Columns: filename, x_mae_mm, y_mae_mm, lm1_x, lm1_y, lm2_x, lm2_y, ...
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ['filename', 'x_mae_mm', 'y_mae_mm']
    for name in landmark_names:
        header.extend([f'{name}_x', f'{name}_y'])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, fname in enumerate(filenames):
            x_mae = float(x_errors_mm[i].mean())
            y_mae = float(y_errors_mm[i].mean())
            row = [fname, f'{x_mae:.4f}', f'{y_mae:.4f}']
            for j in range(len(landmark_names)):
                row.extend([f'{x_errors_mm[i, j]:.4f}', f'{y_errors_mm[i, j]:.4f}'])
            writer.writerow(row)

    print(f"  Saved per-axis errors to {output_path}")


def save_per_sample_clinical_csv(
    output_path: Path,
    filenames: list[str],
    clinical: dict,
):
    """Save per-sample asymmetry measurements and their errors as CSV.

    Columns: midline comparison (midline_std = MSR, midline_old = Cg-ANS line)
    and the ground-truth value, predicted value and error of each deviation
    measurement.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        'filename',
        'midline_std_angle_diff_deg', 'midline_std_x_disp_mm', 'midline_std_cos_sim',
        'midline_old_angle_diff_deg', 'midline_old_x_disp_mm', 'midline_old_cos_sim',
        'gt_lowerface_dev', 'pred_lowerface_dev', 'err_lowerface_dev',
        'gt_lowerface_dir', 'pred_lowerface_dir',
        'gt_midface_dist_R', 'gt_midface_dist_L', 'gt_midface_asymmetry',
        'pred_midface_dist_R', 'pred_midface_dist_L', 'pred_midface_asymmetry',
        'err_midface_asymmetry',
        'gt_dental_upper_midline', 'pred_dental_upper_midline', 'err_dental_upper_midline',
        'gt_dental_crown_midpoint', 'pred_dental_crown_midpoint', 'err_dental_crown_midpoint',
        'gt_canting_diff', 'pred_canting_diff', 'err_canting_diff',
        'gt_canting_dir', 'pred_canting_dir',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, fname in enumerate(filenames):
            row = [fname]
            for col in columns[1:]:
                val = clinical[col][i]
                if isinstance(val, (float, np.floating)):
                    row.append(f'{val:.4f}')
                else:
                    row.append(str(val))
            writer.writerow(row)

    print(f"  Saved clinical metrics to {output_path}")


def get_eval_files(
    data_root: str,
    split: str,
    fold: int,
) -> list[str]:
    """Get file list for evaluation based on split type.

    Args:
        data_root: Path to dataset root.
        split: 'test' or 'val'.
        fold: Fold index (only used for 'val' split).

    Returns:
        List of filename stems.
    """
    splits_path = Path(data_root) / 'fold_splits.json'
    with open(splits_path) as f:
        fold_splits = json.load(f)

    if split == 'test':
        return fold_splits['test_files']
    else:
        return fold_splits['folds'][fold]['val_files']


def evaluate_single(
    checkpoint_dir: Path,
    data_root: str,
    pixel_spacing_map: dict[str, float],
    device: torch.device,
    split: str = 'test',
    batch_size: int = 8,
    num_workers: int = 4,
    vis_dir: Path | None = None,
    original_images_dir: str | None = None,
    output_dir: Path | None = None,
) -> dict | None:
    """Evaluate a single model/fold checkpoint.

    Returns:
        Dict with MRE, SDR, per-landmark, and per-axis results,
        or None if checkpoint not found.
    """
    ckpt_path = checkpoint_dir / 'checkpoint_best.pth'
    if not ckpt_path.exists():
        print(f"  [SKIP] No checkpoint_best.pth in {checkpoint_dir}")
        return None

    # Load training config
    config = load_checkpoint_config(checkpoint_dir)

    # Get evaluation file list
    fold = config.get('fold', 0)
    eval_files = get_eval_files(data_root, split, fold)

    eval_dataset = LandmarkDataset(data_root, eval_files, augment=False)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model (weights come from the checkpoint)
    model = AALNet(
        num_landmarks=config.get('num_landmarks', 33),
        pretrained=False,
        decoder_drop_path=config.get('mlp_fpn_drop_path', 0.275),
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    best_epoch = ckpt.get('epoch', -1)
    best_mre_train = ckpt.get('best_mre', float('nan'))

    print(f"  Loaded checkpoint: epoch={best_epoch}, train_best_mre={best_mre_train:.4f}")

    # Run inference
    pred_lb, gt_lb, filenames = run_inference(model, eval_loader, device)

    # Map from the letterboxed image back to the original image
    params_dir = str(Path(data_root) / 'transform_params')
    pred_orig = batch_coords_to_original(pred_lb, filenames, params_dir)
    gt_orig = batch_coords_to_original(gt_lb, filenames, params_dir)

    # Get per-image pixel spacing
    spacing_array = get_pixel_spacing_array(
        filenames, pixel_spacing_map, allow_missing=True,
    )

    # Compute metrics in mm
    mre_results = compute_mre(pred_orig, gt_orig, spacing_array)
    sdr_results = compute_sdr(pred_orig, gt_orig, spacing_array)
    axis_results = compute_per_axis_mae(pred_orig, gt_orig, spacing_array)

    # Compute clinical metrics
    clinical_results = compute_clinical_errors(pred_orig, gt_orig, spacing_array)

    # Save visualization images (original resolution)
    if vis_dir is not None and original_images_dir is not None:
        save_vis_batch_original(
            pred_orig, gt_orig, filenames, eval_dataset.landmark_names,
            original_images_dir, vis_dir,
            spacing_array=spacing_array,
        )

    # Save per-sample CSVs
    csv_dir = output_dir if output_dir is not None else checkpoint_dir
    csv_prefix = f'{split}_per_sample'
    save_per_sample_csv(
        csv_dir / f'{csv_prefix}_mre_mm.csv',
        filenames,
        mre_results['radial_errors_mm'],
        eval_dataset.landmark_names,
    )
    save_per_sample_axis_csv(
        csv_dir / f'{csv_prefix}_axis_mae_mm.csv',
        filenames,
        axis_results['x_errors_mm'],
        axis_results['y_errors_mm'],
        eval_dataset.landmark_names,
    )
    save_per_sample_clinical_csv(
        csv_dir / f'{csv_prefix}_clinical_mm.csv',
        filenames,
        clinical_results,
    )

    return {
        'mre_mean': mre_results['mre_mean'],
        'mre_std': mre_results['mre_std'],
        'mre_per_landmark': mre_results['mre_per_landmark'],
        'x_mae_per_landmark': axis_results['x_mae_per_landmark'],
        'y_mae_per_landmark': axis_results['y_mae_per_landmark'],
        'num_samples': len(filenames),
        'best_epoch': best_epoch,
        'landmark_names': eval_dataset.landmark_names,
        'clinical': clinical_results,
        **sdr_results,
    }


def discover_experiments(output_root: Path) -> list[tuple[str, int, Path]]:
    """Auto-discover model×fold directories under output_root.

    Expects structure: output_root/{config_name}/fold{N}/checkpoint_best.pth

    Returns:
        List of (config_name, fold, checkpoint_dir) tuples.
    """
    experiments = []
    if not output_root.exists():
        return experiments

    for config_dir in sorted(output_root.iterdir()):
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue
        if config_dir.name in ('summary', 'plots', 'eval'):
            continue

        for fold_dir in sorted(config_dir.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith('fold'):
                continue
            try:
                fold_num = int(fold_dir.name.replace('fold', ''))
            except ValueError:
                continue

            if (fold_dir / 'checkpoint_best.pth').exists():
                experiments.append((config_dir.name, fold_num, fold_dir))

    return experiments


# ── Fold aggregation helpers ─────────────────────────────────────────


def aggregate_folds(
    all_results: list[dict],
    eval_dir: Path,
    split: str,
):
    """Aggregate per-fold results into cross-fold summary CSVs.

    Produces:
        - fold_aggregated_summary.csv: per-config mean±std of MRE and SDR
        - fold_aggregated_per_landmark.csv: per-config per-landmark MRE
        - fold_aggregated_per_axis.csv: per-config per-landmark x/y MAE
    """
    if not all_results:
        return

    # Group by config name
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in all_results:
        by_config[row['config']].append(row)

    landmark_names = all_results[0].get('landmark_names', [])
    num_landmarks = len(landmark_names)

    # ── 1. Aggregated summary (MRE + SDR) ───────────────────────────
    agg_rows = []
    for config_name, folds in sorted(by_config.items()):
        mres = [f['mre_mean_mm'] for f in folds]
        agg = {
            'config': config_name,
            'num_folds': len(folds),
            'mre_mean_mm': float(np.mean(mres)),
            'mre_std_across_folds': float(np.std(mres)),
        }
        # Average SDR across folds
        for sdr_key in ['sdr_2.0mm', 'sdr_2.5mm', 'sdr_3.0mm', 'sdr_4.0mm']:
            vals = [f[sdr_key] for f in folds if sdr_key in f]
            if vals:
                agg[sdr_key] = float(np.mean(vals))
        agg_rows.append(agg)

    summary_path = eval_dir / f'fold_aggregated_summary_{split}.csv'
    _write_dict_csv(summary_path, agg_rows)
    print(f"\nFold-aggregated summary saved to {summary_path}")

    # Print table
    print(f"\n{'Config':<40} {'Folds':>5} {'MRE(mm)':>9} {'±fold':>7} "
          f"{'SDR@2':>6} {'SDR@2.5':>7} {'SDR@3':>6} {'SDR@4':>6}")
    print('-' * 90)
    for row in agg_rows:
        print(f"{row['config']:<40} {row['num_folds']:>5} "
              f"{row['mre_mean_mm']:>9.4f} {row['mre_std_across_folds']:>7.4f} "
              f"{row.get('sdr_2.0mm', 0):>6.1f} {row.get('sdr_2.5mm', 0):>7.1f} "
              f"{row.get('sdr_3.0mm', 0):>6.1f} {row.get('sdr_4.0mm', 0):>6.1f}")

    if num_landmarks == 0:
        return

    # ── 2. Per-landmark MRE aggregated across folds ─────────────────
    lm_rows = []
    for config_name, folds in sorted(by_config.items()):
        # Stack per-landmark arrays: [num_folds, L]
        stacked = np.array([f['mre_per_landmark'] for f in folds])
        mean_per_lm = stacked.mean(axis=0)  # [L]
        std_per_lm = stacked.std(axis=0)  # [L]
        for j, lm_name in enumerate(landmark_names):
            lm_rows.append({
                'config': config_name,
                'landmark': lm_name,
                'mre_mean_mm': float(mean_per_lm[j]),
                'mre_std_mm': float(std_per_lm[j]),
            })

    lm_path = eval_dir / f'fold_aggregated_per_landmark_{split}.csv'
    _write_dict_csv(lm_path, lm_rows)
    print(f"Per-landmark aggregation saved to {lm_path}")

    # ── 3. Per-landmark per-axis MAE aggregated across folds ────────
    axis_rows = []
    for config_name, folds in sorted(by_config.items()):
        x_stacked = np.array([f['x_mae_per_landmark'] for f in folds])
        y_stacked = np.array([f['y_mae_per_landmark'] for f in folds])
        x_mean = x_stacked.mean(axis=0)  # [L]
        y_mean = y_stacked.mean(axis=0)  # [L]
        x_std = x_stacked.std(axis=0)
        y_std = y_stacked.std(axis=0)
        for j, lm_name in enumerate(landmark_names):
            axis_rows.append({
                'config': config_name,
                'landmark': lm_name,
                'x_mae_mean_mm': float(x_mean[j]),
                'x_mae_std_mm': float(x_std[j]),
                'y_mae_mean_mm': float(y_mean[j]),
                'y_mae_std_mm': float(y_std[j]),
            })

    axis_path = eval_dir / f'fold_aggregated_per_axis_{split}.csv'
    _write_dict_csv(axis_path, axis_rows)
    print(f"Per-axis aggregation saved to {axis_path}")

    # ── 4. Clinical metrics aggregated across folds ──────────────
    clinical_metric_keys = [
        ('midline_std_angle', 'midline_std_angle_diff_deg'),
        ('midline_std_x_disp', 'midline_std_x_disp_mm'),
        ('midline_old_angle', 'midline_old_angle_diff_deg'),
        ('midline_old_x_disp', 'midline_old_x_disp_mm'),
        ('lowerface_dev', 'err_lowerface_dev'),
        ('midface_asymmetry', 'err_midface_asymmetry'),
        ('dental_upper_midline', 'err_dental_upper_midline'),
        ('dental_crown_midpoint', 'err_dental_crown_midpoint'),
        ('canting_diff', 'err_canting_diff'),
    ]

    clinical_rows = []
    for config_name, folds in sorted(by_config.items()):
        for metric_type, key in clinical_metric_keys:
            values_per_fold = []
            for f in folds:
                if 'clinical' in f and key in f['clinical']:
                    vals = f['clinical'][key]
                    values_per_fold.append(float(np.nanmean(vals)))

            if values_per_fold:
                clinical_rows.append({
                    'config': config_name,
                    'metric_type': metric_type,
                    'error_mean': float(np.mean(values_per_fold)),
                    'error_std': float(np.std(values_per_fold)),
                })

    if clinical_rows:
        clinical_path = eval_dir / f'fold_aggregated_clinical_{split}.csv'
        _write_dict_csv(clinical_path, clinical_rows)
        print(f"Clinical aggregation saved to {clinical_path}")


def _write_dict_csv(path: Path, rows: list[dict]):
    """Write list of dicts to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate best checkpoints → per-sample MRE (mm) CSV'
    )
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory containing {config}/fold{N}/ outputs '
                             'of scripts/train.py')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the preprocessed dataset root')
    parser.add_argument('--pixel_spacing_file', type=str, required=True,
                        help='Path to per-image pixel spacing JSON '
                             '(from build_pixel_spacing_map.py)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val'],
                        help='Evaluation split: test (fixed test set) '
                             'or val (fold-specific)')
    parser.add_argument('--only', type=str, nargs='+', default=None,
                        help='Only evaluate these config names '
                             '(e.g. --only lambda02 withoutAsyLoss)')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Evaluate only these folds (e.g. --folds 0 1 2)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_vis', action='store_true',
                        help='Disable visualization image generation')
    parser.add_argument('--original_images_dir', type=str, default=None,
                        help='Directory with original-resolution images '
                             '(default: {data_root}/../landmark_detect/images/raw)')
    parser.add_argument('--eval_root', type=str, default=None,
                        help='If provided, write all eval outputs here instead of '
                             'next to checkpoints. --output_root is still used '
                             'for checkpoint discovery only.')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    eval_root = Path(args.eval_root) if args.eval_root else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve original images directory for visualization
    original_images_dir = args.original_images_dir
    if original_images_dir is None and not args.no_vis:
        default_dir = str(Path(args.data_root).resolve().parent / 'landmark_detect' / 'images' / 'raw')
        if Path(default_dir).is_dir():
            original_images_dir = default_dir

    # Load pixel spacing
    pixel_spacing_map = load_pixel_spacing(args.pixel_spacing_file)
    print(f"Loaded pixel spacing for {len(pixel_spacing_map)} images")
    print(f"Evaluation split: {args.split}")
    if not args.no_vis and original_images_dir:
        print(f"Visualization: ON (images from {original_images_dir})")
    else:
        print(f"Visualization: OFF")

    # Discover experiments
    experiments = discover_experiments(output_root)
    if not experiments:
        print(f"[ERROR] No experiments found in {output_root}", file=sys.stderr)
        sys.exit(1)

    # Filter
    if args.only:
        only_set = set(args.only)
        experiments = [(n, f, p) for n, f, p in experiments if n in only_set]
    if args.folds is not None:
        fold_set = set(args.folds)
        experiments = [(n, f, p) for n, f, p in experiments if f in fold_set]

    print(f"Found {len(experiments)} experiment(s) to evaluate")
    print(f"Device: {device}")
    print()

    # Evaluate all
    summary_rows = []
    all_results = []  # For fold aggregation (includes per-landmark arrays)

    for config_name, fold, ckpt_dir in experiments:
        print(f"[{config_name} / fold{fold}]")

        if eval_root is not None:
            eval_fold_dir = eval_root / config_name / f'fold{fold}'
        else:
            eval_fold_dir = ckpt_dir

        vis_dir = eval_fold_dir / f'vis_{args.split}' if (not args.no_vis and original_images_dir) else None

        result = evaluate_single(
            checkpoint_dir=ckpt_dir,
            data_root=args.data_root,
            pixel_spacing_map=pixel_spacing_map,
            device=device,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            vis_dir=vis_dir,
            original_images_dir=original_images_dir,
            output_dir=eval_fold_dir if eval_root else None,
        )

        if result is not None:
            print(f"  MRE: {result['mre_mean']:.4f} ± {result['mre_std']:.4f} mm "
                  f"(N={result['num_samples']}, epoch={result['best_epoch']})")
            print(f"  SDR@2mm={result.get('sdr_2.0mm', 0):.1f}% "
                  f"SDR@2.5mm={result.get('sdr_2.5mm', 0):.1f}% "
                  f"SDR@3mm={result.get('sdr_3.0mm', 0):.1f}% "
                  f"SDR@4mm={result.get('sdr_4.0mm', 0):.1f}%")

            row = {
                'config': config_name,
                'fold': fold,
                'mre_mean_mm': result['mre_mean'],
                'mre_std_mm': result['mre_std'],
                'sdr_2.0mm': result.get('sdr_2.0mm', 0),
                'sdr_2.5mm': result.get('sdr_2.5mm', 0),
                'sdr_3.0mm': result.get('sdr_3.0mm', 0),
                'sdr_4.0mm': result.get('sdr_4.0mm', 0),
                'num_samples': result['num_samples'],
                'best_epoch': result['best_epoch'],
            }
            summary_rows.append(row)

            # Keep full result for aggregation
            all_results.append({
                **row,
                'mre_per_landmark': result['mre_per_landmark'],
                'x_mae_per_landmark': result['x_mae_per_landmark'],
                'y_mae_per_landmark': result['y_mae_per_landmark'],
                'landmark_names': result['landmark_names'],
                'clinical': result.get('clinical'),
            })
        print()

    # Save per-fold summary CSV
    eval_dir = eval_root if eval_root is not None else output_root / 'eval'
    if summary_rows:
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_path = eval_dir / f'summary_mre_mm_{args.split}.csv'
        _write_dict_csv(summary_path, summary_rows)

        print(f"Per-fold summary saved to {summary_path}")
        print()

        # Print per-fold table
        print(f"{'Config':<40} {'Fold':>4} {'MRE(mm)':>9} {'±std':>7} "
              f"{'SDR@2':>6} {'SDR@2.5':>7} {'SDR@3':>6} {'SDR@4':>6} "
              f"{'N':>5} {'Epoch':>6}")
        print('-' * 100)
        for row in summary_rows:
            print(f"{row['config']:<40} {row['fold']:>4} "
                  f"{row['mre_mean_mm']:>9.4f} {row['mre_std_mm']:>7.4f} "
                  f"{row['sdr_2.0mm']:>6.1f} {row['sdr_2.5mm']:>7.1f} "
                  f"{row['sdr_3.0mm']:>6.1f} {row['sdr_4.0mm']:>6.1f} "
                  f"{row['num_samples']:>5} {row['best_epoch']:>6}")

    # Fold aggregation
    if all_results:
        aggregate_folds(all_results, eval_dir, args.split)


if __name__ == '__main__':
    main()
