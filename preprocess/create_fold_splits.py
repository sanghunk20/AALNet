"""Create patient-level fold splits for cross-validation.

Generates a fold_splits.json file that defines:
- Fixed test set (default 10%, patient-level)
- 9-fold CV on the remaining 90% (train:val = 8:1 per fold)

Same patient's all timepoints (init/preop/pod1y/...) are always
in the same partition — no patient-level leakage.

The paper trains five models (folds 0-4) and reports their mean on the
fixed test set.

Usage:
    python preprocess/create_fold_splits.py \
        --data_dir /path/to/landmark_detect_800

    # Custom settings
    python preprocess/create_fold_splits.py \
        --data_dir /path/to/landmark_detect_800 \
        --num_folds 9 --test_ratio 0.1 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def extract_patient_id(filename: str) -> str:
    """Extract patient_id from filename.

    Filename format: {patient_id}_{timepoint}.jpg
    e.g. "P0001_pod1y.jpg" -> "P0001"
         "P0002_init.jpg"  -> "P0002"
    """
    stem = Path(filename).stem  # remove .jpg
    parts = stem.split('_')
    return '_'.join(parts[:-1])  # everything before last '_'


def create_fold_splits(
    data_dir: str,
    num_folds: int = 9,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_name: str = 'fold_splits.json',
):
    data_dir = Path(data_dir)
    img_dir = data_dir / 'images'

    all_files = sorted([f.name for f in img_dir.glob('*.jpg')])
    if not all_files:
        raise FileNotFoundError(f"No .jpg files found in {img_dir}")

    # Group files by patient_id
    patient_to_files: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        pid = extract_patient_id(f)
        patient_to_files[pid].append(f)

    patient_ids = sorted(patient_to_files.keys())
    total_patients = len(patient_ids)

    # Shuffle patients deterministically
    rng = random.Random(seed)
    shuffled_patients = patient_ids.copy()
    rng.shuffle(shuffled_patients)

    # Split: test 10%, non-test 90% (patient-level)
    n_test_patients = max(1, round(total_patients * test_ratio))
    test_patient_ids = set(shuffled_patients[:n_test_patients])
    nontest_patient_ids = shuffled_patients[n_test_patients:]

    test_files = sorted(
        f for pid in sorted(test_patient_ids)
        for f in patient_to_files[pid]
    )

    # Distribute non-test patients into num_folds groups (round-robin)
    fold_patient_groups: list[list[str]] = [[] for _ in range(num_folds)]
    for i, pid in enumerate(nontest_patient_ids):
        fold_patient_groups[i % num_folds].append(pid)

    # Build fold assignments: val = fold k patients, train = rest of non-test
    folds = []
    for k in range(num_folds):
        val_pids = set(fold_patient_groups[k])
        train_pids = set(nontest_patient_ids) - val_pids

        val_files = sorted(
            f for pid in sorted(val_pids)
            for f in patient_to_files[pid]
        )
        train_files = sorted(
            f for pid in sorted(train_pids)
            for f in patient_to_files[pid]
        )
        folds.append({
            'train_files': train_files,
            'val_files': val_files,
        })

    result = {
        'seed': seed,
        'test_ratio': test_ratio,
        'num_folds': num_folds,
        'total_files': len(all_files),
        'total_patients': total_patients,
        'test_patients': n_test_patients,
        'nontest_patients': len(nontest_patient_ids),
        'test_files': test_files,
        'folds': folds,
    }

    output_path = data_dir / output_name
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Print statistics
    print(f"Total files: {len(all_files)}, Total patients: {total_patients}")
    print(f"Test:     {n_test_patients} patients, {len(test_files)} files "
          f"({len(test_files)/len(all_files)*100:.1f}%)")
    print(f"Non-test: {len(nontest_patient_ids)} patients, "
          f"{len(all_files) - len(test_files)} files")
    print()
    for k in range(num_folds):
        n_train = len(folds[k]['train_files'])
        n_val = len(folds[k]['val_files'])
        n_val_patients = len(fold_patient_groups[k])
        print(f"  Fold {k}: train={n_train}, val={n_val} "
              f"(val_patients={n_val_patients}, ratio={n_train/n_val:.1f}:1)")

    # Verify no patient-level leakage
    test_pids = set(extract_patient_id(f) for f in test_files)
    for k in range(num_folds):
        train_pids_k = set(extract_patient_id(f) for f in folds[k]['train_files'])
        val_pids_k = set(extract_patient_id(f) for f in folds[k]['val_files'])

        assert train_pids_k & val_pids_k == set(), \
            f"Fold {k}: patient-level train/val overlap!"
        assert train_pids_k & test_pids == set(), \
            f"Fold {k}: patient-level train/test overlap!"
        assert val_pids_k & test_pids == set(), \
            f"Fold {k}: patient-level val/test overlap!"

        # File-level coverage check
        all_fold_files = (set(folds[k]['train_files'])
                          | set(folds[k]['val_files'])
                          | set(test_files))
        assert all_fold_files == set(all_files), \
            f"Fold {k}: not all files accounted for!"

    print("\nVerification passed: no patient-level overlaps, all files accounted for.")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create patient-level fold splits for cross-validation'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the preprocessed dataset (flat structure)')
    parser.add_argument('--num_folds', type=int, default=9,
                        help='Number of folds (default: 9 → 8:1 train:val)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Fraction of patients for fixed test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_name', type=str, default='fold_splits.json')
    args = parser.parse_args()

    create_fold_splits(
        data_dir=args.data_dir,
        num_folds=args.num_folds,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_name=args.output_name,
    )
