"""Landmark detection dataset.

Loads the preprocessed (ROI-cropped and letterboxed) images and their
transformed landmark coordinates:

    root_dir/images/<name>.jpg
    root_dir/coords/<name>.csv     (3 rows: landmark names / X / Y)
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Left-right landmark pair indices for horizontal flip (0-indexed).
# Each tuple (i, j) means landmark i and j should be swapped after flipping.
# Midline landmarks (ANS, Menton, Crista galli, upper/lower midline) are unchanged.
FLIP_PAIRS = [
    (0, 1),    # Antegonion right <-> left
    (4, 10),   # Right maxillary 6 crown <-> Left maxillary 6 crown
    (5, 11),   # Right maxillary 6 root <-> Left maxillary 6 root
    (6, 8),    # Right maxillary 1 crown <-> Left maxillary 1 crown
    (7, 9),    # Right maxillary 1 root <-> Left maxillary 1 root
    (12, 18),  # Left mandibular 6 crown <-> Right mandibular 6 crown
    (13, 19),  # Left mandibular 6 root <-> Right mandibular 6 root
    (14, 16),  # Left mandibular 1 crown <-> Right mandibular 1 crown
    (15, 17),  # Left mandibular 1 root <-> Right mandibular 1 root
    (21, 22),  # zygomatico-maxillary suture right <-> left
    (23, 24),  # jugal process left <-> right
    (25, 26),  # zygoma right <-> left
    (27, 28),  # mastoid right <-> left
    (29, 30),  # latero-orbital right <-> left
]


class LandmarkDataset(Dataset):
    """Dataset for PA cephalometric landmark detection.

    Each item is ``(image, coords, filename)``:
        image:    [3, H, W] float tensor (grayscale replicated to 3 channels,
                  ImageNet-normalized)
        coords:   [N_landmarks, 2] landmark coordinates (x, y) in input pixels
        filename: image file name without extension

    Args:
        root_dir: Path to the preprocessed dataset.
        file_list: List of image file names (e.g. ['P0001_preop.jpg', ...]).
        augment: Whether to apply data augmentation (training only).
    """

    def __init__(
        self,
        root_dir: str,
        file_list: list[str],
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.augment = augment

        self.img_dir = self.root_dir / 'images'
        self.coord_dir = self.root_dir / 'coords'
        self.samples = sorted([self.img_dir / f for f in file_list])
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

        # Load landmark names from first sample
        first_coord = self.coord_dir / f'{self.samples[0].stem}.csv'
        self.landmark_names = self._load_landmark_names(first_coord)
        self.num_landmarks = len(self.landmark_names)

    def _load_landmark_names(self, csv_path: Path) -> list[str]:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
        return header[1:]  # Skip first empty cell

    def _load_coordinates(self, csv_path: Path) -> np.ndarray:
        """Load coordinates from CSV. Returns [N_landmarks, 2] array (x, y)."""
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        x = np.array([float(v) for v in rows[1][1:]], dtype=np.float32)
        y = np.array([float(v) for v in rows[2][1:]], dtype=np.float32)
        return np.stack([x, y], axis=-1)  # [N, 2]

    def _augment(self, image: np.ndarray, coords: np.ndarray) -> tuple:
        """Apply data augmentation.

            1. Geometric (rotation/scale/translation)
            2. Horizontal flip (p=0.5) + landmark pair swap
            3. Brightness/contrast

        Args:
            image: [H, W, 3] image (uint8).
            coords: [N, 2] coordinates (x, y).

        Returns:
            Tuple of (augmented_image, augmented_coords).
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # --- 1. Geometric: rotation + scale + translation ---
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.9, 1.1)
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        image = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        ones = np.ones((coords.shape[0], 1), dtype=np.float32)
        coords_h = np.concatenate([coords, ones], axis=-1)  # [N, 3]
        coords = (M @ coords_h.T).T.astype(np.float32)  # [N, 2]

        # --- 2. Horizontal flip (p=0.5) with landmark pair swap ---
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)  # flip horizontally
            coords[:, 0] = (w - 1) - coords[:, 0]  # mirror x coordinates
            # Swap left-right landmark pairs
            for i, j in FLIP_PAIRS:
                coords[[i, j]] = coords[[j, i]]

        # --- 3. Brightness/contrast ---
        brightness_delta = np.random.uniform(-30, 30)
        image = np.clip(image.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)

        # Random contrast (multiplicative around mean)
        alpha = np.random.uniform(0.7, 1.3)
        mean_val = image.mean()
        image = np.clip(alpha * (image.astype(np.float32) - mean_val) + mean_val, 0, 255).astype(np.uint8)

        # Clamp coords to image bounds
        coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)

        return image, coords

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        stem = img_path.stem

        # Load image: grayscale duplicated to 3 channels
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = np.stack([img, img, img], axis=-1)  # [H, W, 3]

        # Load coordinates [N_landmarks, 2] in input pixels
        coords = self._load_coordinates(self.coord_dir / f'{stem}.csv')

        if self.augment:
            img, coords = self._augment(img, coords)

        # Normalize image: [H, W, 3] uint8 -> float32 -> normalized
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # [3, H, W]
        coords_tensor = torch.from_numpy(coords)  # [N, 2]

        return img_tensor, coords_tensor, stem
