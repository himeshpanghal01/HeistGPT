"""Dataset and augmentation utilities for semantic segmentation.

This module provides:
- DesertSegmentationDataset: custom PyTorch dataset for image/mask pairs.
- build_train_augmentations / build_val_augmentations: albumentations pipelines
  tuned for synthetic-to-real robustness in off-road desert scenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_train_augmentations(height: int = 512, width: int = 512) -> A.Compose:
    """Aggressive augmentations to improve generalization.

    The augmentations are intentionally strong to bridge the domain gap between
    synthetic imagery and unseen real/sim-like environments, while preserving
    segmentation consistency.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
                mask_value=0,
            ),
            A.RandomCrop(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.25,
                rotate_limit=20,
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.35,
                        contrast_limit=0.35,
                        p=1.0,
                    ),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomGamma(gamma_limit=(70, 140), p=1.0),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 11), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 60.0), p=1.0),
                ],
                p=0.45,
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0),
                    A.OpticalDistortion(distort_limit=0.08, shift_limit=0.08, p=1.0),
                ],
                p=0.25,
            ),
            A.CoarseDropout(
                max_holes=16,
                max_height=int(height * 0.12),
                max_width=int(width * 0.12),
                min_holes=4,
                min_height=int(height * 0.03),
                min_width=int(width * 0.03),
                fill_value=0,
                mask_fill_value=0,
                p=0.35,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=False),
        ]
    )


def build_val_augmentations(height: int = 512, width: int = 512) -> A.Compose:
    """Validation/test preprocessing with deterministic resizing + normalization."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.CenterCrop(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=False),
        ]
    )


class DesertSegmentationDataset(Dataset):
    """PyTorch dataset for semantic segmentation image/mask pairs.

    Expects matching filenames under two folders, e.g.:
        images/frame_001.png
        masks/frame_001.png

    Masks should contain class indices in [0, num_classes - 1].
    """

    SUPPORTED_SUFFIXES: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        return_paths: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.transform = transform
        self.return_paths = return_paths

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.image_paths = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_SUFFIXES
            ]
        )

        if not self.image_paths:
            raise RuntimeError(f"No supported image files found in: {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask.astype(np.int64)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = self._read_image(image_path)

        if self.masks_dir is not None:
            mask_path = self.masks_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

            mask = self._read_mask(mask_path)

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed["image"], transformed["mask"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask)

            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            mask = mask.long()

            if self.return_paths:
                return image, mask, str(image_path)
            return image, mask

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.return_paths:
            return image, str(image_path)
        return image


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization/debugging."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)
