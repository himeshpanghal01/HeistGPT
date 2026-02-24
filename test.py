"""Inference script: load trained weights, save predicted masks, and report mIoU."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DesertSegmentationDataset, build_val_augmentations
from model import build_model
from train import compute_miou, update_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--test-images", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="predictions")

    parser.add_argument("--test-masks", type=str, default=None, help="Optional GT masks for mIoU")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--architecture", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--encoder", type=str, default="resnet34", choices=["resnet34", "efficientnet-b0", "efficientnet-b3"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ignore-index", type=int, default=None)
    return parser.parse_args()


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Save class-index mask as single-channel PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask.astype(np.uint8))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        encoder_weights=None,
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = DesertSegmentationDataset(
        images_dir=args.test_images,
        masks_dir=args.test_masks,
        transform=build_val_augmentations(args.image_size, args.image_size),
        return_paths=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    confmat = torch.zeros((args.num_classes, args.num_classes), dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            if args.test_masks is not None:
                images, masks, paths = batch
                masks = masks.to(device)
            else:
                images, paths = batch
                masks = None

            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            if masks is not None:
                confmat = update_confusion_matrix(confmat, preds, masks, args.num_classes, args.ignore_index)

            for i in range(preds.size(0)):
                img_path = Path(paths[i])
                pred_mask = preds[i].detach().cpu().numpy()
                save_mask(pred_mask, output_dir / f"{img_path.stem}_pred.png")

    if args.test_masks is not None:
        miou = compute_miou(confmat)
        print(f"Test mIoU: {miou:.4f}")
    else:
        print("Inference complete. Ground-truth masks not provided, mIoU not computed.")

    print(f"Saved predictions to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
