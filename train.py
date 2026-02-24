"""Training script for semantic segmentation with AMP, hybrid loss, and mIoU checkpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
from dataset import DesertSegmentationDataset, build_train_augmentations, build_val_augmentations
from model import build_model


class HybridSegLoss(nn.Module):
    """Dice + CrossEntropy loss for robust optimization on class imbalance."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int | None = None) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index is not None else -100)
        self.dice = smp.losses.DiceLoss(
            mode=smp.losses.MULTICLASS_MODE,
            from_logits=True,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)

    valid = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid &= targets != ignore_index

    preds = preds[valid]
    targets = targets[valid]

    indices = targets * num_classes + preds
    bins = torch.bincount(indices, minlength=num_classes * num_classes)
    confmat += bins.reshape(num_classes, num_classes)
    return confmat


def compute_miou(confmat: torch.Tensor, eps: float = 1e-7) -> float:
    intersection = torch.diag(confmat)
    union = confmat.sum(1) + confmat.sum(0) - intersection
    iou = (intersection + eps) / (union + eps)
    valid = union > 0
    if valid.any():
        return iou[valid].mean().item()
    return 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    ignore_index: int | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.detach().item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            confmat = update_confusion_matrix(confmat, preds, masks, num_classes, ignore_index)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    miou = compute_miou(confmat)
    return {"loss": avg_loss, "miou": miou}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic segmentation model for off-road scenes")
    parser.add_argument("--train-images", type=str, required=True)
    parser.add_argument("--train-masks", type=str, required=True)
    parser.add_argument("--val-images", type=str, required=True)
    parser.add_argument("--val-masks", type=str, required=True)
    parser.add_argument("--num-classes", type=int, required=True)

    parser.add_argument("--architecture", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--encoder", type=str, default="resnet34", choices=["resnet34", "efficientnet-b0", "efficientnet-b3"])
    parser.add_argument("--image-size", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ignore-index", type=int, default=None)

    parser.add_argument("--outdir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_ds = DesertSegmentationDataset(
        images_dir=args.train_images,
        masks_dir=args.train_masks,
        transform=build_train_augmentations(args.image_size, args.image_size),
    )
    val_ds = DesertSegmentationDataset(
        images_dir=args.val_images,
        masks_dir=args.val_masks,
        transform=build_val_augmentations(args.image_size, args.image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        encoder_weights="imagenet",
    ).to(device)

    criterion = HybridSegLoss(ce_weight=1.0, dice_weight=1.0, ignore_index=args.ignore_index)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.num_classes,
            optimizer=optimizer,
            scaler=scaler,
            ignore_index=args.ignore_index,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            optimizer=None,
            scaler=None,
            ignore_index=args.ignore_index,
        )
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_mIoU={train_metrics['miou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['miou']:.4f}"
        )

        latest_path = outdir / "latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "val_miou": val_metrics["miou"],
            },
            latest_path,
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_path = outdir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_miou": best_miou,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path} (val_mIoU={best_miou:.4f})")

    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
