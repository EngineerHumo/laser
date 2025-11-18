"""Training script for deep metric learning spot grading."""

from __future__ import annotations

import argparse
import math
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from laser.datasets import (
    SPOT_MEAN,
    SPOT_STD,
    BalancedBatchSampler,
    SpotDataset,
    SpotSample,
    SpotSubsetDataset,
    build_global_transform,
    build_shared_hsv_transform,
    build_spot_transform,
)
from laser.models import DualEncoderMetricModel
from laser.models.losses import ArcMarginProduct, batch_hard_triplet_loss
from laser.utils import VisdomLogger

LOGGER = logging.getLogger("train")


@dataclass
class TrainConfig:
    data_dir: Path
    batch_size: int = 128
    num_epochs: int = 300
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    visdom_env: str = "spot_metric_learning"
    visdom_port: int = 8097
    log_dir: Path = Path("logs")
    amp: bool = True


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    per_class_accuracy: Dict[int, float]
    class_counts: Dict[int, int]
    confusion_matrix: torch.Tensor


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train deep metric learning model for spot grading")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--visdom-env", type=str, default="spot_metric_learning")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        visdom_env=args.visdom_env,
        visdom_port=args.visdom_port,
        log_dir=args.log_dir,
        amp=not args.no_amp,
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: Iterable[SpotSample]) -> Dict[str, torch.Tensor]:
    batch_list = list(batch)
    spot = torch.stack([item.spot_image for item in batch_list])
    spot_context = torch.stack([item.spot_context_image for item in batch_list])
    global_img = torch.stack([item.global_image for item in batch_list])
    labels = torch.tensor([item.label for item in batch_list], dtype=torch.long)
    return {"spot": spot, "spot_context": spot_context, "global": global_img, "labels": labels}


def create_dataloaders(
    train_dataset: SpotDataset,
    val_dataset: SpotDataset,
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_spot_transform = build_spot_transform(augment=True)
    eval_spot_transform = build_spot_transform(augment=False)
    global_transform = build_global_transform()
    shared_hsv_transform = build_shared_hsv_transform()

    train_subset = SpotSubsetDataset(
        train_dataset,
        list(range(len(train_dataset))),
        spot_transform=train_spot_transform,
        global_transform=global_transform,
        pair_transform=shared_hsv_transform,
    )
    val_subset = SpotSubsetDataset(
        val_dataset,
        list(range(len(val_dataset))),
        spot_transform=eval_spot_transform,
        global_transform=global_transform,
    )

    train_sampler = BalancedBatchSampler(
        train_subset,
        config.batch_size,
        extra_classes=[0, 2, 3, 4, 5],
        extras_per_class=2,
    )

    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def train_one_epoch(
    model: DualEncoderMetricModel,
    arcface: ArcMarginProduct,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: amp.GradScaler,
    config: TrainConfig,
    epoch: int,
    vis: VisdomLogger,
) -> Tuple[float, float]:
    model.train()
    arcface.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    ce_loss_fn = nn.CrossEntropyLoss()

    for step, batch in enumerate(train_loader):
        spot_context_batch = batch["spot_context"]
        spot_batch = batch["spot"]
        labels_batch = batch["labels"]

        spot = spot_batch.to(device, non_blocking=True)
        global_img = batch["global"].to(device, non_blocking=True)
        labels = labels_batch.to(device, non_blocking=True)

        if step == 0:
            num_vis = min(16, spot_batch.size(0))
            vis.log_images(
                "train_spot_64",
                spot_batch[:num_vis],
                nrow=4,
                mean=SPOT_MEAN,
                std=SPOT_STD,
            )
            vis.log_images(
                "train_spot_context_128",
                spot_context_batch[:num_vis],
                nrow=4,
                mean=SPOT_MEAN,
                std=SPOT_STD,
            )
            label_lines = [f"Epoch {epoch + 1} | Sample {i + 1}: class {int(label)}" for i, label in enumerate(labels_batch[:num_vis])]
            vis.log_text("train_spot_labels", "<br>".join(label_lines) + "<br><br>")

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=config.amp):
            embeddings, _ = model(spot, global_img)
            logits = arcface(embeddings, labels)
            ce_loss = ce_loss_fn(logits, labels)
            triplet_loss = batch_hard_triplet_loss(embeddings, labels, margin=config.triplet_margin)
            loss = ce_loss + config.triplet_weight * triplet_loss
            #loss = ce_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(arcface.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = torch.argmax(logits.detach(), dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += (preds == labels).sum().item()
        total_samples += batch_size

        global_step = epoch * len(train_loader) + step
        vis.log_scalar("train_loss", global_step, loss.item())
        vis.log_scalar("train_accuracy", global_step, (preds == labels).float().mean().item())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_acc / max(total_samples, 1)
    return avg_loss, avg_acc


def evaluate(
    model: DualEncoderMetricModel,
    arcface: ArcMarginProduct,
    data_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    vis: VisdomLogger,
) -> EvalMetrics:
    model.eval()
    arcface.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    num_classes = arcface.out_features
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    class_totals = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            spot = batch["spot"].to(device, non_blocking=True)
            global_img = batch["global"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            embeddings, _ = model(spot, global_img)
            logits = arcface(embeddings, labels)
            ce_loss = ce_loss_fn(logits, labels)

            triplet_loss = batch_hard_triplet_loss(embeddings, labels, margin=config.triplet_margin)
            loss = ce_loss + config.triplet_weight * triplet_loss
            #loss = ce_loss  

            preds = torch.argmax(arcface.inference(embeddings), dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += (preds == labels).sum().item()
            total_samples += batch_size

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()
            for actual, predicted in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                confusion[actual, predicted] += 1
            class_totals += torch.bincount(labels_cpu, minlength=num_classes)
            matches = preds_cpu == labels_cpu
            if matches.any():
                class_correct += torch.bincount(labels_cpu[matches], minlength=num_classes)

            global_step = epoch * len(data_loader) + step
            vis.log_scalar("val_loss", global_step, loss.item())
            vis.log_scalar("val_accuracy", global_step, (preds == labels).float().mean().item())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_acc / max(total_samples, 1)
    per_class_accuracy: Dict[int, float] = {}
    class_counts: Dict[int, int] = {}
    for cls_idx in range(num_classes):
        total = int(class_totals[cls_idx].item())
        class_counts[cls_idx] = total
        if total == 0:
            per_class_accuracy[cls_idx] = float("nan")
        else:
            per_class_accuracy[cls_idx] = float(class_correct[cls_idx].item()) / total

    return EvalMetrics(
        loss=avg_loss,
        accuracy=avg_acc,
        per_class_accuracy=per_class_accuracy,
        class_counts=class_counts,
        confusion_matrix=confusion,
    )


def run_training(config: TrainConfig) -> None:
    set_seed()
    configure_logging(config.log_dir)
    device = torch.device(config.device)
    train_root = config.data_dir / "train"
    val_root = config.data_dir / "val"

    train_dataset = SpotDataset(train_root)
    val_dataset = SpotDataset(val_root)

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found in {train_root}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation samples found in {val_root}")

    LOGGER.info(
        "Loaded datasets | train: %d samples from %s | val: %d samples from %s",
        len(train_dataset),
        train_root,
        len(val_dataset),
        val_root,
    )

    vis_logger = VisdomLogger(env=config.visdom_env, port=config.visdom_port)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    num_classes = max(train_dataset.num_classes, val_dataset.num_classes)
    model = DualEncoderMetricModel().to(device)
    arcface = ArcMarginProduct(in_features=512, out_features=num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs * len(train_loader))
    scaler = amp.GradScaler(enabled=config.amp)

    best_val_acc = float("-inf")
    best_epoch = -1

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, arcface, train_loader, optimizer, scheduler, device, scaler, config, epoch, vis_logger
        )
        val_metrics = evaluate(model, arcface, val_loader, device, config, epoch, vis_logger)
        val_loss = val_metrics.loss
        val_acc = val_metrics.accuracy

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | train_acc=%.3f | val_loss=%.4f | val_acc=%.3f",
            epoch + 1,
            config.num_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        per_class_log = ", ".join(
            (
                f"class {cls}: {acc:.3f} ({val_metrics.class_counts.get(cls, 0)} samples)"
                if not math.isnan(acc)
                else f"class {cls}: N/A ({val_metrics.class_counts.get(cls, 0)} samples)"
            )
            for cls, acc in sorted(val_metrics.per_class_accuracy.items())
        )
        LOGGER.info(
            "Epoch %d | Validation per-class accuracy: %s",
            epoch + 1,
            per_class_log,
        )
        confusion_array = val_metrics.confusion_matrix.cpu().numpy()
        confusion_rows = ["\t".join(f"{int(value):d}" for value in row) for row in confusion_array]
        header = "\t".join(str(cls) for cls in range(confusion_array.shape[1]))
        matrix_lines = ["Predicted", "\t" + header]
        matrix_lines.extend(
            f"{cls}\t" + row for cls, row in zip(range(confusion_array.shape[0]), confusion_rows)
        )
        matrix_text = "\n".join(matrix_lines)
        LOGGER.info(
            "Epoch %d | Confusion matrix (rows=gt, cols=pred):\n%s",
            epoch + 1,
            matrix_text,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_dir = config.log_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "arcface_state": arcface.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_acc": val_acc,
            }, checkpoint_dir / "best.pt")

    LOGGER.info("Training finished. Best val acc=%.3f at epoch %d", best_val_acc, best_epoch + 1)
    if best_epoch >= 0:
        summary_text = f"Training finished<br>Best val acc: {best_val_acc:.3f} at epoch {best_epoch + 1}"
    else:
        summary_text = "Training finished<br>No validation epochs executed"
    vis_logger.log_text("summary", summary_text)


if __name__ == "main":
    raise SystemExit("Use `python train.py` to start training.")


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
