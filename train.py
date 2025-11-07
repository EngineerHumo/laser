"""Training script for the spot grading model with fixed train/val splits."""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    build_spot_transform,
)
from laser.models import DualEncoderMetricModel
from laser.models.losses import ArcMarginProduct, batch_hard_triplet_loss
from laser.utils import VisdomLogger

LOGGER = logging.getLogger("train")


@dataclass
class TrainConfig:
    train_dir: Path
    val_dir: Path
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    visdom_env: str = "spot_metric_learning"
    visdom_port: int = 8100
    log_dir: Path = Path("logs")
    amp: bool = True
    arcface_only_epochs: int = 200
    combined_epochs: int = 200
    classification_only_epochs: int = 200
    arcface_weight_combined: float = 1.0
    triplet_weight_combined: float = 1.0
    classification_weight_combined: float = 1.0
    classification_weight_final: float = 1.0

    @property
    def total_epochs(self) -> int:
        return self.arcface_only_epochs + self.combined_epochs + self.classification_only_epochs


@dataclass
class LossWeights:
    arcface: float
    triplet: float
    classification: float


@dataclass
class TrainEpochResult:
    loss: float
    arcface_loss: float
    triplet_loss: float
    classification_loss: float
    classification_accuracy: float
    arcface_accuracy: float


@dataclass
class EvalMetrics:
    loss: float
    arcface_loss: float
    triplet_loss: float
    classification_loss: float
    classification_accuracy: float
    arcface_accuracy: float
    per_class_accuracy: Dict[int, float]
    class_counts: Dict[int, int]
    confusion_matrix: torch.Tensor


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train deep metric learning model for spot grading")
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--visdom-env", type=str, default="spot_metric_learning")
    parser.add_argument("--visdom-port", type=int, default=8100)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--arcface-only-epochs", type=int, default=200)
    parser.add_argument("--combined-epochs", type=int, default=200)
    parser.add_argument("--classification-only-epochs", type=int, default=200)
    parser.add_argument("--arcface-weight-combined", type=float, default=1.0)
    parser.add_argument("--triplet-weight-combined", type=float, default=1.0)
    parser.add_argument("--classification-weight-combined", type=float, default=1.0)
    parser.add_argument("--classification-weight-final", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")

    args = parser.parse_args()

    return TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
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
        arcface_only_epochs=args.arcface_only_epochs,
        combined_epochs=args.combined_epochs,
        classification_only_epochs=args.classification_only_epochs,
        arcface_weight_combined=args.arcface_weight_combined,
        triplet_weight_combined=args.triplet_weight_combined,
        classification_weight_combined=args.classification_weight_combined,
        classification_weight_final=args.classification_weight_final,
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

    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    train_subset = SpotSubsetDataset(
        train_dataset,
        train_indices,
        spot_transform=train_spot_transform,
        global_transform=global_transform,
    )
    val_subset = SpotSubsetDataset(
        val_dataset,
        val_indices,
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


def determine_loss_weights(epoch: int, config: TrainConfig) -> LossWeights:
    if epoch < config.arcface_only_epochs:
        return LossWeights(arcface=1.0, triplet=config.triplet_weight, classification=0.0)
    if epoch < config.arcface_only_epochs + config.combined_epochs:
        return LossWeights(
            arcface=config.arcface_weight_combined,
            triplet=config.triplet_weight_combined,
            classification=config.classification_weight_combined,
        )
    return LossWeights(arcface=0.0, triplet=0.0, classification=config.classification_weight_final)


def train_one_epoch(
    model: DualEncoderMetricModel,
    arcface: ArcMarginProduct,
    classifier: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: amp.GradScaler,
    config: TrainConfig,
    epoch: int,
    vis: VisdomLogger,
) -> TrainEpochResult:
    model.train()
    arcface.train()
    classifier.train()

    weights = determine_loss_weights(epoch, config)
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_arcface_loss = 0.0
    total_triplet_loss = 0.0
    total_cls_loss = 0.0
    total_cls_correct = 0
    total_arcface_correct = 0
    total_samples = 0

    vis.log_text(
        "training_phase",
        (
            f"Epoch {epoch + 1} phase weights: arcface={weights.arcface:.3f}, "
            f"triplet={weights.triplet:.3f}, classification={weights.classification:.3f}<br>"
        ),
    )

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
            label_lines = [
                f"Epoch {epoch + 1} | Sample {i + 1}: class {int(label)}"
                for i, label in enumerate(labels_batch[:num_vis])
            ]
            vis.log_text("train_spot_labels", "<br>".join(label_lines) + "<br><br>")

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=config.amp):
            embeddings, fusion_features = model(spot, global_img)
            arcface_logits = arcface(embeddings, labels)
            arcface_loss = ce_loss_fn(arcface_logits, labels)

            classifier_logits = classifier(fusion_features)
            classification_loss = ce_loss_fn(classifier_logits, labels)

            triplet_loss = batch_hard_triplet_loss(embeddings, labels, margin=config.triplet_margin)

            loss = (
                weights.arcface * arcface_loss
                + weights.triplet * triplet_loss
                + weights.classification * classification_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(arcface.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            arcface_preds = torch.argmax(arcface.inference(embeddings.detach()), dim=1)
            cls_preds = torch.argmax(classifier_logits.detach(), dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_arcface_loss += arcface_loss.item() * batch_size
        total_triplet_loss += triplet_loss.item() * batch_size
        total_cls_loss += classification_loss.item() * batch_size
        total_cls_correct += (cls_preds == labels).sum().item()
        total_arcface_correct += (arcface_preds == labels).sum().item()
        total_samples += batch_size

        global_step = epoch * len(train_loader) + step
        vis.log_scalar("train_loss", global_step, loss.item())
        vis.log_scalar("train_arcface_loss", global_step, arcface_loss.item())
        vis.log_scalar("train_triplet_loss", global_step, triplet_loss.item())
        vis.log_scalar("train_classification_loss", global_step, classification_loss.item())
        vis.log_scalar(
            "train_classification_accuracy",
            global_step,
            (cls_preds == labels).float().mean().item(),
        )
        vis.log_scalar(
            "train_arcface_accuracy",
            global_step,
            (arcface_preds == labels).float().mean().item(),
        )

    avg_loss = total_loss / max(total_samples, 1)
    avg_arcface_loss = total_arcface_loss / max(total_samples, 1)
    avg_triplet_loss = total_triplet_loss / max(total_samples, 1)
    avg_cls_loss = total_cls_loss / max(total_samples, 1)
    avg_cls_acc = total_cls_correct / max(total_samples, 1)
    avg_arcface_acc = total_arcface_correct / max(total_samples, 1)

    return TrainEpochResult(
        loss=avg_loss,
        arcface_loss=avg_arcface_loss,
        triplet_loss=avg_triplet_loss,
        classification_loss=avg_cls_loss,
        classification_accuracy=avg_cls_acc,
        arcface_accuracy=avg_arcface_acc,
    )


def evaluate(
    model: DualEncoderMetricModel,
    arcface: ArcMarginProduct,
    classifier: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    vis: VisdomLogger,
) -> EvalMetrics:
    model.eval()
    arcface.eval()
    classifier.eval()

    ce_loss_fn = nn.CrossEntropyLoss()
    weights = determine_loss_weights(epoch, config)

    total_loss = 0.0
    total_arcface_loss = 0.0
    total_triplet_loss = 0.0
    total_cls_loss = 0.0
    total_samples = 0
    total_cls_correct = 0
    total_arcface_correct = 0

    num_classes = arcface.out_features
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    class_totals = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            spot = batch["spot"].to(device, non_blocking=True)
            global_img = batch["global"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            embeddings, fusion_features = model(spot, global_img)
            arcface_logits = arcface(embeddings, labels)
            arcface_loss = ce_loss_fn(arcface_logits, labels)
            classifier_logits = classifier(fusion_features)
            classification_loss = ce_loss_fn(classifier_logits, labels)

            triplet_loss = batch_hard_triplet_loss(embeddings, labels, margin=config.triplet_margin)

            combined_loss = (
                weights.arcface * arcface_loss
                + weights.triplet * triplet_loss
                + weights.classification * classification_loss
            )

            arcface_probs = arcface.inference(embeddings)
            arcface_preds = torch.argmax(arcface_probs, dim=1)
            cls_preds = torch.argmax(classifier_logits, dim=1)

            batch_size = labels.size(0)
            total_loss += combined_loss.item() * batch_size
            total_arcface_loss += arcface_loss.item() * batch_size
            total_triplet_loss += triplet_loss.item() * batch_size
            total_cls_loss += classification_loss.item() * batch_size
            total_samples += batch_size
            total_cls_correct += (cls_preds == labels).sum().item()
            total_arcface_correct += (arcface_preds == labels).sum().item()

            labels_cpu = labels.cpu()
            preds_cpu = cls_preds.cpu()
            for actual, predicted in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                confusion[actual, predicted] += 1
            class_totals += torch.bincount(labels_cpu, minlength=num_classes)
            matches = preds_cpu == labels_cpu
            if matches.any():
                class_correct += torch.bincount(labels_cpu[matches], minlength=num_classes)

            global_step = epoch * len(data_loader) + step
            vis.log_scalar("val_loss", global_step, combined_loss.item())
            vis.log_scalar("val_arcface_loss", global_step, arcface_loss.item())
            vis.log_scalar("val_triplet_loss", global_step, triplet_loss.item())
            vis.log_scalar("val_classification_loss", global_step, classification_loss.item())
            vis.log_scalar(
                "val_classification_accuracy",
                global_step,
                (cls_preds == labels).float().mean().item(),
            )
            vis.log_scalar(
                "val_arcface_accuracy",
                global_step,
                (arcface_preds == labels).float().mean().item(),
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_arcface_loss = total_arcface_loss / max(total_samples, 1)
    avg_triplet_loss = total_triplet_loss / max(total_samples, 1)
    avg_cls_loss = total_cls_loss / max(total_samples, 1)
    avg_cls_acc = total_cls_correct / max(total_samples, 1)
    avg_arcface_acc = total_arcface_correct / max(total_samples, 1)

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
        arcface_loss=avg_arcface_loss,
        triplet_loss=avg_triplet_loss,
        classification_loss=avg_cls_loss,
        classification_accuracy=avg_cls_acc,
        arcface_accuracy=avg_arcface_acc,
        per_class_accuracy=per_class_accuracy,
        class_counts=class_counts,
        confusion_matrix=confusion,
    )


def run_training(config: TrainConfig) -> None:
    set_seed()
    configure_logging(config.log_dir)
    device = torch.device(config.device)

    train_dataset = SpotDataset(config.train_dir)
    val_dataset = SpotDataset(config.val_dir)
    num_classes = max(train_dataset.num_classes, val_dataset.num_classes)

    vis_logger = VisdomLogger(env=config.visdom_env, port=config.visdom_port)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    model = DualEncoderMetricModel().to(device)
    arcface = ArcMarginProduct(in_features=512, out_features=num_classes).to(device)
    classifier = nn.Linear(512, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()) + list(classifier.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_epochs * max(len(train_loader), 1)
    )
    scaler = amp.GradScaler(enabled=config.amp)

    best_val_acc = 0.0
    best_epoch = -1
    checkpoint_dir = config.log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.total_epochs):
        train_metrics = train_one_epoch(
            model,
            arcface,
            classifier,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            config,
            epoch,
            vis_logger,
        )
        val_metrics = evaluate(
            model,
            arcface,
            classifier,
            val_loader,
            device,
            config,
            epoch,
            vis_logger,
        )

        weights = determine_loss_weights(epoch, config)
        LOGGER.info(
            (
                "Epoch %d/%d | weights=(arcface=%.2f, triplet=%.2f, cls=%.2f) | "
                "train_loss=%.4f | train_cls_acc=%.3f | val_loss=%.4f | val_cls_acc=%.3f"
            ),
            epoch + 1,
            config.total_epochs,
            weights.arcface,
            weights.triplet,
            weights.classification,
            train_metrics.loss,
            train_metrics.classification_accuracy,
            val_metrics.loss,
            val_metrics.classification_accuracy,
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

        if val_metrics.classification_accuracy > best_val_acc:
            best_val_acc = val_metrics.classification_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arcface_state": arcface.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "val_classification_acc": val_metrics.classification_accuracy,
                },
                checkpoint_dir / "best.pt",
            )
            vis_logger.log_text(
                "best_model",
                f"Updated best.pt at epoch {epoch + 1} with val classification acc {best_val_acc:.3f}<br>",
            )

    if best_epoch >= 0:
        LOGGER.info(
            "Training finished. Best val classification acc=%.3f at epoch %d",
            best_val_acc,
            best_epoch + 1,
        )
        summary_text = (
            f"Training finished<br>Best val classification acc: {best_val_acc:.3f} "
            f"at epoch {best_epoch + 1}<br>"
        )
    else:
        LOGGER.warning("Training finished but no validation batches were processed.")
        summary_text = "Training finished<br>No validation batches processed<br>"

    vis_logger.log_text("summary", summary_text)


if __name__ == "main":
    raise SystemExit("Use `python train.py` to start training.")


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
