"""Reusable PyTorch training utilities for audio classifiers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    """Pick CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float


class EarlyStopping:
    """Stop training when the monitored metric no longer improves."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def run_classifier_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch_index: int | None = None,
    num_epochs: int | None = None,
    stage_name: str = "train",
    log_interval: int = 50,
) -> tuple[float, list[int], list[int], list[list[float]]]:
    """Run one epoch for either training or validation."""
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_scores: list[list[float]] = []

    total_batches = len(loader)
    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        if training:
            loss.backward()
            optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        losses.append(float(loss.item()))
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_scores.extend(probabilities.detach().cpu().tolist())

        should_log = (
            total_batches > 0
            and (
                batch_index == 1
                or batch_index == total_batches
                or (log_interval > 0 and batch_index % log_interval == 0)
            )
        )
        if should_log:
            prefix = f"[INFO] {stage_name}"
            if epoch_index is not None and num_epochs is not None:
                prefix += f" epoch {epoch_index:03d}/{num_epochs:03d}"
            print(
                f"{prefix} batch {batch_index:04d}/{total_batches:04d} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, all_targets, all_predictions, all_scores


def fit_model(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    num_epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    scheduler=None,
    log_interval: int = 50,
) -> tuple[list[dict], dict]:
    """Train a classifier with early stopping on validation macro F1."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience)
    history: list[dict] = []
    best_bundle: dict | None = None

    for epoch in range(1, num_epochs + 1):
        train_loss, _, _, _ = run_classifier_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epoch_index=epoch,
            num_epochs=num_epochs,
            stage_name="train",
            log_interval=log_interval,
        )
        val_loss, val_targets, val_predictions, val_scores = run_classifier_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            epoch_index=epoch,
            num_epochs=num_epochs,
            stage_name="valid",
            log_interval=log_interval,
        )

        val_accuracy = float(accuracy_score(val_targets, val_predictions))
        val_macro_f1 = float(f1_score(val_targets, val_predictions, average="macro", zero_division=0))
        epoch_result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_f1=val_macro_f1,
        )
        history.append(epoch_result.__dict__)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"[INFO] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )

        if early_stopper.update(val_macro_f1):
            best_bundle = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "val_scores": val_scores,
                "val_targets": val_targets,
                "val_predictions": val_predictions,
            }
            torch.save(best_bundle, checkpoint_path)
            print(f"[INFO] Saved new best checkpoint to {checkpoint_path}", flush=True)

        if early_stopper.should_stop:
            print("[INFO] Early stopping triggered.", flush=True)
            break

    if best_bundle is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")
    return history, best_bundle


def class_weights_from_indices(labels: Sequence[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    labels = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = np.ones(int(unique.max()) + 1, dtype=np.float32)
    for label, count in zip(unique, counts):
        weights[int(label)] = float(total / (len(unique) * count))
    return torch.tensor(weights, dtype=torch.float32)
