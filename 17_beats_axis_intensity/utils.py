"""Training and inference helpers for per-axis BEATs intensity models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import CLASS_SCHEMA, INTENSITY_LEVELS, INTENSITY_LEVEL_TO_DISPLAY
from dataset import AxisWaveformDataset, WaveformDatasetConfig
from model import BEATsAxisIntensity
from shared.audio import load_mono_audio, sliding_window_waveforms


def create_dataloaders(
    train_records,
    val_records,
    *,
    axis_name: str,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int,
):
    train_dataset = AxisWaveformDataset(
        train_records,
        WaveformDatasetConfig(
            axis_name=axis_name,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            training=True,
            augment=True,
        ),
    )
    val_dataset = AxisWaveformDataset(
        val_records,
        WaveformDatasetConfig(
            axis_name=axis_name,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            training=False,
            augment=False,
        ),
    )

    class_counts: dict[int, int] = {}
    for record in train_records:
        key = record.target_for_axis(axis_name)
        class_counts[key] = class_counts.get(key, 0) + 1
    sample_weights = [1.0 / class_counts[record.target_for_axis(axis_name)] for record in train_records]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model(
    axis_name: str,
    pretrained_checkpoint_path: str | Path | None,
    *,
    backbone_config_override: dict | None = None,
) -> BEATsAxisIntensity:
    return BEATsAxisIntensity(
        axis_name=axis_name,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        backbone_config_override=backbone_config_override,
    )


def class_weights_from_labels(labels: list[int], num_classes: int = len(INTENSITY_LEVELS)) -> torch.Tensor:
    values = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    total = counts.sum()
    for value, count in zip(unique, counts):
        weights[int(value)] = float(total / (len(unique) * count))
    return torch.tensor(weights, dtype=torch.float32)


def create_optimizer(
    model: BEATsAxisIntensity,
    *,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": [parameter for parameter in model.backbone.parameters() if parameter.requires_grad],
                "lr": backbone_learning_rate,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("backbone.") and parameter.requires_grad
                ],
                "lr": head_learning_rate,
            },
        ],
        weight_decay=weight_decay,
    )


def _classification_metrics(targets: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "on_off_accuracy": float(
            np.mean([int((target > 0) == (prediction > 0)) for target, prediction in zip(targets, predictions)])
        ),
    }


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch_index: int | None = None,
    num_epochs: int | None = None,
    stage_name: str = "train",
    log_interval: int = 50,
    gradient_clip_norm: float = 1.0,
) -> dict:
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    targets_all: list[int] = []
    predictions_all: list[int] = []

    total_batches = len(loader)
    for batch_index, (waveforms, targets) in enumerate(loader, start=1):
        waveforms = waveforms.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(waveforms)
        loss = criterion(logits, targets)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

        predictions = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)
        losses.append(float(loss.item()))
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

        should_log = total_batches > 0 and (
            batch_index == 1 or batch_index == total_batches or (log_interval > 0 and batch_index % log_interval == 0)
        )
        if should_log:
            prefix = f"[INFO] {stage_name}"
            if epoch_index is not None and num_epochs is not None:
                prefix += f" epoch {epoch_index:03d}/{num_epochs:03d}"
            print(f"{prefix} batch {batch_index:04d}/{total_batches:04d} loss={loss.item():.4f}", flush=True)

    metrics = _classification_metrics(targets_all, predictions_all)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def fit_axis_model(
    *,
    axis_name: str,
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
    gradient_clip_norm: float = 1.0,
    resume_state: dict | None = None,
) -> tuple[list[dict], dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = list(resume_state.get("history", [])) if resume_state else []
    best_bundle: dict | None = None
    best_score: float | None = None
    bad_epochs = 0
    start_epoch = 1

    if resume_state:
        start_epoch = int(resume_state.get("last_epoch", 0)) + 1
        best_score_value = resume_state.get("best_score")
        best_score = float(best_score_value) if best_score_value is not None else None
        bad_epochs = int(resume_state.get("bad_epochs", 0))
        best_bundle = resume_state.get("best_bundle")
        print(
            f"[INFO] Resuming {axis_name}: start_epoch={start_epoch}, "
            f"best_score={best_score}, bad_epochs={bad_epochs}",
            flush=True,
        )

    if start_epoch > num_epochs:
        if best_bundle is None:
            raise RuntimeError("Resume state says training is complete but no best checkpoint exists.")
        print(f"[INFO] {axis_name} already reached requested epochs ({num_epochs}).", flush=True)
        return history, best_bundle

    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epoch_index=epoch,
            num_epochs=num_epochs,
            stage_name=f"{axis_name}/train",
            log_interval=log_interval,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            epoch_index=epoch,
            num_epochs=num_epochs,
            stage_name=f"{axis_name}/valid",
            log_interval=log_interval,
            gradient_clip_norm=gradient_clip_norm,
        )

        composite_score = (
            (0.45 * val_metrics["macro_f1"])
            + (0.30 * val_metrics["on_off_accuracy"])
            + (0.25 * val_metrics["accuracy"])
        )
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        history.append({"epoch": epoch, "train": train_metrics, "valid": val_metrics, "composite_score": composite_score})
        print(
            f"[INFO] {axis_name} Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_onoff_acc={val_metrics['on_off_accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
            f"score={composite_score:.4f}",
            flush=True,
        )

        if best_score is None or composite_score > best_score:
            best_score = composite_score
            bad_epochs = 0
            best_bundle = {
                "axis_name": axis_name,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metrics": val_metrics,
                "composite_score": composite_score,
                "model_config": {
                    "axis_name": axis_name,
                    "pretrained_loaded": getattr(model, "pretrained_loaded", False),
                    "beats_backbone_config": getattr(model, "backbone_config", None),
                },
                "class_schema": CLASS_SCHEMA,
            }
            torch.save(best_bundle, checkpoint_path)
            print(f"[INFO] Saved new best {axis_name} checkpoint to {checkpoint_path}", flush=True)
        else:
            bad_epochs += 1

        latest_bundle = {
            "axis_name": axis_name,
            "last_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "best_score": best_score,
            "bad_epochs": bad_epochs,
            "best_bundle": best_bundle,
            "class_schema": CLASS_SCHEMA,
        }
        latest_path = checkpoint_path.with_name(checkpoint_path.stem.replace("_best_model", "") + "_latest_training.pt")
        torch.save(latest_bundle, latest_path)

        if bad_epochs >= patience:
            print(f"[INFO] {axis_name} early stopping triggered.", flush=True)
            break

    if best_bundle is None:
        raise RuntimeError("Training finished without producing a checkpoint.")
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")
    return history, best_bundle


@torch.inference_mode()
def predict_axis_file(
    model: torch.nn.Module,
    audio_path: str | Path,
    *,
    device: torch.device,
    sample_rate: int,
    segment_seconds: float,
) -> dict:
    model.eval()
    waveform = load_mono_audio(audio_path, sample_rate=sample_rate)
    window_size = int(segment_seconds * sample_rate)
    hop_size = max(window_size // 2, 1)
    windows = sliding_window_waveforms(waveform, window_size=window_size, hop_size=hop_size)

    scores_all = []
    for window in windows:
        waveform_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(waveform_tensor)
        scores_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy()[0])

    scores = np.mean(np.asarray(scores_all, dtype=np.float32), axis=0)
    pred_level = int(np.argmax(scores))
    return {
        "detected": bool(pred_level > 0),
        "intensity_level": pred_level,
        "intensity_display": INTENSITY_LEVEL_TO_DISPLAY[pred_level],
        "confidence": float(scores[pred_level]),
        "scores": {str(level): float(scores[level]) for level in INTENSITY_LEVELS},
    }


def evaluate_axis_records(
    model: torch.nn.Module,
    records,
    *,
    axis_name: str,
    device: torch.device,
    output_dir: str | Path,
    sample_rate: int,
    segment_seconds: float,
    prefix: str = "val",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    true_values: list[int] = []
    pred_values: list[int] = []
    for record in records:
        prediction = predict_axis_file(
            model=model,
            audio_path=record.path,
            device=device,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
        )
        true_level = record.target_for_axis(axis_name)
        pred_level = int(prediction["intensity_level"])
        rows.append(
            {
                "file_name": Path(record.path).name,
                "file_path": record.path,
                "relative_path": record.relative_path,
                "true_level": true_level,
                "true_display": INTENSITY_LEVEL_TO_DISPLAY[true_level],
                "predicted_level": pred_level,
                "predicted_display": prediction["intensity_display"],
                "confidence": prediction["confidence"],
                "correct": int(true_level == pred_level),
                "on_off_correct": int((true_level > 0) == (pred_level > 0)),
                **{f"score_{key}": value for key, value in prediction["scores"].items()},
            }
        )
        true_values.append(true_level)
        pred_values.append(pred_level)

    metrics = _classification_metrics(true_values, pred_values)
    predictions_path = output_dir / f"{prefix}_{axis_name}_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    metrics_path = output_dir / f"{prefix}_{axis_name}_metrics.json"
    metrics_path.write_text(
        json.dumps({**metrics, "predictions_path": str(predictions_path)}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return {**metrics, "predictions_path": str(predictions_path), "metrics_path": str(metrics_path)}
