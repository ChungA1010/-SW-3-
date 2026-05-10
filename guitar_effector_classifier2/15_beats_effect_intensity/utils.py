"""Training and inference helpers for the BEATs effect + intensity model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    EFFECT_CLASS_NAMES,
    INDEX_TO_EFFECT_CLASS,
    INTENSITY_LEVELS,
    INTENSITY_LEVEL_TO_DISPLAY,
)
from dataset import EffectIntensityWaveformDataset, WaveformDatasetConfig
from model import BEATsEffectIntensity
from shared.audio import load_mono_audio, sliding_window_waveforms


def create_dataloaders(
    train_records,
    val_records,
    *,
    sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    num_workers: int,
):
    train_config = WaveformDatasetConfig(
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        training=True,
        augment=True,
    )
    val_config = WaveformDatasetConfig(
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        training=False,
        augment=False,
    )
    train_dataset = EffectIntensityWaveformDataset(train_records, train_config)
    val_dataset = EffectIntensityWaveformDataset(val_records, val_config)

    class_counts: dict[str, int] = {}
    for record in train_records:
        key = f"{record.effect_label}:{record.intensity_level}"
        class_counts[key] = class_counts.get(key, 0) + 1
    sample_weights = [
        1.0 / class_counts[f"{record.effect_label}:{record.intensity_level}"]
        for record in train_records
    ]
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
    pretrained_checkpoint_path: str | Path | None,
    *,
    backbone_config_override: dict | None = None,
) -> BEATsEffectIntensity:
    return BEATsEffectIntensity(
        num_effect_classes=len(EFFECT_CLASS_NAMES),
        num_intensity_classes=len(INTENSITY_LEVELS),
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        backbone_config_override=backbone_config_override,
    )


def class_weights_from_labels(labels: list[int], num_classes: int) -> torch.Tensor:
    values = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    total = counts.sum()
    for value, count in zip(unique, counts):
        weights[int(value)] = float(total / (len(unique) * count))
    return torch.tensor(weights, dtype=torch.float32)


def create_optimizer(
    model: BEATsEffectIntensity,
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


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    effect_criterion,
    intensity_criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch_index: int | None = None,
    num_epochs: int | None = None,
    stage_name: str = "train",
    log_interval: int = 50,
    intensity_loss_weight: float = 1.0,
    gradient_clip_norm: float = 1.0,
) -> dict:
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    effect_losses: list[float] = []
    intensity_losses: list[float] = []
    effect_targets_all: list[int] = []
    effect_predictions_all: list[int] = []
    intensity_targets_all: list[int] = []
    intensity_predictions_all: list[int] = []

    total_batches = len(loader)
    for batch_index, (waveforms, effect_targets, intensity_targets) in enumerate(loader, start=1):
        waveforms = waveforms.to(device)
        effect_targets = effect_targets.to(device)
        intensity_targets = intensity_targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        effect_logits, intensity_logits = model(waveforms)
        effect_loss = effect_criterion(effect_logits, effect_targets)
        intensity_loss = intensity_criterion(intensity_logits, intensity_targets)
        loss = effect_loss + (intensity_loss_weight * intensity_loss)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

        effect_predictions = torch.argmax(torch.softmax(effect_logits.detach(), dim=1), dim=1)
        intensity_predictions = torch.argmax(torch.softmax(intensity_logits.detach(), dim=1), dim=1)

        losses.append(float(loss.item()))
        effect_losses.append(float(effect_loss.item()))
        intensity_losses.append(float(intensity_loss.item()))
        effect_targets_all.extend(effect_targets.detach().cpu().tolist())
        effect_predictions_all.extend(effect_predictions.detach().cpu().tolist())
        intensity_targets_all.extend(intensity_targets.detach().cpu().tolist())
        intensity_predictions_all.extend(intensity_predictions.detach().cpu().tolist())

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
                f"loss={loss.item():.4f} effect_loss={effect_loss.item():.4f} "
                f"intensity_loss={intensity_loss.item():.4f}",
                flush=True,
            )

    effect_accuracy = float(accuracy_score(effect_targets_all, effect_predictions_all))
    intensity_accuracy = float(accuracy_score(intensity_targets_all, intensity_predictions_all))
    joint_accuracy = float(np.mean([
        int(effect_true == effect_pred and intensity_true == intensity_pred)
        for effect_true, effect_pred, intensity_true, intensity_pred in zip(
            effect_targets_all,
            effect_predictions_all,
            intensity_targets_all,
            intensity_predictions_all,
        )
    ]))
    effect_macro_f1 = float(f1_score(effect_targets_all, effect_predictions_all, average="macro", zero_division=0))
    intensity_macro_f1 = float(f1_score(intensity_targets_all, intensity_predictions_all, average="macro", zero_division=0))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "effect_loss": float(np.mean(effect_losses)) if effect_losses else 0.0,
        "intensity_loss": float(np.mean(intensity_losses)) if intensity_losses else 0.0,
        "effect_accuracy": effect_accuracy,
        "intensity_accuracy": intensity_accuracy,
        "joint_accuracy": joint_accuracy,
        "effect_macro_f1": effect_macro_f1,
        "intensity_macro_f1": intensity_macro_f1,
    }


def fit_multitask_model(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    effect_criterion,
    intensity_criterion,
    device: torch.device,
    num_epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    scheduler=None,
    log_interval: int = 50,
    intensity_loss_weight: float = 1.0,
    gradient_clip_norm: float = 1.0,
) -> tuple[list[dict], dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_bundle: dict | None = None
    best_score: float | None = None
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            effect_criterion=effect_criterion,
            intensity_criterion=intensity_criterion,
            device=device,
            optimizer=optimizer,
            epoch_index=epoch,
            num_epochs=num_epochs,
            stage_name="train",
            log_interval=log_interval,
            intensity_loss_weight=intensity_loss_weight,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            effect_criterion=effect_criterion,
            intensity_criterion=intensity_criterion,
            device=device,
            optimizer=None,
            epoch_index=epoch,
            num_epochs=num_epochs,
            stage_name="valid",
            log_interval=log_interval,
            intensity_loss_weight=intensity_loss_weight,
            gradient_clip_norm=gradient_clip_norm,
        )

        composite_score = (
            (0.45 * val_metrics["effect_macro_f1"])
            + (0.25 * val_metrics["effect_accuracy"])
            + (0.15 * val_metrics["intensity_accuracy"])
            + (0.15 * val_metrics["joint_accuracy"])
        )
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": val_metrics,
            "composite_score": composite_score,
        }
        history.append(epoch_summary)

        print(
            f"[INFO] Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_effect_acc={val_metrics['effect_accuracy']:.4f} | "
            f"val_intensity_acc={val_metrics['intensity_accuracy']:.4f} | "
            f"val_joint_acc={val_metrics['joint_accuracy']:.4f} | "
            f"val_effect_f1={val_metrics['effect_macro_f1']:.4f} | "
            f"score={composite_score:.4f}",
            flush=True,
        )

        if best_score is None or composite_score > best_score:
            best_score = composite_score
            bad_epochs = 0
            best_bundle = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metrics": val_metrics,
                "composite_score": composite_score,
            }
            torch.save(best_bundle, checkpoint_path)
            print(f"[INFO] Saved new best checkpoint to {checkpoint_path}", flush=True)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("[INFO] Early stopping triggered.", flush=True)
            break

    if best_bundle is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")
    return history, best_bundle


@torch.inference_mode()
def predict_audio_file(
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

    effect_scores_all = []
    intensity_scores_all = []
    for window in windows:
        waveform_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        effect_logits, intensity_logits = model(waveform_tensor)
        effect_scores_all.append(torch.softmax(effect_logits, dim=1).detach().cpu().numpy()[0])
        intensity_scores_all.append(torch.softmax(intensity_logits, dim=1).detach().cpu().numpy()[0])

    effect_scores = np.mean(np.asarray(effect_scores_all, dtype=np.float32), axis=0)
    intensity_scores = np.mean(np.asarray(intensity_scores_all, dtype=np.float32), axis=0)
    effect_index = int(np.argmax(effect_scores))
    intensity_index = int(np.argmax(intensity_scores))

    return {
        "predicted_effect_label": INDEX_TO_EFFECT_CLASS[effect_index],
        "predicted_effect_confidence": float(effect_scores[effect_index]),
        "predicted_intensity_level": int(intensity_index),
        "predicted_intensity_display": INTENSITY_LEVEL_TO_DISPLAY[intensity_index],
        "predicted_intensity_confidence": float(intensity_scores[intensity_index]),
        "effect_scores": {
            label: float(effect_scores[index])
            for index, label in enumerate(EFFECT_CLASS_NAMES)
        },
        "intensity_scores": {
            str(level): float(intensity_scores[level])
            for level in INTENSITY_LEVELS
        },
    }


def evaluate_records(
    model: torch.nn.Module,
    records,
    *,
    device: torch.device,
    output_dir: str | Path,
    sample_rate: int,
    segment_seconds: float,
    prefix: str = "val",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    effect_true = []
    effect_pred = []
    intensity_true = []
    intensity_pred = []

    for record in records:
        prediction = predict_audio_file(
            model=model,
            audio_path=record.path,
            device=device,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
        )
        rows.append(
            {
                "file_name": Path(record.path).name,
                "file_path": record.path,
                "relative_path": record.relative_path,
                "true_effect_label": record.effect_label,
                "predicted_effect_label": prediction["predicted_effect_label"],
                "true_intensity_level": record.intensity_level,
                "predicted_intensity_level": prediction["predicted_intensity_level"],
                "true_intensity_display": INTENSITY_LEVEL_TO_DISPLAY[record.intensity_level],
                "predicted_intensity_display": prediction["predicted_intensity_display"],
                "joint_correct": int(
                    record.effect_label == prediction["predicted_effect_label"]
                    and record.intensity_level == prediction["predicted_intensity_level"]
                ),
                **{f"effect_score_{key}": value for key, value in prediction["effect_scores"].items()},
                **{f"intensity_score_{key}": value for key, value in prediction["intensity_scores"].items()},
            }
        )
        effect_true.append(record.effect_label)
        effect_pred.append(prediction["predicted_effect_label"])
        intensity_true.append(record.intensity_level)
        intensity_pred.append(prediction["predicted_intensity_level"])

    effect_accuracy = float(accuracy_score(effect_true, effect_pred))
    intensity_accuracy = float(accuracy_score(intensity_true, intensity_pred))
    joint_accuracy = float(np.mean([
        int(effect_t == effect_p and intensity_t == intensity_p)
        for effect_t, effect_p, intensity_t, intensity_p in zip(
            effect_true, effect_pred, intensity_true, intensity_pred
        )
    ]))
    effect_macro_f1 = float(f1_score(effect_true, effect_pred, average="macro", zero_division=0))
    intensity_macro_f1 = float(f1_score(intensity_true, intensity_pred, average="macro", zero_division=0))

    predictions_path = output_dir / f"{prefix}_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    metrics_path = output_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "effect_accuracy": effect_accuracy,
                "intensity_accuracy": intensity_accuracy,
                "joint_accuracy": joint_accuracy,
                "effect_macro_f1": effect_macro_f1,
                "intensity_macro_f1": intensity_macro_f1,
                "predictions_path": str(predictions_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "effect_accuracy": effect_accuracy,
        "intensity_accuracy": intensity_accuracy,
        "joint_accuracy": joint_accuracy,
        "effect_macro_f1": effect_macro_f1,
        "intensity_macro_f1": intensity_macro_f1,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
    }
