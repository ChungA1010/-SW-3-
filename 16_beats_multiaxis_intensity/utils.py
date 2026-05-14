"""Training and inference helpers for the BEATs multi-axis intensity model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import AXIS_NAMES, INTENSITY_LEVELS, INTENSITY_LEVEL_TO_DISPLAY
from dataset import MultiAxisWaveformDataset, WaveformDatasetConfig
from model import BEATsMultiAxisIntensity
from shared.audio import load_mono_audio, sliding_window_waveforms


def create_dataloaders(train_records, val_records, *, sample_rate: int, segment_seconds: float, batch_size: int, num_workers: int):
    train_dataset = MultiAxisWaveformDataset(
        train_records,
        WaveformDatasetConfig(sample_rate=sample_rate, segment_seconds=segment_seconds, training=True, augment=True),
    )
    val_dataset = MultiAxisWaveformDataset(
        val_records,
        WaveformDatasetConfig(sample_rate=sample_rate, segment_seconds=segment_seconds, training=False, augment=False),
    )

    class_counts: dict[str, int] = {}
    for record in train_records:
        key = f"{record.drive_level}:{record.phase_level}:{record.space_level}"
        class_counts[key] = class_counts.get(key, 0) + 1
    sample_weights = [
        1.0 / class_counts[f"{record.drive_level}:{record.phase_level}:{record.space_level}"]
        for record in train_records
    ]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def build_model(pretrained_checkpoint_path: str | Path | None, *, backbone_config_override: dict | None = None) -> BEATsMultiAxisIntensity:
    return BEATsMultiAxisIntensity(
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        backbone_config_override=backbone_config_override,
    )


def class_weights_from_labels(labels: list[int], num_classes: int = 5) -> torch.Tensor:
    values = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    total = counts.sum()
    for value, count in zip(unique, counts):
        weights[int(value)] = float(total / (len(unique) * count))
    return torch.tensor(weights, dtype=torch.float32)


def create_optimizer(model: BEATsMultiAxisIntensity, *, backbone_learning_rate: float, head_learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
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


def _axis_metrics(targets: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    drive_criterion,
    phase_criterion,
    space_criterion,
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
    drive_true: list[int] = []
    phase_true: list[int] = []
    space_true: list[int] = []
    drive_pred: list[int] = []
    phase_pred: list[int] = []
    space_pred: list[int] = []

    total_batches = len(loader)
    for batch_index, (waveforms, drive_targets, phase_targets, space_targets) in enumerate(loader, start=1):
        waveforms = waveforms.to(device)
        drive_targets = drive_targets.to(device)
        phase_targets = phase_targets.to(device)
        space_targets = space_targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        drive_logits, phase_logits, space_logits = model(waveforms)
        drive_loss = drive_criterion(drive_logits, drive_targets)
        phase_loss = phase_criterion(phase_logits, phase_targets)
        space_loss = space_criterion(space_logits, space_targets)
        loss = drive_loss + phase_loss + space_loss

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

        drive_predictions = torch.argmax(torch.softmax(drive_logits.detach(), dim=1), dim=1)
        phase_predictions = torch.argmax(torch.softmax(phase_logits.detach(), dim=1), dim=1)
        space_predictions = torch.argmax(torch.softmax(space_logits.detach(), dim=1), dim=1)

        losses.append(float(loss.item()))
        drive_true.extend(drive_targets.detach().cpu().tolist())
        phase_true.extend(phase_targets.detach().cpu().tolist())
        space_true.extend(space_targets.detach().cpu().tolist())
        drive_pred.extend(drive_predictions.detach().cpu().tolist())
        phase_pred.extend(phase_predictions.detach().cpu().tolist())
        space_pred.extend(space_predictions.detach().cpu().tolist())

        should_log = total_batches > 0 and (batch_index == 1 or batch_index == total_batches or (log_interval > 0 and batch_index % log_interval == 0))
        if should_log:
            prefix = f"[INFO] {stage_name}"
            if epoch_index is not None and num_epochs is not None:
                prefix += f" epoch {epoch_index:03d}/{num_epochs:03d}"
            print(
                f"{prefix} batch {batch_index:04d}/{total_batches:04d} "
                f"loss={loss.item():.4f} drive_loss={drive_loss.item():.4f} "
                f"phase_loss={phase_loss.item():.4f} space_loss={space_loss.item():.4f}",
                flush=True,
            )

    drive_metrics = _axis_metrics(drive_true, drive_pred)
    phase_metrics = _axis_metrics(phase_true, phase_pred)
    space_metrics = _axis_metrics(space_true, space_pred)
    joint_accuracy = float(np.mean([
        int(d_t == d_p and p_t == p_p and s_t == s_p)
        for d_t, d_p, p_t, p_p, s_t, s_p in zip(drive_true, drive_pred, phase_true, phase_pred, space_true, space_pred)
    ]))

    mean_axis_accuracy = float(np.mean([drive_metrics["accuracy"], phase_metrics["accuracy"], space_metrics["accuracy"]]))
    mean_axis_f1 = float(np.mean([drive_metrics["macro_f1"], phase_metrics["macro_f1"], space_metrics["macro_f1"]]))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "joint_accuracy": joint_accuracy,
        "mean_axis_accuracy": mean_axis_accuracy,
        "mean_axis_f1": mean_axis_f1,
        "drive_accuracy": drive_metrics["accuracy"],
        "phase_accuracy": phase_metrics["accuracy"],
        "space_accuracy": space_metrics["accuracy"],
        "drive_macro_f1": drive_metrics["macro_f1"],
        "phase_macro_f1": phase_metrics["macro_f1"],
        "space_macro_f1": space_metrics["macro_f1"],
    }


def fit_multiaxis_model(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    drive_criterion,
    phase_criterion,
    space_criterion,
    device: torch.device,
    num_epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    scheduler=None,
    log_interval: int = 50,
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
            model=model, loader=train_loader, drive_criterion=drive_criterion, phase_criterion=phase_criterion, space_criterion=space_criterion,
            device=device, optimizer=optimizer, epoch_index=epoch, num_epochs=num_epochs, stage_name="train",
            log_interval=log_interval, gradient_clip_norm=gradient_clip_norm,
        )
        val_metrics = _run_epoch(
            model=model, loader=val_loader, drive_criterion=drive_criterion, phase_criterion=phase_criterion, space_criterion=space_criterion,
            device=device, optimizer=None, epoch_index=epoch, num_epochs=num_epochs, stage_name="valid",
            log_interval=log_interval, gradient_clip_norm=gradient_clip_norm,
        )

        composite_score = (
            (0.40 * val_metrics["mean_axis_f1"])
            + (0.25 * val_metrics["mean_axis_accuracy"])
            + (0.35 * val_metrics["joint_accuracy"])
        )
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        history.append({"epoch": epoch, "train": train_metrics, "valid": val_metrics, "composite_score": composite_score})
        print(
            f"[INFO] Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"drive_acc={val_metrics['drive_accuracy']:.4f} | phase_acc={val_metrics['phase_accuracy']:.4f} | "
            f"space_acc={val_metrics['space_accuracy']:.4f} | joint_acc={val_metrics['joint_accuracy']:.4f} | score={composite_score:.4f}",
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
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")
    return history, best_bundle


@torch.inference_mode()
def predict_audio_file(model: torch.nn.Module, audio_path: str | Path, *, device: torch.device, sample_rate: int, segment_seconds: float) -> dict:
    model.eval()
    waveform = load_mono_audio(audio_path, sample_rate=sample_rate)
    window_size = int(segment_seconds * sample_rate)
    hop_size = max(window_size // 2, 1)
    windows = sliding_window_waveforms(waveform, window_size=window_size, hop_size=hop_size)

    drive_scores_all = []
    phase_scores_all = []
    space_scores_all = []
    for window in windows:
        waveform_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        drive_logits, phase_logits, space_logits = model(waveform_tensor)
        drive_scores_all.append(torch.softmax(drive_logits, dim=1).detach().cpu().numpy()[0])
        phase_scores_all.append(torch.softmax(phase_logits, dim=1).detach().cpu().numpy()[0])
        space_scores_all.append(torch.softmax(space_logits, dim=1).detach().cpu().numpy()[0])

    axis_probabilities = {
        "drive": np.mean(np.asarray(drive_scores_all, dtype=np.float32), axis=0),
        "phase": np.mean(np.asarray(phase_scores_all, dtype=np.float32), axis=0),
        "space": np.mean(np.asarray(space_scores_all, dtype=np.float32), axis=0),
    }

    axes = {}
    for axis_name in AXIS_NAMES:
        scores = axis_probabilities[axis_name]
        pred_level = int(np.argmax(scores))
        axes[axis_name] = {
            "detected": bool(pred_level > 0),
            "intensity_level": pred_level,
            "intensity_display": INTENSITY_LEVEL_TO_DISPLAY[pred_level],
            "confidence": float(scores[pred_level]),
            "scores": {str(level): float(scores[level]) for level in INTENSITY_LEVELS},
        }

    active_axes = {axis_name: payload for axis_name, payload in axes.items() if payload["detected"]}
    dominant_axis = None
    if active_axes:
        dominant_axis = max(active_axes.items(), key=lambda item: item[1]["confidence"])[0]
    return {
        "format_version": "multiaxis-v2",
        "mode": "dedicated_multiaxis_heads",
        "dominant_axis": dominant_axis,
        "axes": axes,
    }


def evaluate_records(model: torch.nn.Module, records, *, device: torch.device, output_dir: str | Path, sample_rate: int, segment_seconds: float, prefix: str = "val") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    joint_correct = []
    axis_correct: dict[str, list[int]] = {axis_name: [] for axis_name in AXIS_NAMES}
    axis_true: dict[str, list[int]] = {axis_name: [] for axis_name in AXIS_NAMES}
    axis_pred: dict[str, list[int]] = {axis_name: [] for axis_name in AXIS_NAMES}

    for record in records:
        prediction = predict_audio_file(model=model, audio_path=record.path, device=device, sample_rate=sample_rate, segment_seconds=segment_seconds)
        row = {"file_name": Path(record.path).name, "file_path": record.path, "relative_path": record.relative_path}
        correct_flags = []
        for axis_name in AXIS_NAMES:
            true_level = getattr(record, f"{axis_name}_level")
            pred_level = prediction["axes"][axis_name]["intensity_level"]
            row[f"true_{axis_name}_level"] = true_level
            row[f"predicted_{axis_name}_level"] = pred_level
            row[f"predicted_{axis_name}_display"] = prediction["axes"][axis_name]["intensity_display"]
            row[f"{axis_name}_confidence"] = prediction["axes"][axis_name]["confidence"]
            is_correct = int(true_level == pred_level)
            row[f"{axis_name}_correct"] = is_correct
            correct_flags.append(is_correct)
            axis_correct[axis_name].append(is_correct)
            axis_true[axis_name].append(true_level)
            axis_pred[axis_name].append(pred_level)
        row["joint_correct"] = int(all(correct_flags))
        joint_correct.append(row["joint_correct"])
        row["multiaxis_json"] = json.dumps(prediction, ensure_ascii=False)
        rows.append(row)

    predictions_path = output_dir / f"{prefix}_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    metrics = {
        "joint_accuracy": float(np.mean(joint_correct)),
        "predictions_path": str(predictions_path),
    }
    for axis_name in AXIS_NAMES:
        metrics[f"{axis_name}_accuracy"] = float(np.mean(axis_correct[axis_name]))
        metrics[f"{axis_name}_macro_f1"] = float(f1_score(axis_true[axis_name], axis_pred[axis_name], average="macro", zero_division=0))

    metrics_path = output_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics

