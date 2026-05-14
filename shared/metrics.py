"""Evaluation helpers shared by every experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_probabilities(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def ensure_probabilities(y_scores: np.ndarray) -> np.ndarray:
    """Normalize outputs so downstream reporting can always use class scores."""
    y_scores = np.asarray(y_scores, dtype=np.float32)
    if y_scores.ndim != 2:
        raise ValueError("Expected y_scores to have shape [num_samples, num_classes]")
    row_sums = np.sum(y_scores, axis=1, keepdims=True)
    if np.allclose(row_sums, 1.0, atol=1e-3):
        return y_scores
    return _as_probabilities(y_scores)


def align_scores_to_class_names(
    y_scores: np.ndarray,
    source_class_names: Sequence[str],
    target_class_names: Sequence[str],
) -> np.ndarray:
    """Reorder score columns to a desired class-name order."""
    y_scores = np.asarray(y_scores, dtype=np.float32)
    if list(source_class_names) == list(target_class_names):
        return y_scores
    source_index = {label: idx for idx, label in enumerate(source_class_names)}
    aligned = np.zeros((y_scores.shape[0], len(target_class_names)), dtype=np.float32)
    for target_index, label in enumerate(target_class_names):
        aligned[:, target_index] = y_scores[:, source_index[label]]
    return aligned


def build_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
) -> dict:
    """Compute the main classification metrics requested for the project."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(class_names), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(class_names), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(class_names), average="macro", zero_division=0)),
        "report": report,
    }


def save_classification_outputs(
    output_dir: str | Path,
    prefix: str,
    class_names: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_scores: np.ndarray,
    sample_ids: Iterable[str] | None = None,
) -> dict:
    """Save metrics JSON, predictions CSV, and confusion-matrix PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_scores = ensure_probabilities(np.asarray(y_scores, dtype=np.float32))
    metrics = build_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    matrix = confusion_matrix(y_true, y_pred, labels=list(class_names))
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )

    payload = {
        "class_names": list(class_names),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }
    metrics_path = output_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    prediction_rows = []
    sample_ids = list(sample_ids) if sample_ids is not None else [f"sample_{idx:04d}" for idx in range(len(y_true))]
    for sample_id, truth, prediction, scores in zip(sample_ids, y_true, y_pred, y_scores):
        row = {
            "sample_id": sample_id,
            "true_label": truth,
            "predicted_label": prediction,
        }
        for class_name, score in zip(class_names, scores.tolist()):
            row[f"score_{class_name}"] = float(score)
        prediction_rows.append(row)

    predictions_path = output_dir / f"{prefix}_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{prefix} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                format(matrix[row_index, col_index], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > threshold else "black",
            )

    figure.tight_layout()
    confusion_path = output_dir / f"{prefix}_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print_metrics(payload["metrics"], matrix_frame)
    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "confusion_matrix_path": str(confusion_path),
    }


def print_metrics(metrics: dict, matrix_frame: pd.DataFrame) -> None:
    """Print a concise evaluation summary to the terminal."""
    report_frame = pd.DataFrame(metrics["report"]).transpose()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(report_frame.to_string())
    print("\nConfusion matrix:")
    print(matrix_frame.to_string())
