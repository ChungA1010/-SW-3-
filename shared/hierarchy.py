"""Helpers for fine-to-coarse hierarchical classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import COARSE_CLASS_NAMES, FINE_CLASS_NAMES, FINE_TO_COARSE
from .metrics import save_classification_outputs


def map_fine_label_to_coarse(label: str) -> str:
    """Map a fine effect label to its coarse label."""
    return FINE_TO_COARSE[str(label)]


def fine_scores_to_coarse_scores(
    fine_scores: np.ndarray,
    fine_class_names: Sequence[str] = FINE_CLASS_NAMES,
    coarse_class_names: Sequence[str] = COARSE_CLASS_NAMES,
) -> np.ndarray:
    """Aggregate fine-class scores into clean/drive/space/phase scores."""
    fine_scores = np.asarray(fine_scores, dtype=np.float32)
    coarse_scores = np.zeros((fine_scores.shape[0], len(coarse_class_names)), dtype=np.float32)
    coarse_index = {label: idx for idx, label in enumerate(coarse_class_names)}
    for fine_index, fine_label in enumerate(fine_class_names):
        mapped_label = map_fine_label_to_coarse(fine_label)
        coarse_scores[:, coarse_index[mapped_label]] += fine_scores[:, fine_index]
    return coarse_scores


def prediction_dict_from_fine_scores(
    fine_scores_1d: np.ndarray,
    fine_class_names: Sequence[str] = FINE_CLASS_NAMES,
    coarse_class_names: Sequence[str] = COARSE_CLASS_NAMES,
) -> dict:
    """Build a unified inference output from one fine-score vector."""
    fine_scores_1d = np.asarray(fine_scores_1d, dtype=np.float32)
    if fine_scores_1d.ndim != 1:
        raise ValueError("Expected a 1D fine score vector for inference.")
    fine_index = int(np.argmax(fine_scores_1d))
    predicted_fine_label = fine_class_names[fine_index]
    coarse_scores = fine_scores_to_coarse_scores(
        fine_scores_1d[np.newaxis, :],
        fine_class_names=fine_class_names,
        coarse_class_names=coarse_class_names,
    )[0]
    coarse_index = int(np.argmax(coarse_scores))
    return {
        "predicted_fine_label": predicted_fine_label,
        "predicted_coarse_label": coarse_class_names[coarse_index],
        "fine_scores": {label: float(fine_scores_1d[idx]) for idx, label in enumerate(fine_class_names)},
        "coarse_scores": {label: float(coarse_scores[idx]) for idx, label in enumerate(coarse_class_names)},
    }


def save_hierarchical_outputs(
    *,
    output_dir: str | Path,
    prefix: str,
    fine_truth: Sequence[str],
    fine_pred: Sequence[str],
    fine_scores: np.ndarray,
    sample_ids: Sequence[str],
    fine_class_names: Sequence[str] = FINE_CLASS_NAMES,
    coarse_class_names: Sequence[str] = COARSE_CLASS_NAMES,
) -> dict:
    """Save both fine-label and mapped coarse-label evaluation outputs."""
    output_dir = Path(output_dir)
    fine_scores = np.asarray(fine_scores, dtype=np.float32)

    fine_results = save_classification_outputs(
        output_dir=output_dir,
        prefix=f"{prefix}_fine13",
        class_names=fine_class_names,
        y_true=fine_truth,
        y_pred=fine_pred,
        y_scores=fine_scores,
        sample_ids=sample_ids,
    )

    coarse_truth = [map_fine_label_to_coarse(label) for label in fine_truth]
    coarse_pred = [map_fine_label_to_coarse(label) for label in fine_pred]
    coarse_scores = fine_scores_to_coarse_scores(
        fine_scores,
        fine_class_names=fine_class_names,
        coarse_class_names=coarse_class_names,
    )
    coarse_results = save_classification_outputs(
        output_dir=output_dir,
        prefix=f"{prefix}_coarse4",
        class_names=coarse_class_names,
        y_true=coarse_truth,
        y_pred=coarse_pred,
        y_scores=coarse_scores,
        sample_ids=sample_ids,
    )

    rows = []
    for sample_id, fine_true_label, fine_pred_label, row_scores, row_coarse_scores in zip(
        sample_ids,
        fine_truth,
        fine_pred,
        fine_scores,
        coarse_scores,
    ):
        row = {
            "sample_id": sample_id,
            "true_fine_label": fine_true_label,
            "predicted_fine_label": fine_pred_label,
            "true_coarse_label": map_fine_label_to_coarse(fine_true_label),
            "predicted_coarse_label": map_fine_label_to_coarse(fine_pred_label),
        }
        for label, score in zip(fine_class_names, row_scores.tolist()):
            row[f"fine_score_{label}"] = float(score)
        for label, score in zip(coarse_class_names, row_coarse_scores.tolist()):
            row[f"coarse_score_{label}"] = float(score)
        rows.append(row)

    combined_predictions_path = output_dir / f"{prefix}_combined_predictions.csv"
    pd.DataFrame(rows).to_csv(combined_predictions_path, index=False)
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fine_metrics_path": fine_results["metrics_path"],
                "coarse_metrics_path": coarse_results["metrics_path"],
                "combined_predictions_path": str(combined_predictions_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "fine": fine_results,
        "coarse": coarse_results,
        "combined_predictions_path": str(combined_predictions_path),
        "summary_path": str(summary_path),
    }
