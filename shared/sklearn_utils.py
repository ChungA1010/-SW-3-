"""Shared sklearn workflows for feature-based experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio import dump_json, ensure_directory, scan_dataset
from .constants import COARSE_CLASS_NAMES, FINE_CLASS_NAMES
from .features import (
    dataframe_to_matrix,
    extract_baseline_features,
    extract_engineered_features,
    extract_feature_table,
    feature_columns_from_dataframe,
)
from .hierarchy import prediction_dict_from_fine_scores, save_hierarchical_outputs
from .metrics import align_scores_to_class_names
from .splits import create_train_val_split, load_split_manifest, save_split_manifest


def _pipeline_classes(pipeline: Pipeline) -> list[str] | None:
    if hasattr(pipeline, "classes_"):
        return [str(label) for label in pipeline.classes_]
    final_estimator = pipeline.steps[-1][1]
    if hasattr(final_estimator, "classes_"):
        return [str(label) for label in final_estimator.classes_]
    return None


def _predict_scores(pipeline: Pipeline, matrix: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(matrix)
        pipeline_classes = _pipeline_classes(pipeline)
        if pipeline_classes is not None:
            return align_scores_to_class_names(scores, pipeline_classes, class_names)
        return scores
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(matrix)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    predictions = pipeline.predict(matrix)
    scores = np.zeros((len(predictions), len(class_names)), dtype=np.float32)
    class_to_index = {label: idx for idx, label in enumerate(class_names)}
    for row_index, label in enumerate(predictions):
        scores[row_index, class_to_index[str(label)]] = 1.0
    return scores


def build_feature_cache(
    data_dir: str | Path,
    extractor_name: str,
    split_manifest_path: str | Path,
    feature_cache_path: str | Path,
    sample_rate: int,
    val_size: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Create a reproducible split and cache the extracted features on disk."""
    records = scan_dataset(data_dir, class_names=FINE_CLASS_NAMES)
    split = create_train_val_split(records, val_size=val_size, seed=seed, class_names=FINE_CLASS_NAMES)
    save_split_manifest(split, split_manifest_path)

    train_frame = extract_feature_table(split["train_records"], extractor_name=extractor_name, sample_rate=sample_rate)
    train_frame["split"] = "train"
    val_frame = extract_feature_table(split["val_records"], extractor_name=extractor_name, sample_rate=sample_rate)
    val_frame["split"] = "val"

    combined = pd.concat([train_frame, val_frame], axis=0, ignore_index=True)
    feature_cache_path = Path(feature_cache_path)
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(feature_cache_path, index=False)
    return split, combined


def _build_pipeline(estimator, scale_features: bool) -> Pipeline:
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def train_feature_experiment(
    *,
    data_dir: str | Path,
    artifact_path: str | Path,
    feature_cache_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    extractor_name: str,
    estimator,
    scale_features: bool,
    sample_rate: int,
    val_size: float,
    seed: int,
    extra_config: dict | None = None,
) -> dict:
    """Train a classical ML experiment and save a reusable artifact."""
    artifact_path = Path(artifact_path)
    feature_cache_path = Path(feature_cache_path)
    split_manifest_path = Path(split_manifest_path)
    output_dir = ensure_directory(output_dir)

    split, combined = build_feature_cache(
        data_dir=data_dir,
        extractor_name=extractor_name,
        split_manifest_path=split_manifest_path,
        feature_cache_path=feature_cache_path,
        sample_rate=sample_rate,
        val_size=val_size,
        seed=seed,
    )

    feature_columns = feature_columns_from_dataframe(combined)
    train_frame = combined[combined["split"] == "train"].reset_index(drop=True)
    val_frame = combined[combined["split"] == "val"].reset_index(drop=True)
    x_train = dataframe_to_matrix(train_frame, feature_columns)
    x_val = dataframe_to_matrix(val_frame, feature_columns)
    y_train = train_frame["label"].astype(str).tolist()
    y_val = val_frame["label"].astype(str).tolist()

    pipeline = _build_pipeline(estimator=estimator, scale_features=scale_features)
    pipeline.fit(x_train, y_train)

    val_scores = _predict_scores(pipeline, x_val, class_names=FINE_CLASS_NAMES)
    val_predictions = pipeline.predict(x_val).tolist()
    outputs = save_hierarchical_outputs(
        output_dir=output_dir,
        prefix="val",
        fine_truth=y_val,
        fine_pred=val_predictions,
        fine_scores=val_scores,
        sample_ids=val_frame["relative_path"].tolist(),
    )

    artifact = {
        "fine_class_names": FINE_CLASS_NAMES,
        "coarse_class_names": COARSE_CLASS_NAMES,
        "extractor_name": extractor_name,
        "feature_columns": feature_columns,
        "pipeline": pipeline,
        "split_manifest_path": str(split_manifest_path.resolve()),
        "feature_cache_path": str(feature_cache_path.resolve()),
        "metrics_path": outputs["coarse"]["metrics_path"],
        "scale_features": scale_features,
        "extra_config": extra_config or {},
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    dump_json(
        {
            "artifact_path": str(artifact_path.resolve()),
            "feature_cache_path": str(feature_cache_path.resolve()),
            "split_manifest_path": str(split_manifest_path.resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "extractor_name": extractor_name,
            "scale_features": scale_features,
            "extra_config": extra_config or {},
        },
        artifact_path.with_suffix(".metadata.json"),
    )
    print(f"[INFO] Saved artifact: {artifact_path.resolve()}")
    return artifact


def evaluate_feature_experiment(
    *,
    artifact_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Re-run validation evaluation from a saved artifact."""
    artifact = joblib.load(artifact_path)
    feature_cache_path = Path(artifact["feature_cache_path"])
    split_manifest_path = Path(artifact["split_manifest_path"])
    output_dir = ensure_directory(output_dir)

    if feature_cache_path.exists():
        combined = pd.read_csv(feature_cache_path)
    else:
        split = load_split_manifest(split_manifest_path)
        train_frame = extract_feature_table(split["train_records"], extractor_name=artifact["extractor_name"], sample_rate=artifact["extra_config"]["sample_rate"])
        train_frame["split"] = "train"
        val_frame = extract_feature_table(split["val_records"], extractor_name=artifact["extractor_name"], sample_rate=artifact["extra_config"]["sample_rate"])
        val_frame["split"] = "val"
        combined = pd.concat([train_frame, val_frame], axis=0, ignore_index=True)
        combined.to_csv(feature_cache_path, index=False)

    val_frame = combined[combined["split"] == "val"].reset_index(drop=True)
    x_val = dataframe_to_matrix(val_frame, artifact["feature_columns"])
    y_val = val_frame["label"].astype(str).tolist()
    scores = _predict_scores(artifact["pipeline"], x_val, class_names=artifact["fine_class_names"])
    predictions = artifact["pipeline"].predict(x_val).tolist()
    return save_hierarchical_outputs(
        output_dir=output_dir,
        prefix="val",
        fine_truth=y_val,
        fine_pred=predictions,
        fine_scores=scores,
        sample_ids=val_frame["relative_path"].tolist(),
    )


def infer_feature_experiment(
    *,
    artifact_path: str | Path,
    audio_path: str | Path,
) -> dict:
    """Predict a single wav file using fine classification and coarse mapping."""
    artifact = joblib.load(artifact_path)
    extractor_name = artifact["extractor_name"]
    sample_rate = artifact["extra_config"].get("sample_rate")
    extractor = extract_baseline_features if extractor_name == "baseline" else extract_engineered_features
    feature_dict = extractor(audio_path, sample_rate=sample_rate)
    feature_row = {column: 0.0 for column in artifact["feature_columns"]}
    feature_row.update(feature_dict)
    matrix = pd.DataFrame([feature_row])[artifact["feature_columns"]].fillna(0.0).to_numpy(dtype=np.float32)
    scores = _predict_scores(artifact["pipeline"], matrix, class_names=artifact["fine_class_names"])[0]
    return prediction_dict_from_fine_scores(
        scores,
        fine_class_names=artifact["fine_class_names"],
        coarse_class_names=artifact["coarse_class_names"],
    )
