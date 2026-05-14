"""Leakage-aware train/validation split helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .audio import AudioRecord, deserialize_records, dump_json, serialize_records
from .constants import FINE_CLASS_NAMES


def _class_group_counts(records: Iterable[AudioRecord]) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[record.label].add(record.group_id)
    return {label: len(grouped[label]) for label in grouped}


def create_train_val_split(
    records: list[AudioRecord],
    val_size: float,
    seed: int,
    class_names: list[str] | None = None,
) -> dict:
    """
    Create a deterministic train/validation split.

    The function first attempts a grouped split if filename patterns suggest
    multiple related clips might exist. If that is not possible, it falls back
    to stratified file-level splitting.
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")

    labels = np.array([record.label for record in records], dtype=object)
    groups = np.array([record.group_id for record in records], dtype=object)

    expected_class_names = class_names or FINE_CLASS_NAMES
    unique_labels = sorted(set(labels.tolist()))
    if sorted(unique_labels) != sorted(expected_class_names):
        raise ValueError(
            f"Expected exactly these labels: {expected_class_names}, but found {unique_labels}"
        )

    strategy = "stratified_shuffle"
    notes: list[str] = []
    train_indices: np.ndarray
    val_indices: np.ndarray

    unique_group_count = len(set(groups.tolist()))
    if unique_group_count < len(records):
        group_counts = _class_group_counts(records)
        min_groups_per_class = min(group_counts.values())
        desired_splits = max(2, int(round(1.0 / val_size)))
        n_splits = min(desired_splits, min_groups_per_class)

        if n_splits >= 2:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            train_indices, val_indices = next(
                splitter.split(
                    X=np.zeros(len(records), dtype=np.float32),
                    y=labels,
                    groups=groups,
                )
            )
            strategy = f"stratified_group_kfold_{n_splits}fold"
            notes.append(
                "Group-aware split was used because repeated filename stems were detected."
            )
        else:
            notes.append(
                "Detected related filenames, but there were too few groups per class to keep group-aware stratification."
            )

    if strategy == "stratified_shuffle":
        label_counts = {label: int(np.sum(labels == label)) for label in unique_labels}
        if min(label_counts.values()) < 2:
            raise ValueError(
                "Each class needs at least 2 files for a stable train/validation split."
            )
        train_indices, val_indices = train_test_split(
            np.arange(len(records)),
            test_size=val_size,
            random_state=seed,
            stratify=labels,
        )

    train_records = [records[index] for index in sorted(train_indices.tolist())]
    val_records = [records[index] for index in sorted(val_indices.tolist())]

    return {
        "class_names": expected_class_names,
        "seed": seed,
        "val_size": val_size,
        "strategy": strategy,
        "notes": notes,
        "train_records": train_records,
        "val_records": val_records,
    }


def save_split_manifest(split: dict, output_path: str | Path) -> Path:
    """Serialize a split manifest so evaluation can reuse the exact same files."""
    payload = {
        "class_names": split["class_names"],
        "seed": split["seed"],
        "val_size": split["val_size"],
        "strategy": split["strategy"],
        "notes": split["notes"],
        "train_records": serialize_records(split["train_records"]),
        "val_records": serialize_records(split["val_records"]),
    }
    return dump_json(payload, output_path)


def load_split_manifest(input_path: str | Path) -> dict:
    """Load a split manifest saved by `save_split_manifest`."""
    input_path = Path(input_path)
    payload = input_path.read_text(encoding="utf-8")
    data = __import__("json").loads(payload)
    return {
        "class_names": data["class_names"],
        "seed": data["seed"],
        "val_size": data["val_size"],
        "strategy": data["strategy"],
        "notes": data.get("notes", []),
        "train_records": deserialize_records(data["train_records"]),
        "val_records": deserialize_records(data["val_records"]),
    }
