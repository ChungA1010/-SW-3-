"""Dataset helpers for per-axis BEATs intensity models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

from config import (
    ALL_KNOWN_EFFECTS,
    AXIS_NAMES,
    CLASS_SCHEMA,
    EFFECT_TO_AXIS,
    INTENSITY_DISPLAY_TO_LEVEL,
    TARGET_EFFECTS,
    TARGET_INTENSITY_DISPLAYS,
    TOP_LEVEL_TO_AXIS,
)
from shared.audio import load_mono_audio, pad_or_crop_waveform, random_waveform_augmentation

TOP_LEVEL_DIRS = {"Clean", "Drive", "Phase", "Space"}
SUPPORTED_EXTENSIONS = {".wav", ".wave"}
EFFECT_NAME_PATTERN = re.compile(
    rf"^(?P<effect>{'|'.join(TARGET_EFFECTS)})_(?P<intensity>50|100)_.+$",
    re.IGNORECASE,
)
MULTI_EFFECT_PAIR_PATTERN = re.compile(
    rf"(?P<effect>{'|'.join(TARGET_EFFECTS)})_(?P<intensity>50|100)",
    re.IGNORECASE,
)
ANY_EFFECT_PAIR_PATTERN = re.compile(
    rf"(?P<effect>{'|'.join(ALL_KNOWN_EFFECTS)})_(?P<intensity>25|50|75|100)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AxisRecord:
    path: str
    drive_level: int
    phase_level: int
    space_level: int
    relative_path: str
    group_id: str

    def target_for_axis(self, axis_name: str) -> int:
        if axis_name not in AXIS_NAMES:
            raise ValueError(f"Unsupported axis: {axis_name}")
        return int(getattr(self, f"{axis_name}_level"))


def _root_label(root: Path) -> str:
    return root.name or str(root)


def _parse_single_effect_record(audio_path: Path, root: Path, source_name: str) -> AxisRecord | None:
    relative_path = audio_path.relative_to(root)
    if len(relative_path.parts) < 2 or relative_path.parts[0] not in TOP_LEVEL_DIRS:
        return None

    top_level = relative_path.parts[0]
    stem = audio_path.stem
    levels = {axis_name: 0 for axis_name in AXIS_NAMES}
    clip_id = stem

    if top_level != "Clean":
        match = EFFECT_NAME_PATTERN.match(stem)
        if match is None:
            return None
        effect_name = match.group("effect").lower()
        if EFFECT_TO_AXIS[effect_name] != TOP_LEVEL_TO_AXIS[top_level]:
            return None
        level = INTENSITY_DISPLAY_TO_LEVEL[match.group("intensity")]
        axis_name = EFFECT_TO_AXIS[effect_name]
        levels[axis_name] = level
        clip_id = stem.split("_")[-1]

    group_id = f"{source_name}/{audio_path.parent.name}/{clip_id}"
    return AxisRecord(
        path=str(audio_path.resolve()),
        drive_level=levels["drive"],
        phase_level=levels["phase"],
        space_level=levels["space"],
        relative_path=f"{source_name}/{relative_path.as_posix()}",
        group_id=group_id,
    )


def _parse_multi_effect_record(audio_path: Path, root: Path, source_name: str) -> AxisRecord | None:
    relative_path = audio_path.relative_to(root)
    all_pairs = list(ANY_EFFECT_PAIR_PATTERN.finditer(audio_path.stem))
    if not all_pairs:
        return None
    for match in all_pairs:
        effect_name = match.group("effect").lower()
        intensity = match.group("intensity")
        if effect_name not in EFFECT_TO_AXIS or intensity not in TARGET_INTENSITY_DISPLAYS:
            return None

    matches = list(MULTI_EFFECT_PAIR_PATTERN.finditer(audio_path.stem))
    if not matches:
        return None

    levels = {axis_name: 0 for axis_name in AXIS_NAMES}
    for match in matches:
        effect_name = match.group("effect").lower()
        intensity = match.group("intensity")
        axis_name = EFFECT_TO_AXIS[effect_name]
        levels[axis_name] = max(levels[axis_name], INTENSITY_DISPLAY_TO_LEVEL[intensity])

    return AxisRecord(
        path=str(audio_path.resolve()),
        drive_level=levels["drive"],
        phase_level=levels["phase"],
        space_level=levels["space"],
        relative_path=f"{source_name}/{relative_path.as_posix()}",
        group_id=f"{source_name}/{audio_path.stem}",
    )


def _parse_record(audio_path: Path, root: Path, source_name: str) -> AxisRecord | None:
    single_record = _parse_single_effect_record(audio_path, root, source_name)
    if single_record is not None:
        return single_record
    return _parse_multi_effect_record(audio_path, root, source_name)


def _normalize_data_dirs(data_dirs: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(data_dirs, (str, Path)):
        return [Path(data_dirs).expanduser().resolve()]
    return [Path(data_dir).expanduser().resolve() for data_dir in data_dirs]


def scan_dataset(data_dirs: str | Path | Sequence[str | Path]) -> list[AxisRecord]:
    roots = _normalize_data_dirs(data_dirs)
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: list[AxisRecord] = []
    for root in roots:
        source_name = _root_label(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            record = _parse_record(path, root, source_name)
            if record is not None:
                records.append(record)

    if not records:
        root_list = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No labeled audio files were found under: {root_list}")
    return records


def create_train_val_split(records: list[AxisRecord], *, axis_name: str, val_size: float, seed: int) -> dict:
    if axis_name not in AXIS_NAMES:
        raise ValueError(f"Unsupported axis: {axis_name}")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")

    labels = np.array([record.target_for_axis(axis_name) for record in records], dtype=np.int64)
    groups = np.array([record.group_id for record in records], dtype=object)
    unique_group_count = len(set(groups.tolist()))

    strategy = "stratified_shuffle"
    notes: list[str] = []

    if unique_group_count < len(records):
        group_counts: dict[int, set[str]] = {}
        for record, label in zip(records, labels.tolist()):
            group_counts.setdefault(int(label), set()).add(record.group_id)
        min_groups_per_label = min(len(item) for item in group_counts.values())
        desired_splits = max(2, int(round(1.0 / val_size)))
        n_splits = min(desired_splits, min_groups_per_label)
        if n_splits >= 2:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_indices, val_indices = next(
                splitter.split(
                    X=np.zeros(len(records), dtype=np.float32),
                    y=labels,
                    groups=groups,
                )
            )
            strategy = f"stratified_group_kfold_{n_splits}fold"
            notes.append("Group-aware split was used to keep matching riffs together.")
        else:
            train_indices, val_indices = train_test_split(
                np.arange(len(records)),
                test_size=val_size,
                random_state=seed,
                stratify=labels,
            )
    else:
        train_indices, val_indices = train_test_split(
            np.arange(len(records)),
            test_size=val_size,
            random_state=seed,
            stratify=labels,
        )

    return {
        "axis_name": axis_name,
        "seed": seed,
        "val_size": val_size,
        "strategy": strategy,
        "notes": notes,
        "train_records": [records[index] for index in sorted(train_indices.tolist())],
        "val_records": [records[index] for index in sorted(val_indices.tolist())],
    }


def save_split_manifest(split: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "axis_name": split["axis_name"],
        "seed": split["seed"],
        "val_size": split["val_size"],
        "strategy": split["strategy"],
        "notes": split["notes"],
        "data_dirs": split.get("data_dirs", []),
        "class_schema": split.get("class_schema", CLASS_SCHEMA),
        "train_records": [asdict(record) for record in split["train_records"]],
        "val_records": [asdict(record) for record in split["val_records"]],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


@dataclass
class WaveformDatasetConfig:
    axis_name: str
    sample_rate: int
    segment_seconds: float
    training: bool
    augment: bool


class AxisWaveformDataset(Dataset):
    def __init__(self, records: list[AxisRecord], config: WaveformDatasetConfig) -> None:
        self.records = records
        self.config = config
        self.target_num_samples = int(config.sample_rate * config.segment_seconds)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        waveform = load_mono_audio(record.path, sample_rate=self.config.sample_rate)
        waveform = pad_or_crop_waveform(
            waveform,
            target_num_samples=self.target_num_samples,
            training=self.config.training,
        )
        if self.config.training and self.config.augment:
            waveform = random_waveform_augmentation(waveform, np.random.default_rng())
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        target = torch.tensor(record.target_for_axis(self.config.axis_name), dtype=torch.long)
        return waveform_tensor, target
