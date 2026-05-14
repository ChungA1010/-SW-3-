"""Dataset helpers for the BEATs multi-axis intensity experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

from config import AXIS_NAMES, INTENSITY_DISPLAY_TO_LEVEL, TOP_LEVEL_TO_AXIS
from shared.audio import load_mono_audio, pad_or_crop_waveform, random_waveform_augmentation

TOP_LEVEL_DIRS = {"Clean", "Drive", "Phase", "Space"}
SUPPORTED_EXTENSIONS = {".wav", ".wave"}
EFFECT_NAME_PATTERN = re.compile(
    r"^(?P<effect>dist|chorus|phaser|delay|reverb)_(?P<intensity>25|50|75|100)_.+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MultiAxisRecord:
    path: str
    drive_level: int
    phase_level: int
    space_level: int
    relative_path: str
    group_id: str


def _parse_record(audio_path: Path, root: Path) -> MultiAxisRecord | None:
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
        level = INTENSITY_DISPLAY_TO_LEVEL[match.group("intensity")]
        axis_name = TOP_LEVEL_TO_AXIS[top_level]
        levels[axis_name] = level
        clip_id = stem.split("_")[-1]

    group_id = f"{audio_path.parent.name}/{clip_id}"
    return MultiAxisRecord(
        path=str(audio_path.resolve()),
        drive_level=levels["drive"],
        phase_level=levels["phase"],
        space_level=levels["space"],
        relative_path=str(relative_path),
        group_id=group_id,
    )


def scan_dataset(data_dir: str | Path) -> list[MultiAxisRecord]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: list[MultiAxisRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        record = _parse_record(path, root)
        if record is not None:
            records.append(record)

    if not records:
        raise FileNotFoundError(f"No labeled audio files were found under {root}")
    return records


def _joint_label(record: MultiAxisRecord) -> str:
    return f"{record.drive_level}:{record.phase_level}:{record.space_level}"


def create_train_val_split(records: list[MultiAxisRecord], val_size: float, seed: int) -> dict:
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")

    labels = np.array([_joint_label(record) for record in records], dtype=object)
    groups = np.array([record.group_id for record in records], dtype=object)
    unique_group_count = len(set(groups.tolist()))

    strategy = "stratified_shuffle"
    notes: list[str] = []

    if unique_group_count < len(records):
        group_counts: dict[str, set[str]] = {}
        for record, label in zip(records, labels.tolist()):
            group_counts.setdefault(label, set()).add(record.group_id)
        min_groups_per_label = min(len(item) for item in group_counts.values())
        desired_splits = max(2, int(round(1.0 / val_size)))
        n_splits = min(desired_splits, min_groups_per_label)
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
        "seed": split["seed"],
        "val_size": split["val_size"],
        "strategy": split["strategy"],
        "notes": split["notes"],
        "train_records": [asdict(record) for record in split["train_records"]],
        "val_records": [asdict(record) for record in split["val_records"]],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


@dataclass
class WaveformDatasetConfig:
    sample_rate: int
    segment_seconds: float
    training: bool
    augment: bool


class MultiAxisWaveformDataset(Dataset):
    def __init__(self, records: list[MultiAxisRecord], config: WaveformDatasetConfig) -> None:
        self.records = records
        self.config = config
        self.target_num_samples = int(config.sample_rate * config.segment_seconds)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return (
            waveform_tensor,
            torch.tensor(record.drive_level, dtype=torch.long),
            torch.tensor(record.phase_level, dtype=torch.long),
            torch.tensor(record.space_level, dtype=torch.long),
        )

