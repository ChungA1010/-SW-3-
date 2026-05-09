"""Dataset helpers for the effect + intensity 2D CNN experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

from config import (
    EFFECT_CLASS_NAMES,
    EFFECT_CLASS_TO_INDEX,
    INTENSITY_DISPLAY_TO_LEVEL,
)
from shared.audio import load_mono_audio, pad_or_crop_waveform, random_waveform_augmentation

TOP_LEVEL_DIRS = {"Clean", "Drive", "Phase", "Space"}
SUPPORTED_EXTENSIONS = {".wav", ".wave"}
EFFECT_NAME_PATTERN = re.compile(
    r"^(?P<effect>dist|chorus|phaser|delay|reverb)_(?P<intensity>25|50|75|100)_.+$",
    re.IGNORECASE,
)


def apply_spec_augment(mel_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply simple time and frequency masking."""
    augmented = mel_db.copy()
    num_mels, num_frames = augmented.shape

    if num_frames > 4:
        max_time_mask = max(1, num_frames // 10)
        time_mask = int(rng.integers(0, max_time_mask + 1))
        if time_mask > 0:
            time_start = int(rng.integers(0, max(1, num_frames - time_mask + 1)))
            augmented[:, time_start : time_start + time_mask] = augmented.mean()

    if num_mels > 4:
        max_freq_mask = max(1, num_mels // 8)
        freq_mask = int(rng.integers(0, max_freq_mask + 1))
        if freq_mask > 0:
            freq_start = int(rng.integers(0, max(1, num_mels - freq_mask + 1)))
            augmented[freq_start : freq_start + freq_mask, :] = augmented.mean()

    return augmented


def waveform_to_mel_tensor(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    training: bool,
) -> torch.Tensor:
    """Convert waveform into a normalized log-mel tensor."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel + 1e-10)
    if training:
        mel_db = apply_spec_augment(mel_db, np.random.default_rng())
    mel_db = (mel_db - float(np.mean(mel_db))) / float(np.std(mel_db) + 1e-6)
    return torch.tensor(mel_db[np.newaxis, :, :], dtype=torch.float32)


@dataclass(frozen=True)
class EffectIntensityRecord:
    path: str
    effect_label: str
    intensity_level: int
    relative_path: str
    group_id: str


def _parse_record(audio_path: Path, root: Path) -> EffectIntensityRecord | None:
    relative_path = audio_path.relative_to(root)
    if len(relative_path.parts) < 2 or relative_path.parts[0] not in TOP_LEVEL_DIRS:
        return None

    stem = audio_path.stem
    top_level = relative_path.parts[0]
    if top_level == "Clean":
        effect_label = "clean"
        intensity_level = 0
        clip_id = stem
    else:
        match = EFFECT_NAME_PATTERN.match(stem)
        if match is None:
            return None
        effect_label = match.group("effect").lower()
        intensity_level = INTENSITY_DISPLAY_TO_LEVEL[match.group("intensity")]
        clip_id = stem.split("_")[-1]

    if effect_label not in EFFECT_CLASS_TO_INDEX:
        return None

    group_id = f"{audio_path.parent.name}/{clip_id}"
    return EffectIntensityRecord(
        path=str(audio_path.resolve()),
        effect_label=effect_label,
        intensity_level=intensity_level,
        relative_path=str(relative_path),
        group_id=group_id,
    )


def scan_dataset(data_dir: str | Path) -> list[EffectIntensityRecord]:
    """Scan the single_effect dataset and parse effect/intensity labels from filenames."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: list[EffectIntensityRecord] = []
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


def _joint_label(record: EffectIntensityRecord) -> str:
    return f"{record.effect_label}:{record.intensity_level}"


def create_train_val_split(
    records: list[EffectIntensityRecord],
    val_size: float,
    seed: int,
) -> dict:
    """Create a leakage-aware split using joint effect/intensity labels."""
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
    """Serialize the split manifest."""
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
class MelDatasetConfig:
    sample_rate: int
    segment_seconds: float
    n_fft: int
    hop_length: int
    n_mels: int
    training: bool
    augment: bool


class EffectIntensityMelDataset(Dataset):
    """Convert fixed-size waveform clips into mel spectrogram tensors."""

    def __init__(self, records: list[EffectIntensityRecord], config: MelDatasetConfig) -> None:
        self.records = records
        self.config = config
        self.target_num_samples = int(config.sample_rate * config.segment_seconds)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        waveform = load_mono_audio(record.path, sample_rate=self.config.sample_rate)
        waveform = pad_or_crop_waveform(
            waveform,
            target_num_samples=self.target_num_samples,
            training=self.config.training,
        )
        if self.config.training and self.config.augment:
            waveform = random_waveform_augmentation(waveform, np.random.default_rng())
        features = waveform_to_mel_tensor(
            waveform=waveform,
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            training=self.config.training and self.config.augment,
        )
        effect_target = torch.tensor(EFFECT_CLASS_TO_INDEX[record.effect_label], dtype=torch.long)
        intensity_target = torch.tensor(record.intensity_level, dtype=torch.long)
        return features, effect_target, intensity_target
