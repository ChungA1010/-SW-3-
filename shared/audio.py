"""Audio IO helpers and dataset scanning utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import librosa
import numpy as np

from .constants import (
    DEFAULT_SAMPLE_RATE,
    FINE_CLASS_DIRECTORY_ALIASES,
    FINE_CLASS_NAMES,
    SUPPORTED_AUDIO_EXTENSIONS,
)


@dataclass(frozen=True)
class AudioRecord:
    """Metadata describing one audio file in the dataset."""

    path: str
    label: str
    relative_path: str
    group_id: str

    def as_path(self) -> Path:
        return Path(self.path)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist yet."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def infer_group_id(audio_path: Path) -> str:
    """
    Infer a weak grouping key from the filename to reduce leakage risk.

    If multiple segments or augmented renders were created from the same source
    and the filenames follow common suffix patterns such as `_chunk03`,
    `_seg1`, `_aug2`, the split code can keep them in the same fold.
    """
    stem = audio_path.stem.lower()
    stem = re.sub(
        r"([_-](chunk|seg|segment|clip|part|take|crop|aug)\d+)+$",
        "",
        stem,
    )
    return f"{audio_path.parent.name}/{stem}"


def scan_dataset(data_dir: str | Path, class_names: Iterable[str] = FINE_CLASS_NAMES) -> List[AudioRecord]:
    """Scan `data/{label}/*.wav` style folders and return deterministic records."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: List[AudioRecord] = []
    missing_label_dirs = []

    for label in class_names:
        candidate_names = FINE_CLASS_DIRECTORY_ALIASES.get(label, [label])
        matching_dirs = []
        seen_paths = set()
        for candidate_name in candidate_names:
            candidate_dir = root / candidate_name
            if candidate_dir.exists():
                resolved = str(candidate_dir.resolve())
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    matching_dirs.append(candidate_dir)

        if not matching_dirs:
            missing_label_dirs.append(f"{label}: {', '.join(str(root / name) for name in candidate_names)}")
            continue

        for label_dir in matching_dirs:
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                    continue
                records.append(
                    AudioRecord(
                        path=str(path.resolve()),
                        label=label,
                        # Keep the displayed sample path relative to the user-facing
                        # dataset root. This also works when class directories are
                        # symlinked to an external dataset location.
                        relative_path=str(path.relative_to(root)),
                        group_id=infer_group_id(path),
                    )
                )

    if missing_label_dirs:
        raise FileNotFoundError(
            "Missing required class folders:\n" + "\n".join(missing_label_dirs)
        )

    if not records:
        raise FileNotFoundError(
            f"No audio files were found under {root}. Expected fine-label folders matching: {list(class_names)}"
        )

    return records


def serialize_records(records: Iterable[AudioRecord]) -> list[dict]:
    """Convert records into plain dictionaries for JSON serialization."""
    return [asdict(record) for record in records]


def deserialize_records(payload: Iterable[dict]) -> List[AudioRecord]:
    """Create `AudioRecord` objects from a JSON payload."""
    return [AudioRecord(**item) for item in payload]


def dump_json(data: dict, output_path: str | Path) -> Path:
    """Save a JSON file with UTF-8 disabled to keep it portable."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_mono_audio(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize_peak: bool = False,
) -> np.ndarray:
    """Load a mono waveform as float32."""
    audio_path = Path(audio_path)
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = np.nan_to_num(waveform.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    if normalize_peak:
        peak = float(np.max(np.abs(waveform)))
        if peak > 1e-8:
            waveform = waveform / peak

    return waveform


def pad_or_crop_waveform(
    waveform: np.ndarray,
    target_num_samples: int,
    training: bool,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fit a waveform to a fixed-size segment using crop or zero-pad."""
    if waveform.shape[0] == target_num_samples:
        return waveform.astype(np.float32)

    if waveform.shape[0] > target_num_samples:
        if training:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, waveform.shape[0] - target_num_samples + 1))
        else:
            start = max(0, (waveform.shape[0] - target_num_samples) // 2)
        end = start + target_num_samples
        return waveform[start:end].astype(np.float32)

    pad_amount = target_num_samples - waveform.shape[0]
    if training:
        rng = rng or np.random.default_rng()
        left_pad = int(rng.integers(0, pad_amount + 1))
    else:
        left_pad = pad_amount // 2
    right_pad = pad_amount - left_pad
    padded = np.pad(waveform, (left_pad, right_pad), mode="constant")
    return padded.astype(np.float32)


def sliding_window_waveforms(
    waveform: np.ndarray,
    window_size: int,
    hop_size: int,
) -> List[np.ndarray]:
    """Split a waveform into overlapping windows for robust file-level inference."""
    if waveform.shape[0] <= window_size:
        return [pad_or_crop_waveform(waveform, window_size, training=False)]

    windows: List[np.ndarray] = []
    for start in range(0, waveform.shape[0] - window_size + 1, hop_size):
        windows.append(waveform[start : start + window_size].astype(np.float32))

    if not windows:
        windows.append(pad_or_crop_waveform(waveform, window_size, training=False))
        return windows

    last_start = waveform.shape[0] - window_size
    if last_start > 0 and (len(windows) == 0 or not np.array_equal(windows[-1], waveform[last_start:last_start + window_size])):
        windows.append(waveform[last_start : last_start + window_size].astype(np.float32))
    return windows


def random_waveform_augmentation(
    waveform: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply lightweight waveform augmentation that is safe for small datasets."""
    augmented = waveform.astype(np.float32).copy()

    gain = float(rng.uniform(0.8, 1.2))
    augmented *= gain

    if rng.random() < 0.5:
        noise_scale = float(rng.uniform(0.0005, 0.005))
        augmented += rng.normal(0.0, noise_scale, size=augmented.shape).astype(np.float32)

    if rng.random() < 0.5 and augmented.shape[0] > 8:
        max_shift = max(1, augmented.shape[0] // 20)
        shift = int(rng.integers(-max_shift, max_shift + 1))
        augmented = np.roll(augmented, shift)

    return np.clip(augmented, -1.0, 1.0).astype(np.float32)
