from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


def load_mono_audio(audio_path: str | Path, sample_rate: int) -> np.ndarray:
    waveform, _ = librosa.load(Path(audio_path), sr=sample_rate, mono=True)
    waveform = np.nan_to_num(waveform.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if waveform.size == 0:
        raise ValueError("Audio file is empty.")
    return waveform


def pad_or_crop_waveform(waveform: np.ndarray, target_num_samples: int) -> np.ndarray:
    if waveform.shape[0] == target_num_samples:
        return waveform.astype(np.float32)
    if waveform.shape[0] > target_num_samples:
        start = max(0, (waveform.shape[0] - target_num_samples) // 2)
        end = start + target_num_samples
        return waveform[start:end].astype(np.float32)
    pad_amount = target_num_samples - waveform.shape[0]
    left_pad = pad_amount // 2
    right_pad = pad_amount - left_pad
    return np.pad(waveform, (left_pad, right_pad), mode="constant").astype(np.float32)


def sliding_window_waveforms(waveform: np.ndarray, window_size: int, hop_size: int) -> list[np.ndarray]:
    if waveform.shape[0] <= window_size:
        return [pad_or_crop_waveform(waveform, window_size)]

    windows: list[np.ndarray] = []
    for start in range(0, waveform.shape[0] - window_size + 1, hop_size):
        windows.append(waveform[start : start + window_size].astype(np.float32))

    last_start = waveform.shape[0] - window_size
    if last_start > 0:
        last_window = waveform[last_start : last_start + window_size].astype(np.float32)
        if not windows or not np.array_equal(windows[-1], last_window):
            windows.append(last_window)
    return windows


def waveform_to_sequence_tensor(
    waveform: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    hop_length: int,
) -> torch.Tensor:
    # NOTE: Keep this aligned with the original 09_gru training pipeline.
    # If you retrain with different features, update this function first.
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    sequence = np.concatenate([mfcc, delta, delta2], axis=0)
    sequence = (sequence - np.mean(sequence, axis=1, keepdims=True)) / (
        np.std(sequence, axis=1, keepdims=True) + 1e-6
    )
    return torch.tensor(sequence, dtype=torch.float32)
