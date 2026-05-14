"""Feature extraction used by classical ML experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import librosa
import numpy as np
import pandas as pd

from .audio import AudioRecord, load_mono_audio
from .constants import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MFCC,
    DEFAULT_SAMPLE_RATE,
)


def _sanitize_feature_array(values: np.ndarray) -> np.ndarray:
    """Convert arbitrary librosa outputs into a stable 2D float array."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[np.newaxis, :]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _statistic_functions() -> dict[str, Callable[[np.ndarray], float]]:
    return {
        "mean": lambda row: float(np.mean(row)),
        "std": lambda row: float(np.std(row)),
        "max": lambda row: float(np.max(row)),
        "min": lambda row: float(np.min(row)),
        "median": lambda row: float(np.median(row)),
        "q25": lambda row: float(np.percentile(row, 25)),
        "q75": lambda row: float(np.percentile(row, 75)),
    }


def summarize_feature(
    values: np.ndarray,
    prefix: str,
    statistics: Sequence[str],
) -> Dict[str, float]:
    """Summarize frame-wise features into fixed-size statistics."""
    array = _sanitize_feature_array(values)
    stat_functions = _statistic_functions()
    summary: Dict[str, float] = {}

    for row_index, row in enumerate(array, start=1):
        base_name = prefix if array.shape[0] == 1 else f"{prefix}_{row_index:02d}"
        for statistic in statistics:
            summary[f"{base_name}_{statistic}"] = stat_functions[statistic](row)
    return summary


def _safe_tonnetz(signal: np.ndarray, sr: int) -> np.ndarray:
    try:
        harmonic, _ = librosa.effects.hpss(signal)
        return librosa.feature.tonnetz(y=harmonic, sr=sr)
    except Exception:
        return np.zeros((6, 1), dtype=np.float32)


def extract_baseline_features(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mfcc: int = 13,
) -> Dict[str, float]:
    """Baseline feature set close to the original RandomForest reference."""
    signal = load_mono_audio(audio_path, sample_rate=sample_rate)
    sr = sample_rate

    rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=hop_length)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length)

    features: Dict[str, float] = {
        "duration_seconds": float(librosa.get_duration(y=signal, sr=sr)),
        "tempo_bpm": float(np.mean(np.atleast_1d(tempo))),
    }
    baseline_stats = ("mean", "std", "max")
    features.update(summarize_feature(rms, "rms", baseline_stats))
    features.update(summarize_feature(zcr, "zero_crossing_rate", baseline_stats))
    features.update(summarize_feature(centroid, "spectral_centroid", baseline_stats))
    features.update(summarize_feature(bandwidth, "spectral_bandwidth", baseline_stats))
    features.update(summarize_feature(rolloff, "spectral_rolloff", baseline_stats))
    features.update(summarize_feature(contrast, "spectral_contrast", baseline_stats))
    features.update(summarize_feature(mfcc, "mfcc", baseline_stats))
    features.update(summarize_feature(onset, "onset_strength", baseline_stats))
    features.update(summarize_feature(np.mean(tempogram, axis=1), "tempogram", baseline_stats))
    return features


def extract_engineered_features(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mfcc: int = DEFAULT_N_MFCC,
) -> Dict[str, float]:
    """Richer feature set for stronger small-data classical baselines."""
    signal = load_mono_audio(audio_path, sample_rate=sample_rate)
    sr = sample_rate

    rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=32,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel + 1e-10)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    tonnetz = _safe_tonnetz(signal, sr=sr)
    onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=hop_length)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length)

    harmonic, percussive = librosa.effects.hpss(signal)
    harmonic_energy = float(np.mean(np.square(harmonic)) + 1e-8)
    percussive_energy = float(np.mean(np.square(percussive)) + 1e-8)

    engineered_stats = ("mean", "std", "max", "min", "median", "q25", "q75")
    features: Dict[str, float] = {
        "duration_seconds": float(librosa.get_duration(y=signal, sr=sr)),
        "tempo_bpm": float(np.mean(np.atleast_1d(tempo))),
        "harmonic_to_percussive_ratio": float(harmonic_energy / percussive_energy),
        "signal_abs_mean": float(np.mean(np.abs(signal))),
        "signal_std": float(np.std(signal)),
    }
    features.update(summarize_feature(rms, "rms", engineered_stats))
    features.update(summarize_feature(zcr, "zero_crossing_rate", engineered_stats))
    features.update(summarize_feature(centroid, "spectral_centroid", engineered_stats))
    features.update(summarize_feature(bandwidth, "spectral_bandwidth", engineered_stats))
    features.update(summarize_feature(rolloff, "spectral_rolloff", engineered_stats))
    features.update(summarize_feature(contrast, "spectral_contrast", engineered_stats))
    features.update(summarize_feature(flatness, "spectral_flatness", engineered_stats))
    features.update(summarize_feature(chroma, "chroma", engineered_stats))
    features.update(summarize_feature(mel_db, "mel", engineered_stats))
    features.update(summarize_feature(mfcc, "mfcc", engineered_stats))
    features.update(summarize_feature(delta, "mfcc_delta", engineered_stats))
    features.update(summarize_feature(delta2, "mfcc_delta2", engineered_stats))
    features.update(summarize_feature(tonnetz, "tonnetz", engineered_stats))
    features.update(summarize_feature(onset, "onset_strength", engineered_stats))
    features.update(summarize_feature(np.mean(tempogram, axis=1), "tempogram", engineered_stats))
    return features


def extract_feature_table(
    records: Iterable[AudioRecord],
    extractor_name: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> pd.DataFrame:
    """Extract features for a list of records and return a dataframe."""
    if extractor_name not in {"baseline", "engineered"}:
        raise ValueError(f"Unsupported extractor_name: {extractor_name}")

    extractor = extract_baseline_features if extractor_name == "baseline" else extract_engineered_features

    rows: list[dict] = []
    for index, record in enumerate(records, start=1):
        row = {
            "filename": Path(record.path).name,
            "relative_path": record.relative_path,
            "label": record.label,
            "group_id": record.group_id,
        }
        row.update(extractor(record.path, sample_rate=sample_rate))
        rows.append(row)
        if index % 10 == 0:
            print(f"[INFO] Extracted {index} files for {extractor_name} features")
    return pd.DataFrame(rows)


def feature_columns_from_dataframe(dataframe: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns in deterministic order."""
    excluded = {"filename", "relative_path", "label", "split", "group_id"}
    return [column for column in dataframe.columns if column not in excluded]


def dataframe_to_matrix(dataframe: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    """Convert a feature dataframe to a numeric matrix with NaNs removed."""
    matrix = dataframe[list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix.to_numpy(dtype=np.float32)

