"""
1단계: 시계열 피처 추출 (기존 aggregate 방식과 병행)
- onset 타임스탬프 배열
- pitch(Hz) → MIDI 번호 배열
- 프레임 단위 크로마 배열 (보조)
"""
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeSeriesFeatures:
    onset_times: np.ndarray       # (N,) 초 단위 onset 시각
    pitch_hz: np.ndarray          # (M,) 프레임별 추정 pitch (Hz, unvoiced=NaN)
    pitch_midi: np.ndarray        # (M,) Hz→MIDI 변환 (unvoiced=NaN)
    pitch_times: np.ndarray       # (M,) pitch 프레임 중심 시각
    pitch_confidence: np.ndarray  # (M,) voiced 확률
    duration: float               # 전체 길이(초)
    bpm: float                    # 전체 평균 BPM
    # 0529/음정 신뢰도 판정용: 무음 프레임 비율. delay가 노트 사이를 채우면 낮아진다.
    #       실험상 dist/phaser=16~20%, delay=7~12%로 분리됨 → 게이팅(조건 통과 못하면 차단)에 사용.
    silence_ratio: float = 0.0


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Hz → MIDI 번호. 0 이하 또는 NaN은 NaN 유지."""
    midi = np.full_like(hz, np.nan, dtype=np.float64)
    valid = np.isfinite(hz) & (hz > 0)
    midi[valid] = 69 + 12 * np.log2(hz[valid] / 440.0)
    return midi


def extract_timeseries(y: np.ndarray, sr: int = 22050,
                       fmin: float = 65.0,   # C2
                       fmax: float = 2093.0  # C7
                       ) -> TimeSeriesFeatures:
    """원본/유저 음원 하나에 대해 시계열 피처를 추출한다."""

    # ── Onset detection ──
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # ── Pitch tracking (pyin) ──
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr
    )
    pitch_times = librosa.times_like(f0, sr=sr)
    pitch_midi = hz_to_midi(f0)

    # ── BPM ──
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # 0529/무음 비율: 최대 RMS의 5% 미만인 프레임의 비율.
    #       delay(잔향)가 노트 사이 빈 구간을 채우면 이 값이 떨어진다.
    rms = librosa.feature.rms(y=y)[0]
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    silence_ratio = float(np.mean(rms < peak_rms * 0.05)) if peak_rms > 0 else 0.0

    return TimeSeriesFeatures(
        onset_times=onset_times,
        pitch_hz=f0,
        pitch_midi=pitch_midi,
        pitch_times=pitch_times,
        pitch_confidence=voiced_prob,
        duration=len(y) / sr,
        bpm=bpm,
        silence_ratio=silence_ratio,
    )
