"""
preprocessor.py
---------------
전처리 파이프라인:
  1. RMS 정규화 (진폭 보정)
  2. DTW 기반 시간 축 정렬
  3. 길이 강제 정렬 (리샘플링)
"""

import numpy as np
import librosa
from dtaidistance import dtw_ndim
from scipy.interpolate import interp1d


# ──────────────────────────────────────────
# 1. RMS 정규화
# ──────────────────────────────────────────

def rms_normalize(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    오디오 신호를 목표 RMS 레벨로 정규화한다.
    볼륨(에너지 레벨) 차이로 인한 분석 왜곡을 방지.

    Args:
        y          : 오디오 시계열 (1-D numpy array)
        target_rms : 목표 RMS 값 (기본값 0.1)

    Returns:
        정규화된 오디오 시계열
    """
    current_rms = np.sqrt(np.mean(y ** 2))

    # 묵음 방지
    if current_rms < 1e-6:
        return y

    gain = target_rms / current_rms
    normalized = y * gain

    # 클리핑 방지: [-1.0, 1.0] 범위 클램핑
    return np.clip(normalized, -1.0, 1.0)


# ──────────────────────────────────────────
# 2. DTW 시간 축 정렬
# ──────────────────────────────────────────

def extract_chroma_for_dtw(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    DTW 정렬에 사용할 크로마그램 피처를 추출한다.
    크로마(음정 클래스)는 이펙터 음색 변화에 강인하고
    시간 정렬에 적합한 피처다.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    # shape: (12, T) → (T, 12) 로 전치 (dtaidistance 입력 형식)
    return chroma.T


def dtw_align(
    y_ref: np.ndarray,
    y_user: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DTW(Dynamic Time Warping)를 이용해 두 음원의 시간 축을 정렬한다.

    사용자의 미세한 박자 밀림·BPM 차이를 흡수하여
    의미 있는 구간끼리 대응(warping)시킨다.

    Args:
        y_ref   : 원본 오디오 (reference)
        y_user  : 사용자 연주 오디오
        sr      : 샘플링 레이트

    Returns:
        (정렬된 y_ref, 정렬된 y_user) — 같은 길이
    """
    # 크로마 피처 추출 (T x 12)
    chroma_ref  = extract_chroma_for_dtw(y_ref,  sr, hop_length)
    chroma_user = extract_chroma_for_dtw(y_user, sr, hop_length)

    # DTW 경로 계산 (다차원 시계열)
    path = dtw_ndim.warping_path(
        chroma_ref.astype(np.double),
        chroma_user.astype(np.double)
    )

    # 경로에서 각 시계열의 인덱스 추출
    idx_ref  = np.array([p[0] for p in path])
    idx_user = np.array([p[1] for p in path])

    # hop_length 단위 인덱스 → 샘플 인덱스로 변환
    ref_samples  = idx_ref  * hop_length
    user_samples = idx_user * hop_length

    # 음원 길이를 초과하지 않도록 클램핑
    ref_samples  = np.clip(ref_samples,  0, len(y_ref)  - 1)
    user_samples = np.clip(user_samples, 0, len(y_user) - 1)

    # 워핑 경로에 따라 신호 재배열
    y_ref_aligned  = y_ref[ref_samples]
    y_user_aligned = y_user[user_samples]

    return y_ref_aligned, y_user_aligned


# ──────────────────────────────────────────
# 3. 길이 강제 정렬 (리샘플링)
# ──────────────────────────────────────────

def force_length_match(
    y_ref: np.ndarray,
    y_user: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    두 음원의 길이를 짧은 쪽에 맞춰 강제로 통일한다.
    DTW 후 잔여 길이 차이를 제거하는 최종 단계.

    전략: 선형 보간(interpolation)으로 긴 쪽을 축소.
    """
    len_ref  = len(y_ref)
    len_user = len(y_user)
    target_len = min(len_ref, len_user)

    def resample_to(y: np.ndarray, target: int) -> np.ndarray:
        if len(y) == target:
            return y
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, target)
        interpolator = interp1d(x_old, y, kind="linear")
        return interpolator(x_new)

    y_ref_matched  = resample_to(y_ref,  target_len)
    y_user_matched = resample_to(y_user, target_len)

    return y_ref_matched, y_user_matched

# ──────────────────────────────────────────
# BPM 추정 + trim 후 길이 기록
# ──────────────────────────────────────────

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """librosa의 beat tracking으로 BPM을 추정한다."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        return float(tempo[0])
    return float(tempo)


# ──────────────────────────────────────────
# 통합 전처리 함수
# ──────────────────────────────────────────

def preprocess(
    y_ref: np.ndarray,
    y_user: np.ndarray,
    sr: int,
    target_rms: float = 0.1,
) -> dict:
    """
    전처리 후 정렬된 신호와 함께 템포 분석 정보를 반환한다.

    Returns:
        dict with keys:
            'y_ref':          정렬된 레퍼런스 신호
            'y_user':         정렬된 사용자 신호
            'ref_duration':   trim 후 레퍼런스 길이(초)
            'user_duration':  trim 후 사용자 길이(초)
            'ref_bpm':        레퍼런스 추정 BPM
            'user_bpm':       사용자 추정 BPM
    """
    print("[전처리] Step 1/3: 무음 제거 및 템포 분석 …")
    y_ref, _  = librosa.effects.trim(y_ref)
    y_user, _ = librosa.effects.trim(y_user)

    # ★ trim 직후 길이 기록 (force_length_match 전)
    ref_duration  = len(y_ref)  / sr
    user_duration = len(y_user) / sr

    # ★ BPM 추정
    ref_bpm  = estimate_bpm(y_ref,  sr)
    user_bpm = estimate_bpm(y_user, sr)

    print(f"  레퍼런스: {ref_duration:.2f}s / {ref_bpm:.1f} BPM")
    print(f"  사용자:   {user_duration:.2f}s / {user_bpm:.1f} BPM")

    print("[전처리] Step 2/3: RMS 정규화 …")
    y_ref  = rms_normalize(y_ref,  target_rms)
    y_user = rms_normalize(y_user, target_rms)

    print("[전처리] Step 3/3: 길이 강제 정렬 …")
    y_ref, y_user = force_length_match(y_ref, y_user)

    print(f"[전처리] 완료 — 최종 길이: {len(y_ref)} samples ({len(y_ref)/sr:.2f}s)")

    return {
        'y_ref':          y_ref,
        'y_user':         y_user,
        'ref_duration':   ref_duration,
        'user_duration':  user_duration,
        'ref_bpm':        ref_bpm,
        'user_bpm':       user_bpm,
    }