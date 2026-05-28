"""
similarity.py
-------------
피처 벡터 간 코사인 유사도 계산 + 가중 합산 최종 점수.

피처별 가중치 설계 철학:
  - 인간 청각 인지 연구(MDS 기반)에서 '음색'이 가장 강력한 인지 척도임을 반영.
  - MFCC / Spectral Contrast에 높은 가중치 부여.
  - 단순 에너지(RMS)는 정규화 후이므로 낮은 가중치.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from dataclasses import dataclass
from feature_extractor import AudioFeatures


# ──────────────────────────────────────────
# 피처별 유사도 결과 컨테이너
# ──────────────────────────────────────────

@dataclass
class SimilarityResult:
    """각 피처별 코사인 유사도 + 최종 가중 점수."""
    mfcc:                float = 0.0
    spectral_centroid:   float = 0.0
    spectral_bandwidth:  float = 0.0
    spectral_rolloff:    float = 0.0
    spectral_contrast:   float = 0.0
    chroma:              float = 0.0
    zcr:                 float = 0.0
    rms:                 float = 0.0
    tonnetz:             float = 0.0
    duration_similarity: float = 0.0
    bpm_similarity:      float = 0.0
    final_score:         float = 0.0


# ──────────────────────────────────────────
# 피처별 가중치 정의
# ──────────────────────────────────────────

FEATURE_WEIGHTS = {
    "mfcc":               0.35,
    "spectral_contrast":  0.18,
    "spectral_centroid":  0.10,
    "chroma":             0.08,
    "tonnetz":            0.05,
    "spectral_bandwidth": 0.03,
    "spectral_rolloff":   0.02,
    "zcr":                0.01,
    "rms":                0.01,
    "duration_similarity": 0.07,
    "bpm_similarity":      0.10,
}

assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 1e-9, "가중치 합계가 1.0이어야 합니다."


# ──────────────────────────────────────────
# 유사도 헬퍼 함수
# ──────────────────────────────────────────

def duration_sim(ref_dur: float, user_dur: float) -> float:
    """trim 후 연주 길이 비율로 속도 유사도를 계산. ratio=1이면 1.0."""
    if ref_dur < 0.1:
        return 0.0
    ratio = user_dur / ref_dur
    return max(0.0, 1.0 - abs(ratio - 1.0))


def bpm_sim(ref_bpm: float, user_bpm: float) -> float:
    """BPM 차이 기반 유사도. 차이 0이면 1.0, ±5 BPM 이내면 만점에 가깝도록 설계."""
    diff = abs(ref_bpm - user_bpm)
    return 1.0 / (1.0 + (diff / 10.0) ** 2)


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 1D 벡터의 코사인 유사도를 [0, 1]로 반환.

    0.75 이하는 0점, 0.75~1.0 구간을 0.0~1.0으로 stretch해 변별력을 확보.
    """
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0

    a = normalize(vec_a.reshape(1, -1))
    b = normalize(vec_b.reshape(1, -1))
    sim = cosine_similarity(a, b)[0][0]

    threshold = 0.75
    if sim <= threshold:
        return 0.0
    return float((sim - threshold) / (1.0 - threshold))


# ──────────────────────────────────────────
# 전체 유사도 계산
# ──────────────────────────────────────────

def compute_similarity(
    feat_ref: AudioFeatures,
    feat_user: AudioFeatures,
    weights: dict = None,
    tempo_info: dict = None,
) -> SimilarityResult:
    """
    두 음원의 AudioFeatures를 받아 피처별 유사도와 최종 점수를 계산.

    Args:
        feat_ref   : 원본 음원 피처
        feat_user  : 사용자 연주 피처
        weights    : 커스텀 가중치 dict (None이면 기본값 사용)
        tempo_info : ref_duration, user_duration, ref_bpm, user_bpm 포함 dict

    Returns:
        SimilarityResult (각 피처 점수 + final_score)
    """
    if weights is None:
        weights = FEATURE_WEIGHTS

    result = SimilarityResult()

    for name in [
        "mfcc", "spectral_centroid", "spectral_bandwidth",
        "spectral_rolloff", "spectral_contrast", "chroma",
        "zcr", "rms", "tonnetz",
    ]:
        vec_ref  = getattr(feat_ref,  name, None)
        vec_user = getattr(feat_user, name, None)
        if vec_ref is not None and vec_user is not None and vec_ref.size > 0:
            setattr(result, name, cosine_sim(vec_ref, vec_user))

    if tempo_info:
        result.duration_similarity = duration_sim(
            tempo_info["ref_duration"], tempo_info["user_duration"]
        )
        result.bpm_similarity = bpm_sim(
            tempo_info["ref_bpm"], tempo_info["user_bpm"]
        )

    result.final_score = sum(
        getattr(result, name, 0.0) * w for name, w in weights.items()
    )

    return result
