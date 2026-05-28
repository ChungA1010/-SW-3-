"""
similarity.py
-------------
피처 벡터 간 코사인 유사도 계산 + 가중 합산 최종 점수.

피처별 가중치 설계 철학:
  - 인간 청각 인지 연구(MDS 기반)에서 '음색'이 가장 강력한 인지 척도임을 반영.
  - VGGish / MFCC / Spectral Contrast에 높은 가중치 부여.
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
    mfcc:               float = 0.0
    spectral_centroid:  float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff:   float = 0.0
    spectral_contrast:  float = 0.0
    chroma:             float = 0.0
    zcr:                float = 0.0
    rms:                float = 0.0
    tonnetz:            float = 0.0
    # vggish_embedding:   float = 0.0
    # yamnet_embedding:   float = 0.0
    duration_similarity: float = 0.0   # 연주 길이 유사도 (0~1)
    bpm_similarity:      float = 0.0   # BPM 유사도 (0~1)
    final_score:        float = 0.0  # 0.0 ~ 1.0 (0% ~ 100%)


# ── 새로운 유사도 계산 함수 추가 ──
def duration_sim(ref_dur: float, user_dur: float) -> float:
    """
    trim 후 연주 길이 비율로 속도 유사도를 계산한다.
    비율이 1.0에 가까울수록 유사도가 높다.
    """
    if ref_dur < 0.1:
        return 0.0
    ratio = user_dur / ref_dur
    # ratio=1 → 1.0,  ratio=0.5 or 2.0 → 낮은 점수
    return max(0.0, 1.0 - abs(ratio - 1.0))


def bpm_sim(ref_bpm: float, user_bpm: float) -> float:
    """
    BPM 차이 기반 유사도. 차이가 0이면 1.0, 차이가 클수록 0에 수렴.
    허용 오차 ±5 BPM 이내면 만점에 가깝도록 설계.
    """
    diff = abs(ref_bpm - user_bpm)
    # 시그모이드 스타일 감쇠: 5 BPM 차이 → ~0.92, 15 BPM → ~0.67
    return 1.0 / (1.0 + (diff / 10.0) ** 2)

# ──────────────────────────────────────────
# 피처별 가중치 정의
# ──────────────────────────────────────────

# 총합 = 1.0 (확률적 해석 가능)
FEATURE_WEIGHTS = {
    # "vggish_embedding":   0.00,  # 가장 높음: 딥러닝 음색 지문, 검증된 성능
    "mfcc":               0.35,  # 음색/공진 특성의 핵심
    "spectral_contrast":  0.18,  # 배음 구조 → 이펙터 왜곡에 민감
    "spectral_centroid":  0.10,  # 밝기 (드라이브 세기 반영)
    "chroma":             0.08,  # 음정 일치도
    "tonnetz":            0.05,  # 조성 공간
    # "yamnet_embedding":   0.08,  # 경량 보조 확인
    "spectral_bandwidth": 0.03,
    "spectral_rolloff":   0.02,
    "zcr":                0.01,
    "rms":                0.01,
    "duration_similarity": 0.07,   # 연주 속도(길이) 평가
    "bpm_similarity":      0.10,   # BPM 평가
}

assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 1e-9, "가중치 합계가 1.0이어야 합니다."


# ──────────────────────────────────────────
# 단일 피처 코사인 유사도
# ──────────────────────────────────────────

def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 1D 벡터의 코사인 유사도를 반환한다. [-1, 1] → [0, 1]로 정규화.

    코사인 유사도는 벡터의 방향(비율)만 비교하므로
    RMS 정규화 후에도 볼륨 차이 잔여분에 강인하다.
    """
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0

    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)

    # L2 정규화
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    
    a = normalize(a)
    b = normalize(b)

    sim = cosine_similarity(a, b)[0][0]  # [-1, 1]

    # [0, 1] 범위로 스케일
    # return float((sim + 1.0) / 2.0)
    # [엄격한 스케일링] <- 너무 후하게 점수를 줘서 변별이 안되는 문제
    # 0.75 이하의 유사도는 아예 0점으로 처리하고, 
    # 0.75 ~ 1.0 구간을 0.0 ~ 1.0 점수로 쫙 늘려서(Stretch) 변별력을 줌
    threshold = 0.75
    if sim <= threshold:
        return 0.0
    
    strict_sim = (sim - threshold) / (1.0 - threshold)
    
    return float(strict_sim)


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
        feat_ref  : 원본 음원 피처
        feat_user : 사용자 연주 피처
        weights   : 커스텀 가중치 dict (None이면 기본값 사용)

    Returns:
        SimilarityResult (각 피처 점수 + final_score)
    """
    if weights is None:
        weights = FEATURE_WEIGHTS

    result = SimilarityResult()

    feature_names = [
        "mfcc", "spectral_centroid", "spectral_bandwidth",
        "spectral_rolloff", "spectral_contrast", "chroma",
        "zcr", "rms", "tonnetz",
        # "vggish_embedding", "yamnet_embedding",
    ]
    for name in feature_names:
        vec_ref  = getattr(feat_ref,  name, None)
        vec_user = getattr(feat_user, name, None)
        if vec_ref is not None and vec_user is not None and vec_ref.size > 0:
            setattr(result, name, cosine_sim(vec_ref, vec_user))

    # ★ 템포 유사도 계산
    if tempo_info:
        result.duration_similarity = duration_sim(
            tempo_info['ref_duration'], tempo_info['user_duration']
        )
        result.bpm_similarity = bpm_sim(
            tempo_info['ref_bpm'], tempo_info['user_bpm']
        )

    # 가중 합산
    total = 0.0
    for name, w in weights.items():
        total += getattr(result, name, 0.0) * w
    result.final_score = total

    return result

def duration_sim(ref_dur, user_dur):
    """길이 비율 → 유사도. ratio=1이면 1.0"""
    ratio = user_dur / ref_dur
    return max(0.0, 1.0 - abs(ratio - 1.0))

def bpm_sim(ref_bpm, user_bpm):
    """BPM 차이 → 유사도. 차이 0이면 1.0, ±10 BPM → 0.5"""
    diff = abs(ref_bpm - user_bpm)
    return 1.0 / (1.0 + (diff / 10.0) ** 2)

