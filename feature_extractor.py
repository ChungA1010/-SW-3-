"""
feature_extractor.py
--------------------
librosa 기반 피처 추출 + VGGish / YAMNet 임베딩.

추출 피처 목록:
  - MFCC (음색/음소 특성)
  - Spectral Centroid (음색 밝기)
  - Spectral Bandwidth (음색 넓이)
  - Spectral Rolloff (고음 에너지 분포)
  - Spectral Contrast (배음 구조)
  - Chroma (음정 클래스)
  - Zero Crossing Rate (파형 거칠기 / 노이즈)
  - RMS Energy (다이나믹)
  - Tonnetz (조성 공간)
  - VGGish / YAMNet 딥러닝 임베딩
  - BPM (템포) - 추후 추가 가능
"""

import numpy as np
import librosa
# import tensorflow as tf
# import tensorflow_hub as hub
from dataclasses import dataclass, field


# ──────────────────────────────────────────
# 피처 컨테이너
# ──────────────────────────────────────────

@dataclass
class AudioFeatures:
    """하나의 음원에서 추출된 모든 피처를 담는 컨테이너."""
    mfcc:               np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_centroid:  np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_bandwidth: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_rolloff:   np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_contrast:  np.ndarray = field(default_factory=lambda: np.array([]))
    chroma:             np.ndarray = field(default_factory=lambda: np.array([]))
    zcr:                np.ndarray = field(default_factory=lambda: np.array([]))
    rms:                np.ndarray = field(default_factory=lambda: np.array([]))
    tonnetz:            np.ndarray = field(default_factory=lambda: np.array([]))
    # vggish_embedding:   np.ndarray = field(default_factory=lambda: np.array([]))
    # yamnet_embedding:   np.ndarray = field(default_factory=lambda: np.array([]))


# ──────────────────────────────────────────
# librosa 기반 피처 추출
# ──────────────────────────────────────────

def extract_librosa_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    hop_length: int = 512,
) -> dict:
    """
    librosa를 사용하여 오디오에서 다양한 스펙트럴 피처를 추출한다.

    각 피처는 시계열 평균(mean)으로 집계되어
    고정 크기 벡터로 반환된다 (시간 불변 표현).
    """
    features = {}

    # MFCC: 인간 청각 인지를 모사하는 Mel 주파수 기반 계수
    # → 음색, 공진 특성, 이펙터 특성 포착에 핵심
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    features["mfcc"] = mfcc  # shape: (n_mfcc, T)

    # Spectral Centroid: 스펙트럼 무게 중심 → 음색 밝기
    # 디스토션/오버드라이브가 많을수록 일반적으로 높아짐
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    features["spectral_centroid"] = sc

    # Spectral Bandwidth: 스펙트럼 대역폭 → 음색 풍성함
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    features["spectral_bandwidth"] = sb

    # Spectral Rolloff: 85% 에너지 경계 주파수 → 고음 vs 저음 분포
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    features["spectral_rolloff"] = sr_feat

    # Spectral Contrast: 배음 피크와 골의 차이 → 이펙터 왜곡 민감
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    features["spectral_contrast"] = contrast

    # Chroma: 12개 음정 클래스 에너지 → 음정 일치도
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    features["chroma"] = chroma 

    # Zero Crossing Rate: 파형의 부호 변화율 → 노이즈/거칠기
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    features["zcr"] = zcr

    # RMS Energy: 순간 에너지 → 다이나믹 패턴
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    features["rms"] = rms

    # Tonnetz: 5도권 기반 조성 공간 표현 → 화성 특성
    try:
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    except:
        # 에러 발생 시 6차원 0 벡터로 채워서 멈추지 않게 함
        tonnetz = np.zeros((6, chroma.shape[1]))

    features["tonnetz"] = tonnetz

    return features


def aggregate_features(features: dict) -> dict:
    """
    시계열 피처를 [평균, 표준편차] 로 집계하여 고정 크기 벡터 생성.
    평균만 쓰면 정보 손실이 크므로 표준편차(다이나믹 변화량)도 포함.
    """
    aggregated = {}
    for key, feat in features.items():
        mean = np.mean(feat, axis=1)   # 시간 축 평균
        std  = np.std(feat,  axis=1)   # 시간 축 표준편차
        aggregated[key] = np.concatenate([mean, std])
    return aggregated


# # ──────────────────────────────────────────
# # VGGish 임베딩
# # ──────────────────────────────────────────

# VGGISH_URL = "https://tfhub.dev/google/vggish/1"
# _vggish_model = None  # 싱글턴


# def get_vggish_model():
#     global _vggish_model
#     if _vggish_model is None:
#         print("[VGGish] 모델 로딩 중...")
#         _vggish_model = hub.load(VGGISH_URL)
#         print("[VGGish] 로드 완료.")
#     return _vggish_model


# def extract_vggish_embedding(y: np.ndarray, sr: int) -> np.ndarray:
#     """
#     VGGish: 오디오를 0.96초 단위 청크로 분할,
#     각 청크를 128차원 벡터로 임베딩.
#     최종적으로 모든 청크의 평균을 반환.

#     Args:
#         y  : 오디오 시계열 (float32, mono)
#         sr : 샘플링 레이트 (VGGish는 16000Hz 기준)

#     Returns:
#         128차원 임베딩 벡터
#     """
#     model = get_vggish_model()

#     # VGGish는 16kHz 모노 입력 필요
#     if sr != 16000:
#         y = librosa.resample(y, orig_sr=sr, target_sr=16000)

#     # float32 보장
#     y = y.astype(np.float32)

#     # TF 텐서로 변환
#     waveform_tensor = tf.constant(y, dtype=tf.float32)
#     embeddings = model(waveform_tensor)  # shape: (N_chunks, 128)

#     # 청크 평균 → 단일 벡터
#     return np.mean(embeddings.numpy(), axis=0)  # (128,)


# # ──────────────────────────────────────────
# # YAMNet 임베딩
# # ──────────────────────────────────────────

# YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
# _yamnet_model = None  # 싱글턴


# def get_yamnet_model():
#     global _yamnet_model
#     if _yamnet_model is None:
#         print("[YAMNet] 모델 로딩 중...")
#         _yamnet_model = hub.load(YAMNET_URL)
#         print("[YAMNet] 로드 완료.")
#     return _yamnet_model


# def extract_yamnet_embedding(y: np.ndarray, sr: int) -> np.ndarray:
#     """
#     YAMNet: 경량 MobileNet 기반.
#     오디오를 분석하여 1024차원 임베딩을 반환.
#     실시간 / 준실시간 서비스에 적합한 속도.

#     Returns:
#         1024차원 임베딩 벡터 (프레임 평균)
#     """
#     model = get_yamnet_model()

#     # YAMNet 역시 16kHz 모노
#     if sr != 16000:
#         y = librosa.resample(y, orig_sr=sr, target_sr=16000)

#     y = y.astype(np.float32)
#     waveform_tensor = tf.constant(y, dtype=tf.float32)

#     # YAMNet 출력: (scores, embeddings, spectrogram)
#     _, embeddings, _ = model(waveform_tensor)

#     return np.mean(embeddings.numpy(), axis=0)  # (1024,)


# ──────────────────────────────────────────
# 통합 피처 추출
# ──────────────────────────────────────────

def extract_all_features(
    y: np.ndarray,
    sr: int,
    # use_vggish: bool = True,
    # use_yamnet: bool = True,
) -> AudioFeatures:
    """
    librosa 피처 + 딥러닝 임베딩을 모두 추출하여
    AudioFeatures 객체로 반환.
    """
    raw = extract_librosa_features(y, sr)
    agg = aggregate_features(raw)

    feat = AudioFeatures(
        mfcc               = agg["mfcc"],
        spectral_centroid  = agg["spectral_centroid"],
        spectral_bandwidth = agg["spectral_bandwidth"],
        spectral_rolloff   = agg["spectral_rolloff"],
        spectral_contrast  = agg["spectral_contrast"],
        chroma             = agg["chroma"],
        zcr                = agg["zcr"],
        rms                = agg["rms"],
        tonnetz            = agg["tonnetz"],
    )

    # if use_vggish:
    #     feat.vggish_embedding = extract_vggish_embedding(y, sr)

    # if use_yamnet:
    #     feat.yamnet_embedding = extract_yamnet_embedding(y, sr)

    return feat
