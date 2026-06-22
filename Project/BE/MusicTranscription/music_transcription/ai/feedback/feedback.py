"""
feedback.py
-----------
유사도 결과 + 피처 차이를 바탕으로
인간 청각 인지 기준에 맞는 rule-based 피드백 생성.

피드백 카테고리:
  1. 게인 / 드라이브 (이펙터 왜곡 세기)
  2. 톤 (Tone) / EQ (음역대 밝기·어두움)
  3. 음정 정확도
  4. 다이나믹 / 아티큘레이션
  5. 공간감 (리버브·딜레이)
"""

from dataclasses import dataclass, field
from .similarity import SimilarityResult
from .feature_extractor import AudioFeatures
import numpy as np


# ──────────────────────────────────────────
# 피드백 컨테이너
# ──────────────────────────────────────────

@dataclass
class FeedbackReport:
    """최종 피드백 보고서."""
    overall_score:  float = 0.0   # 0~100 (%)
    grade:          str   = ""    # S / A / B / C / D
    issues:         list  = field(default_factory=list)   # 발견된 문제
    suggestions:    list  = field(default_factory=list)   # 구체적 제안
    strengths:      list  = field(default_factory=list)   # 잘한 점

    def summary(self) -> str:
        lines = [
            f"=== 기타 톤 분석 결과 ===",
            f"종합 점수: {self.overall_score:.1f}% ({self.grade})",
            "",
            "[ 개선 포인트 ]",
        ]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        lines.append("")
        lines.append("[ 구체적 제안 ]")
        for s in self.suggestions:
            lines.append(f"  * {s}")
        lines.append("")
        lines.append("[ 잘한 점 ]")
        for strength in self.strengths:
            lines.append(f"  + {strength}")
        return "\n".join(lines)


# ──────────────────────────────────────────
# 등급 결정
# ──────────────────────────────────────────

def score_to_grade(score: float) -> str:
    """0~100 점수를 S/A/B/C/D 등급으로 변환."""
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    return "D"


# ──────────────────────────────────────────
# 피처 차이 분석 헬퍼
# ──────────────────────────────────────────

# 드라이브 계열 (Drive, Dist, Fuzz): * zcr_ratio를 보면 됩니다. 퍼즈(Fuzz)는 ZCR이 폭발하고, 드라이브는 적당히 높겠죠?

# 피드백: "소리가 너무 부드러워요. Fuzz나 Dist를 켜서 더 거칠게 만드세요!"

# 공간계 (Delay, Reverb): * Spectral Bandwidth가 담당합니다.

# 피드백: "소리가 너무 건조해요. Delay를 추가해 메아리 효과를 주세요."

# 변조계 (Chorus, Tremolo):

# Chorus: 소리가 화성적으로 풍성해지므로 Tonnetz 유사도가 낮으면 "코러스를 켜보세요"라고 할 수 있습니다.

# Tremolo: 볼륨이 울렁이는 게 특징이라, 우리가 aggregate_features에서 뽑은 RMS의 표준편차(std) 값이 원본보다 낮으면 "트레몰로의 깊이(Depth)를 키우세요"라고 하면 됩니다.

def get_centroid_ratio(feat_ref: AudioFeatures, feat_user: AudioFeatures) -> float:
    """
    Spectral Centroid 비율 계산.
    ratio > 1.1 → 사용자 쪽이 더 밝음(고음 에너지 과다)
    ratio < 0.9 → 사용자 쪽이 더 어두움(저음 편향)
    """
    mean_ref  = feat_ref.spectral_centroid[0]   # 평균값 (첫 번째 원소)
    mean_user = feat_user.spectral_centroid[0]
    if mean_ref < 1e-6:
        return 1.0
    return mean_user / mean_ref


def get_zcr_ratio(feat_ref: AudioFeatures, feat_user: AudioFeatures) -> float:
    """
    Zero Crossing Rate 비율.
    높을수록 파형 왜곡(디스토션/오버드라이브) 강도 높음.
    """
    zcr_ref  = feat_ref.zcr[0]
    zcr_user = feat_user.zcr[0]
    if zcr_ref < 1e-9:
        return 1.0
    return zcr_user / zcr_ref


def get_contrast_diff(feat_ref: AudioFeatures, feat_user: AudioFeatures) -> float:
    """
    Spectral Contrast 평균 차이.
    원본보다 낮으면 배음 에너지가 부족 → 드라이브/게인 부족.
    """
    contrast_ref  = np.mean(feat_ref.spectral_contrast[:len(feat_ref.spectral_contrast)//2])
    contrast_user = np.mean(feat_user.spectral_contrast[:len(feat_user.spectral_contrast)//2])
    return contrast_user - contrast_ref   # 양수: 사용자가 더 선명, 음수: 부족


# ──────────────────────────────────────────
# Rule-Based 피드백 규칙 엔진
# ──────────────────────────────────────────

def generate_feedback(
    sim_result: SimilarityResult,
    feat_ref: AudioFeatures,
    feat_user: AudioFeatures,
    tempo_info: dict = None,
) -> FeedbackReport:
    """
    유사도 점수와 피처 차이를 분석하여
    구체적이고 실행 가능한 피드백을 생성한다.

    규칙 기반(Rule-Based) 접근:
      - 각 피처 유사도의 임계값(threshold)과
      - 원본 대비 사용자 값의 비율(ratio)을 조합
    """
    report = FeedbackReport()
    report.overall_score = round(sim_result.final_score * 100, 1)
    report.grade         = score_to_grade(report.overall_score)

    issues      = []
    suggestions = []
    strengths   = []

    # ── 보조 분석 ──────────────────────────
    centroid_ratio = get_centroid_ratio(feat_ref, feat_user)
    zcr_ratio      = get_zcr_ratio(feat_ref, feat_user)
    contrast_diff  = get_contrast_diff(feat_ref, feat_user)

    # # ══════════════════════════════════════════
    # # 규칙 1: 음색 전반 (VGGish 임베딩 유사도)
    # # ══════════════════════════════════════════
    # if sim_result.vggish_embedding < 0.70:
    #     issues.append(
    #         f"전반적인 음색(톤)이 원본과 많이 다릅니다. "
    #         f"(VGGish 유사도: {sim_result.vggish_embedding:.0%})"
    #     )
    #     suggestions.append(
    #         "앰프 채널과 이펙터 체인 전체를 원본 연주자의 세팅과 비교해보세요. "
    #         "특히 앰프 타입(클린/오버드라이브/디스토션)이 다를 가능성이 높습니다."
    #     )
    # elif sim_result.vggish_embedding >= 0.85:
    #     strengths.append(
    #         f"전반적인 음색이 원본과 매우 유사합니다. (VGGish: {sim_result.vggish_embedding:.0%})"
    #     )

    # ══════════════════════════════════════════
    # 규칙 2: 이펙터 왜곡 세기 (ZCR + Spectral Contrast)
    # ══════════════════════════════════════════
    if zcr_ratio < 0.5: # 0.75보다 더 엄격하게 잡아서 '확실히 깨끗한 소리'일 때
        issues.append("현재 연주가 원본에 비해 매우 깨끗한 '클린 톤'으로 들립니다.")
        suggestions.append("의도한 연주라면 좋지만, 원본의 거친 느낌을 내려면 디스토션 이펙터를 켜주세요.")
    elif 0.5 <= zcr_ratio < 0.85:
        issues.append("파형의 왜곡 밀도(ZCR)가 약간 낮습니다.")
        suggestions.append("게인(Gain) 노브를 조금만 더 올려서 질감을 살려보세요.")
    elif zcr_ratio > 1.35:
        issues.append("파형의 왜곡 밀도(ZCR)가 과도합니다. 음이 너무 뭉개질 수 있어요.")
        suggestions.append("드라이브를 줄여서 소리의 명료함을 확보해 보세요.")

    if contrast_diff < -5.0:
        issues.append(
            "배음 구조(Spectral Contrast)가 원본보다 약합니다. "
            "이펙터 이후 음이 뭉개지고 있을 수 있습니다."
        )
        suggestions.append(
            "앰프의 Presence 또는 Treble 노브를 약간 올려 "
            "배음 에너지를 살려보세요."
        )

    # ══════════════════════════════════════════
    # 규칙 3: 음색 밝기 / EQ 세팅 (Spectral Centroid)
    # ══════════════════════════════════════════
    if sim_result.spectral_centroid < 0.75:
        if centroid_ratio > 1.15:
            issues.append(
                f"음색이 원본보다 {(centroid_ratio-1)*100:.0f}% 밝습니다. "
                "고음역 에너지가 과다합니다."
            )
            suggestions.append(
                "앰프의 Treble 노브를 낮추거나, 톤(Tone) 컨트롤을 반시계 방향으로 돌려보세요. "
                "기타 픽업이 브릿지 픽업이라면 넥 픽업으로 전환을 고려하세요."
            )
        elif centroid_ratio < 0.85:
            issues.append(
                f"음색이 원본보다 {(1-centroid_ratio)*100:.0f}% 어둡습니다. "
                "고음역이 부족합니다."
            )
            suggestions.append(
                "앰프의 Treble 또는 Presence 노브를 높이거나, "
                "기타 톤 노브를 시계 방향으로 돌려보세요."
            )
    elif sim_result.spectral_centroid >= 0.88:
        strengths.append("음색 밝기(EQ 세팅)가 원본과 잘 일치합니다.")

    # ══════════════════════════════════════════
    # 규칙 4: 음정 정확도 (Chroma)
    # ══════════════════════════════════════════
    if sim_result.chroma < 0.70:
        issues.append(
            f"음정 패턴이 원본과 크게 다릅니다. "
            f"(코드/음계 유사도: {sim_result.chroma:.0%})"
        )
        suggestions.append(
            "음정 정확도를 점검하세요. "
            "기타 튜닝이 맞는지 확인하고, 코드 보이싱과 음계 운지법을 다시 연습하세요."
        )
    elif sim_result.chroma >= 0.85:
        strengths.append(
            f"음정(코드/음계) 정확도가 높습니다. ({sim_result.chroma:.0%})"
        )

    # ══════════════════════════════════════════
    # 규칙 5: MFCC 기반 음색 질감
    # ══════════════════════════════════════════
    if sim_result.mfcc < 0.72:
        issues.append(
            f"음색 질감(MFCC)이 원본과 다릅니다. ({sim_result.mfcc:.0%})"
        )
        suggestions.append(
            "앰프 시뮬레이터/캡처 세팅을 원본에 가까운 프리셋으로 변경하거나, "
            "픽업 포지션(브릿지/미들/넥)을 바꿔보세요. "
            "픽의 재질과 두께도 음색에 큰 영향을 줍니다."
        )

    # ══════════════════════════════════════════
    # 규칙 6: 다이나믹 / 아티큘레이션 (RMS)
    # ══════════════════════════════════════════
    if sim_result.rms < 0.65:
        issues.append(
            "다이나믹(강약 변화 패턴)이 원본과 다릅니다."
        )
        suggestions.append(
            "오리지널 연주의 강약 변화를 분석하고, "
            "같은 구간에서 피킹(picking) 강도를 맞춰보세요. "
            "레가토/스타카토 같은 아티큘레이션을 의식적으로 재현해보세요."
        )

    # ══════════════════════════════════════════
    # 규칙 7: 연주 속도 (trim 후 길이 비교)
    # ══════════════════════════════════════════
    if tempo_info:
        ref_dur  = tempo_info['ref_duration']
        user_dur = tempo_info['user_duration']
        ratio    = user_dur / ref_dur if ref_dur > 0.1 else 1.0

        if sim_result.duration_similarity < 0.85:
            if ratio > 1.0:
                pct = (ratio - 1.0) * 100
                issues.append(
                    f"연주 길이가 원본보다 {pct:.0f}% 길어 — 전체적으로 느리게 연주했습니다. "
                    f"(원본 {ref_dur:.1f}s → 사용자 {user_dur:.1f}s)"
                )
                suggestions.append(
                    "메트로놈을 원본 BPM에 맞추고, 곡 전체를 끊지 않고 통으로 연습하세요. "
                    "느려지는 구간(보통 어려운 코드 체인지 부분)을 집중 반복하세요."
                )
            else:
                pct = (1.0 - ratio) * 100
                issues.append(
                    f"연주 길이가 원본보다 {pct:.0f}% 짧아 — 전체적으로 빠르게 연주했습니다. "
                    f"(원본 {ref_dur:.1f}s → 사용자 {user_dur:.1f}s)"
                )
                suggestions.append(
                    "템포를 의식적으로 늦추세요. "
                    "메트로놈 없이 빨라지는 습관이 있다면, 원곡 BPM보다 5~10 낮게 설정해서 연습하세요."
                )
        elif sim_result.duration_similarity >= 0.95:
            strengths.append(
                f"연주 속도가 원본과 거의 일치합니다. "
                f"(원본 {ref_dur:.1f}s / 사용자 {user_dur:.1f}s)"
            )

    # ══════════════════════════════════════════
    # 규칙 8: BPM 비교
    # ══════════════════════════════════════════
    if tempo_info:
        ref_bpm  = tempo_info['ref_bpm']
        user_bpm = tempo_info['user_bpm']
        bpm_diff = user_bpm - ref_bpm

        if sim_result.bpm_similarity < 0.80:
            issues.append(
                f"BPM 차이가 큽니다. (원본 {ref_bpm:.0f} BPM → 사용자 {user_bpm:.0f} BPM, "
                f"차이 {bpm_diff:+.0f})"
            )
            if bpm_diff > 0:
                suggestions.append(
                    f"연주 BPM이 원본보다 {bpm_diff:.0f}만큼 빠릅니다. "
                    f"메트로놈을 {ref_bpm:.0f} BPM으로 설정하고 정확히 맞춰 연습하세요."
                )
            else:
                suggestions.append(
                    f"연주 BPM이 원본보다 {abs(bpm_diff):.0f}만큼 느립니다. "
                    f"점진적으로 BPM을 올려서 {ref_bpm:.0f} BPM에 도달하도록 연습하세요."
                )
        elif sim_result.bpm_similarity >= 0.92:
            strengths.append(
                f"BPM이 원본과 잘 일치합니다. "
                f"(원본 {ref_bpm:.0f} / 사용자 {user_bpm:.0f} BPM)"
            )

    # ══════════════════════════════════════════
    # 최우수 점수 특별 메시지
    # ══════════════════════════════════════════
    if report.overall_score >= 90:
        strengths.append(
            "원본 연주자의 이펙터 세팅과 연주 스타일을 매우 훌륭하게 재현했습니다!"
        )

    report.issues      = issues
    report.suggestions = suggestions
    report.strengths   = strengths
    return report
