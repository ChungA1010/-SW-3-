"""
3단계: 정렬된 시계열에서 음정·박자 오차를 계산
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TimeSeriesErrors:
    pitch_mae_semitone: float = 0.0   # 평균 절대 음정 오차(반음)
    onset_mae_ms: float = 0.0         # 평균 절대 타이밍 오차(ms)
    pitch_score: float = 0.0          # 0~100
    rhythm_score: float = 0.0         # 0~100

    # 0529/음정 점수 신뢰 가능 여부. delay(잔향)가 감지되면 False.
    #       False면 pyin 음정 추적이 잔향에 교란돼 점수가 부정확하므로 '참고용'으로 다룬다.
    pitch_reliable: bool = True

    pitch_issues: list[str] = field(default_factory=list)
    rhythm_issues: list[str] = field(default_factory=list)


def _score_from_mae(mae: float, perfect: float, fail: float) -> float:
    """mae가 perfect 이하면 100, fail 이상이면 0, 사이는 선형."""
    if mae <= perfect:
        return 100.0
    if mae >= fail:
        return 0.0
    return 100.0 * (fail - mae) / (fail - perfect)


def compute_timeseries_errors(
    aligned_ref_midi: np.ndarray,
    aligned_user_midi: np.ndarray,
    aligned_ref_times: np.ndarray,
    aligned_ref_onsets: np.ndarray,
    aligned_user_onsets: np.ndarray,
    pitch_issue_threshold: float = 1.0,   # 반음
    onset_issue_threshold: float = 80.0,  # ms
    pitch_reliable: bool = True,          # 0529/delay 감지 시 False로 호출 → 음정 hedge
) -> TimeSeriesErrors:

    result = TimeSeriesErrors()
    result.pitch_reliable = pitch_reliable  # 0529/

    # ── 1. 음정 오차 평가 ──
    both_voiced = np.isfinite(aligned_ref_midi) & np.isfinite(aligned_user_midi)
    errs = aligned_user_midi[both_voiced] - aligned_ref_midi[both_voiced]

    abs_errs = np.abs(errs)
    result.pitch_mae_semitone = float(np.mean(abs_errs)) if len(abs_errs) else 0.0
    # MAE를 기반으로 0~100 점수 계산 (0.5 반음 이내는 완벽, 6 반음 이상은 실패)
    result.pitch_score = _score_from_mae(result.pitch_mae_semitone, perfect=0.5, fail=6.0)  

    # 점수 기반 심플 음정 피드백
    if len(abs_errs) > 0:
        if not pitch_reliable:
            # 0529/delay(잔향) 감지 시: 음정 추적이 교란되므로 단정하지 않고 '가능성'으로 안내.
            result.pitch_issues.append(
                f"딜레이(잔향)의 영향으로 음정이 실제와 다르게 측정됐을 가능성이 있습니다. "
                f"(추정 {result.pitch_score:.1f}점)"
            )
        elif result.pitch_score >= 90.0:
            result.pitch_issues.append(f"음정 점수 {result.pitch_score:.1f}점: 아주 정확하게 연주했습니다!")
        elif result.pitch_score >= 70.0:
            result.pitch_issues.append(f"음정 점수 {result.pitch_score:.1f}점: 대체로 맞지만 연주 중간에 틀린 음이 약간 섞여 있습니다.")
        else:
            result.pitch_issues.append(f"음정 점수 {result.pitch_score:.1f}점: 원곡과 다르게 연주된(틀린) 음정이 꽤 많습니다.")


    # ── 2. 박자(onset) 오차 평가 ──
    onset_diffs_ms = (aligned_user_onsets - aligned_ref_onsets) * 1000.0

    abs_onset = np.abs(onset_diffs_ms)
    result.onset_mae_ms = float(np.mean(abs_onset)) if len(abs_onset) else 0.0
    # MAE를 기반으로 0~100 점수 계산 (30ms 이내는 완벽, 400ms 이상은 실패)
    result.rhythm_score = _score_from_mae(result.onset_mae_ms, perfect=30.0, fail=400.0)
    
    # 박자 피드백 (rhythm_score 기준)
    if len(abs_onset) > 0:
        score = result.rhythm_score
        if score >= 90.0:
            result.rhythm_issues.append(f"박자 점수 {score:.1f}점: 흔들림 없이 박자가 아주 정확합니다!")
        elif score >= 70.0:
            result.rhythm_issues.append(f"박자 점수 {score:.1f}점: 전체적으로 박자가 불안정하게 흔들립니다.")
        else:
            mean_diff = float(np.mean(onset_diffs_ms))
            if mean_diff > 30.0:
                result.rhythm_issues.append(f"박자 점수 {score:.1f}점: 전체적으로 박자가 원곡보다 느리게 밀리는 경향이 있습니다.")
            elif mean_diff < -30.0:
                result.rhythm_issues.append(f"박자 점수 {score:.1f}점: 전체적으로 박자가 원곡보다 성급하고 빠르게 연주되었습니다.")
            else:
                result.rhythm_issues.append(f"박자 점수 {score:.1f}점: 전체적으로 박자가 불안정하게 흔들립니다.")

    return result