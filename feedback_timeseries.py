"""
4단계: 기존 FeedbackReport 구조에 시계열 오차 데이터를 주입
기존 feedback.py의 generate_feedback을 감싸는 래퍼.
"""
from dataclasses import dataclass, field
from typing import Optional


def score_to_grade(score: float) -> str:
    if score >= 95: return "S"
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 50: return "C"
    return "D"


@dataclass
class TimeSeriesFeedbackReport:
    pitch_score: float = 0.0
    rhythm_score: float = 0.0
    combined_score: float = 0.0
    grade: str = "D"
    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def generate_timeseries_feedback(
    ts_errors,              # TimeSeriesErrors from timeseries_scoring
    pitch_weight: float = 0.6,
    rhythm_weight: float = 0.4,
) -> TimeSeriesFeedbackReport:
    """시계열 오차 → 피드백 리포트 생성."""

    report = TimeSeriesFeedbackReport()
    report.pitch_score = ts_errors.pitch_score
    report.rhythm_score = ts_errors.rhythm_score
    report.combined_score = (
        pitch_weight * ts_errors.pitch_score
        + rhythm_weight * ts_errors.rhythm_score
    )
    report.grade = score_to_grade(report.combined_score)

    # ── Issues: 3단계에서 생성된 텍스트를 그대로 주입 ──
    report.issues.extend(ts_errors.pitch_issues)
    report.issues.extend(ts_errors.rhythm_issues)

    # ── Strengths ──
    if ts_errors.pitch_score >= 85:
        report.strengths.append("음정 정확도가 우수합니다.")
    if ts_errors.rhythm_score >= 85:
        report.strengths.append("박자 타이밍이 안정적입니다.")

    # ── Suggestions ──
    if ts_errors.pitch_mae_semitone > 1.0:
        report.suggestions.append(
            f"평균 음정 오차가 {ts_errors.pitch_mae_semitone:.1f}반음입니다. "
            "느린 템포로 한 음씩 정확도를 잡아 보세요."
        )
    if ts_errors.onset_mae_ms > 50:
        report.suggestions.append(
            f"평균 타이밍 오차가 {ts_errors.onset_mae_ms:.0f}ms입니다. "
            "메트로놈에 맞춰 연습해 보세요."
        )
    if not report.suggestions:
        report.suggestions.append("현재 수준이 훌륭합니다. 세부 뉘앙스를 다듬어 보세요.")

    return report
