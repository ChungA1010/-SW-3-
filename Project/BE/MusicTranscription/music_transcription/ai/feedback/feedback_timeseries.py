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
    pitch_reliable: bool = True   # 0529/음정 점수 신뢰 여부 (delay면 False, 참고용)
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
    report.pitch_reliable = ts_errors.pitch_reliable  # 0529/

    # 0529/delay로 음정 신뢰도가 낮으면 종합 점수·등급을 '박자만'으로 산출한다.
    #       부정확한 음정 점수가 등급을 끌어내려 정확한 연주를 오판하는 것을 막는다.
    if ts_errors.pitch_reliable:
        report.combined_score = (
            pitch_weight * ts_errors.pitch_score
            + rhythm_weight * ts_errors.rhythm_score
        )
    else:
        report.combined_score = ts_errors.rhythm_score
        # 0601/delay 감지 시 combined_score가 박자만으로 바뀐다는 사실을 사용자가 알 수 있도록
        #       issues에 텍스트로 명시. 프론트가 issues를 그대로 렌더링하므로 별도 필드 불필요.
        report.issues.append("딜레이(잔향)가 감지되어 종합 점수는 박자 점수만으로 산출되었습니다.")
    report.grade = score_to_grade(report.combined_score)

    # ── Issues: 3단계에서 생성된 텍스트를 그대로 주입 ──
    report.issues.extend(ts_errors.pitch_issues)
    report.issues.extend(ts_errors.rhythm_issues)

    # ── Strengths ──
    # 0529/음정 칭찬은 신뢰 가능할 때만 (delay면 점수가 부정확하므로 칭찬 보류)
    if ts_errors.pitch_reliable and ts_errors.pitch_score >= 85:
        report.strengths.append("음정 정확도가 우수합니다.")
    if ts_errors.rhythm_score >= 85:
        report.strengths.append("박자 타이밍이 안정적입니다.")

    # ── Suggestions ──
    # 0529/음정 기반 제안도 신뢰 가능할 때만 (부정확한 오차로 연습 제안하지 않음)
    if ts_errors.pitch_reliable and ts_errors.pitch_mae_semitone > 1.0:
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