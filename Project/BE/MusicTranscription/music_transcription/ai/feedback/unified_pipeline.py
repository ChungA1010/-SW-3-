"""
unified_pipeline.py
-------------------
이펙터 피드백(tone_match_model) + 연주 피드백(tone / timeseries)
두 파이프라인을 통합하여 단일 dict로 반환.

Usage:
    from unified_pipeline import run_unified_feedback
    result = run_unified_feedback("ref.wav", "copy.wav")
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import librosa

from .tone_match_model import ToneMatchModel
from .pipeline_timeseries import run_timeseries_pipeline
# 0529/연주 피드백에서 음색(tone) 리포트 제거 — 음색 분석은 이펙터 피드백(ToneMatchModel)과
#       중복이라 삭제. 연주 피드백은 음정(pitch)·박자(rhythm) 시계열만 담당한다.
#       이로써 preprocessor / feature_extractor / similarity / feedback import 불필요.


def run_unified_feedback(
    ref_path: str | Path,
    copy_path: str | Path,
    sr_playing: int = 22050,
    active_effects: list[str] | None = None,
) -> dict:
    """
    이펙터 피드백 + 연주 피드백을 통합 실행하여 단일 dict 반환.

    Parameters
    ----------
    ref_path       : 기준 음원 경로
    copy_path      : 비교 음원 경로
    sr_playing     : 연주 분석용 샘플링 레이트 (기본 22050)
    active_effects : 이펙터 목록 override. None이면 파일명 자동 감지.
                     예) ["dist", "delay"]
    """
    ref_path  = Path(ref_path)
    copy_path = Path(copy_path)

    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink):
        # ── 1. 이펙터 피드백 ──────────────────────────────────────────
        effect_result = ToneMatchModel().compare(
            ref_path, copy_path, active_effects=active_effects
        )

        # ── 2. 연주용 오디오 로드 ─────────────────────────────────────
        y_ref,  _ = librosa.load(ref_path,  sr=sr_playing, mono=True)
        y_copy, _ = librosa.load(copy_path, sr=sr_playing, mono=True)

        # ── 3. 연주 분석 (음정 · 박자 시계열) ────────────────────────
        ts_report, ts_errors = run_timeseries_pipeline(y_ref, y_copy, sr_playing)

    # ── 4. 통합 반환 ────────────────────────────────────────────────
    return {
        "reference_path": str(ref_path),
        "copy_path":      str(copy_path),
        "effect_feedback": {
            "overall_similarity": effect_result["overall_similarity"],
            "axes":               effect_result["axes"],
        },
        # 0529/연주 피드백 = 음정(pitch) + 박자(rhythm) 시계열만. tone(음색) 리포트 제거.
        "playing_feedback": {
            "pitch_score":        ts_report.pitch_score,
            "rhythm_score":       ts_report.rhythm_score,
            "combined_score":     ts_report.combined_score,
            "grade":              ts_report.grade,
            "pitch_reliable":     ts_report.pitch_reliable,  # 0529/False면 delay로 음정 부정확 가능성
            "pitch_mae_semitone": ts_errors.pitch_mae_semitone,
            "onset_mae_ms":       ts_errors.onset_mae_ms,
            "issues":             ts_report.issues,
            "suggestions":        ts_report.suggestions,
            "strengths":          ts_report.strengths,
        },
    }