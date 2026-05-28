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

from tone_match_model import ToneMatchModel
from preprocessor import preprocess
from feature_extractor import extract_all_features
from similarity import compute_similarity
from feedback import generate_feedback
from pipeline_timeseries import run_timeseries_pipeline


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

    # ── 1. 이펙터 피드백 ──────────────────────────────────────────────
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink):
        effect_result = ToneMatchModel().compare(
            ref_path, copy_path, active_effects=active_effects
        )

    # ── 2. 연주용 오디오 로드 ─────────────────────────────────────────
    y_ref,  _ = librosa.load(ref_path,  sr=sr_playing, mono=True)
    y_copy, _ = librosa.load(copy_path, sr=sr_playing, mono=True)

    # ── 3. 전처리 + 톤 분석 ──────────────────────────────────────────
    prep = preprocess(y_ref, y_copy, sr_playing)
    feat_ref  = extract_all_features(prep["y_ref"],  sr_playing)
    feat_copy = extract_all_features(prep["y_user"], sr_playing)

    tempo_info = {
        "ref_duration":  prep["ref_duration"],
        "user_duration": prep["user_duration"],
        "ref_bpm":       prep["ref_bpm"],
        "user_bpm":      prep["user_bpm"],
    }

    sim_result  = compute_similarity(feat_ref, feat_copy, tempo_info=tempo_info)
    tone_report = generate_feedback(sim_result, feat_ref, feat_copy, tempo_info=tempo_info)

    # ── 4. 시계열 분석 (음정 · 박자) ─────────────────────────────────
    ts_report, ts_errors = run_timeseries_pipeline(y_ref, y_copy, sr_playing)

    # ── 5. 통합 반환 ─────────────────────────────────────────────────
    return {
        "reference_path": str(ref_path),
        "copy_path":      str(copy_path),
        "effect_feedback": {
            "overall_similarity": effect_result["overall_similarity"],
            "axes":               effect_result["axes"],
        },
        "playing_feedback": {
            "tone": {
                "overall_score": tone_report.overall_score,
                "grade":         tone_report.grade,
                "issues":        tone_report.issues,
                "suggestions":   tone_report.suggestions,
                "strengths":     tone_report.strengths,
            },
            "timeseries": {
                "pitch_score":        ts_report.pitch_score,
                "rhythm_score":       ts_report.rhythm_score,
                "combined_score":     ts_report.combined_score,
                "grade":              ts_report.grade,
                "pitch_mae_semitone": ts_errors.pitch_mae_semitone,
                "onset_mae_ms":       ts_errors.onset_mae_ms,
                "issues":             ts_report.issues,
                "suggestions":        ts_report.suggestions,
                "strengths":          ts_report.strengths,
            },
        },
    }
