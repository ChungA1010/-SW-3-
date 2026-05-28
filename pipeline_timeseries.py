"""
통합 파이프라인: 1~4단계를 순차 실행
main.py에서 이 함수만 호출하면 된다.

사용 예:
    from pipeline_timeseries import run_timeseries_pipeline
    report = run_timeseries_pipeline(ref_y, user_y, sr=22050)
    print(report.grade, report.issues)
"""
import numpy as np
from feature_extractor_timeseries import extract_timeseries
from timeseries_align import (
    dtw_align_onsets, dtw_align_pitch, build_aligned_arrays,
)
from timeseries_scoring import compute_timeseries_errors
from feedback_timeseries import generate_timeseries_feedback


def run_timeseries_pipeline(ref_y: np.ndarray,
                            user_y: np.ndarray,
                            sr: int = 22050):
    # 1단계: 피처 추출
    ref_feat = extract_timeseries(ref_y, sr)
    user_feat = extract_timeseries(user_y, sr)

    # 2단계: DTW 정렬
    onset_path = dtw_align_onsets(ref_feat.onset_times, user_feat.onset_times)
    pitch_path = dtw_align_pitch(ref_feat.pitch_midi, user_feat.pitch_midi)

    aligned_ref_onsets, aligned_user_onsets = build_aligned_arrays(
        ref_feat.onset_times, user_feat.onset_times, onset_path,
    )
    aligned_ref_midi, aligned_user_midi = build_aligned_arrays(
        ref_feat.pitch_midi, user_feat.pitch_midi, pitch_path,
    )
    aligned_ref_pitch_times, _ = build_aligned_arrays(
        ref_feat.pitch_times, user_feat.pitch_times, pitch_path,
    )

    # 3단계: 오차 계산
    ts_errors = compute_timeseries_errors(
        aligned_ref_midi=aligned_ref_midi,
        aligned_user_midi=aligned_user_midi,
        aligned_ref_times=aligned_ref_pitch_times,
        aligned_ref_onsets=aligned_ref_onsets,
        aligned_user_onsets=aligned_user_onsets,
    )

    # 4단계: 피드백
    report = generate_timeseries_feedback(ts_errors)
    return report, ts_errors
