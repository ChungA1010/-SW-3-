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


# 0529/무음 비율이 이 값 미만이면 delay(잔향)로 판단 → 음정 분석 신뢰도 낮음.
#       검증(노래 2/5/6): 솔로는 완벽 분리(no-delay≥15.6%, delay≤12%) → 핵심 케이스 정확.
#       한계: chord_2(지속화음, no-delay 11%)는 과임될 수 있고, powerchord_2+delay_100(15%)은
#             놓칠 수 있으나, 둘 다 무해(과임=안전, 미탐은 근음 지속이라 음정 손상 없음).
#             지속음에서 단일 임계값 완전분리는 불가하나 솔로 정확도를 위해 0.14 채택.
DELAY_SILENCE_THRESHOLD = 0.14


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

    # 0529/음정 신뢰도 판정: ref·user . 어느 쪽이든 delay면 음정 hedge(~일것같다).
    pitch_reliable = (
        ref_feat.silence_ratio  >= DELAY_SILENCE_THRESHOLD
        and user_feat.silence_ratio >= DELAY_SILENCE_THRESHOLD
    )

    # 3단계: 오차 계산
    ts_errors = compute_timeseries_errors(
        aligned_ref_midi=aligned_ref_midi,
        aligned_user_midi=aligned_user_midi,
        aligned_ref_times=aligned_ref_pitch_times,
        aligned_ref_onsets=aligned_ref_onsets,
        aligned_user_onsets=aligned_user_onsets,
        pitch_reliable=pitch_reliable,  # 0529/
    )

    # 4단계: 피드백
    report = generate_timeseries_feedback(ts_errors)
    return report, ts_errors
