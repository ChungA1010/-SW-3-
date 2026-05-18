"""
2단계: 시계열 DTW 정렬
- Onset 타임스탬프 배열 정렬
- Pitch(MIDI) 배열 정렬
오디오 파형은 건드리지 않는다.
"""
import numpy as np
from dtaidistance import dtw


def dtw_align_onsets(ref_onsets: np.ndarray,
                     user_onsets: np.ndarray) -> list[tuple[int, int]]:
    """
    두 onset 시각 배열을 DTW로 매칭.
    Returns: [(ref_idx, user_idx), ...] 매칭 쌍 리스트
    """
    if len(ref_onsets) == 0 or len(user_onsets) == 0:
        return []

    # 1-D 시계열로 간주하여 DTW warping path 계산
    path = dtw.warping_path(
        ref_onsets.astype(np.double),
        user_onsets.astype(np.double),
    )
    return path  # List[(i, j)]


def dtw_align_pitch(ref_midi: np.ndarray,
                    user_midi: np.ndarray) -> list[tuple[int, int]]:
    """
    두 MIDI 피치 배열을 DTW로 매칭.
    NaN(unvoiced) 프레임은 0으로 치환 후 정렬한다.
    """
    def _fill(arr):
        a = arr.copy()
        a[np.isnan(a)] = 0.0
        return a.astype(np.double)

    if len(ref_midi) == 0 or len(user_midi) == 0:
        return []

    path = dtw.warping_path(_fill(ref_midi), _fill(user_midi))
    return path


def build_aligned_arrays(ref_vals: np.ndarray,
                         user_vals: np.ndarray,
                         path: list[tuple[int, int]]):
    """
    warping path를 이용해 두 배열을 같은 길이로 정렬.
    Returns: (aligned_ref, aligned_user) 각각 (len(path),)
    """
    ri = np.array([p[0] for p in path])
    ui = np.array([p[1] for p in path])
    return ref_vals[ri], user_vals[ui]
