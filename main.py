"""
main.py
-------
기타 톤 유사도 분석 + 시계열(음정·박자) 분석 통합 파이프라인

사용법:
  python main.py --ref path/to/reference.wav --user path/to/user.wav

선택 옵션:
  --no-vggish    VGGish 임베딩 비활성화 (빠른 실행)
  --no-yamnet    YAMNet 임베딩 비활성화
  --no-timeseries  시계열 분석 비활성화
  --sr           타겟 샘플링 레이트 (기본 22050)
  --save         결과를 JSON으로 저장할 경로
"""

import argparse
import librosa
import numpy as np
import json

# ── 기존 톤 분석 모듈 ──
from preprocessor     import preprocess
from feature_extractor import extract_all_features
from similarity        import compute_similarity
from feedback          import generate_feedback

# ── 시계열 분석 모듈 (신규) ──
from pipeline_timeseries import run_timeseries_pipeline


# ──────────────────────────────────────────
# 오디오 로드
# ──────────────────────────────────────────

def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """오디오 파일을 모노로 로드."""
    print(f"[로드] {path}")
    y, sr_loaded = librosa.load(path, sr=sr, mono=True)
    print(f"  -> {len(y)} samples | {len(y)/sr:.2f}s | sr={sr}")
    return y, sr


# ──────────────────────────────────────────
# 피처 벡터 직렬화 (JSON 저장용)
# ──────────────────────────────────────────

def features_to_dict(feat) -> dict:
    """AudioFeatures를 JSON 직렬화 가능한 dict로 변환."""
    import dataclasses
    d = {}
    for f in dataclasses.fields(feat):
        val = getattr(feat, f.name)
        if isinstance(val, np.ndarray):
            d[f.name] = val.tolist()
        else:
            d[f.name] = val
    return d


# ──────────────────────────────────────────
# 메인 분석 함수
# ──────────────────────────────────────────

def analyze(
    ref_path: str,
    user_path: str,
    sr: int = 22050,
    use_vggish: bool = True,
    use_yamnet: bool = True,
    use_timeseries: bool = True,
    save_report: str = None,
) -> dict:
    """전체 분석 파이프라인 실행."""

    # ── Step 1: 오디오 로드 ──────────────────
    print("\n[1/5] 오디오 로드")
    y_ref, sr = load_audio(ref_path, sr)
    y_user, sr = load_audio(user_path, sr)

    # ── Step 2: 전처리 ─────────────────────
    print("\n[2/5] 전처리")
    prep = preprocess(y_ref, y_user, sr)

    y_ref_clean  = prep['y_ref']
    y_user_clean = prep['y_user']

    tempo_info = {
        'ref_duration':  prep['ref_duration'],
        'user_duration': prep['user_duration'],
        'ref_bpm':       prep['ref_bpm'],
        'user_bpm':      prep['user_bpm'],
    }

    # ── Step 3: 톤 피처 추출 ────────────────
    print("\n[3/5] 톤 피처 추출")
    print("  > 원본 피처 추출 중...")
    feat_ref  = extract_all_features(y_ref_clean, sr
                                    #  , use_vggish, use_yamnet
                                     )

    print("  > 사용자 피처 추출 중...")
    feat_user = extract_all_features(y_user_clean, sr
                                     #  , use_vggish, use_yamnet
                                     )

    # ── Step 4: 톤 유사도 + 피드백 ──────────
    print("\n[4/5] 톤 유사도 계산 및 피드백 생성")
    sim_result = compute_similarity(feat_ref, feat_user, tempo_info=tempo_info)
    tone_report = generate_feedback(sim_result, feat_ref, feat_user, tempo_info=tempo_info)

    # ── Step 5: 시계열 분석 (신규) ──────────
    ts_report = None
    ts_errors = None

    if use_timeseries:
        print("\n[5/5] 시계열 분석 (음정·박자)")
        ts_report, ts_errors = run_timeseries_pipeline(y_ref, y_user, sr)
    else:
        print("\n[5/5] 시계열 분석 건너뜀")

    # ══════════════════════════════════════════
    # 결과 출력
    # ══════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  [ A. 톤 분석 결과 ]")
    print("=" * 60)
    print(tone_report.summary())

    # 피처별 상세 유사도
    print("\n[ 피처별 상세 유사도 ]")
    for key in [
        # "vggish_embedding", 
        "mfcc", "spectral_contrast",
        "spectral_centroid", "chroma", "tonnetz",
        # "yamnet_embedding", 
        "spectral_bandwidth", "zcr", "rms",
    ]:
        score = getattr(sim_result, key)
        bar   = "#" * int(score * 20)
        print(f"  {key:<25} {score:.3f}  [{bar:<20}]")

    print(f"\n  톤 최종 점수: {sim_result.final_score:.4f} "
          f"({tone_report.overall_score:.1f}%) [{tone_report.grade}]")

    # ── 시계열 결과 출력 ──
    if ts_report:
        print("\n" + "=" * 60)
        print("  [ B. 시계열 분석 결과 (음정·박자) ]")
        print("=" * 60)
        print(f"  음정 점수: {ts_report.pitch_score:.1f}")
        print(f"  박자 점수: {ts_report.rhythm_score:.1f}")
        print(f"  종합 점수: {ts_report.combined_score:.1f}% [{ts_report.grade}]")

        if ts_report.strengths:
            print("\n  ✅ 잘한 점:")
            for s in ts_report.strengths:
                print(f"    - {s}")

        if ts_report.issues:
            print("\n  ⚠️ 문제 구간:")
            for issue in ts_report.issues:
                print(f"    - {issue}")

        if ts_report.suggestions:
            print("\n  💡 제안:")
            for sug in ts_report.suggestions:
                print(f"    - {sug}")

        if ts_errors:
            print(f"\n  [상세] 평균 음정 오차: {ts_errors.pitch_mae_semitone:.2f}반음")
            print(f"  [상세] 평균 타이밍 오차: {ts_errors.onset_mae_ms:.0f}ms")

    print("\n" + "=" * 60)

    # ══════════════════════════════════════════
    # 결과 딕셔너리 (JSON 저장용)
    # ══════════════════════════════════════════

    result = {
        "tone_analysis": {
            "overall_score": tone_report.overall_score,
            "grade": tone_report.grade,
            "similarities": {
                # "vggish_embedding":    sim_result.vggish_embedding,
                "mfcc":                sim_result.mfcc,
                "spectral_contrast":   sim_result.spectral_contrast,
                "spectral_centroid":   sim_result.spectral_centroid,
                "chroma":              sim_result.chroma,
                "tonnetz":             sim_result.tonnetz,
                # "yamnet_embedding":    sim_result.yamnet_embedding,
                "spectral_bandwidth":  sim_result.spectral_bandwidth,
                "zcr":                 sim_result.zcr,
                "rms":                 sim_result.rms,
            },
            "final_score": sim_result.final_score,
            "feedback": {
                "issues":      tone_report.issues,
                "suggestions": tone_report.suggestions,
                "strengths":   tone_report.strengths,
            },
        },
        "tempo_info": tempo_info,
    }

    if ts_report:
        result["timeseries_analysis"] = {
            "pitch_score":       ts_report.pitch_score,
            "rhythm_score":      ts_report.rhythm_score,
            "combined_score":    ts_report.combined_score,
            "grade":             ts_report.grade,
            "pitch_mae_semitone": ts_errors.pitch_mae_semitone,
            "onset_mae_ms":      ts_errors.onset_mae_ms,
            "feedback": {
                "issues":      ts_report.issues,
                "suggestions": ts_report.suggestions,
                "strengths":   ts_report.strengths,
            },
        }

    if save_report:
        with open(save_report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] 분석 결과 -> {save_report}")

    return result


# ──────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="기타 톤 유사도 + 시계열(음정·박자) 분석기"
    )
    parser.add_argument(
        "--ref", required=True,
        help="원본(기타 분리) 음원 경로 (.wav / .mp3)"
    )
    parser.add_argument(
        "--user", required=True,
        help="사용자 연주 음원 경로 (.wav / .mp3)"
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="샘플링 레이트 (기본: 22050)"
    )
    parser.add_argument(
        "--no-vggish", action="store_true",
        help="VGGish 임베딩 비활성화"
    )
    parser.add_argument(
        "--no-yamnet", action="store_true",
        help="YAMNet 임베딩 비활성화"
    )
    parser.add_argument(
        "--no-timeseries", action="store_true",
        help="시계열(음정·박자) 분석 비활성화"
    )
    parser.add_argument(
        "--save", default=None,
        help="결과를 JSON으로 저장할 경로"
    )

    args = parser.parse_args()

    analyze(
        ref_path       = args.ref,
        user_path      = args.user,
        sr             = args.sr,
        # use_vggish     = not args.no_vggish,
        # use_yamnet     = not args.no_yamnet,
        use_timeseries = not args.no_timeseries,
        save_report    = args.save,
    )
