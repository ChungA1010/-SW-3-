"""
random_pair_test.py
-------------------
통합 파이프라인 스모크 테스트.
폴더에서 wav 파일 2개를 무작위로 뽑아 N번 실행하고 눈으로 결과를 확인.
"""

import sys
import io
import re
import random
import argparse
import contextlib
from pathlib import Path
from collections import defaultdict

# Windows 콘솔 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_EFFECT_RE  = re.compile(r"(dist|delay|phaser)_(\d+)")
_PLAY_RE    = re.compile(r"(powerchord|chord|solo)_(\d+)\.wav$", re.IGNORECASE)
_EXCLUDE_RE = re.compile(r"(chorus|reverb|\+)")

_AXIS_TO_EFFECT = {"drive": "dist", "space": "delay", "phase": "phaser"}
_AXIS_LABEL     = {"drive": "드라이브", "space": "공간계 ", "phase": "위상계 "}

VALID_NUMS = {2, 5, 6}


# ──────────────────────────────────────────────────────────────
# 파싱
# ──────────────────────────────────────────────────────────────

def parse_strengths(filename: str) -> dict[str, int]:
    """파일명에서 이펙터 강도 추출. 화이트리스트(dist/delay/phaser)만."""
    return {e: int(v) for e, v in _EFFECT_RE.findall(filename)}


def parse_play(filename: str) -> tuple[str, int] | None:
    """파일명에서 연주타입·인덱스 추출. powerchord를 chord보다 먼저 매칭."""
    m = _PLAY_RE.search(filename)
    return (m.group(1).lower(), int(m.group(2))) if m else None


# ──────────────────────────────────────────────────────────────
# 파일 탐색
# ──────────────────────────────────────────────────────────────

def discover_wavs(audio_dir: Path) -> list[tuple[Path, dict, tuple | None]]:
    """
    유효한 wav 파일 목록 반환.
    - chorus/reverb/+ 포함 제외
    - 이펙터 최소 1개 있는 것만
    - 곡 번호가 VALID_NUMS(2, 5, 6)인 것만
    """
    candidates = []
    for f in sorted(audio_dir.glob("*.wav")):
        if _EXCLUDE_RE.search(f.name):
            continue
        strengths = parse_strengths(f.name)
        if not strengths:
            continue
        play = parse_play(f.name)
        if play is None or play[1] not in VALID_NUMS:
            continue
        candidates.append((f, strengths, play))
    return candidates


# ──────────────────────────────────────────────────────────────
# 검증 로직
# ──────────────────────────────────────────────────────────────

def expected_action(ref_val: int | None, copy_val: int | None) -> str:
    """파일명 강도로 기대 action 계산. 한쪽/양쪽 없으면 keep."""
    if ref_val is None or copy_val is None:
        return "keep"
    diff = copy_val - ref_val
    if diff > 30:  return "much_lower"
    if diff > 0:   return "lower"
    if diff < -30: return "much_raise"
    if diff < 0:   return "raise"
    return "keep"


def action_matches(expected: str, actual: str) -> bool:
    """방향(lower계열/raise계열/keep)이 일치하면 True."""
    def direction(a: str) -> str:
        if "lower" in a: return "lower"
        if "raise" in a: return "raise"
        return "keep"
    return direction(expected) == direction(actual)


# ──────────────────────────────────────────────────────────────
# 쌍 선택
# ──────────────────────────────────────────────────────────────

def _build_groups(candidates: list, strict: bool) -> list[list]:
    """
    candidates를 그룹으로 묶어 반환.
    항상 (playtype, idx, frozenset(effects)) 기준 — 연주타입·번호·이펙터 종류 모두 동일.
    strict=True(--same-group)도 동일 키 사용 (이미 가장 엄격한 기준).
    """
    groups = defaultdict(list)
    for item in candidates:
        _, strengths, play = item
        key = (play[0], play[1], frozenset(strengths.keys()))
        groups[key].append(item)
    return [g for g in groups.values() if len(g) >= 2]


def pick_pair(
    candidates: list,
    same_group: bool,
    rng: random.Random,
) -> tuple | None:
    """
    ref, copy 한 쌍을 뽑아 반환. 순서도 무작위.
    항상 같은 (playtype, idx) 내에서 뽑음.
    --same-group이면 이펙터 종류까지 동일한 그룹에서, 강도만 다른 쌍 선택.
    """
    valid_groups = _build_groups(candidates, strict=same_group)
    if not valid_groups:
        return None

    for _ in range(30):
        group = rng.choice(valid_groups)
        ref, copy = rng.sample(group, 2)
        # --same-group일 때 강도 dict가 완전히 같은 쌍은 제외
        if same_group and ref[1] == copy[1]:
            continue
        pair = [ref, copy]
        rng.shuffle(pair)
        return tuple(pair)
    return None


# ──────────────────────────────────────────────────────────────
# 단일 trial 실행
# ──────────────────────────────────────────────────────────────

def run_one_trial(
    trial_num: int,
    total: int,
    ref_item: tuple,
    copy_item: tuple,
    effect_only: bool,
    sr: int,
) -> tuple[int, int]:
    """한 쌍을 실행하고 출력. (matched, verifiable) 반환."""
    ref_path,  ref_strengths,  _ = ref_item
    copy_path, copy_strengths, _ = copy_item

    print(f"\n{'═' * 54}")
    print(f"Trial {trial_num} / {total}")
    print(f"{'═' * 54}")
    print(f"ref : {ref_path.name}")
    print(f"copy: {copy_path.name}")
    print("파일명 강도:")
    print(f"  ref  → {', '.join(f'{e}={v}' for e, v in ref_strengths.items()) or '(없음)'}")
    print(f"  copy → {', '.join(f'{e}={v}' for e, v in copy_strengths.items()) or '(없음)'}")

    # ── 파이프라인 실행 ──────────────────────────────
    try:
        _sink = io.StringIO()
        if effect_only:
            from tone_match_model import ToneMatchModel
            with contextlib.redirect_stdout(_sink):
                result = ToneMatchModel().compare(ref_path, copy_path, active_effects=None)
            axes      = result["axes"]
            tone_data = None
        else:
            from unified_pipeline import run_unified_feedback
            with contextlib.redirect_stdout(_sink):
                data = run_unified_feedback(ref_path, copy_path, sr_playing=sr, active_effects=None)
            axes      = data["effect_feedback"]["axes"]
            tone_data = data["playing_feedback"]
    except Exception as e:
        print(f"  [건너뜀] 오류 발생: {e}")
        return 0, 0

    # ── 이펙터 피드백 출력 ──────────────────────────
    print("\n[이펙터 피드백]")
    matched    = 0
    verifiable = 0

    for ax in axes:
        axis_name  = ax["axis"]
        action     = ax["action"]
        message    = ax.get("message", "")
        effect     = _AXIS_TO_EFFECT[axis_name]
        label      = _AXIS_LABEL[axis_name]

        ref_v  = ref_strengths.get(effect)
        copy_v = copy_strengths.get(effect)
        rv_str = str(ref_v)  if ref_v  is not None else "-"
        cv_str = str(copy_v) if copy_v is not None else "-"

        if axis_name == "phase":
            print(f"[{label}] {effect}  {rv_str}→{cv_str}  diff=0    → keep   (on/off만 판정)")
            print(f"  {message}")
            continue

        diff     = (copy_v or 0) - (ref_v or 0)
        diff_str = f"{diff:+d}" if (ref_v is not None and copy_v is not None) else "N/A"
        exp      = expected_action(ref_v, copy_v)
        ok       = action_matches(exp, action)
        mark     = "✅" if ok else f"❌(기대 {exp})"

        verifiable += 1
        if ok:
            matched += 1

        print(f"[{label}] {effect}  {rv_str}→{cv_str}  diff={diff_str}  → {action}   {mark}")
        print(f"  {message}")

    print(f"drive/space 검증: {matched}/{verifiable} 일치")

    # ── 연주 피드백 출력 ──────────────────────────────
    if tone_data is not None:
        tone = tone_data["tone"]
        ts   = tone_data["timeseries"]
        print("\n[연주 피드백]")
        print(f"  톤   : {tone['overall_score']:.1f}% ({tone['grade']})")
        print(f"  음정 : {ts['pitch_score']:.1f}점 (MAE {ts['pitch_mae_semitone']:.2f}반음)")
        print(f"  박자 : {ts['rhythm_score']:.1f}점 (MAE {ts['onset_mae_ms']:.0f}ms)")

    return matched, verifiable


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="통합 파이프라인 스모크 테스트")
    parser.add_argument("--audio-dir",   default="./play", help="wav 폴더 (기본: ./play)")
    parser.add_argument("--num-trials",  type=int, default=5, help="반복 횟수 (기본: 5)")
    parser.add_argument("--seed",        type=int, default=None, help="랜덤 시드 (기본: None)")
    parser.add_argument("--sr",          type=int, default=22050, help="샘플링 레이트 (기본: 22050)")
    parser.add_argument("--effect-only", action="store_true",
                        help="이펙터 피드백만 실행 — 연주 파이프라인 스킵 (빠름)")
    parser.add_argument("--same-group",  action="store_true",
                        help="같은 연주타입·인덱스·이펙터 종류 조합에서 강도만 다른 쌍 뽑기")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"오류: --audio-dir '{audio_dir}'가 존재하지 않습니다.")
        sys.exit(1)

    candidates = discover_wavs(audio_dir)
    if len(candidates) < 2:
        print(f"오류: 유효한 wav 파일이 {len(candidates)}개뿐입니다 (최소 2개 필요).")
        sys.exit(1)

    mode_str  = "effect-only" if args.effect_only else "전체 파이프라인"
    group_str = "same-group" if args.same_group else "완전 무작위"
    print(f"스캔 완료: {len(candidates)}개 유효 파일 ({audio_dir})")
    print(f"모드: {mode_str} | {group_str} | N={args.num_trials} | seed={args.seed}")

    rng = random.Random(args.seed)
    total_matched    = 0
    total_verifiable = 0

    for i in range(1, args.num_trials + 1):
        pair = pick_pair(candidates, args.same_group, rng)
        if pair is None:
            print(f"\nTrial {i}: 유효한 쌍을 찾지 못했습니다. 건너뜁니다.")
            continue
        m, v = run_one_trial(i, args.num_trials, pair[0], pair[1], args.effect_only, args.sr)
        total_matched    += m
        total_verifiable += v

    print(f"\n{'═' * 54}")
    print(f"총 {args.num_trials}쌍 중 drive/space 검증 {total_matched}/{total_verifiable} 일치")
    print(f"{'═' * 54}")
    print("\n사용법 예시:")
    print(f"  python random_pair_test.py --audio-dir {args.audio_dir} --num-trials 10 --seed 42")
    print(f"  python random_pair_test.py --audio-dir {args.audio_dir} --effect-only --num-trials 3")
    print(f"  python random_pair_test.py --audio-dir {args.audio_dir} --same-group --num-trials 5")


if __name__ == "__main__":
    main()
