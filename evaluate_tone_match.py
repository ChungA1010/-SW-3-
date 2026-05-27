#!/usr/bin/env python3
"""
evaluate_tone_match.py — ToneMatchModel 정확도 평가 스크립트.

Usage:
    python evaluate_tone_match.py --audio-dir ./fb --num-pairs 100
    python evaluate_tone_match.py --audio-dir ./fb --num-pairs 5 --seed 42
    python evaluate_tone_match.py --audio-dir ./fb --num-pairs 500 --output results.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

sys.path.insert(0, str(Path(__file__).parent))
from tone_match_model import ToneMatchModel  # noqa: E402


# ── Regex ──────────────────────────────────────────────────────────────────────
_EFFECT_RE  = re.compile(r"(dist|delay|phaser)_(\d+)")
# powerchord 먼저 — 그냥 chord 를 쓰면 "powerchord" 안의 "chord" 에 먼저 걸림
_PLAY_RE    = re.compile(r"(powerchord|chord|solo)_(\d+)\.wav$", re.IGNORECASE)
_EXCLUDE_RE = re.compile(r"(chorus|reverb|\+)")

# ── Constants ──────────────────────────────────────────────────────────────────
EFFECT_TO_AXIS: dict[str, str] = {"dist": "drive", "delay": "space", "phaser": "phase"}
AXIS_TO_EFFECT: dict[str, str] = {"drive": "dist", "space": "delay", "phase": "phaser"}
ALL_AXES = ["drive", "space", "phase"]

_LOWER_GROUP = {"lower", "much_lower"}
_RAISE_GROUP = {"raise", "much_raise"}

CSV_FIELDS = [
    "pair_idx", "ref_file", "copy_file", "effects_in_group",
    "drive_ref_strength", "drive_copy_strength", "drive_expected", "drive_actual", "drive_diff", "drive_strict", "drive_lenient",
    "space_ref_strength", "space_copy_strength", "space_expected", "space_actual", "space_diff", "space_strict", "space_lenient",
    "phase_ref_strength", "phase_copy_strength", "phase_expected", "phase_actual", "phase_diff", "phase_strict", "phase_lenient",
    "overall_strict", "overall_lenient",
]


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class ParsedFile:
    path: Path
    play_type: str
    play_idx: int
    effects: dict[str, int]   # {"dist": 50, "delay": 75}
    effects_key: frozenset    # frozenset({"dist", "delay"})


@dataclass
class PairInfo:
    ref: ParsedFile
    copy: ParsedFile
    group_key: tuple          # (play_type, play_idx, effects_key)


# ── Filename parsing ───────────────────────────────────────────────────────────
def parse_filename(path: Path) -> ParsedFile | None:
    name = path.name
    if _EXCLUDE_RE.search(name):
        return None
    play_m = _PLAY_RE.search(name)
    if not play_m:
        return None
    play_type = play_m.group(1).lower()
    play_idx  = int(play_m.group(2))
    effects: dict[str, int] = {
        m.group(1): int(m.group(2)) for m in _EFFECT_RE.finditer(name)
    }
    if not effects:
        return None
    return ParsedFile(
        path=path,
        play_type=play_type,
        play_idx=play_idx,
        effects=effects,
        effects_key=frozenset(effects.keys()),
    )


# ── Discovery & grouping ───────────────────────────────────────────────────────
def discover_and_group(audio_dir: Path) -> dict[tuple, list[ParsedFile]]:
    groups: dict[tuple, list[ParsedFile]] = {}
    for wav in sorted(audio_dir.glob("*.wav")):
        pf = parse_filename(wav)
        if pf is None:
            continue
        key = (pf.play_type, pf.play_idx, pf.effects_key)
        groups.setdefault(key, []).append(pf)
    return groups


# ── Pair generation ────────────────────────────────────────────────────────────
def generate_pairs(groups: dict[tuple, list[ParsedFile]]) -> list[PairInfo]:
    """All unordered (n choose 2) pairs within each group, excluding same-effect dicts."""
    candidates: list[PairInfo] = []
    for key, files in groups.items():
        if len(files) < 2:
            continue
        for a, b in itertools.combinations(files, 2):
            if a.effects == b.effects:
                continue
            candidates.append(PairInfo(ref=a, copy=b, group_key=key))
    return candidates


def sample_pairs(candidates: list[PairInfo], n: int, rng: random.Random) -> list[PairInfo]:
    """Random sample; ref/copy order is randomised per pair."""
    sampled = rng.sample(candidates, min(n, len(candidates)))
    result: list[PairInfo] = []
    for pair in sampled:
        if rng.random() < 0.5:
            pair = PairInfo(ref=pair.copy, copy=pair.ref, group_key=pair.group_key)
        result.append(pair)
    return result


# ── Expected action ────────────────────────────────────────────────────────────
def expected_action(ref_s: int | None, copy_s: int | None) -> str:
    """
    diff = copy_s - ref_s
    |diff| >= 50 → much_lower / much_raise
    |diff|  < 50 → lower / raise
    diff == 0    → keep
    """
    if ref_s is None or copy_s is None:
        return "keep"
    diff = copy_s - ref_s
    if diff == 0:
        return "keep"
    if abs(diff) >= 50:
        return "much_lower" if diff > 0 else "much_raise"
    return "lower" if diff > 0 else "raise"


# ── Correctness ────────────────────────────────────────────────────────────────
def is_correct(expected: str, actual: str) -> bool:
    """
    strict & lenient share the same rule here (lenient kept as a separate column
    in case the definition diverges later):
      keep     → keep only
      lower    → lower or much_lower
      raise    → raise or much_raise
      much_lower → much_lower or lower
      much_raise → much_raise or raise
    """
    if expected == actual:
        return True
    if expected == "keep":
        return False
    if expected in _LOWER_GROUP and actual in _LOWER_GROUP:
        return True
    if expected in _RAISE_GROUP and actual in _RAISE_GROUP:
        return True
    return False


# ── Evaluate one pair ──────────────────────────────────────────────────────────
def evaluate_pair(model: ToneMatchModel, pair: PairInfo, pair_idx: int) -> dict:
    # Suppress the [auto-detected] debug line printed by compare()
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink):
        result = model.compare(pair.ref.path, pair.copy.path)

    axes_map    = {a["axis"]: a for a in result["axes"]}
    effects_key = pair.group_key[2]

    row: dict = {
        "pair_idx":         pair_idx,
        "ref_file":         pair.ref.path.name,
        "copy_file":        pair.copy.path.name,
        "effects_in_group": "+".join(sorted(effects_key)),
    }

    all_strict = all_lenient = True

    for axis in ALL_AXES:
        effect = AXIS_TO_EFFECT[axis]
        # None when the effect is not part of this group
        in_group = effect in effects_key
        ref_s    = pair.ref.effects.get(effect)  if in_group else None
        copy_s   = pair.copy.effects.get(effect) if in_group else None
        # phase: classifier가 phaser on/off를 판정하므로 강도 차이는 항상 keep 기대
        if axis == "phase":
            exp = "keep"
        else:
            exp = expected_action(ref_s, copy_s)
        act      = axes_map.get(axis, {}).get("action", "keep")
        diff     = (copy_s - ref_s) if (ref_s is not None and copy_s is not None) else None

        strict  = is_correct(exp, act)
        lenient = is_correct(exp, act)

        row[f"{axis}_ref_strength"]  = ref_s  if ref_s  is not None else ""
        row[f"{axis}_copy_strength"] = copy_s if copy_s is not None else ""
        row[f"{axis}_expected"]      = exp
        row[f"{axis}_actual"]        = act
        row[f"{axis}_diff"]          = diff if diff is not None else ""
        row[f"{axis}_strict"]        = strict
        row[f"{axis}_lenient"]       = lenient

        if not strict:  all_strict  = False
        if not lenient: all_lenient = False

    row["overall_strict"]  = all_strict
    row["overall_lenient"] = all_lenient
    return row


# ── Console summary ────────────────────────────────────────────────────────────
def summarize(results: list[dict], n_candidates: int, elapsed_ms: float) -> None:
    n = len(results)
    if n == 0:
        print("결과 없음.")
        return

    W = 72

    def _pct(it) -> str:
        c = sum(1 for r in results if it(r))
        return f"{c}/{n} ({100*c/n:.1f}%)"

    print(f"\n[note] phase 축은 강도 추정 안 함 (classifier가 phaser on/off 판정) — 항상 keep 기대")
    print(f"\n{'━'*W}")
    print(f"  {n} pairs sampled from {n_candidates:,} candidates")
    print(f"  평균 처리 시간: {elapsed_ms / n:.1f} ms/pair")
    print(f"{'━'*W}")

    # ── Overall ──────────────────────────────────────────────────────────────
    print(f"\n  전체 정확도")
    print(f"    strict : {_pct(lambda r: r['overall_strict'])}")
    print(f"    lenient: {_pct(lambda r: r['overall_lenient'])}")

    # ── Per-axis ─────────────────────────────────────────────────────────────
    print(f"\n  축별 정확도")
    for axis in ALL_AXES:
        s = sum(1 for r in results if r[f"{axis}_strict"])
        l = sum(1 for r in results if r[f"{axis}_lenient"])
        print(f"    {axis:6s}  strict={s}/{n} ({100*s/n:.1f}%)  lenient={l}/{n} ({100*l/n:.1f}%)")

    # ── Per-combo ─────────────────────────────────────────────────────────────
    combo_map: dict[str, list[dict]] = {}
    for r in results:
        combo_map.setdefault(r["effects_in_group"], []).append(r)

    print(f"\n  이펙터 조합별 정확도 (strict / lenient)")
    # sort: single effects first (len 1), then by name
    for combo in sorted(combo_map, key=lambda x: (len(x.split("+")), x)):
        rows = combo_map[combo]
        s = sum(1 for r in rows if r["overall_strict"])
        l = sum(1 for r in rows if r["overall_lenient"])
        m = len(rows)
        print(f"    {combo:25s}  strict={s}/{m} ({100*s/m:.1f}%)  lenient={l}/{m} ({100*l/m:.1f}%)")

    # ── Failure cases top 10 ──────────────────────────────────────────────────
    failures = [r for r in results if not r["overall_strict"]]
    if not failures:
        print("\n  ✅ 실패 케이스 없음")
        return

    print(f"\n  실패 케이스 상위 10개 (overall_strict=False)")
    for i, r in enumerate(failures[:10], 1):
        play_m   = _PLAY_RE.search(r["ref_file"])
        play_tag = f"{play_m.group(1)}_{play_m.group(2)}" if play_m else r["ref_file"]

        parts = []
        for axis in ALL_AXES:
            effect = AXIS_TO_EFFECT[axis]
            rs = r[f"{axis}_ref_strength"]
            cs = r[f"{axis}_copy_strength"]
            parts.append(f"{effect}:{rs}→{cs}" if rs != "" else f"{effect}:-")

        fail_axes = [
            f"{axis} expected={r[f'{axis}_expected']} got={r[f'{axis}_actual']}"
            for axis in ALL_AXES if not r[f"{axis}_strict"]
        ]
        print(f"  [{i:2d}] {play_tag}  {' '.join(parts)}  |  {' | '.join(fail_axes)}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="ToneMatchModel 정확도 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio-dir",   type=Path, required=True,
                        help="WAV 폴더 경로")
    parser.add_argument("--num-pairs",   type=int,  default=100,
                        help="샘플링할 쌍 수 (기본 100)")
    parser.add_argument("--seed",        type=int,  default=None,
                        help="랜덤 시드 (기본 없음 — 매번 다른 결과)")
    parser.add_argument("--output",      type=Path, default=Path("eval_results.csv"),
                        help="CSV 저장 경로 (기본 eval_results.csv)")
    parser.add_argument("--sample-rate", type=int,  default=32000)
    parser.add_argument("--hop-length",  type=int,  default=256)
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()
    if not audio_dir.exists():
        print(f"[ERROR] 폴더 없음: {audio_dir}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    print(f"\n  스캔 중: {audio_dir}")
    groups     = discover_and_group(audio_dir)
    candidates = generate_pairs(groups)
    pairs      = sample_pairs(candidates, args.num_pairs, rng)

    n_candidates = len(candidates)
    n_pairs      = len(pairs)
    print(f"  그룹 수: {len(groups):,}  후보 쌍: {n_candidates:,}  평가 쌍: {n_pairs}")

    if n_pairs == 0:
        print("[ERROR] 평가할 쌍이 없습니다.", file=sys.stderr)
        return 1

    model   = ToneMatchModel(sample_rate=args.sample_rate, hop_length=args.hop_length)
    results: list[dict] = []

    t0 = time.perf_counter()

    if HAS_TQDM:
        it = tqdm(enumerate(pairs, 1), total=n_pairs, desc="  평가", unit="pair")
    else:
        it = enumerate(pairs, 1)

    for i, pair in it:
        results.append(evaluate_pair(model, pair, i))
        if not HAS_TQDM and i % 20 == 0:
            print(f"  [{i}/{n_pairs}]")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV 저장: {args.output.resolve()}")

    summarize(results, n_candidates, elapsed_ms)
    return 0 if all(r["overall_strict"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
