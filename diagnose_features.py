#!/usr/bin/env python3
"""
diagnose_features.py — Per-feature cross-axis contamination diagnosis.

Measures how much each ToneMatchModel feature changes when one effect's
intensity shifts from --low to --high (other effects held fixed), so you
can see which features are responsible for cross-axis contamination.

Usage:
    python diagnose_features.py --audio-dir ./fb --effect dist --fix-others isolated
    python diagnose_features.py --audio-dir ./fb --effect delay --fix-others any
    python diagnose_features.py --audio-dir ./fb --effect phaser --low 25 --high 75
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

sys.path.insert(0, str(Path(__file__).parent))
from tone_match_model import ToneMatchModel, EPSILON  # noqa: E402

# evaluate_tone_match.py replaces sys.stdout at module level, which would close
# the buffer already wrapped above.  Copy the two helpers instead of importing.
# Source: evaluate_tone_match.py
import itertools
import re
from dataclasses import dataclass

_EFFECT_RE  = re.compile(r"(dist|delay|phaser)_(\d+)")
_PLAY_RE    = re.compile(r"(powerchord|chord|solo)_(\d+)\.wav$", re.IGNORECASE)
_EXCLUDE_RE = re.compile(r"(chorus|reverb|\+)")


@dataclass
class ParsedFile:
    path: Path
    play_type: str
    play_idx: int
    effects: dict[str, int]
    effects_key: frozenset


def parse_filename(path: Path) -> "ParsedFile | None":
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
        path=path, play_type=play_type, play_idx=play_idx,
        effects=effects, effects_key=frozenset(effects.keys()),
    )


def discover_and_group(audio_dir: Path) -> dict:
    groups: dict = {}
    for wav in sorted(audio_dir.glob("*.wav")):
        pf = parse_filename(wav)
        if pf is None:
            continue
        key = (pf.play_type, pf.play_idx, pf.effects_key)
        groups.setdefault(key, []).append(pf)
    return groups


# ── Constants ──────────────────────────────────────────────────────────────────
EFFECT_TO_AXIS: dict[str, str] = {"dist": "drive", "delay": "space", "phaser": "phase"}

CSV_FIELDS = [
    "feature", "n_pairs",
    "mean_pct", "median_pct", "std_pct", "min_pct", "max_pct",
    "direction", "used_in", "suspect_cross_axis",
]


# ── Pair collection ────────────────────────────────────────────────────────────
def collect_pairs(
    groups: dict,
    effect: str,
    low: int,
    high: int,
    fix_others: str,
) -> list[tuple]:
    """Return (low_file, high_file) pairs where only `effect` intensity differs.

    fix_others="isolated" : effects_key must equal {effect} exactly.
    fix_others="any"      : other effects allowed, but same intensity in both files.
    Pair order is always (low_file, high_file) for direction consistency.
    """
    pairs: list[tuple] = []
    for key, files in groups.items():
        _play_type, _play_idx, effects_key = key

        if effect not in effects_key:
            continue
        if fix_others == "isolated" and effects_key != frozenset({effect}):
            continue

        low_files  = [f for f in files if f.effects.get(effect) == low]
        high_files = [f for f in files if f.effects.get(effect) == high]
        if not low_files or not high_files:
            continue

        other_effects = effects_key - {effect}
        for lf in low_files:
            for hf in high_files:
                if all(lf.effects.get(e) == hf.effects.get(e) for e in other_effects):
                    pairs.append((lf, hf))

    return pairs


# ── Feature aggregation ────────────────────────────────────────────────────────
def aggregate_features(
    pairs: list[tuple],
    model: ToneMatchModel,
) -> dict[str, list[float]]:
    """For each pair, extract features and accumulate % change per feature name."""
    feature_pcts: dict[str, list[float]] = defaultdict(list)
    n = len(pairs)

    iterator = (
        tqdm(enumerate(pairs, 1), total=n, desc="  분석", unit="pair")
        if HAS_TQDM
        else enumerate(pairs, 1)
    )

    for i, (lf, hf) in iterator:
        low_audio  = model._load_audio(lf.path)
        high_audio = model._load_audio(hf.path)
        length     = min(low_audio.size, high_audio.size)
        low_feats  = model._extract_features(low_audio[:length])
        high_feats = model._extract_features(high_audio[:length])

        for feat_name, low_val in low_feats.items():
            high_val = high_feats[feat_name]
            pct      = (high_val - low_val) / (abs(low_val) + EPSILON) * 100.0
            feature_pcts[feat_name].append(pct)

        if not HAS_TQDM and i % 20 == 0:
            print(f"  [{i}/{n}]")

    return dict(feature_pcts)


# ── Rule map ───────────────────────────────────────────────────────────────────
def get_rule_map(model: ToneMatchModel) -> dict[str, list[str]]:
    """feature_name → [axis, ...] from each axis's FeatureRule list."""
    rule_map: dict[str, list[str]] = {}
    for axis, rules in [
        ("drive", model._drive_rules()),
        ("space", model._space_rules()),
        ("phase", model._phase_rules()),
    ]:
        for rule in rules:
            rule_map.setdefault(rule.name, []).append(axis)
    return rule_map


# ── Stats ──────────────────────────────────────────────────────────────────────
def compute_stats(pcts: list[float]) -> dict[str, float]:
    arr = np.array(pcts, dtype=float)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


# ── Console output ─────────────────────────────────────────────────────────────
def print_table(
    rows: list[dict],
    effect: str,
    low: int,
    high: int,
    fix_others: str,
    n_pairs: int,
    audio_dir: Path,
) -> None:
    main_axis = EFFECT_TO_AXIS[effect]
    W = 82

    print(f"\n{'='*W}")
    print(f"  {effect} 강도 {low}→{high} 변화 시 feature 변동")
    print(f"  fix-others: {fix_others}   n_pairs: {n_pairs}")
    print(f"{'='*W}")

    hdr_feat = "feature"
    print(
        f"  {hdr_feat:<32}  {'mean_pct':>9}  {'median_pct':>10}  "
        f"{'std_pct':>8}  {'dir':>3}  used_in"
    )
    print(
        f"  {'-'*32}  {'-'*9}  {'-'*10}  "
        f"{'-'*8}  {'-'*3}  {'-'*24}"
    )

    suspects: list[dict] = []
    for row in rows:
        warn    = "  ⚠️" if row["suspect"] else ""
        used    = row["used_in"] if row["used_in"] else "—"
        print(
            f"  {row['feature']:<32}  "
            f"{row['mean']:>+8.1f}%  "
            f"{row['median']:>+9.1f}%  "
            f"{row['std']:>7.1f}%  "
            f"{row['direction']:>3}  "
            f"{used}{warn}"
        )
        if row["suspect"]:
            suspects.append(row)

    # ── Cross-axis contamination summary ──────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  cross-axis contamination 의심 feature")
    print(f"{'='*W}")
    if suspects:
        print(f"  다음 feature는 {effect} 강도 변화에 크게 반응하지만 다른 축 룰에서도 쓰임:\n")
        for row in suspects:
            other_axes = [a for a in row["used_in"].split(",") if a and a != main_axis]
            print(
                f"  {row['feature']:<32}  "
                f"|mean_pct|={abs(row['mean']):.1f}%   used in {row['used_in']}  "
                f"(non-main: {', '.join(other_axes)})"
            )
        print(f"""
  권장 조치:
    - 해당 feature를 다른 축 룰에서 제거하거나 가중치 낮추기
    - {effect}와 무관한 feature로 대체 후보 검토
""")
    else:
        print(
            f"  (|mean_pct| > 10% 이면서 다른 축 룰 포함) 조건을 만족하는 feature 없음.\n"
        )

    # ── Recommended next commands ──────────────────────────────────────────────
    d = str(audio_dir)
    print(f"{'─'*W}")
    print("  다음 명령들로 세 이펙터 모두 진단해보세요:")
    print(f"    python diagnose_features.py --audio-dir {d} --effect dist   --fix-others isolated")
    print(f"    python diagnose_features.py --audio-dir {d} --effect delay  --fix-others isolated")
    print(f"    python diagnose_features.py --audio-dir {d} --effect phaser --fix-others isolated")
    print(
        "  그 다음 --fix-others any로도 돌려보면 다른 이펙터가 같이 있을 때의 "
        "노이즈도 확인 가능합니다.\n"
    )


# ── CSV ────────────────────────────────────────────────────────────────────────
def write_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "feature":            row["feature"],
                "n_pairs":            row["n_pairs"],
                "mean_pct":           round(row["mean"],   2),
                "median_pct":         round(row["median"], 2),
                "std_pct":            round(row["std"],    2),
                "min_pct":            round(row["min"],    2),
                "max_pct":            round(row["max"],    2),
                "direction":          row["direction"],
                "used_in":            row["used_in"],
                "suspect_cross_axis": row["suspect"],
            })


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feature-level cross-axis contamination diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio-dir",   type=Path, required=True,
                        help="WAV 폴더 경로")
    parser.add_argument("--effect",      choices=["dist", "delay", "phaser"], required=True,
                        help="분석할 이펙터")
    parser.add_argument("--low",         type=int, default=25,
                        help="낮은 강도 (기본 25)")
    parser.add_argument("--high",        type=int, default=100,
                        help="높은 강도 (기본 100)")
    parser.add_argument("--fix-others",  choices=["isolated", "any"], default="any",
                        help="isolated=단독 파일만, any=다른 이펙터 허용(단 강도 동일)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="랜덤 시드 (현재 미사용 — 향후 샘플링용)")
    parser.add_argument("--output",      type=Path, default=None,
                        help="CSV 저장 경로 (기본: feature_diagnosis_{effect}_{low}to{high}_{fix_others}.csv)")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--hop-length",  type=int, default=256)
    args = parser.parse_args()

    if args.low >= args.high:
        print(
            f"[ERROR] --low ({args.low}) must be less than --high ({args.high})",
            file=sys.stderr,
        )
        return 1

    audio_dir = args.audio_dir.resolve()
    if not audio_dir.exists():
        print(f"[ERROR] 폴더 없음: {audio_dir}", file=sys.stderr)
        return 1

    output_path = args.output or Path(
        f"feature_diagnosis_{args.effect}_{args.low}to{args.high}_{args.fix_others}.csv"
    )

    # ── Discover & filter ──────────────────────────────────────────────────────
    print(f"\n  스캔 중: {audio_dir}")
    groups = discover_and_group(audio_dir)
    pairs  = collect_pairs(groups, args.effect, args.low, args.high, args.fix_others)

    print(f"  그룹 수: {len(groups):,}  대상 쌍: {len(pairs)}")
    if not pairs:
        print(
            "[ERROR] 조건에 맞는 쌍이 없습니다. "
            "--low, --high, --effect, --fix-others 조합을 확인하세요.",
            file=sys.stderr,
        )
        return 1

    # ── Extract & aggregate ────────────────────────────────────────────────────
    model     = ToneMatchModel(sample_rate=args.sample_rate, hop_length=args.hop_length)
    rule_map  = get_rule_map(model)
    main_axis = EFFECT_TO_AXIS[args.effect]

    t0           = time.perf_counter()
    feature_pcts = aggregate_features(pairs, model)
    elapsed_ms   = (time.perf_counter() - t0) * 1000
    print(f"  완료: {elapsed_ms:.0f} ms  ({elapsed_ms / len(pairs):.0f} ms/pair)")

    # ── Build rows ─────────────────────────────────────────────────────────────
    rows: list[dict] = []
    for feat_name, pcts in feature_pcts.items():
        stats    = compute_stats(pcts)
        mean_pct = stats["mean"]
        direction = "↑" if mean_pct > 5 else ("↓" if mean_pct < -5 else "→")

        axes    = rule_map.get(feat_name, [])
        used_in = ",".join(axes)

        non_main = [a for a in axes if a != main_axis]
        suspect  = abs(mean_pct) > 10.0 and bool(non_main)

        rows.append({
            "feature":   feat_name,
            "n_pairs":   len(pcts),
            "mean":      mean_pct,
            "median":    stats["median"],
            "std":       stats["std"],
            "min":       stats["min"],
            "max":       stats["max"],
            "direction": direction,
            "used_in":   used_in,
            "suspect":   suspect,
        })

    rows.sort(key=lambda r: abs(r["mean"]), reverse=True)

    # ── Output ─────────────────────────────────────────────────────────────────
    print_table(rows, args.effect, args.low, args.high, args.fix_others, len(pairs), audio_dir)

    write_csv(rows, output_path)
    print(f"  CSV 저장: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
