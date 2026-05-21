#!/usr/bin/env python3
"""
Boundary Value Analysis (BVA) test runner for ToneMatchModel.

Usage:
    python bva_test.py
    python bva_test.py --wav-dir ./play --recording solo
    python bva_test.py --wav-dir ./play --recording chord --output results.json

WAV naming convention expected in --wav-dir:
    clean_{rec}_5.wav
    dist_{25|50|75|100}_{rec}_5.wav
    reverb_{25|50|75|100}_{rec}_5.wav
    chorus_{25|50|75|100}_{rec}_5.wav
  where {rec} = "solo" | "chord"
"""

from __future__ import annotations
import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Force UTF-8 output on Windows consoles (cp949 can't handle emoji/Korean arrows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from tone_match_model import ToneMatchModel  # noqa: E402

# ── Pass/Fail helpers ─────────────────────────────────────────────────────────
POSITIVE_ACTIONS = {"lower", "turn_off"}   # copy has MORE effect than ref
NEGATIVE_ACTIONS = {"raise", "turn_on"}    # copy has LESS effect than ref
NEUTRAL_ACTIONS  = {"keep"}
STATUS_MARK      = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}


# ── Test case data ─────────────────────────────────────────────────────────────
@dataclass
class BVACase:
    label: str         # "Drive 0→25"
    axis: str          # target axis: "drive" | "space" | "phase"
    ref_path: Path
    copy_path: Path
    expected_dir: str  # "+" = copy has MORE effect, "-" = copy has LESS effect
    note: str = ""     # e.g. approximation notes


def _nearest(intensity: int) -> int:
    """Snap intensity to nearest available level: 25 / 50 / 75 / 100."""
    return min([25, 50, 75, 100], key=lambda x: abs(x - intensity))


def build_test_cases(wav_dir: Path, rec: str) -> list[BVACase]:
    """Build predefined BVA scenarios, mapping intensity levels to file paths."""

    def wav(effect: str | None, intensity: int) -> Path:
        if intensity == 0 or effect is None:
            return wav_dir / f"clean_{rec}_5.wav"
        nearest = _nearest(intensity)
        return wav_dir / f"{effect}_{nearest}_{rec}_5.wav"

    return [
        # ── Drive (distortion) ───────────────────────────────────────────────
        # ref=clean → copy=dist_25: copy has MORE drive  → expected: lower/turn_off
        BVACase("Drive   0→25", "drive", wav(None,   0), wav("dist",  25), "+"),
        # ref=dist_25 → copy=dist_75: copy has MORE drive
        BVACase("Drive  25→75", "drive", wav("dist", 25), wav("dist", 75), "+"),
        # ref=dist_75 → copy=dist_25: copy has LESS drive → expected: raise/turn_on
        BVACase("Drive  75→25", "drive", wav("dist", 75), wav("dist", 25), "-"),
        # ref=dist_50 → copy=clean:   copy is clean       → expected: raise/turn_on
        BVACase("Drive  50→0",  "drive", wav("dist", 50), wav(None,   0),  "-"),

        # ── Space (reverb) ───────────────────────────────────────────────────
        # "30" mapped to reverb_25 (nearest available)
        BVACase("Space   0→30",  "space", wav(None,     0), wav("reverb",  25), "+", "30≈reverb_25"),
        BVACase("Space  30→100", "space", wav("reverb", 25), wav("reverb", 100), "+", "30≈reverb_25"),
        BVACase("Space 100→30",  "space", wav("reverb", 100), wav("reverb", 25), "-", "30≈reverb_25"),
        BVACase("Space  30→0",   "space", wav("reverb", 25), wav(None,      0),  "-", "30≈reverb_25"),

        # ── Phase (chorus) ───────────────────────────────────────────────────
        BVACase("Phase Clean→25",  "phase", wav(None,     0), wav("chorus", 25), "+"),
        BVACase("Phase   25→50",   "phase", wav("chorus", 25), wav("chorus", 50), "+"),
        BVACase("Phase   25→75",   "phase", wav("chorus", 25), wav("chorus", 75), "+"),
        BVACase("Phase  25→Clean", "phase", wav("chorus", 25), wav(None,      0),  "-"),
    ]


# ── Result ─────────────────────────────────────────────────────────────────────
@dataclass
class BVAResult:
    case: BVACase
    axis_result: dict       # target axis data from model
    all_axes: list[dict]    # all three axes
    status: str             # "PASS" | "FAIL" | "WARN"
    diagnosis: str


def _axis_by_name(axes: list[dict], name: str) -> dict:
    return next((a for a in axes if a["axis"] == name), {})


def run_case(model: ToneMatchModel, case: BVACase) -> BVAResult:
    result   = model.compare(case.ref_path, case.copy_path)
    all_axes = result["axes"]
    axis_r   = _axis_by_name(all_axes, case.axis)

    if not axis_r:
        return BVAResult(case, {}, all_axes, "FAIL",
                         f"Axis '{case.axis}' missing from model output")

    action = axis_r["action"]
    diff   = axis_r["difference"]

    expect_positive = case.expected_dir == "+"
    got_positive    = action in POSITIVE_ACTIONS
    got_negative    = action in NEGATIVE_ACTIONS
    got_neutral     = action in NEUTRAL_ACTIONS

    # ── Primary axis verdict ──────────────────────────────────────────────────
    if got_neutral:
        status = "FAIL"
        expected_label = "lower/turn_off" if expect_positive else "raise/turn_on"
        diagnosis = (
            f"'keep' (diff={diff:+d}) — {expected_label} 기대. "
            "물리적으로 변화량이 임계값 미만이거나 피처 억제 과잉 가능성."
        )
    elif (expect_positive and got_positive) or (not expect_positive and got_negative):
        status = "PASS"
        diagnosis = "방향 정확 ✓"
    else:
        status = "FAIL"
        expected_label = "lower/turn_off" if expect_positive else "raise/turn_on"
        diagnosis = (
            f"역방향 오판! expected={expected_label}, got='{action}' (diff={diff:+d}). "
            "교차검증 게이팅 또는 피처 방향 로직 확인 필요."
        )

    # ── Cross-axis contamination check ────────────────────────────────────────
    spurious = [
        a for a in all_axes
        if a["axis"] != case.axis and a["action"] not in NEUTRAL_ACTIONS
    ]
    if spurious and status == "PASS":
        spill = ", ".join(
            f"{a['axis']}={a['action']}(diff={a['difference']:+d})" for a in spurious
        )
        status = "WARN"
        diagnosis += f" | 교차축 오염: {spill}"

    return BVAResult(case, axis_r, all_axes, status, diagnosis)


# ── Table printer ──────────────────────────────────────────────────────────────
_COL = dict(label=18, axis=5, diff=5, action=9, drv=5, spc=5, phs=5, status=3)
_SEP = "+" + "+".join("-" * (w + 2) for w in _COL.values()) + "+"
_SEP_THICK = _SEP.replace("-", "=")


def _cell(text, width: int) -> str:
    return f" {str(text)[:width].ljust(width)} "


def _row(*cells) -> str:
    parts = [_cell(c, w) for c, w in zip(cells, _COL.values())]
    return "|" + "|".join(parts) + "|"


def print_table(results: list[BVAResult]) -> None:
    total_width = sum(_COL.values()) + len(_COL) * 3 + 1
    print()
    print("━" * total_width)
    print("  BOUNDARY VALUE ANALYSIS — ToneMatchModel")
    print("━" * total_width)
    print(_SEP)
    print(_row("Test Case", "Axis", "Diff", "Action", "DRV", "SPC", "PHS", "OK"))
    print(_SEP_THICK)

    pass_n = fail_n = warn_n = 0
    for r in results:
        ax   = r.axis_result
        diff_str   = f"{ax.get('difference', 0):+d}"
        action_str = ax.get("action", "N/A")
        mark       = STATUS_MARK.get(r.status, "?")

        # per-axis diffs for cross-contamination overview
        def _d(name: str) -> str:
            a = _axis_by_name(r.all_axes, name)
            return f"{a.get('difference', 0):+d}" if a else "N/A"

        print(_row(r.case.label, r.case.axis, diff_str, action_str,
                   _d("drive"), _d("space"), _d("phase"), mark))

        if r.status == "PASS":   pass_n += 1
        elif r.status == "FAIL": fail_n += 1
        elif r.status == "WARN": warn_n += 1

    print(_SEP)
    print(f"\n  총 {len(results)}건 ── ✅ PASS {pass_n}  ⚠️  WARN {warn_n}  ❌ FAIL {fail_n}")
    print()


def print_diagnosis(results: list[BVAResult]) -> None:
    failed = [r for r in results if r.status in ("FAIL", "WARN")]
    if not failed:
        print("  모든 테스트 통과 — 진단 항목 없음.\n")
        return

    total_width = 72
    print("━" * total_width)
    print("  실패 / 경고 상세 진단")
    print("━" * total_width)

    for r in failed:
        ax = r.axis_result
        print(f"\n[{r.status}] {r.case.label}  (ref={r.case.ref_path.name}  copy={r.case.copy_path.name})")
        print(f"  판정 : {r.diagnosis}")
        print(f"  메시지: {ax.get('message', '—')}")

        features = ax.get("features") or []
        if features:
            print("  주요 피처 변화 (상위 5개, |%변화량| 내림차순):")
            print(f"    {'피처명':35s}  {'ref값':>10}  {'copy값':>10}  {'변화율':>8}  {'예상방향':8}  판정")
            print(f"    {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  ----")
            for f in features:
                arrow   = "↑" if f["delta"] > 0 else "↓"
                exp_dir = f["expected_direction"]
                ok      = "✓" if (arrow == "↑") == (exp_dir == "increase") else "✗"
                print(
                    f"    {ok} {f['feature']:33s}  "
                    f"{f['reference_value']:>10.5f}  {f['copy_value']:>10.5f}  "
                    f"{f['percent_change']:>+7.1f}%  {exp_dir:8s}  {arrow}"
                )

        # Show all-axis overview for this case
        print("  전체 축 결과:")
        for a in r.all_axes:
            tag = " ◀ 대상" if a["axis"] == r.case.axis else ""
            print(f"    {a['axis']:6s}  diff={a['difference']:+4d}  action={a['action']:9s}  {a['message']}{tag}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="BVA test runner for ToneMatchModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--wav-dir",     type=Path, default=Path("play"),
                        help="WAV 파일 디렉토리 (기본값: ./play)")
    parser.add_argument("--recording",   choices=["solo", "chord"], default="solo",
                        help="녹음 유형 접미사 (기본값: solo)")
    parser.add_argument("--output",      type=Path, default=None,
                        help="JSON 결과 저장 경로")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--hop-length",  type=int, default=256)
    args = parser.parse_args()

    wav_dir = args.wav_dir.resolve()
    if not wav_dir.exists():
        print(f"[ERROR] WAV 디렉토리 없음: {wav_dir}", file=sys.stderr)
        return 1

    test_cases = build_test_cases(wav_dir, args.recording)

    missing = sorted({
        str(p)
        for tc in test_cases
        for p in (tc.ref_path, tc.copy_path)
        if not p.exists()
    })
    if missing:
        print("[ERROR] 다음 WAV 파일을 찾을 수 없습니다:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    print(f"\n  녹음 유형 : {args.recording}")
    print(f"  WAV 경로  : {wav_dir}")
    print(f"  테스트 수 : {len(test_cases)}")

    model   = ToneMatchModel(sample_rate=args.sample_rate, hop_length=args.hop_length)
    results: list[BVAResult] = []

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i:02d}/{len(test_cases)}] {tc.label} ...", end="\r", flush=True)
        results.append(run_case(model, tc))

    print(" " * 60, end="\r")

    print_table(results)
    print_diagnosis(results)

    if args.output:
        out_data = {
            "wav_dir":    str(wav_dir),
            "recording":  args.recording,
            "summary": {
                "total": len(results),
                "pass":  sum(1 for r in results if r.status == "PASS"),
                "warn":  sum(1 for r in results if r.status == "WARN"),
                "fail":  sum(1 for r in results if r.status == "FAIL"),
            },
            "cases": [
                {
                    "label":        r.case.label,
                    "axis":         r.case.axis,
                    "expected_dir": r.case.expected_dir,
                    "status":       r.status,
                    "diagnosis":    r.diagnosis,
                    "ref_path":     str(r.case.ref_path),
                    "copy_path":    str(r.case.copy_path),
                    "note":         r.case.note,
                    "target_axis":  r.axis_result,
                    "all_axes":     r.all_axes,
                }
                for r in results
            ],
        }
        args.output.write_text(
            json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  JSON 저장 완료: {args.output.resolve()}")

    return 0 if all(r.status in ("PASS", "WARN") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
