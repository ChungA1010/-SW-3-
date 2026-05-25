#!/usr/bin/env python3
"""
Extended tone test runner for ToneMatchModel.
Tests all effect intensity combinations exhaustively.

Usage:
    python extended_tone_test.py
    python extended_tone_test.py --wav-dir ./play --recording solo
    python extended_tone_test.py --wav-dir ./play --recording chord --output results.json
    python extended_tone_test.py --effects dist reverb --nums 2 5

WAV naming convention expected in --wav-dir:
    clean_{rec}_{num}.wav                  (intensity = 0)
    {effect}_{intensity}_{rec}_{num}.wav   (intensity = 25|50|75|100)
  where {rec} = "solo" | "chord",  {num} = 2 | 5 | 6
"""

from __future__ import annotations
import argparse
import io
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from tone_match_model import ToneMatchModel  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
POSITIVE_ACTIONS = {"lower", "much_lower"}
NEGATIVE_ACTIONS = {"raise", "much_raise"}
NEUTRAL_ACTIONS  = {"keep"}
STATUS_MARK      = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}

EFFECT_AXIS: dict[str, str] = {
    "dist":   "drive",
    "reverb": "space",
    "chorus": "phase",
}

INTENSITY_PAIRS: list[tuple[int, int]] = [
    (0, 25),  (0, 50),  (0, 75),  (0, 100),
    (25, 0),  (25, 50), (25, 75), (25, 100),
    (50, 0),  (50, 25), (50, 75), (50, 100),
    (75, 0),  (75, 25), (75, 50), (75, 100),
    (100, 0), (100, 25),(100, 50),(100, 75),
]

ALL_EFFECTS: list[str] = ["dist", "reverb", "chorus"]
ALL_NUMS:    list[int]  = [2, 5, 6]


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class TestCase:
    label: str
    effect: str
    axis: str
    num: int
    ref_intensity: int
    copy_intensity: int
    ref_path: Path
    copy_path: Path
    expected_dir: str   # "+" = copy has MORE effect,  "-" = copy has LESS


@dataclass
class TestResult:
    case: TestCase
    axis_result: dict
    all_axes: list[dict]
    status: str         # "PASS" | "FAIL" | "WARN"
    diagnosis: str


# ── File path helper ──────────────────────────────────────────────────────────
def _wav_path(wav_dir: Path, effect: str, intensity: int, rec: str, num: int) -> Path:
    if intensity == 0:
        return wav_dir / f"clean_{rec}_{num}.wav"
    return wav_dir / f"{effect}_{intensity}_{rec}_{num}.wav"


# ── Test case builder ─────────────────────────────────────────────────────────
def build_test_cases(
    wav_dir: Path,
    rec: str,
    effects: list[str],
    nums: list[int],
) -> list[TestCase]:
    cases: list[TestCase] = []
    for effect, num, (ref_i, copy_i) in product(effects, nums, INTENSITY_PAIRS):
        axis = EFFECT_AXIS[effect]
        cases.append(TestCase(
            label=f"{effect.upper():6s} #{num} {ref_i:3d}→{copy_i:3d}",
            effect=effect,
            axis=axis,
            num=num,
            ref_intensity=ref_i,
            copy_intensity=copy_i,
            ref_path=_wav_path(wav_dir, effect, ref_i,  rec, num),
            copy_path=_wav_path(wav_dir, effect, copy_i, rec, num),
            expected_dir="+" if copy_i > ref_i else "-",
        ))
    return cases


# ── File existence check ──────────────────────────────────────────────────────
def filter_available(cases: list[TestCase]) -> tuple[list[TestCase], list[TestCase]]:
    """Split into (runnable, skipped). Skipped = either file missing."""
    runnable, skipped = [], []
    for tc in cases:
        if tc.ref_path.exists() and tc.copy_path.exists():
            runnable.append(tc)
        else:
            skipped.append(tc)
    return runnable, skipped


# ── Axis lookup ───────────────────────────────────────────────────────────────
def _axis_by_name(axes: list[dict], name: str) -> dict:
    return next((a for a in axes if a["axis"] == name), {})


# ── Run single test case ──────────────────────────────────────────────────────
def run_case(model: ToneMatchModel, case: TestCase) -> TestResult:
    # 정답 이펙터를 단일 원소 리스트로 모델에 주입.
    # 해당 축만 채점되며, 나머지 축은 즉시 difference=0 반환 → 교차 오염 원천 차단.
    active_effects = [case.effect]
    result   = model.compare(case.ref_path, case.copy_path, active_effects=active_effects)
    all_axes = result["axes"]
    axis_r   = _axis_by_name(all_axes, case.axis)

    if not axis_r:
        return TestResult(case, {}, all_axes, "FAIL",
                          f"Axis '{case.axis}' missing from model output")

    action = axis_r["action"]
    diff   = axis_r["difference"]
    expect_positive = case.expected_dir == "+"

    # ── Primary verdict ───────────────────────────────────────────────────────
    if action in NEUTRAL_ACTIONS:
        status = "FAIL"
        expected_label = "lower/turn_off" if expect_positive else "raise/turn_on"
        diagnosis = (
            f"'keep' (diff={diff:+d}) — {expected_label} 기대. "
            "변화량이 임계값 미만이거나 피처 방향 확인 필요."
        )
    elif (expect_positive and action in POSITIVE_ACTIONS) or \
         (not expect_positive and action in NEGATIVE_ACTIONS):
        status = "PASS"
        diagnosis = "방향 정확 ✓"
    else:
        status = "FAIL"
        expected_label = "lower/turn_off" if expect_positive else "raise/turn_on"
        diagnosis = (
            f"역방향 오판! expected={expected_label}, got='{action}' (diff={diff:+d}). "
            "피처 방향 로직 확인 필요."
        )

    # ── Cross-axis contamination check (classifier 모드에서는 항상 0이나 호환성 유지) ──
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

    return TestResult(case, axis_r, all_axes, status, diagnosis)


# ── Table layout ──────────────────────────────────────────────────────────────
_COL       = dict(label=22, axis=6, diff=5, action=9, drv=5, spc=5, phs=5, ok=3)
_SEP       = "+" + "+".join("-" * (w + 2) for w in _COL.values()) + "+"
_SEP_THICK = _SEP.replace("-", "=")
_WIDTH     = sum(_COL.values()) + len(_COL) * 3 + 1


def _cell(text: object, width: int) -> str:
    return f" {str(text)[:width].ljust(width)} "


def _row(*cells: object) -> str:
    return "|" + "|".join(_cell(c, w) for c, w in zip(cells, _COL.values())) + "|"


# ── Per-group table ───────────────────────────────────────────────────────────
def print_group_table(results: list[TestResult], group_label: str) -> tuple[int, int, int]:
    """Print the result table for one (effect, num) group. Returns (pass, warn, fail)."""
    print(f"\n  ── {group_label} ──")
    print(_SEP)
    print(_row("Test Case", "Axis", "Diff", "Action", "DRV", "SPC", "PHS", "OK"))
    print(_SEP_THICK)

    pass_n = fail_n = warn_n = 0
    for r in results:
        ax = r.axis_result

        def _d(name: str) -> str:
            a = _axis_by_name(r.all_axes, name)
            return f"{a.get('difference', 0):+d}" if a else "N/A"

        print(_row(
            r.case.label,
            r.case.axis,
            f"{ax.get('difference', 0):+d}",
            ax.get("action", "N/A"),
            _d("drive"), _d("space"), _d("phase"),
            STATUS_MARK.get(r.status, "?"),
        ))

        if r.status == "PASS":   pass_n += 1
        elif r.status == "FAIL": fail_n += 1
        elif r.status == "WARN": warn_n += 1

    print(_SEP)
    print(f"  소계: ✅ PASS {pass_n}  ⚠️  WARN {warn_n}  ❌ FAIL {fail_n}")
    return pass_n, warn_n, fail_n


# ── Overall summary ───────────────────────────────────────────────────────────
def print_summary(results: list[TestResult], skipped: list[TestCase], effects: list[str]) -> None:
    pass_n = sum(1 for r in results if r.status == "PASS")
    warn_n = sum(1 for r in results if r.status == "WARN")
    fail_n = sum(1 for r in results if r.status == "FAIL")

    print()
    print("━" * _WIDTH)
    print("  EXTENDED TONE TEST — 전체 요약")
    print("━" * _WIDTH)
    print(f"  실행: {len(results)}건  ──  ✅ PASS {pass_n}  ⚠️  WARN {warn_n}  ❌ FAIL {fail_n}")
    if skipped:
        print(f"  건너뜀: {len(skipped)}건 (파일 없음)")
    print()

    for effect in effects:
        eff = [r for r in results if r.case.effect == effect]
        if not eff:
            continue
        ep = sum(1 for r in eff if r.status == "PASS")
        ew = sum(1 for r in eff if r.status == "WARN")
        ef = sum(1 for r in eff if r.status == "FAIL")
        axis = EFFECT_AXIS[effect]
        bar_total = len(eff)
        bar_pass  = int(ep / bar_total * 20) if bar_total else 0
        bar_warn  = int(ew / bar_total * 20) if bar_total else 0
        bar_fail  = 20 - bar_pass - bar_warn
        bar = "█" * bar_pass + "▒" * bar_warn + "░" * bar_fail
        print(f"  {effect.upper():6s} [{axis:5s}]  {bar}  ✅{ep:3d}  ⚠️ {ew:3d}  ❌{ef:3d}  / {bar_total}")

    print()


# ── Failure / warning diagnosis ───────────────────────────────────────────────
def print_diagnosis(results: list[TestResult]) -> None:
    failed = [r for r in results if r.status in ("FAIL", "WARN")]
    if not failed:
        print("  모든 테스트 통과 — 진단 항목 없음.\n")
        return

    print("━" * 72)
    print("  실패 / 경고 상세 진단")
    print("━" * 72)

    for r in failed:
        ax = r.axis_result
        print(f"\n[{r.status}] {r.case.label}"
              f"  (ref={r.case.ref_path.name}  copy={r.case.copy_path.name})")
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

        print("  전체 축 결과:")
        for a in r.all_axes:
            tag = " ◀ 대상" if a["axis"] == r.case.axis else ""
            print(f"    {a['axis']:6s}  diff={a['difference']:+4d}  "
                  f"action={a['action']:9s}  {a['message']}{tag}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extended tone test runner for ToneMatchModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--wav-dir",      type=Path, default=Path("play"),
                        help="WAV 파일 디렉토리 (기본값: ./play)")
    parser.add_argument("--recording",    choices=["solo", "chord"], default="solo",
                        help="녹음 유형 (기본값: solo)")
    parser.add_argument("--effects",      nargs="+", choices=list(EFFECT_AXIS),
                        default=ALL_EFFECTS,
                        help="테스트할 이펙터 (기본값: dist reverb chorus)")
    parser.add_argument("--nums",         nargs="+", type=int, choices=ALL_NUMS,
                        default=ALL_NUMS,
                        help="테스트할 파일 번호 (기본값: 2 5 6)")
    parser.add_argument("--output",       type=Path, default=None,
                        help="JSON 결과 저장 경로")
    parser.add_argument("--sample-rate",  type=int, default=32000)
    parser.add_argument("--hop-length",   type=int, default=256)
    parser.add_argument("--no-diagnosis", action="store_true",
                        help="상세 진단 출력 생략")
    args = parser.parse_args()

    wav_dir = args.wav_dir.resolve()
    if not wav_dir.exists():
        print(f"[ERROR] WAV 디렉토리 없음: {wav_dir}", file=sys.stderr)
        return 1

    all_cases = build_test_cases(wav_dir, args.recording, args.effects, args.nums)
    runnable, skipped = filter_available(all_cases)

    if skipped:
        missing_files = sorted({
            str(p)
            for tc in skipped
            for p in (tc.ref_path, tc.copy_path)
            if not p.exists()
        })
        print(f"  [SKIP] 파일 없음으로 {len(skipped)}건 건너뜀 ({len(missing_files)}개 파일 부재)")
        for m in missing_files:
            print(f"    {m}")

    if not runnable:
        print("[ERROR] 실행 가능한 테스트 케이스가 없습니다.", file=sys.stderr)
        return 1

    print()
    print("━" * _WIDTH)
    print("  EXTENDED TONE TEST — ToneMatchModel 전수 검사  [정답 주입 모드]")
    print("━" * _WIDTH)
    print(f"  녹음 유형 : {args.recording}")
    print(f"  WAV 경로  : {wav_dir}")
    print(f"  이펙터    : {', '.join(args.effects)}")
    print(f"  번호      : {', '.join(str(n) for n in args.nums)}")
    print(f"  채점 방식 : active_effects=[case.effect] 자동 주입 — 정답 축만 채점")
    print(f"  전체: {len(all_cases)}건  (실행: {len(runnable)}건  건너뜀: {len(skipped)}건)")

    model   = ToneMatchModel(sample_rate=args.sample_rate, hop_length=args.hop_length)
    results: list[TestResult] = []

    for i, tc in enumerate(runnable, 1):
        print(f"  [{i:03d}/{len(runnable)}] {tc.label} ...", end="\r", flush=True)
        results.append(run_case(model, tc))

    print(" " * 70, end="\r")

    # ── Per-group tables ──────────────────────────────────────────────────────
    for effect in args.effects:
        for num in args.nums:
            group = [r for r in results if r.case.effect == effect and r.case.num == num]
            if group:
                print_group_table(group, f"{effect.upper()} #{num}  [{EFFECT_AXIS[effect]}]")

    # ── Overall summary ───────────────────────────────────────────────────────
    print_summary(results, skipped, args.effects)

    # ── Detailed diagnosis ────────────────────────────────────────────────────
    if not args.no_diagnosis:
        print_diagnosis(results)

    # ── JSON export ───────────────────────────────────────────────────────────
    if args.output:
        out_data = {
            "wav_dir":   str(wav_dir),
            "recording": args.recording,
            "effects":   args.effects,
            "nums":      args.nums,
            "summary": {
                "total": len(results),
                "pass":  sum(1 for r in results if r.status == "PASS"),
                "warn":  sum(1 for r in results if r.status == "WARN"),
                "fail":  sum(1 for r in results if r.status == "FAIL"),
            },
            "per_effect": {
                effect: {
                    "total": len(g := [r for r in results if r.case.effect == effect]),
                    "pass":  sum(1 for r in g if r.status == "PASS"),
                    "warn":  sum(1 for r in g if r.status == "WARN"),
                    "fail":  sum(1 for r in g if r.status == "FAIL"),
                }
                for effect in args.effects
            },
            "cases": [
                {
                    "label":          r.case.label,
                    "effect":         r.case.effect,
                    "axis":           r.case.axis,
                    "num":            r.case.num,
                    "ref_intensity":  r.case.ref_intensity,
                    "copy_intensity": r.case.copy_intensity,
                    "expected_dir":   r.case.expected_dir,
                    "status":         r.status,
                    "diagnosis":      r.diagnosis,
                    "ref_path":       str(r.case.ref_path),
                    "copy_path":      str(r.case.copy_path),
                    "target_axis":    r.axis_result,
                    "all_axes":       r.all_axes,
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
