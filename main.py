"""
main.py
-------
이펙터 피드백 + 연주 피드백 통합 파이프라인 CLI

사용법:
  python main.py --ref path/to/reference.wav --copy path/to/copy.wav

선택 옵션:
  --active-effects  이펙터 목록 (예: dist delay). None이면 파일명 자동 감지.
  --sr              연주 분석용 샘플링 레이트 (기본 22050)
  --save            결과를 JSON으로 저장할 경로
"""

import argparse
import io
import json
import sys

from unified_pipeline import run_unified_feedback


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="기타 이펙터 피드백 + 연주 피드백 통합 분석기"
    )
    parser.add_argument("--ref",  required=True, help="기준 음원 경로 (.wav / .mp3)")
    parser.add_argument("--copy", required=True, help="비교 음원 경로 (.wav / .mp3)")
    parser.add_argument(
        "--active-effects", nargs="*", default=None,
        help="이펙터 목록 override (예: --active-effects dist delay). 생략 시 파일명 자동 감지."
    )
    parser.add_argument("--sr",   type=int, default=22050, help="샘플링 레이트 (기본: 22050)")
    parser.add_argument("--save", default=None, help="결과를 JSON으로 저장할 경로")

    args = parser.parse_args()

    result = run_unified_feedback(
        ref_path=args.ref,
        copy_path=args.copy,
        sr_playing=args.sr,
        active_effects=args.active_effects,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n[저장] {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
