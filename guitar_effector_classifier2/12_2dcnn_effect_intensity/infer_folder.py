#!/usr/bin/env python3
"""Run folder-level effect + intensity inference and print a summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_OUTPUT_DIR,
    EFFECT_CLASS_NAMES,
    INTENSITY_LEVEL_TO_DISPLAY,
)
from model import EffectIntensityCNN  # noqa: E402
from utils import predict_audio_file  # noqa: E402

EXPECTED_PATTERN = re.compile(
    r"^(?P<effect>clean|dist|chorus|phaser|delay|reverb)(?:_(?P<intensity>25|50|75|100))?.*$",
    re.IGNORECASE,
)


def infer_truth_from_name(file_name: str) -> tuple[str | None, int | None]:
    """Infer effect/intensity labels from common single_effect file naming."""
    stem = Path(file_name).stem.lower()
    match = EXPECTED_PATTERN.match(stem)
    if match is None:
        return None, None

    effect = match.group("effect")
    if effect not in EFFECT_CLASS_NAMES:
        return None, None
    if effect == "clean":
        return "clean", 0

    intensity_text = match.group("intensity")
    if intensity_text is None:
        return effect, None

    return effect, {"25": 1, "50": 2, "75": 3, "100": 4}[intensity_text]


def build_summary(rows: list[dict]) -> list[str]:
    """Build a concise text summary for effect/intensity predictions."""
    total_files = len(rows)
    effect_valid = [row for row in rows if row["expected_effect_label"]]
    intensity_valid = [row for row in rows if row["expected_intensity_level"] is not None]
    joint_valid = [
        row
        for row in rows
        if row["expected_effect_label"] and row["expected_intensity_level"] is not None
    ]

    effect_correct = sum(
        1
        for row in effect_valid
        if row["expected_effect_label"] == row["predicted_effect_label"]
    )
    intensity_correct = sum(
        1
        for row in intensity_valid
        if row["expected_intensity_level"] == row["predicted_intensity_level"]
    )
    joint_correct = sum(
        1
        for row in joint_valid
        if row["expected_effect_label"] == row["predicted_effect_label"]
        and row["expected_intensity_level"] == row["predicted_intensity_level"]
    )

    lines = [
        "결과 요약",
        "",
        f"전체 파일 수: {total_files}",
        f"이펙터 정답 판정 가능 파일 수: {len(effect_valid)}",
        f"세기 정답 판정 가능 파일 수: {len(intensity_valid)}",
        f"이펙터 정답 수: {effect_correct}",
        f"세기 정답 수: {intensity_correct}",
        f"이펙터 정확도: {(effect_correct / len(effect_valid) * 100):.2f}%" if effect_valid else "이펙터 정확도: N/A",
        f"세기 정확도: {(intensity_correct / len(intensity_valid) * 100):.2f}%" if intensity_valid else "세기 정확도: N/A",
        f"종류+세기 동시 정답 수: {joint_correct}",
        f"종류+세기 동시 정확도: {(joint_correct / len(joint_valid) * 100):.2f}%" if joint_valid else "종류+세기 동시 정확도: N/A",
        "",
        "이펙터별 정확도",
    ]

    for effect_label in EFFECT_CLASS_NAMES:
        subset = [row for row in effect_valid if row["expected_effect_label"] == effect_label]
        if not subset:
            continue
        correct = sum(
            1 for row in subset if row["predicted_effect_label"] == row["expected_effect_label"]
        )
        lines.append(f"{effect_label}: {correct} / {len(subset)} = {(correct / len(subset) * 100):.2f}%")

    lines.extend(["", "세기별 정확도"])
    for intensity_level, display in INTENSITY_LEVEL_TO_DISPLAY.items():
        subset = [row for row in intensity_valid if row["expected_intensity_level"] == intensity_level]
        if not subset:
            continue
        correct = sum(
            1 for row in subset if row["predicted_intensity_level"] == row["expected_intensity_level"]
        )
        lines.append(f"{display}: {correct} / {len(subset)} = {(correct / len(subset) * 100):.2f}%")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer effect + intensity for every wav file in a folder.")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH), help="Saved model checkpoint")
    parser.add_argument("--input-dir", required=True, help="Folder containing wav files")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories")
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_DIR / "single_folder_inference.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    artifact = torch.load(args.artifact_path, map_location="cpu")
    model = EffectIntensityCNN(
        num_effect_classes=artifact["model_config"]["num_effect_classes"],
        num_intensity_classes=artifact["model_config"]["num_intensity_classes"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    cfg = artifact["training_config"]
    input_dir = Path(args.input_dir).expanduser().resolve()
    wav_paths = sorted(input_dir.rglob("*.wav") if args.recursive else input_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No wav files were found in {input_dir}")

    rows: list[dict] = []
    for wav_path in wav_paths:
        prediction = predict_audio_file(
            model=model,
            audio_path=wav_path,
            device=torch.device("cpu"),
            sample_rate=cfg["sample_rate"],
            segment_seconds=cfg["segment_seconds"],
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
        )
        expected_effect_label, expected_intensity_level = infer_truth_from_name(wav_path.name)
        row = {
            "file_name": wav_path.name,
            "file_path": str(wav_path),
            "expected_effect_label": expected_effect_label or "",
            "predicted_effect_label": prediction["predicted_effect_label"],
            "expected_intensity_level": expected_intensity_level,
            "predicted_intensity_level": prediction["predicted_intensity_level"],
            "expected_intensity_display": (
                INTENSITY_LEVEL_TO_DISPLAY[expected_intensity_level]
                if expected_intensity_level is not None
                else ""
            ),
            "predicted_intensity_display": prediction["predicted_intensity_display"],
            "predicted_effect_confidence": prediction["predicted_effect_confidence"],
            "predicted_intensity_confidence": prediction["predicted_intensity_confidence"],
        }
        row.update({f"effect_score_{key}": value for key, value in prediction["effect_scores"].items()})
        row.update({f"intensity_score_{key}": value for key, value in prediction["intensity_scores"].items()})
        rows.append(row)
        print(
            f"{wav_path.name}: effect={prediction['predicted_effect_label']} ({prediction['predicted_effect_confidence']:.4f}), "
            f"intensity={prediction['predicted_intensity_display']} ({prediction['predicted_intensity_confidence']:.4f})",
            flush=True,
        )

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n".join(build_summary(rows)), flush=True)
    print(f"[INFO] Saved CSV to {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
