#!/usr/bin/env python3
"""Run folder inference with the BEATs effect + intensity model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_ARTIFACT_PATH, EFFECT_CLASS_NAMES, INTENSITY_LEVEL_TO_DISPLAY  # noqa: E402
from shared.audio import ensure_directory  # noqa: E402
from shared.torch_training import choose_device  # noqa: E402
from utils import build_model, predict_audio_file  # noqa: E402

SUPPORTED_EXTENSIONS = {".wav", ".wave"}
FILE_PATTERN = re.compile(r"^(?P<effect>clean|dist|chorus|phaser|delay|reverb)_(?P<intensity>25|50|75|100)_.+$", re.IGNORECASE)


def iter_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [
        path
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def parse_targets(path: Path) -> tuple[str | None, int | None]:
    stem = path.stem
    if stem.lower().startswith("clean"):
        return "clean", 0
    match = FILE_PATTERN.match(stem)
    if match is None:
        return None, None
    effect = match.group("effect").lower()
    intensity_display = match.group("intensity")
    inverse_map = {value: key for key, value in INTENSITY_LEVEL_TO_DISPLAY.items()}
    return effect, inverse_map[intensity_display]


def build_summary(rows: list[dict]) -> list[str]:
    total_files = len(rows)
    effect_evaluable = [row for row in rows if row["true_effect_label"] is not None]
    intensity_evaluable = [row for row in rows if row["true_intensity_level"] is not None]
    effect_correct = sum(int(row["effect_correct"]) for row in effect_evaluable)
    intensity_correct = sum(int(row["intensity_correct"]) for row in intensity_evaluable)
    joint_correct = sum(int(row["joint_correct"]) for row in intensity_evaluable)

    lines = [
        "결과 요약",
        "",
        f"전체 파일 수: {total_files}",
        f"이펙터 정답 판정 가능 파일 수: {len(effect_evaluable)}",
        f"세기 정답 판정 가능 파일 수: {len(intensity_evaluable)}",
        f"이펙터 정답 수: {effect_correct}",
        f"세기 정답 수: {intensity_correct}",
        f"이펙터 정확도: {(effect_correct / len(effect_evaluable) * 100):.2f}%" if effect_evaluable else "이펙터 정확도: N/A",
        f"세기 정확도: {(intensity_correct / len(intensity_evaluable) * 100):.2f}%" if intensity_evaluable else "세기 정확도: N/A",
        f"종류+세기 동시 정답 수: {joint_correct}",
        f"종류+세기 동시 정확도: {(joint_correct / len(intensity_evaluable) * 100):.2f}%" if intensity_evaluable else "종류+세기 동시 정확도: N/A",
        "",
        "이펙터별 정확도",
    ]

    for effect_label in EFFECT_CLASS_NAMES:
        subset = [row for row in effect_evaluable if row["true_effect_label"] == effect_label]
        correct = sum(int(row["effect_correct"]) for row in subset)
        total = len(subset)
        accuracy = (correct / total * 100) if total else 0.0
        lines.append(f"{effect_label}: {correct} / {total} = {accuracy:.2f}%")

    lines.append("")
    lines.append("세기별 정확도")
    for intensity_level, display in INTENSITY_LEVEL_TO_DISPLAY.items():
        subset = [row for row in intensity_evaluable if row["true_intensity_level"] == intensity_level]
        correct = sum(int(row["intensity_correct"]) for row in subset)
        total = len(subset)
        accuracy = (correct / total * 100) if total else 0.0
        lines.append(f"{display}: {correct} / {total} = {accuracy:.2f}%")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BEATs effect + intensity inference on a folder of WAV files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing WAV files")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH), help="Path to a saved checkpoint")
    parser.add_argument("--recursive", action="store_true", help="Search for WAV files recursively")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for the CSV")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    device = choose_device()
    checkpoint = torch.load(args.artifact_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = build_model(
        pretrained_checkpoint_path=None,
        backbone_config_override=model_config.get("beats_backbone_config"),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model weights.")
    model.load_state_dict(state_dict)

    config = checkpoint.get("training_config", {})
    sample_rate = int(config.get("sample_rate", 16000))
    segment_seconds = float(config.get("segment_seconds", 3.0))

    audio_files = iter_audio_files(input_dir, recursive=args.recursive)
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found under {input_dir}")

    output_dir = ensure_directory(args.output_dir) if args.output_dir else ensure_directory(CURRENT_DIR / "outputs")
    rows: list[dict] = []
    for path in audio_files:
        prediction = predict_audio_file(
            model=model,
            audio_path=path,
            device=device,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
        )
        true_effect_label, true_intensity_level = parse_targets(path)
        true_intensity_display = (
            INTENSITY_LEVEL_TO_DISPLAY[true_intensity_level]
            if true_intensity_level is not None
            else None
        )
        effect_correct = (
            true_effect_label == prediction["predicted_effect_label"]
            if true_effect_label is not None
            else None
        )
        intensity_correct = (
            true_intensity_level == prediction["predicted_intensity_level"]
            if true_intensity_level is not None
            else None
        )
        joint_correct = (
            bool(effect_correct and intensity_correct)
            if true_intensity_level is not None and true_effect_label is not None
            else None
        )
        row = {
            "file_name": path.name,
            "file_path": str(path),
            "true_effect_label": true_effect_label,
            "predicted_effect_label": prediction["predicted_effect_label"],
            "predicted_effect_confidence": prediction["predicted_effect_confidence"],
            "true_intensity_level": true_intensity_level,
            "predicted_intensity_level": prediction["predicted_intensity_level"],
            "true_intensity_display": true_intensity_display,
            "predicted_intensity_display": prediction["predicted_intensity_display"],
            "predicted_intensity_confidence": prediction["predicted_intensity_confidence"],
            "effect_correct": effect_correct,
            "intensity_correct": intensity_correct,
            "joint_correct": joint_correct,
            **{f"effect_score_{key}": value for key, value in prediction["effect_scores"].items()},
            **{f"intensity_score_{key}": value for key, value in prediction["intensity_scores"].items()},
        }
        rows.append(row)
        print(
            f"{path.name}: effect={prediction['predicted_effect_label']} ({prediction['predicted_effect_confidence']:.4f}), "
            f"intensity={prediction['predicted_intensity_display']} ({prediction['predicted_intensity_confidence']:.4f})",
            flush=True,
        )

    csv_path = Path(output_dir) / "single_folder_inference.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV to {csv_path}", flush=True)
    print("\n".join(build_summary(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
