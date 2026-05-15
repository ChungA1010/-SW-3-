#!/usr/bin/env python3
"""Run folder inference with three independent BEATs axis models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AXIS_NAMES  # noqa: E402
from shared.audio import ensure_directory  # noqa: E402
from shared.torch_training import choose_device  # noqa: E402
from utils import build_model, predict_axis_file  # noqa: E402

SUPPORTED_EXTENSIONS = {".wav", ".wave"}


def iter_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return [path for path in sorted(iterator) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]


def load_axis_model(axis_name: str, artifact_dir: Path, device: torch.device):
    artifact_path = artifact_dir / f"{axis_name}_best_model.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing {axis_name} checkpoint: {artifact_path}")
    checkpoint = torch.load(artifact_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = build_model(
        axis_name,
        pretrained_checkpoint_path=None,
        backbone_config_override=model_config.get("beats_backbone_config"),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError(f"Checkpoint does not contain model weights: {artifact_path}")
    model.load_state_dict(state_dict)
    return model, checkpoint.get("training_config", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run folder inference with three independent BEATs axis models.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--artifact-dir", default=str(CURRENT_DIR / "artifacts"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    device = choose_device()
    artifact_dir = Path(args.artifact_dir)
    loaded = {axis_name: load_axis_model(axis_name, artifact_dir, device) for axis_name in AXIS_NAMES}

    audio_files = iter_audio_files(input_dir, recursive=args.recursive)
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found under {input_dir}")

    output_dir = ensure_directory(args.output_dir) if args.output_dir else ensure_directory(CURRENT_DIR / "outputs")
    rows = []
    for path in audio_files:
        axes = {}
        for axis_name, (model, training_config) in loaded.items():
            axes[axis_name] = predict_axis_file(
                model=model,
                audio_path=path,
                device=device,
                sample_rate=int(training_config.get("sample_rate", 16000)),
                segment_seconds=float(training_config.get("segment_seconds", 3.0)),
            )

        active_axes = {axis_name: payload for axis_name, payload in axes.items() if payload["detected"]}
        dominant_axis = None
        if active_axes:
            dominant_axis = max(active_axes.items(), key=lambda item: item[1]["confidence"])[0]
        prediction = {
            "format_version": "axis-ensemble-v1",
            "mode": "three_independent_axis_models",
            "dominant_axis": dominant_axis,
            "axes": axes,
        }

        row = {"file_name": path.name, "file_path": str(path), "multiaxis_json": json.dumps(prediction, ensure_ascii=False)}
        parts = []
        for axis_name in AXIS_NAMES:
            payload = axes[axis_name]
            row[f"{axis_name}_intensity_level"] = payload["intensity_level"]
            row[f"{axis_name}_intensity_display"] = payload["intensity_display"]
            row[f"{axis_name}_confidence"] = payload["confidence"]
            parts.append(f"{axis_name}={payload['intensity_display']} (conf={payload['confidence']:.4f})")
        rows.append(row)
        print(f"{path.name}: " + " | ".join(parts), flush=True)

    csv_path = Path(output_dir) / "axis_folder_inference.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV to {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
