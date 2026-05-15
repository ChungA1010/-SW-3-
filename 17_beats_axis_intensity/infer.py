#!/usr/bin/env python3
"""Run one-file inference with the three BEATs per-axis intensity models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AXIS_NAMES, artifact_path_for_axis  # noqa: E402
from shared.torch_training import choose_device  # noqa: E402
from utils import build_model, predict_axis_file  # noqa: E402


def load_axis_model(axis_name: str, artifact_path: Path, device: torch.device):
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
    training_config = checkpoint.get("training_config", {})
    return model, training_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run three-axis inference on a single WAV file.")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--artifact-dir", default=str(CURRENT_DIR / "artifacts"))
    args = parser.parse_args()

    device = choose_device()
    artifact_dir = Path(args.artifact_dir)
    axes = {}
    for axis_name in AXIS_NAMES:
        artifact_path = artifact_dir / f"{axis_name}_best_model.pt"
        model, training_config = load_axis_model(axis_name, artifact_path, device)
        axes[axis_name] = predict_axis_file(
            model=model,
            audio_path=args.audio_path,
            device=device,
            sample_rate=int(training_config.get("sample_rate", 16000)),
            segment_seconds=float(training_config.get("segment_seconds", 3.0)),
        )

    active_axes = {axis_name: payload for axis_name, payload in axes.items() if payload["detected"]}
    dominant_axis = None
    if active_axes:
        dominant_axis = max(active_axes.items(), key=lambda item: item[1]["confidence"])[0]
    print(
        json.dumps(
            {
                "format_version": "axis-ensemble-v1",
                "mode": "three_independent_axis_models",
                "dominant_axis": dominant_axis,
                "axes": axes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
