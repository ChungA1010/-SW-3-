#!/usr/bin/env python3
"""Run inference on a single audio file with the BEATs effect + intensity model."""

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

from config import DEFAULT_ARTIFACT_PATH  # noqa: E402
from shared.torch_training import choose_device  # noqa: E402
from utils import build_model, predict_audio_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BEATs effect + intensity inference on a single file.")
    parser.add_argument("--audio-path", required=True, help="Path to a WAV file")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH), help="Path to a saved checkpoint")
    args = parser.parse_args()

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

    result = predict_audio_file(
        model=model,
        audio_path=args.audio_path,
        device=device,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
