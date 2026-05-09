#!/usr/bin/env python3
"""Run single-file effect + intensity inference with the 2D CNN."""

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
from model import EffectIntensityCNN  # noqa: E402
from utils import predict_audio_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer effect type and intensity from one wav file.")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH), help="Saved model checkpoint")
    parser.add_argument("--audio-path", required=True, help="Input wav file")
    args = parser.parse_args()

    artifact = torch.load(args.artifact_path, map_location="cpu")
    model = EffectIntensityCNN(
        num_effect_classes=artifact["model_config"]["num_effect_classes"],
        num_intensity_classes=artifact["model_config"]["num_intensity_classes"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    cfg = artifact["training_config"]
    prediction = predict_audio_file(
        model=model,
        audio_path=args.audio_path,
        device=torch.device("cpu"),
        sample_rate=cfg["sample_rate"],
        segment_seconds=cfg["segment_seconds"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
    )
    print(json.dumps(prediction, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
