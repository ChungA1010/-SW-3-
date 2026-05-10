"""Configuration for the BEATs effect + intensity experiment."""

from __future__ import annotations

from pathlib import Path

from shared.constants import DEFAULT_RANDOM_SEED

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "single_effect"
DEFAULT_ARTIFACT_PATH = EXPERIMENT_DIR / "artifacts" / "best_model.pt"
DEFAULT_SPLIT_PATH = EXPERIMENT_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_PRETRAINED_CHECKPOINT_PATH = EXPERIMENT_DIR / "artifacts" / "pretrained" / "beats_iter3plus_as2m.pt"

DEFAULT_SEED = DEFAULT_RANDOM_SEED
DEFAULT_VAL_SIZE = 0.2
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_SEGMENT_SECONDS = 3.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_BACKBONE_LEARNING_RATE = 2e-5
DEFAULT_HEAD_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 0
DEFAULT_INTENSITY_LOSS_WEIGHT = 1.1
DEFAULT_GRAD_CLIP_NORM = 1.0

EFFECT_CLASS_NAMES = [
    "clean",
    "dist",
    "chorus",
    "phaser",
    "delay",
    "reverb",
]
EFFECT_CLASS_TO_INDEX = {label: index for index, label in enumerate(EFFECT_CLASS_NAMES)}
INDEX_TO_EFFECT_CLASS = {index: label for label, index in EFFECT_CLASS_TO_INDEX.items()}

INTENSITY_LEVELS = [0, 1, 2, 3, 4]
INTENSITY_LEVEL_TO_DISPLAY = {
    0: "clean",
    1: "25",
    2: "50",
    3: "75",
    4: "100",
}
INTENSITY_DISPLAY_TO_LEVEL = {value: key for key, value in INTENSITY_LEVEL_TO_DISPLAY.items()}
