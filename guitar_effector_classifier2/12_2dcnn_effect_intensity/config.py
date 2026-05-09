"""Configuration for the 2D CNN effect + intensity experiment."""

from __future__ import annotations

from pathlib import Path

from shared.constants import DEFAULT_RANDOM_SEED, DEFAULT_SAMPLE_RATE

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "single_effect"
DEFAULT_ARTIFACT_PATH = EXPERIMENT_DIR / "artifacts" / "best_model.pt"
DEFAULT_SPLIT_PATH = EXPERIMENT_DIR / "artifacts" / "split.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

DEFAULT_SEED = DEFAULT_RANDOM_SEED
DEFAULT_VAL_SIZE = 0.2
DEFAULT_SAMPLE_RATE_HZ = DEFAULT_SAMPLE_RATE
DEFAULT_SEGMENT_SECONDS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 40
DEFAULT_PATIENCE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_N_MELS_BANDS = 96
DEFAULT_N_FFT_SIZE = 1024
DEFAULT_HOP_LENGTH = 256
DEFAULT_NUM_WORKERS = 0

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

# We use five ordered stages:
# 0 = clean / no effect, 1 = 25, 2 = 50, 3 = 75, 4 = 100.
INTENSITY_LEVELS = [0, 1, 2, 3, 4]
INTENSITY_LEVEL_TO_DISPLAY = {
    0: "clean",
    1: "25",
    2: "50",
    3: "75",
    4: "100",
}
INTENSITY_DISPLAY_TO_LEVEL = {value: key for key, value in INTENSITY_LEVEL_TO_DISPLAY.items()}
