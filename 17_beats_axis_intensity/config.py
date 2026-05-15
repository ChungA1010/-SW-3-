"""Configuration for the BEATs per-axis intensity experiments."""

from __future__ import annotations

from pathlib import Path

from shared.constants import DEFAULT_RANDOM_SEED

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "single_effect"
DEFAULT_MULTI_DATA_DIR = WORKSPACE_ROOT / "multi_effect"
DEFAULT_DATA_DIRS = [DEFAULT_DATA_DIR, DEFAULT_MULTI_DATA_DIR]
DEFAULT_ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_PRETRAINED_CHECKPOINT_PATH = (
    WORKSPACE_ROOT
    / "guitar_effector_classifier2"
    / "15_beats_effect_intensity"
    / "artifacts"
    / "pretrained"
    / "beats_iter3plus_as2m.pt"
)

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
DEFAULT_GRAD_CLIP_NORM = 1.0

AXIS_NAMES = ["drive", "phase", "space"]
TARGET_EFFECTS = ["dist", "delay", "phaser"]
INTENSITY_LEVELS = [0, 1, 2]
INTENSITY_LEVEL_TO_DISPLAY = {
    0: "off",
    1: "50",
    2: "100",
}
INTENSITY_DISPLAY_TO_LEVEL = {value: key for key, value in INTENSITY_LEVEL_TO_DISPLAY.items()}
TARGET_INTENSITY_DISPLAYS = set(INTENSITY_DISPLAY_TO_LEVEL)
CLASS_SCHEMA = {
    "effects": TARGET_EFFECTS,
    "intensity_levels": INTENSITY_LEVELS,
    "intensity_displays": INTENSITY_LEVEL_TO_DISPLAY,
}

TOP_LEVEL_TO_AXIS = {
    "Drive": "drive",
    "Phase": "phase",
    "Space": "space",
}

EFFECT_TO_AXIS = {
    "dist": "drive",
    "phaser": "phase",
    "delay": "space",
}
ALL_KNOWN_EFFECTS = [
    "dist",
    "drive",
    "fuzz",
    "chorus",
    "phaser",
    "flanger",
    "tremolo",
    "delay",
    "reverb",
]


def artifact_path_for_axis(axis_name: str) -> Path:
    return DEFAULT_ARTIFACT_DIR / f"{axis_name}_best_model.pt"


def split_path_for_axis(axis_name: str) -> Path:
    return DEFAULT_ARTIFACT_DIR / f"{axis_name}_split.json"


def output_dir_for_axis(axis_name: str) -> Path:
    return DEFAULT_OUTPUT_DIR / axis_name
