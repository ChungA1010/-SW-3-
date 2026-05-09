"""Project-wide constants used by every experiment."""

from __future__ import annotations

COARSE_CLASS_NAMES = ["clean", "drive", "space", "phase"]
COARSE_CLASS_TO_INDEX = {label: index for index, label in enumerate(COARSE_CLASS_NAMES)}
INDEX_TO_COARSE_CLASS = {index: label for label, index in COARSE_CLASS_TO_INDEX.items()}

# Backward-compatible aliases used across the existing experiment code.
CLASS_NAMES = COARSE_CLASS_NAMES
CLASS_TO_INDEX = COARSE_CLASS_TO_INDEX
INDEX_TO_CLASS = INDEX_TO_COARSE_CLASS

FINE_CLASS_NAMES = [
    "clean",
    "Blues_Driver",
    "Tube_Screamer",
    "RAT",
    "Chorus",
    "Flanger",
    "Phaser",
    "Tape_Echo",
    "Digital_Delay",
    "Sweep_Echo",
    "Plate_Reverb",
    "Hall_Reverb",
    "Spring_Reverb",
]
FINE_CLASS_TO_INDEX = {label: index for index, label in enumerate(FINE_CLASS_NAMES)}
INDEX_TO_FINE_CLASS = {index: label for label, index in FINE_CLASS_TO_INDEX.items()}

FINE_CLASS_DIRECTORY_ALIASES = {
    "clean": ["clean", "Clean"],
    "Blues_Driver": ["Blues_Driver", "BluesDriver"],
    "Tube_Screamer": ["Tube_Screamer", "TubeScreamer"],
    "RAT": ["RAT"],
    "Chorus": ["Chorus"],
    "Flanger": ["Flanger"],
    "Phaser": ["Phaser"],
    "Tape_Echo": ["Tape_Echo", "TapeEcho"],
    "Digital_Delay": ["Digital_Delay", "Digital Delay", "Digital-Delay"],
    "Sweep_Echo": ["Sweep_Echo", "Sweep Echo", "Sweep-Echo"],
    "Plate_Reverb": ["Plate_Reverb", "Plate Reverb", "Plate-Reverb"],
    "Hall_Reverb": ["Hall_Reverb", "Hall Reverb", "Hall-Reverb"],
    "Spring_Reverb": ["Spring_Reverb", "Spring Reverb", "Spring-Reverb"],
}

FINE_TO_COARSE = {
    "clean": "clean",
    "Blues_Driver": "drive",
    "Tube_Screamer": "drive",
    "RAT": "drive",
    "Chorus": "phase",
    "Flanger": "phase",
    "Phaser": "phase",
    "Tape_Echo": "space",
    "Digital_Delay": "space",
    "Sweep_Echo": "space",
    "Plate_Reverb": "space",
    "Hall_Reverb": "space",
    "Spring_Reverb": "space",
}
COARSE_TO_FINE = {
    coarse_label: [fine_label for fine_label, mapped in FINE_TO_COARSE.items() if mapped == coarse_label]
    for coarse_label in COARSE_CLASS_NAMES
}

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".wave"}

DEFAULT_RANDOM_SEED = 42
DEFAULT_VALIDATION_SIZE = 0.2

DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 96
DEFAULT_N_MFCC = 20
