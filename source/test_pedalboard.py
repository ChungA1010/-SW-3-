import os

from pedalboard import Pedalboard, Reverb, Delay, Distortion, Chorus, Phaser
from pedalboard.io import AudioFile

INPUT_FILE = "test_effector/pedalboard/clean_solo_5.wav"
OUTPUT_DIR = "test_effector/pedalboard"

# Distortion max gain: 30.0
# Reverb max room_size: 1.0
# Delay max delay_time_ms: 1000.0
# Chorus max rate_hz: 3.0
# Phaser max rate_hz: 3.0
EFFECTS = [
    "dist",
    "reverb",
    "delay",
    "chorus",
    "phaser",
]
PERCENTS = [25, 50, 75, 100]


def shaped_factor(percent, exponent):
    return (percent / 100.0) ** exponent


def make_effect(effect_name, percent):
    factor = percent / 100.0
    distortion_factor = shaped_factor(percent, 0.5)
    space_factor = shaped_factor(percent, 0.6)
    delay_factor = shaped_factor(percent, 0.65)
    motion_factor = shaped_factor(percent, 0.6)

    if effect_name == "dist":
        return Distortion(drive_db=30.0 * distortion_factor)
    if effect_name == "reverb":
        return Reverb(
            damping=space_factor,
            dry_level=0.4 * space_factor,
            room_size=space_factor,
            wet_level=0.33 * space_factor,
            width=space_factor,
        )
    if effect_name == "delay":
        return Delay(
            delay_seconds=0.2 * delay_factor,
            feedback=0.7 * delay_factor,
            mix=0.8 * delay_factor,
        )
    if effect_name == "chorus":
        return Chorus(
            centre_delay_ms=5.0 * motion_factor,
            depth=0.2 * motion_factor,
            feedback=0.7 * motion_factor,
            mix=0.8 * motion_factor,
            rate_hz=3.0 * motion_factor,
        )
    if effect_name == "phaser":
        return Phaser(
            centre_frequency_hz=900.0 * motion_factor,
            depth=0.35 * motion_factor,
            feedback=0.7 * motion_factor,
            mix=0.8 * motion_factor,
            rate_hz=3.0 * motion_factor,
        )

    raise ValueError(f"Unknown effect: {effect_name}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with AudioFile(INPUT_FILE) as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

for effect_name in EFFECTS:
    for percent in PERCENTS:
        effect = make_effect(effect_name, percent)
        board = Pedalboard([effect])
        processed = board(audio, samplerate)

        output_filename = f"{effect_name}_{percent}_solo_5.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with AudioFile(output_path, "w", samplerate, processed.shape[0]) as f:
            f.write(processed)
