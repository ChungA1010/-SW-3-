import os
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Distortion, Reverb, Delay, Chorus, Phaser

# Create a pedalboard
# Distortion max gain: 30.0
# Reverb max room_size: 1.0
# Delay max delay_time_ms: 1000.0
# Chorus max rate_hz: 3.0
# Phaser max rate_hz: 3.0


def shaped_factor(percent, exponent):
    return (percent / 100.0) ** exponent


def make_effect(effect_name, percent):
    factor = percent / 100.0
    distortion_factor = shaped_factor(percent, 0.5)
    space_factor = shaped_factor(percent, 0.6)
    delay_factor = shaped_factor(percent, 0.65)
    motion_factor = shaped_factor(percent, 0.6)

    if effect_name == "Distortion":
        return Distortion(drive_db=30.0 * distortion_factor)
    if effect_name == "Reverb":
        return Reverb(
            damping=space_factor,
            dry_level=0.4 * space_factor,
            room_size=space_factor,
            wet_level=0.33 * space_factor,
            width=space_factor,
        )
    if effect_name == "Delay":
        return Delay(
            delay_seconds=1.0 * delay_factor,
            feedback=0.7 * delay_factor,
            mix=0.8 * delay_factor,
        )
    if effect_name == "Chorus":
        return Chorus(
            centre_delay_ms=5.0 * motion_factor,
            depth=0.2 * motion_factor,
            feedback=0.7 * motion_factor,
            mix=0.8 * motion_factor,
            rate_hz=3.0 * motion_factor,
        )
    if effect_name == "Phaser":
        return Phaser(
            centre_frequency_hz=900.0 * motion_factor,
            depth=0.35 * motion_factor,
            feedback=0.7 * motion_factor,
            mix=0.8 * motion_factor,
            rate_hz=3.0 * motion_factor,
        )

    raise ValueError(f"Unknown effect: {effect_name}")


def process_chord_with_effects(input_file, output_dir):
    """
    Process a single clean chord file with various effects and intensities.
    Output filename format: <effect>_<intensity>_chord_5.wav
    For clean signal, intensity is omitted.
    """
    # Define effect configurations: (effect_name, effect_short_name, percentages)
    effects_config = [
        ("Distortion", "dist", [25, 50, 75, 100]),
        ("Reverb", "reverb", [25, 50, 75, 100]),
        ("Delay", "delay", [25, 50, 75, 100]),
        ("Chorus", "chorus", [25, 50, 75, 100]),
        ("Phaser", "phaser", [25, 50, 75, 100]),
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    print(f"Processing: {input_file}")
    
    # Load clean audio once
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    
    # First, save clean signal
    clean_filename = "clean_chord_5.wav"
    clean_file = os.path.join(output_dir, clean_filename)
    with AudioFile(clean_file, 'w', samplerate, audio.shape[0]) as f:
        f.write(audio)
    print(f"  Saved: {clean_filename}")
    
    # Process each effect type with different intensities
    for effect_name, effect_short_name, percentages in effects_config:
        print(f"  Processing {effect_name}...")
        
        for percent in percentages:
            # Create effect
            effect = make_effect(effect_name, percent)
            
            # Create pedalboard with single effect
            board = Pedalboard([effect])
            
            # Process audio
            processed_audio = board(audio, samplerate)
            
            # Generate output filename: <effect>_<intensity>_chord_5.wav
            output_filename = f"{effect_short_name}_{percent}_chord_5.wav"
            output_file = os.path.join(output_dir, output_filename)
            
            # Save processed audio
            with AudioFile(output_file, 'w', samplerate, processed_audio.shape[0]) as f:
                f.write(processed_audio)
            print(f"    Saved: {output_filename}")
    
    print(f"Completed processing {input_file}")


if __name__ == "__main__":
    # Process clean_chord_5.wav
    input_file = "test_effector/pedalboard/clean_chord_5.wav"
    output_dir = "test_effector/pedalboard/play"
    
    process_chord_with_effects(input_file, output_dir)
