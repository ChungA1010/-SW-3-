import os
from itertools import combinations, product
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Distortion, Delay, Phaser

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


def list_wav_files_recursive(root_dir):
    wav_files = []
    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".wav"):
                wav_files.append(os.path.join(current_root, filename))
    return wav_files


def build_multi_effect_configs(min_effect_types=2, max_effect_types=3):
    """
    Build multi-effect combinations with category constraints.
    Same category cannot appear twice in one combination.
    """
    categories = {
        "drive": [("Distortion", "dist")],
        "space": [("Delay", "delay")],
        "phase": [("Phaser", "phaser")],
    }

    category_names = list(categories.keys())
    configs = []
    for k in range(min_effect_types, max_effect_types + 1):
        for selected_categories in combinations(category_names, k):
            options_by_category = [categories[cat] for cat in selected_categories]
            for picked_effects in product(*options_by_category):
                configs.append(list(picked_effects))
    return configs


def process_clean_single_note_multi_effects(
    clean_root_dir,
    output_dir,
    intensities=(25, 50, 75, 100),
    min_effect_types=2,
    max_effect_types=3,
):
    """
    Process all wav files under Clean recursively and generate multi-effect samples.

    Output filename format:
    <effect>_<intensity>_<effect>_<intensity>[_<effect>_<intensity>]_<pickup_info>_<original_name>.wav
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(clean_root_dir):
        print(f"Error: Clean root {clean_root_dir} does not exist")
        return

    wav_files = list_wav_files_recursive(clean_root_dir)
    # Filter to only clean files (contain "clean" and exclude effect-related terms)
    clean_wav_files = [f for f in wav_files if "clean" in os.path.basename(f).lower() 
                       and not any(x in os.path.basename(f).lower() for x in ["effect", "fx", "effected", "processed"])]
    wav_files = clean_wav_files
    if not wav_files:
        print(f"No clean wav files found under {clean_root_dir}")
        return

    effect_configs = build_multi_effect_configs(
        min_effect_types=min_effect_types,
        max_effect_types=max_effect_types,
    )

    print(f"Found {len(wav_files)} clean wav files")
    print(f"Using {len(effect_configs)} multi-effect configurations")

    generated_count = 0
    failed_count = 0

    for input_file in wav_files:
        rel_path = os.path.relpath(input_file, clean_root_dir)
        rel_dir = os.path.dirname(rel_path)
        pickup_info = rel_dir.replace("\\", "_").replace("/", "_") if rel_dir else "no_pickup"
        original_name = os.path.splitext(os.path.basename(input_file))[0]

        print(f"Processing clean source: {rel_path}")

        try:
            with AudioFile(input_file) as f:
                audio = f.read(f.frames)
                samplerate = f.samplerate
        except Exception as e:
            print(f"  Failed to load {input_file}: {e}")
            failed_count += 1
            continue

        for config in effect_configs:
            intensity_products = product(intensities, repeat=len(config))

            for intensity_values in intensity_products:
                board_effects = []
                filename_tokens = []

                for (effect_name, effect_short_name), intensity in zip(config, intensity_values):
                    board_effects.append(make_effect(effect_name, intensity))
                    filename_tokens.extend([effect_short_name, str(intensity)])

                board = Pedalboard(board_effects)
                processed_audio = board(audio, samplerate)

                output_filename = "_".join(filename_tokens + [pickup_info, original_name]) + ".wav"
                output_file = os.path.join(output_dir, output_filename)

                try:
                    with AudioFile(output_file, "w", samplerate, processed_audio.shape[0]) as f:
                        f.write(processed_audio)
                    generated_count += 1
                except Exception as e:
                    print(f"  Failed to save {output_filename}: {e}")
                    failed_count += 1

    print("\nCompleted multi-effect generation")
    print(f"Generated files: {generated_count}")
    print(f"Failed files: {failed_count}")


if __name__ == "__main__":
    clean_root_dir = "test_effector/handmade_new/single"
    output_dir = "test_effector/pedalboard/play"

    process_clean_single_note_multi_effects(
        clean_root_dir=clean_root_dir,
        output_dir=output_dir,
        intensities=(25, 50, 75, 100),
        min_effect_types=1,
        max_effect_types=3,
    )
