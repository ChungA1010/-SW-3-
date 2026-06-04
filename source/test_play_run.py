#!/usr/bin/env python3
"""
Test runner for play dataset generation (handmade_new/single clean → pedalboard/play)
Tests with a small sample of files before full run.
"""
import os
import sys
import importlib.util

# Load mix_pedalboard module by file path
spec = importlib.util.spec_from_file_location("mix_pedalboard", "source/mix_pedalboard.py")
mix_pedalboard = importlib.util.module_from_spec(spec)
sys.modules["mix_pedalboard"] = mix_pedalboard
spec.loader.exec_module(mix_pedalboard)

def test_play_generation(num_files=2):
    """
    Test play dataset generation with a small number of files.
    """
    clean_root = "test_effector/handmade_new/single"
    output_dir = "test_effector/pedalboard/play_test"
    
    # Get all clean files
    all_wavs = mix_pedalboard.list_wav_files_recursive(clean_root)
    clean_files = [f for f in all_wavs if "clean" in os.path.basename(f).lower() 
                   and not any(x in os.path.basename(f).lower() for x in ["effect", "fx", "effected", "processed"])]
    
    print(f"Found {len(clean_files)} total clean files")
    test_files = clean_files[:num_files]
    print(f"Testing with {len(test_files)} files: {[os.path.basename(f) for f in test_files]}")
    
    # Create test configs (single + 2 + 3 effects)
    effect_configs = mix_pedalboard.build_multi_effect_configs(min_effect_types=1, max_effect_types=3)
    print(f"Testing with {len(effect_configs)} multi-effect configurations (min=1, max=3)")
    print(f"  Configs: {effect_configs}")
    
    # Manually process test files
    os.makedirs(output_dir, exist_ok=True)
    generated_count = 0
    
    for input_file in test_files:
        rel_path = os.path.relpath(input_file, clean_root)
        rel_dir = os.path.dirname(rel_path)
        pickup_info = rel_dir.replace("\\", "_").replace("/", "_") if rel_dir else "no_pickup"
        original_name = os.path.splitext(os.path.basename(input_file))[0]
        
        print(f"\nProcessing: {rel_path}")
        
        try:
            from pedalboard.io import AudioFile
            with AudioFile(input_file) as f:
                audio = f.read(f.frames)
                samplerate = f.samplerate
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        
        for config in effect_configs:
            intensities = (25, 50, 75, 100)
            from itertools import product
            intensity_products = product(intensities, repeat=len(config))
            
            for intensity_values in intensity_products:
                board_effects = []
                filename_tokens = []
                
                for (effect_name, effect_short_name), intensity in zip(config, intensity_values):
                    board_effects.append(mix_pedalboard.make_effect(effect_name, intensity))
                    filename_tokens.extend([effect_short_name, str(intensity)])
                
                from pedalboard import Pedalboard
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
    
    print(f"\n✓ Test completed: Generated {generated_count} files in {output_dir}")
    
    # List sample outputs
    output_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    print(f"Sample outputs (first 10 of {len(output_files)}):")
    for fname in sorted(output_files)[:10]:
        print(f"  - {fname}")

if __name__ == "__main__":
    test_play_generation(num_files=2)
