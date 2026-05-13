import os
from pedalboard.io import AudioFile
<<<<<<< HEAD
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

# Define effect configurations: (output_folder, effect_name, effect_short_name, percentages, max_value)
effects_config = [
    ("Drive", "Distortion", "dist", [25, 50, 75, 100]),
    ("Space", "Reverb", "reverb", [25, 50, 75, 100]),
    ("Space", "Delay", "delay", [25, 50, 75, 100]),
    ("Phase", "Chorus", "chorus", [25, 50, 75, 100]),
    ("Phase", "Phaser", "phaser", [25, 50, 75, 100]),
]

# Base paths
clean_base_path = "test_effector/pedalboard/Clean"
output_base_path = "test_effector/pedalboard"

# Pickup positions (subfolder names in Clean)
pickup_positions = ["Bridge", "Bridge-Middle", "Middle", "Middle-Neck", "Neck"]

# Create output directories
for folder in ["Drive", "Space", "Phase"]:
    for pickup in pickup_positions:
        os.makedirs(os.path.join(output_base_path, folder, pickup), exist_ok=True)

# Process each effect type
for output_folder, effect_name, effect_short_name, percentages in effects_config:
    print(f"\n=== Processing {effect_name} ===")
    
    # Process each pickup position
    for pickup in pickup_positions:
        clean_dir = os.path.join(clean_base_path, pickup)
        
        # Get all wav files in this pickup directory
        if not os.path.exists(clean_dir):
            print(f"Warning: {clean_dir} does not exist")
            continue
            
        wav_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        
        print(f"  Processing {pickup}: {len(wav_files)} files")
        
        for wav_file in wav_files:
            input_file = os.path.join(clean_dir, wav_file)
            
            # Process each intensity level
            for percent in percentages:
                # Load audio
                with AudioFile(input_file) as f:
                    audio = f.read(f.frames)
                    samplerate = f.samplerate

                # Create effect bundle from the same percentage factor
                effect = make_effect(effect_name, percent)
                
                # Create pedalboard with single effect
                board = Pedalboard([effect])
                
                # Process audio
                processed_audio = board(audio, samplerate)
                
                # Generate output filename
                filename_without_ext = os.path.splitext(wav_file)[0]
                output_filename = f"{effect_short_name}_{percent}_{pickup}_{filename_without_ext}.wav"
                output_file = os.path.join(output_base_path, output_folder, pickup, output_filename)
                
                # Save processed audio
                with AudioFile(output_file, 'w', samplerate, processed_audio.shape[0]) as f:
                    f.write(processed_audio)

print("\n=== Processing completed ===")
=======
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Distortion, Phaser

class PedalboardProcessor:
    def __init__(self, effects):
        """
        effects: list of pedalboard effect instances
        """
        self.board = Pedalboard(effects)

    @staticmethod
    def build_random_board():
        # Create random effect chain for testing / automated generation
        import random

        effects = []

        if random.random() < 0.5:
            effects.append(Distortion(drive_db=random.uniform(10, 40)))

        if random.random() < 0.5:
            effects.append(Delay(
                delay_seconds=random.uniform(0.05, 0.3),
                feedback=random.uniform(0.1, 0.5),
                mix=random.uniform(0.1, 0.5)
            ))

        if random.random() < 0.5:
            effects.append(Reverb(room_size=random.uniform(0.2, 0.8)))

        if random.random() < 0.5:
            effects.append(Chorus(
                rate_hz=random.uniform(0.5, 3),
                depth=random.uniform(0.1, 0.5)
            ))
        
        if random.random() < 0.5:
            effects.append(Phaser(
                rate_hz=random.uniform(0.5, 3),
                depth=random.uniform(0.1, 0.5)
            ))

        return PedalboardProcessor(effects)
    
    def get_effect_names(self):
        effects = []
        for eff in self.board:
            if isinstance(eff, Distortion):
                effects.append("dist")
            elif isinstance(eff, Reverb):
                effects.append("reverb")
            elif isinstance(eff, Delay):
                effects.append("delay")
            elif isinstance(eff, Chorus):
                effects.append("chorus")
            else:
                effects.append("unknown")
        return "+".join(effects)

    def process(self, input_file, output_dir="test_effector/pedalboard"):
        # effect name 생성
        effect_type = self.get_effect_names()
        print(f"Effect type: {effect_type}")

        # output 파일명 생성
        output_filename = os.path.basename(input_file).replace("clean", effect_type)
        output_file = os.path.join(output_dir, output_filename)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 오디오 읽기
        with AudioFile(input_file) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        # 처리
        processed_audio = self.board(audio, samplerate)

        # 저장
        with AudioFile(output_file, 'w', samplerate, processed_audio.shape[0]) as f:
            f.write(processed_audio)

        return output_file

if __name__ == "__main__":
    # Create a pedalboard
    effects = [
        # Distortion(drive_db=25.0),
        Delay(delay_seconds=0.1, feedback=0.3, mix=0.1),
        Reverb(room_size=0.5),
        # Chorus(rate_hz=1.5, depth=0.3)
    ]

    processor = PedalboardProcessor(effects)

    input_file = "test_effector/handmade/test_clean_solo_2.wav"
    output_file = processor.process(input_file)
>>>>>>> 1ceaf2600f1e5ac2ea9e1591e9eb165a97f32edf
