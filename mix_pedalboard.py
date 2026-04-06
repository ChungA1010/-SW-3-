import os
from pedalboard.io import AudioFile
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