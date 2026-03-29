from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Distortion

# Create a pedalboard
board = Pedalboard([
    Distortion(drive_db=20.0),
    # Reverb(room_size=0.5),
    # Delay(delay_seconds=0.1, feedback=0.3, mix=0.1),
    # Chorus(rate_hz=1.5, depth=0.7)
])

# set audio file name by effect type
# i.e. drive / dist / reverb / delay / chorus
# i.e. drive+reverb / dist+delay / etc. (if multiple effecters are used)
effects = []
for eff in board:
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

effect_type = "+".join(effects)

print(f"Effect type: {effect_type}")

# Load an audio file, set output file name
input_file = "test_effector/test_clean_solo_3.wav"
output_file = input_file.replace("clean", effect_type)

# Process the audio through the pedalboard
with AudioFile(input_file) as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

processed_audio = board(audio, samplerate)

# Save the processed audio to a new file
with AudioFile(output_file, 'w', samplerate, processed_audio.shape[0]) as f:
    f.write(processed_audio)