import os
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Distortion

# Create a pedalboard
board = Pedalboard([
    # Distortion(drive_db=25.0),
    Delay(delay_seconds=0.1, feedback=0.3, mix=0.1),
    Reverb(room_size=0.5),
    # Chorus(rate_hz=1.5, depth=0.3)
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
input_file = "test_effector/handmade/test_clean_solo_2.wav"
output_filename = os.path.basename(input_file).replace("clean", effect_type)
output_file = os.path.join("test_effector", "pedalboard", output_filename)

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Process the audio through the pedalboard
with AudioFile(input_file) as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

processed_audio = board(audio, samplerate)

# Save the processed audio to a new file
with AudioFile(output_file, 'w', samplerate, processed_audio.shape[0]) as f:
    f.write(processed_audio)