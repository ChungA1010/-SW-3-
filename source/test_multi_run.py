import importlib.util
import os
from pedalboard.io import AudioFile
from pedalboard import Pedalboard

# Load mix_pedalboard module by file path to avoid package import issues
mix_path = os.path.join(os.path.dirname(__file__), "mix_pedalboard.py")
spec = importlib.util.spec_from_file_location("mix_pedalboard", mix_path)
mix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mix)
make_effect = mix.make_effect
from itertools import combinations, product
import os

input_file = "test_effector/pedalboard/Clean/Bridge/1-0.wav"
output_dir = "test_effector/pedalboard/Multi_test"
intensities = (50, 100)

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_file):
    print("Input file not found:", input_file)
    raise SystemExit(1)

with AudioFile(input_file) as f:
    audio = f.read(f.frames)
    sr = f.samplerate

effects = [("Distortion", "dist"), ("Delay", "delay"), ("Phaser", "phaser")]

configs = []
for k in (2, 3):
    for cats in combinations(range(len(effects)), k):
        opts = [[effects[i]] for i in cats]
        for picked in product(*opts):
            configs.append(list(picked))

print(f"Will generate for {len(configs)} configs")

generated = 0
for config in configs:
    for vals in product(intensities, repeat=len(config)):
        board_effects = [make_effect(name, v) for (name, short), v in zip(config, vals)]
        board = Pedalboard(board_effects)
        processed = board(audio, sr)
        tokens = []
        for (name, short), v in zip(config, vals):
            tokens.extend([short, str(v)])
        outfn = "_".join(tokens + ["Bridge_1-0"]) + ".wav"
        outpath = os.path.join(output_dir, outfn)
        with AudioFile(outpath, "w", sr, processed.shape[0]) as fo:
            fo.write(processed)
        generated += 1

print("Generated files:", generated)
print("Sample files:")
for i, fn in enumerate(sorted(os.listdir(output_dir))[:10]):
    print(" ", fn)
