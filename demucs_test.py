from demucs import pretrained
from demucs.apply import apply_model
import torch
import librosa
import soundfile as sf

# Use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pre-trained model
model = pretrained.get_model('htdemucs')
model.to(device) # move model to selected device
model.eval()

# Load audio file
y, sr = librosa.load("test_data/_original.wav", sr=44100, mono=False)
wav = torch.tensor(y)

# Make audio stereo if mono
if wav.ndim == 1:
    wav = wav.unsqueeze(0)

if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)

# shape: (channels, samples) → (1, channels, samples)
wav = wav.unsqueeze(0)

# move audio to selected device
wav = wav.to(device)

# Separate
with torch.no_grad():
    sources = apply_model(model, wav)

# Results
# sources shape: (1, 4, channels, samples)
sources = sources[0]

names = ["drums", "bass", "other", "vocals"]

for i, name in enumerate(names):
    audio = sources[i].cpu().numpy().T  # (samples, channels)
    sf.write(f"test_separation/{name}.wav", audio, 44100)