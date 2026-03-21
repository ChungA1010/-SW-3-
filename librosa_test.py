import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display


# Testing numpy / matplotlib
"""
t = np.linspace(0,1,100)
f = 1 #1Hz

plt.plot(t, 1 * np.sin(2*np.pi*f*t + 0), "-", 
         label='sin(2$\pi$ft)')
plt.plot(t, 0.7 * np.sin(2*np.pi*f*t - 1), ls="--", 
         label='0.7sin(2$\pi$ft-1)')

plt.xlabel("Time")
plt.title("Sine Wave")
plt.legend();plt.grid();plt.show()

"""

# Testing librosa with test_data/test.mp3

# Load audio file with librosa package
file = "test_data/test_clean.wav"
signal, sample_rate = librosa.load(file, sr = None, mono = True)

# Draw waveform of audio as simple sin-wave graph
FIG_SIZE = (12, 7)
plt.figure(figsize=FIG_SIZE)
librosa.display.waveshow(signal, sr = sample_rate, alpha = 0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Original waveform")
plt.show()

# Draw Spectogram of audio
# *Spectogram: Graph consisted of time as x-asis and frequency as y-axis
"""
Caculate STFT(Short time Fourier transformation)
Parameters:
- n_fft: Number of fast Fourier transformation
         = How many samples will be analized = window size
         = Larger -> Higher resolution of freq -> more precise pitch analizing
         ** Need to be larger for drive effecters analizing **
         (Since they generate high freqs)
- hop_length: How much the window moves(jumps) per each analyzing
         = Smaller -> Higher resolution of time
         ** Need to be larger for phase effecters analizing **
         (Since they manipulate phase over time)
"""
FRAME_SIZE = 2048
HOP_SIZE = 512 # Usually, hop_length = n_fft/4

S_scale = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_scale) # Result is form of complex number