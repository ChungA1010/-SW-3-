import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
# For librosa library reference:
# https://librosa.org/doc/latest/index.html

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

#%% Testing librosa with test_data/test.mp3

# Load audio file with librosa package
# NOTE: original sample rate of test file is 44100 (44.1kHz)
#       Also, 44100 is common sample rate for most of audio file
file = "test_effector/test_clean_solo_3.wav"
signal, sample_rate = librosa.load(file, sr = 44100, mono = True)

#%% Draw waveform of audio as simple sin-wave graph
FIG_SIZE = (12, 5)
plt.figure(figsize=FIG_SIZE)
librosa.display.waveshow(signal, sr = sample_rate, alpha = 0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Original waveform" + "\n" + file)
plt.show()

#%% Draw Spectogram of audio
# *Spectogram: Graph consisted of time as x-asis
#  and frequency as y-axis
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
# print(S_scale)
# Result is form of complex number
# So to draw graph from this, convert values into absolute values
# TODO: Need to study for the mathmatical background of this
Y_scale = np.abs(S_scale) ** 2

# Scale it into log-amplitude
# since human ear recognize as log scale of freq
Y_log_scale = librosa.power_to_db(Y_scale)

plt.figure(figsize=FIG_SIZE)
img = librosa.display.specshow(
    Y_log_scale, sr=sample_rate, hop_length=HOP_SIZE,
    x_axis="time", y_axis="log")
plt.title("Log-scaled spectrogram (amplitude/frequency)" + "\n" + file)
plt.colorbar(format="%+2.f dB")

plt.show()


#%% Draw Mel-spectogram of audio
# Mel-spectogram: Spectogram that Frequency is transformed into Mel-scale
# Mel-scale: pitch scale based on human listening feature
# FYI: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

# Scale into Mel-scale
M_scale = librosa.feature.melspectrogram(S=Y_scale, sr=sample_rate)
M_log_scale = librosa.power_to_db(M_scale, ref=np.max)

plt.figure(figsize=FIG_SIZE)
img = librosa.display.specshow(M_log_scale, sr=sample_rate, x_axis="time", y_axis="mel")
plt.title("Mel-spectrogram" + "\n" + file)
plt.colorbar(format="%+2.f dB")

plt.show()