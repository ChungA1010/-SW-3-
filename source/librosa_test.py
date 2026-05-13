#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from typing import Tuple, Callable, Any
# For librosa library reference:
# https://librosa.org/doc/latest/index.html

#%% Define functions for analysis
def analyze_waveform(file: str) -> Tuple[np.ndarray, float]:
    # Load audio file with librosa package
    # NOTE: original sample rate of test file is 44100 (44.1kHz)
    #       Also, 44100 is common sample rate for most of audio files
    signal, sample_rate = librosa.load(file, sr=44100, mono=True)
    return signal, float(sample_rate)

# Spectogram: Graph consisted of time as x-asis, and frequency as y-axis
def analyze_spectrogram(file: str) -> Tuple[np.ndarray, float, int]:
    signal, sample_rate = librosa.load(file, sr=44100, mono=True)
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
    HOP_SIZE = 512  # Usually, hop_length = n_fft/4

    S_scale = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    # Result is form of complex number
    # So to draw graph from this, convert values into absolute values
    # TODO: Need to study for the mathmatical background of this
    Y_scale = np.abs(S_scale) ** 2

    # Scale it into log-amplitude
    # since human ear recognize as log scale of freq
    Y_log_scale = librosa.power_to_db(Y_scale)
    return Y_log_scale, sample_rate, HOP_SIZE

# Mel-spectogram: Spectogram that Frequency is transformed into Mel-scale
# Mel-scale: pitch scale based on human listening feature
# FYI: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
def analyze_mel_spectrogram(file: str) -> Tuple[np.ndarray, float]:
    signal, sample_rate = librosa.load(file, sr=44100, mono=True)
    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S_scale = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    Y_scale = np.abs(S_scale) ** 2

    # Scale into Mel-scale
    M_scale = librosa.feature.melspectrogram(S=Y_scale, sr=sample_rate)
    M_log_scale = librosa.power_to_db(M_scale, ref=np.max)
    return M_log_scale, sample_rate

# Visualize the results of analysis
def visualize_analysis(analysis_func: Callable[[str], Any], file: str) -> None:
    FIG_SIZE = (12, 5)
    if analysis_func == analyze_waveform:
        signal, sample_rate = analysis_func(file)
        plt.figure(figsize=FIG_SIZE)
        librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Original waveform" + "\n" + file)
        plt.show()
    elif analysis_func == analyze_spectrogram:
        Y_log_scale, sample_rate, HOP_SIZE = analysis_func(file)
        plt.figure(figsize=FIG_SIZE)
        img = librosa.display.specshow(
            Y_log_scale, sr=sample_rate, hop_length=HOP_SIZE,
            x_axis="time", y_axis="log")
        plt.title("Log-scaled spectrogram (amplitude/frequency)" + "\n" + file)
        plt.colorbar(format="%+2.f dB")
        plt.show()
    elif analysis_func == analyze_mel_spectrogram:
        M_log_scale, sample_rate = analysis_func(file)
        plt.figure(figsize=FIG_SIZE)
        img = librosa.display.specshow(M_log_scale, sr=sample_rate, x_axis="time", y_axis="mel")
        plt.title("Mel-spectrogram" + "\n" + file)
        plt.colorbar(format="%+2.f dB")
        plt.show()

#%% Example usage
file = "test_effector/test_clean_solo_3.wav"
visualize_analysis(analyze_waveform, file)
visualize_analysis(analyze_spectrogram, file)
visualize_analysis(analyze_mel_spectrogram, file)