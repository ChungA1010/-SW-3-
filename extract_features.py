import numpy as np
import librosa
from scipy.signal import hilbert

########## Time Domain Features ##########

def compute_rms(y):
    """
    Compute RMS (Root Mean Square) energy of the signal.
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: RMS energy
    """
    print("Starting RMS computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    result = np.sqrt(np.mean(y**2))
    print("Done.")
    return result


def compute_zero_crossing_rate(y):
    """
    Compute average ZCR (zero crossing rate) of the signal.
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Average zero crossing rate
    """
    print("Starting zero crossing rate computation...")
    if len(y) <= 1:
        print("Done.")
        return 0.0
    # Count zero crossings
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y))))
    result = zero_crossings / (2 * (len(y) - 1))
    print("Done.")
    return result


def compute_crest_factor(y):
    """
    Compute crest factor = peak amplitude / RMS.
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Crest factor
    """
    print("Starting crest factor computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    rms = compute_rms(y)
    if rms == 0:
        print("Done.")
        return 0.0
    peak = np.max(np.abs(y))
    result = peak / rms
    print("Done.")
    return result


def compute_temporal_centroid(y, sr):
    """
    Compute temporal centroid (energy-weighted center of mass in time domain).
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Temporal centroid in seconds
    """
    print("Starting temporal centroid computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    t = np.arange(len(y)) / sr
    energy = y ** 2
    total_energy = np.sum(energy)
    if total_energy == 0:
        print("Done.")
        return 0.0
    centroid = np.sum(t * energy) / total_energy
    print("Done.")
    return centroid


def compute_sustain_ratio(y):
    """
    Compute sustain ratio (ratio of energy after peak vs total energy).
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Sustain ratio
    """
    print("Starting sustain ratio computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    peak_idx = np.argmax(np.abs(y))
    if peak_idx == len(y) - 1:
        print("Done.")
        return 0.0
    energy_after = np.sum(y[peak_idx:] ** 2)
    total_energy = np.sum(y ** 2)
    if total_energy == 0:
        print("Done.")
        return 0.0
    result = energy_after / total_energy
    print("Done.")
    return result


def compute_post_onset_energy_ratio(y, sr):
    """
    Compute post-onset energy ratio (energy after onset relative to total energy).
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Post-onset energy ratio
    """
    print("Starting post-onset energy ratio computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='samples')
    if len(onset_frames) == 0:
        onset_sample = 0
    else:
        onset_sample = onset_frames[0]
    energy_after = np.sum(y[onset_sample:] ** 2)
    total_energy = np.sum(y ** 2)
    if total_energy == 0:
        print("Done.")
        return 0.0
    result = energy_after / total_energy
    print("Done.")
    return result


########## Frequency Domain Features ##########

def compute_spectral_centroid(y, sr):
    """
    Compute spectral centroid using librosa.
    Means overall brightness of the sound.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Mean spectral centroid
    """
    print("Starting spectral centroid computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    result = np.mean(centroid)
    print("Done.")
    return result


def compute_spectral_bandwidth(y, sr):
    """
    Compute spectral bandwidth using librosa.
    Means spread of spectrum around centroid.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Mean spectral bandwidth
    """
    print("Starting spectral bandwidth computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    result = np.mean(bandwidth)
    print("Done.")
    return result


def compute_spectral_rolloff(y, sr):
    """
    Compute spectral rolloff using librosa.
    Means frequency limit which contains 85% of total spectral energy.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Mean spectral rolloff
    """
    print("Starting spectral rolloff computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    result = np.mean(rolloff)
    print("Done.")
    return result


def compute_spectral_flatness(y):
    """
    Compute spectral flatness using librosa.
    Means noisiness vs tonality of the sound.
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Mean spectral flatness
    """
    print("Starting spectral flatness computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    flatness = librosa.feature.spectral_flatness(y=y)
    result = np.mean(flatness)
    print("Done.")
    return result

########## Harmonic/Structural Features ##########

def compute_harmonic_ratio(y, sr):
    """
    Compute harmonic ratio = harmonic_energy / total_energy using harmonic extraction.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Harmonic ratio
    """
    print("Starting harmonic ratio computation...")
    if len(y) < 2048:  # Minimum length
        print("Done.")
        return 0.0
    try:
        harmonic = librosa.effects.harmonic(y)
        harmonic_energy = np.sum(harmonic**2)
        total_energy = np.sum(y**2)
        if total_energy == 0:
            print("Done.")
            return 0.0
        result = harmonic_energy / total_energy
        print("Done.")
        return result
    except Exception:
        # Fallback if harmonic extraction fails
        print("Done.")
        return 0.0


def compute_autocorrelation(y):
    """
    Compute normalized autocorrelation peak (excluding lag 0).
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Normalized autocorrelation peak
    """
    print("Starting autocorrelation computation...")
    if len(y) <= 1:
        print("Done.")
        return 0.0
    y = y - np.mean(y)
    fft = np.fft.fft(y, n=2*len(y))
    power = fft * np.conj(fft)
    autocorr = np.fft.ifft(power).real
    autocorr = autocorr[:len(y)]
    # Get positive lags (excluding lag 0)
    positive_lags = autocorr[1:]
    if len(positive_lags) == 0:
        print("Done.")
        return 0.0
    lag_zero = autocorr[0]
    if lag_zero == 0:
        print("Done.")
        return 0.0
    peak = np.max(positive_lags)
    result = peak / lag_zero
    print("Done.")
    return result

########## Time Structures ##########

def compute_energy_decay(y):
    """
    Compute energy decay slope by analyzing RMS over time frames.
    How fast the energy decays after initial attack.
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Energy decay slope (negative indicates decay)
    """
    print("Starting energy decay computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    # Frame the signal
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    
    if frames.shape[1] == 0:
        print("Done.")
        return 0.0
    
    # Compute RMS per frame
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    
    if len(rms_frames) <= 1:
        print("Done.")
        return 0.0
    
    # Avoid log of zero
    rms_frames = np.maximum(rms_frames, 1e-10)
    
    # Time points (frame indices)
    time_points = np.arange(len(rms_frames))
    
    # Linear regression on log(RMS)
    log_rms = np.log(rms_frames)
    slope = np.polyfit(time_points, log_rms, 1)[0]
    
    print("Done.")
    return slope


def compute_secondary_peak_count(y):
    """
    Compute secondary peak count (number of peaks after main peak in RMS envelope).
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Secondary peak count
    """
    print("Starting secondary peak count computation...")
    if len(y) < 2048:
        print("Done.")
        return 0.0
    # Frame the signal
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    
    if len(rms_frames) <= 1:
        print("Done.")
        return 0.0
    
    # Find peaks
    peak_idx = np.argmax(rms_frames)
    if peak_idx == len(rms_frames) - 1:
        print("Done.")
        return 0.0
    
    # Simple peak detection: local maxima after main peak
    rms_after = rms_frames[peak_idx + 1:]
    if len(rms_after) <= 1:
        print("Done.")
        return 0.0
    
    # Threshold: 50% of max after peak
    threshold = 0.5 * np.max(rms_after)
    peaks = []
    for i in range(1, len(rms_after) - 1):
        if rms_after[i] > rms_after[i-1] and rms_after[i] > rms_after[i+1] and rms_after[i] > threshold:
            peaks.append(i)
    
    result = len(peaks)
    print("Done.")
    return result


def compute_inter_peak_interval_variance(y):
    """
    Compute inter-peak interval variance (variance of time intervals between peaks).
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Inter-peak interval variance
    """
    print("Starting inter-peak interval variance computation...")
    if len(y) < 2048:
        print("Done.")
        return 0.0
    # Frame the signal
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    
    if len(rms_frames) <= 1:
        print("Done.")
        return 0.0
    
    # Find all peaks
    peaks = []
    for i in range(1, len(rms_frames) - 1):
        if rms_frames[i] > rms_frames[i-1] and rms_frames[i] > rms_frames[i+1]:
            peaks.append(i)
    
    if len(peaks) <= 1:
        print("Done.")
        return 0.0
    
    # Intervals in frames
    intervals = np.diff(peaks)
    result = np.var(intervals)
    print("Done.")
    return result

########## Modulation / Phase Effects Features ##########

def compute_amplitude_modulation(y):
    """
    Compute variance of amplitude envelope using Hilbert transform.
    How much the amplitude envelope shakes over time
    
    Args:
        y (np.ndarray): Audio signal
        
    Returns:
        float: Variance of amplitude envelope
    """
    print("Starting amplitude modulation computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    envelope = np.abs(hilbert(y))
    result = np.var(envelope)
    print("Done.")
    return result


def compute_spectral_flux(y, sr):
    """
    Compute spectral flux approximation using onset strength.
    How much the spectrum changes over time.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Mean spectral flux
    """
    print("Starting spectral flux computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    result = np.mean(onset_strength)
    print("Done.")
    return result


def compute_centroid_modulation_depth(y, sr):
    """
    Compute centroid modulation depth (standard deviation of spectral centroid over time).
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Centroid modulation depth
    """
    print("Starting centroid modulation depth computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    result = np.std(centroid)
    print("Done.")
    return result


def compute_spectral_flux_variance(y, sr):
    """
    Compute spectral flux variance (variance of spectral flux).
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Spectral flux variance
    """
    print("Starting spectral flux variance computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    result = np.var(onset_strength)
    print("Done.")
    return result


def compute_modulation_energy(y, sr):
    """
    Compute modulation energy (energy in low-frequency modulation band 0.1-10 Hz of amplitude envelope).
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        float: Modulation energy
    """
    print("Starting modulation energy computation...")
    if len(y) == 0:
        print("Done.")
        return 0.0
    envelope = np.abs(hilbert(y))
    # FFT of envelope
    fft_env = np.fft.fft(envelope)
    freqs = np.fft.fftfreq(len(envelope), d=1/sr)
    # Positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_pos = fft_env[pos_mask]
    # Band 0.1-10 Hz
    band_mask = (freqs_pos >= 0.1) & (freqs_pos <= 10)
    if not np.any(band_mask):
        print("Done.")
        return 0.0
    energy_mod = np.sum(np.abs(fft_pos[band_mask]) ** 2)
    total_energy = np.sum(np.abs(fft_pos) ** 2)
    if total_energy == 0:
        print("Done.")
        return 0.0
    result = energy_mod / total_energy
    print("Done.")
    return result


def extract_features(y, sr):
    """
    Extract all audio features and return as dictionary.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        dict: Dictionary of features
    """
    features = {
        "rms": compute_rms(y),
        "zcr": compute_zero_crossing_rate(y),
        "crest_factor": compute_crest_factor(y),
        "temporal_centroid": compute_temporal_centroid(y, sr),
        "sustain_ratio": compute_sustain_ratio(y),
        "post_onset_energy_ratio": compute_post_onset_energy_ratio(y, sr),
        "spectral_centroid": compute_spectral_centroid(y, sr),
        "spectral_bandwidth": compute_spectral_bandwidth(y, sr),
        "spectral_rolloff": compute_spectral_rolloff(y, sr),
        "spectral_flatness": compute_spectral_flatness(y),
        "harmonic_ratio": compute_harmonic_ratio(y, sr),
        "autocorrelation": compute_autocorrelation(y),
        "energy_decay": compute_energy_decay(y),
        "secondary_peak_count": compute_secondary_peak_count(y),
        "inter_peak_interval_variance": compute_inter_peak_interval_variance(y),
        "amplitude_modulation": compute_amplitude_modulation(y),
        "spectral_flux": compute_spectral_flux(y, sr),
        "centroid_modulation_depth": compute_centroid_modulation_depth(y, sr),
        "spectral_flux_variance": compute_spectral_flux_variance(y, sr),
        "modulation_energy": compute_modulation_energy(y, sr)
    }
    return features

def plot_features(features, y, sr):
    """
    Plot audio features in grouped bar plots with normalization and time series.
    
    Args:
        features (dict): Dictionary of extracted features
        y (np.ndarray): Audio signal
        sr (int): Sampling rate
    """
    import matplotlib.pyplot as plt
    # Define feature groups based on function comments
    groups = {
        "Time Domain Features": ["rms", "zcr", "crest_factor", "temporal_centroid", "sustain_ratio", "post_onset_energy_ratio"],
        "Frequency Domain Features": ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness"],
        "Harmonic/Structural Features": ["harmonic_ratio", "autocorrelation"],
        "Time Structures": ["energy_decay", "secondary_peak_count", "inter_peak_interval_variance"],
        "Modulation / Phase Effects Features": ["amplitude_modulation", "spectral_flux", "centroid_modulation_depth", "spectral_flux_variance", "modulation_energy"]
    }
    
    # Create subplots: 2x3 grid (5 groups + 1 time series)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Audio Feature Analysis", fontsize=16)
    
    # Plot bar plots for each group
    for i, (group_name, feature_list) in enumerate(groups.items()):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        
        # Get values
        values = [features[feat] for feat in feature_list if feat in features]
        
        # Min-max normalization
        if len(values) > 1 and max(values) != min(values):
            values_norm = [(v - min(values)) / (max(values) - min(values)) for v in values]
        else:
            values_norm = values  # If all same or single value
        
        # Bar plot
        bars = ax.bar(feature_list, values_norm, color='skyblue', edgecolor='black')
        ax.set_title(group_name)
        ax.set_ylabel("Normalized Value")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot spectral centroid over time
    ax_time = axes[1, 2]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(centroid))
    t = librosa.frames_to_time(frames, sr=sr)
    ax_time.plot(t, centroid, color='orange')
    ax_time.set_title("Spectral Centroid Over Time")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Frequency (Hz)")
    ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os
    import csv
    
    # Get all wav files in test_effector directory and subdirectories
    effector_dir = "test_effector"
    if not os.path.exists(effector_dir):
        print(f"Directory {effector_dir} not found!")
        exit(1)
    
    wav_files = []
    for root, dirs, files in os.walk(effector_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                wav_files.append(full_path)
    
    if not wav_files:
        print("No wav files found in test_effector directory!")
        exit(1)
    
    print(f"Found {len(wav_files)} wav files. Processing...")
    
    # Extract features for all files
    results = []
    feature_keys = None
    
    for i, full_path in enumerate(wav_files):
        filename = os.path.basename(full_path)
        print(f"Processing {i+1}/{len(wav_files)}: {filename} ...\n")
        try:
            y, sr = librosa.load(full_path, sr=44100, mono=True)
            features = extract_features(y, sr)
            if feature_keys is None:
                feature_keys = sorted(features.keys())
            row = [filename] + [features[key] for key in feature_keys]
            results.append(row)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            print(f"Finished processing {filename}.\n")
    
    # Save to CSV
    csv_filename = "audio_features.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + feature_keys
        writer.writerow(header)
        writer.writerows(results)
    
    print(f"Features saved to {csv_filename}")
    
    # Visualize only the first file
    if results:
        first_full_path = wav_files[0]
        first_filename = os.path.basename(first_full_path)
        print(f"Visualizing features for {first_filename}")
        y, sr = librosa.load(first_full_path, sr=44100, mono=True)
        features = extract_features(y, sr)
        plot_features(features, y, sr)
