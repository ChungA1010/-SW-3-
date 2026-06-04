import os
import argparse
import numpy as np
import librosa
from scipy.signal import hilbert

########## Time Domain Features ##########

def compute_rms(y):
    if len(y) == 0:
        return 0.0
    return np.sqrt(np.mean(y**2))


def compute_zero_crossing_rate(y):
    if len(y) <= 1:
        return 0.0
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y))))
    return zero_crossings / (2 * (len(y) - 1))


def compute_crest_factor(y):
    if len(y) == 0:
        return 0.0
    rms = compute_rms(y)
    if rms == 0:
        return 0.0
    peak = np.max(np.abs(y))
    return peak / rms


def compute_temporal_centroid(y, sr):
    if len(y) == 0:
        return 0.0
    t = np.arange(len(y)) / sr
    energy = y ** 2
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 0.0
    centroid = np.sum(t * energy) / total_energy
    return centroid


def compute_sustain_ratio(y):
    if len(y) == 0:
        return 0.0
    peak_idx = np.argmax(np.abs(y))
    if peak_idx == len(y) - 1:
        return 0.0
    energy_after = np.sum(y[peak_idx:] ** 2)
    total_energy = np.sum(y ** 2)
    if total_energy == 0:
        return 0.0
    return energy_after / total_energy


def compute_post_onset_energy_ratio(y, sr):
    if len(y) == 0:
        return 0.0
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='samples')
    onset_sample = onset_frames[0] if len(onset_frames) else 0
    energy_after = np.sum(y[onset_sample:] ** 2)
    total_energy = np.sum(y ** 2)
    if total_energy == 0:
        return 0.0
    return energy_after / total_energy


########## Frequency Domain Features ##########

def compute_spectral_centroid(y, sr):
    if len(y) == 0:
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid))


def compute_spectral_bandwidth(y, sr):
    if len(y) == 0:
        return 0.0
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return float(np.mean(bandwidth))


def compute_spectral_rolloff(y, sr):
    if len(y) == 0:
        return 0.0
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return float(np.mean(rolloff))


def compute_spectral_flatness(y):
    if len(y) == 0:
        return 0.0
    flatness = librosa.feature.spectral_flatness(y=y)
    return float(np.mean(flatness))


########## Harmonic/Structural Features ##########

def compute_harmonic_ratio(y, sr):
    if len(y) < 2048:
        return 0.0
    try:
        harmonic = librosa.effects.harmonic(y)
        harmonic_energy = np.sum(harmonic**2)
        total_energy = np.sum(y**2)
        if total_energy == 0:
            return 0.0
        return harmonic_energy / total_energy
    except Exception:
        return 0.0


def compute_autocorrelation(y):
    if len(y) <= 1:
        return 0.0
    y = y - np.mean(y)
    fft = np.fft.fft(y, n=2*len(y))
    power = fft * np.conj(fft)
    autocorr = np.fft.ifft(power).real
    autocorr = autocorr[:len(y)]
    positive_lags = autocorr[1:]
    if len(positive_lags) == 0:
        return 0.0
    lag_zero = autocorr[0]
    if lag_zero == 0:
        return 0.0
    peak = np.max(positive_lags)
    return peak / lag_zero


########## Time Structures ##########

def compute_energy_decay(y):
    if len(y) == 0:
        return 0.0
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    if frames.shape[1] == 0:
        return 0.0
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    if len(rms_frames) <= 1:
        return 0.0
    rms_frames = np.maximum(rms_frames, 1e-10)
    time_points = np.arange(len(rms_frames))
    log_rms = np.log(rms_frames)
    slope = np.polyfit(time_points, log_rms, 1)[0]
    return slope


def compute_secondary_peak_count(y):
    if len(y) < 2048:
        return 0.0
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    if len(rms_frames) <= 1:
        return 0.0
    peak_idx = np.argmax(rms_frames)
    if peak_idx == len(rms_frames) - 1:
        return 0.0
    rms_after = rms_frames[peak_idx + 1:]
    if len(rms_after) <= 1:
        return 0.0
    threshold = 0.5 * np.max(rms_after)
    peaks = []
    for i in range(1, len(rms_after) - 1):
        if rms_after[i] > rms_after[i-1] and rms_after[i] > rms_after[i+1] and rms_after[i] > threshold:
            peaks.append(i)
    return len(peaks)


def compute_inter_peak_interval_variance(y):
    if len(y) < 2048:
        return 0.0
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    if len(rms_frames) <= 1:
        return 0.0
    peaks = []
    for i in range(1, len(rms_frames) - 1):
        if rms_frames[i] > rms_frames[i-1] and rms_frames[i] > rms_frames[i+1]:
            peaks.append(i)
    if len(peaks) <= 1:
        return 0.0
    intervals = np.diff(peaks)
    return float(np.var(intervals))


########## Modulation / Phase Effects Features ##########

def compute_amplitude_modulation(y):
    if len(y) == 0:
        return 0.0
    envelope = np.abs(hilbert(y))
    return float(np.var(envelope))


def compute_spectral_flux(y, sr):
    if len(y) == 0:
        return 0.0
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.mean(onset_strength))


def compute_centroid_modulation_depth(y, sr):
    if len(y) == 0:
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.std(centroid))


def compute_spectral_flux_variance(y, sr):
    if len(y) == 0:
        return 0.0
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.var(onset_strength))


def compute_modulation_energy(y, sr):
    if len(y) == 0:
        return 0.0
    envelope = np.abs(hilbert(y))
    fft_env = np.fft.fft(envelope)
    freqs = np.fft.fftfreq(len(envelope), d=1/sr)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_pos = fft_env[pos_mask]
    band_mask = (freqs_pos >= 0.1) & (freqs_pos <= 10)
    if not np.any(band_mask):
        return 0.0
    energy_mod = np.sum(np.abs(fft_pos[band_mask]) ** 2)
    total_energy = np.sum(np.abs(fft_pos) ** 2)
    if total_energy == 0:
        return 0.0
    return float(energy_mod / total_energy)


########## Additional Spectral Features ##########

def compute_spectral_contrast(y, sr):
    """Spectral contrast: average contrast across frequency bands (useful for distortion detection)"""
    if len(y) == 0:
        return 0.0
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        return float(np.mean(contrast))
    except Exception:
        return np.nan


########## MFCC and Derivatives ##########

def compute_mfcc_features(y, sr, n_mfcc=13):
    """Extract MFCC coefficients (useful for timbre analysis across effect types)"""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features = {}
        for i in range(1, n_mfcc + 1):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i-1, :]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i-1, :]))
        return features
    except Exception:
        return {f'mfcc_{i}_{stat}': np.nan for i in range(1, n_mfcc + 1) for stat in ['mean', 'std']}


def compute_delta_mfcc_features(y, sr, n_mfcc=13):
    """Extract delta MFCC (temporal derivatives of MFCC - useful for detecting modulation effects)"""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        features = {}
        for i in range(1, n_mfcc + 1):
            features[f'delta_mfcc_{i}_mean'] = float(np.mean(delta_mfcc[i-1, :]))
            features[f'delta_mfcc_{i}_std'] = float(np.std(delta_mfcc[i-1, :]))
        return features
    except Exception:
        return {f'delta_mfcc_{i}_{stat}': np.nan for i in range(1, n_mfcc + 1) for stat in ['mean', 'std']}


def compute_chroma_features(y, sr):
    """Extract chroma features (useful for harmonic content and pitch-related effects)"""
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features = {
            'chroma_mean': float(np.mean(chroma)),
            'chroma_std': float(np.std(chroma))
        }
        return features
    except Exception:
        return {'chroma_mean': np.nan, 'chroma_std': np.nan}



def extract_features(y, sr):
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
        "modulation_energy": compute_modulation_energy(y, sr),
        "spectral_contrast": compute_spectral_contrast(y, sr),
    }
    
    # Add MFCC features (13 coefficients * 2 stats = 26 features)
    features.update(compute_mfcc_features(y, sr))
    
    # Add delta MFCC features (13 coefficients * 2 stats = 26 features)
    features.update(compute_delta_mfcc_features(y, sr))
    
    # Add chroma features
    features.update(compute_chroma_features(y, sr))
    
    return features


def process_audio_file(args):
    full_path, effector_dir = args
    filename_only = os.path.basename(full_path)
    try:
        y, sr = librosa.load(full_path, sr=44100, mono=True)
        features = extract_features(y, sr)
        return (filename_only, features, None)
    except Exception as e:
        return (filename_only, None, str(e))


if __name__ == "__main__":
    import csv
    from concurrent.futures import ThreadPoolExecutor

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    csv_dir = os.path.join(project_root, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Extract audio features from pedalboard datasets")
    parser.add_argument(
        "--input-dir",
        default=os.path.join("test_effector", "pedalboard", "Multi"),
        help="Input directory (relative to project root)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (relative to csv dir or absolute path)",
    )
    args = parser.parse_args()

    effector_dir = os.path.join(project_root, "test_effector")
    subdir_path = os.path.join(project_root, args.input_dir)

    if not os.path.exists(subdir_path):
        print(f"Directory {subdir_path} not found!")
        exit(1)

    wav_files = []
    for current_root, _, files in os.walk(subdir_path):
        for entry in files:
            if entry.lower().endswith('.wav'):
                wav_files.append(os.path.join(current_root, entry))

    wav_files.sort()

    if not wav_files:
        print(f"No wav files found in {subdir_path}! Exiting.")
        exit(0)

    print(f"Found {len(wav_files)} wav files in {subdir_path}. Processing in parallel...")

    def parse_pedalboard_filename(basename):
        name = os.path.splitext(os.path.basename(basename))[0]
        parts = name.split('_')
        if len(parts) > 0 and parts[0].lower() == 'pedalboard':
            parts = parts[1:]

        known_effects = {
            'clean', 'dist', 'distortion', 'reverb', 'delay', 'chorus', 'phaser',
            'overdrive', 'od', 'drive', 'phase'
        }

        effect_pairs = []
        i = 0
        while i + 1 < len(parts):
            effect_token = parts[i].lower()
            intensity_token = parts[i + 1]
            if effect_token in known_effects and intensity_token.isdigit():
                effect_pairs.append((parts[i], intensity_token))
                i += 2
            else:
                break

        if effect_pairs:
            effect = '+'.join([e.lower() for e, _ in effect_pairs])
            intensity = '_'.join([f"{e.lower()}{v}" for e, v in effect_pairs])
            play = '_'.join(parts[i:]) if i < len(parts) else ''
        else:
            effect = parts[0] if len(parts) > 0 else ''
            intensity = ''
            play = '_'.join(parts[1:]) if len(parts) > 1 else ''

        # Determine top category
        e = effect.lower()
        cats = []
        effect_tokens_for_category = e.split('+') if '+' in e else [e]
        for token in effect_tokens_for_category:
            if 'clean' in token:
                cats.append('clean')
            elif any(k in token for k in ('drive', 'dist', 'overdrive', 'od')):
                cats.append('drive')
            elif any(k in token for k in ('reverb', 'delay')):
                cats.append('space')
            elif any(k in token for k in ('phaser', 'chorus', 'phase')):
                cats.append('phase')
            else:
                cats.append('unknown')

        unique_cats = []
        for c in cats:
            if c not in unique_cats:
                unique_cats.append(c)

        if unique_cats == ['clean']:
            top = 'clean'
        else:
            order = ['drive', 'space', 'phase', 'unknown', 'clean']
            ordered_cats = [c for c in order if c in unique_cats]
            top = '+'.join(ordered_cats) if ordered_cats else 'unknown'
        return top, effect, intensity, play

    results = []
    feature_keys = None

    with ThreadPoolExecutor(max_workers=4) as executor:
        args_list = [(fpath, effector_dir) for fpath in wav_files]
        for i, (filename_with_path, features, error) in enumerate(executor.map(process_audio_file, args_list), 1):
            print(f"Processed {i}/{len(wav_files)}: {filename_with_path}")
            if error:
                print(f"  Error: {error}")
                continue
            if feature_keys is None:
                feature_keys = sorted(features.keys())
            base = os.path.basename(filename_with_path)
            top_cat, spec_eff, intensity, play = parse_pedalboard_filename(base)
            row = [top_cat, spec_eff, intensity, play] + [features[k] for k in feature_keys]
            results.append(row)

    if args.output_csv:
        if os.path.isabs(args.output_csv):
            csv_filename = args.output_csv
        else:
            csv_filename = os.path.join(csv_dir, args.output_csv)
    else:
        input_dir_name = os.path.basename(os.path.normpath(subdir_path)).lower()
        if input_dir_name == 'multi':
            csv_filename = os.path.join(csv_dir, "features_pedalboard_multi.csv")
        else:
            csv_filename = os.path.join(csv_dir, "features_pedalboarded_handmade.csv")

    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["top_category", "effect_kind", "intensity", "play_type_and_number"] + feature_keys
        writer.writerow(header)
        writer.writerows(results)

    print(f"Features saved to {csv_filename}")
