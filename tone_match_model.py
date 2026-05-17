from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks


EPSILON = 1e-8


@dataclass(frozen=True)
class FeatureRule:
    name: str
    direction: str
    meaning: str
    weight: float
    low: float
    high: float
    invert: bool = False


@dataclass
class FeatureComparison:
    feature: str
    reference_value: float
    copy_value: float
    delta: float
    percent_change: float
    expected_direction: str
    meaning: str


@dataclass
class AxisFeedback:
    axis: str
    reference_amount: int
    copy_amount: int
    difference: int
    similarity: float
    action: str
    message: str
    features: list[FeatureComparison]


class ToneMatchModel:
    """Feature-based tone matcher for reference/copy effector comparison.

    This model does not classify effect types. It compares two audio files:
    a reference tone and a user's copied tone. The output tells whether the
    copy needs more or less drive, space, or phase-style processing.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        hop_length: int = 256,
        max_analysis_seconds: float = 20.0,
        action_threshold: int = 12,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_analysis_seconds = max_analysis_seconds
        self.action_threshold = action_threshold

    def compare(self, reference_path: str | Path, copy_path: str | Path) -> dict:
        reference_audio = self._load_audio(reference_path)
        copy_audio = self._load_audio(copy_path)
        length = min(reference_audio.size, copy_audio.size)

        reference_features = self._extract_features(reference_audio[:length])
        copy_features = self._extract_features(copy_audio[:length])

        axes = [
            self._build_axis("drive", self._drive_rules(), reference_features, copy_features),
            self._build_axis("space", self._space_rules(), reference_features, copy_features),
            self._build_axis("phase", self._phase_rules(), reference_features, copy_features),
        ]

        total_similarity = float(np.mean([axis.similarity for axis in axes])) if axes else 0.0
        return {
            "reference_path": str(reference_path),
            "copy_path": str(copy_path),
            "overall_similarity": round(total_similarity, 3),
            "axes": [asdict(axis) for axis in axes],
        }

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        audio, _ = librosa.load(Path(audio_path), sr=self.sample_rate, mono=True)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        max_samples = int(self.sample_rate * self.max_analysis_seconds)
        return audio[:max_samples] if audio.size > max_samples else audio

    def _extract_features(self, audio: np.ndarray) -> dict[str, float]:
        n_fft = 2048
        hop_length = max(128, self.hop_length)
        frame_length = min(n_fft, max(512, int(self.sample_rate * 0.064)))

        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=48,
        )
        mel_db = librosa.power_to_db(mel + EPSILON)

        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = float(np.mean(np.square(harmonic)) + EPSILON)
        percussive_energy = float(np.mean(np.square(percussive)) + EPSILON)
        signal_rms = float(np.sqrt(np.mean(np.square(audio)) + EPSILON))
        peak = float(np.max(np.abs(audio)) + EPSILON)
        flux_frames = np.sqrt(np.mean(np.diff(mel_db, axis=1) ** 2, axis=0)) if mel_db.shape[1] > 1 else np.zeros(1)

        onset = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=hop_length)
        peaks, _ = find_peaks(onset, distance=max(1, int(self.sample_rate / hop_length * 0.12)))
        peak_intervals = np.diff(peaks) * hop_length / self.sample_rate if peaks.size > 1 else np.asarray([], dtype=np.float32)

        early = rms[: max(1, len(rms) // 4)]
        tail = rms[max(0, len(rms) * 3 // 4) :]
        voiced_threshold = max(float(np.max(rms)) * 0.25, EPSILON)
        centroid_mean = float(np.mean(centroid) + EPSILON)

        return {
            "amplitude_modulation": float(np.std(rms) / (np.mean(rms) + EPSILON)),
            "rms": signal_rms,
            "crest_factor": float(peak / signal_rms),
            "spectral_bandwidth": float(np.mean(bandwidth) / max(self.sample_rate / 2, EPSILON)),
            "spectral_flatness": float(np.mean(flatness)),
            "autocorrelation": self._autocorrelation_repetition(audio),
            "harmonic_ratio": float(harmonic_energy / (harmonic_energy + percussive_energy)),
            "zcr": float(np.mean(zcr)),
            "spectral_flux": float(np.mean(flux_frames)),
            "spectral_flux_variance": float(np.var(flux_frames)),
            "energy_decay": float(np.mean(tail) / (np.mean(early) + EPSILON)),
            "sustain_ratio": float(np.mean(rms > voiced_threshold)),
            "inter_peak_interval_variance": float(np.std(peak_intervals) / (np.mean(peak_intervals) + EPSILON))
            if peak_intervals.size
            else 0.0,
            "spectral_contrast": float(np.mean(contrast)),
            "secondary_peak_count": float(max(0, len(peaks) - 1) / max(librosa.get_duration(y=audio, sr=self.sample_rate), 1.0)),
            "modulation_energy": self._modulation_energy(rms, hop_length),
            "delta_mfcc_std": float(np.mean(np.std(librosa.feature.delta(mfcc), axis=1))),
            "centroid_modulation_depth": float(np.std(centroid) / centroid_mean),
            "mfcc_modulation_variance": float(np.mean(np.var(mfcc, axis=1))),
        }

    def _build_axis(
        self,
        axis: str,
        rules: list[FeatureRule],
        reference_features: dict[str, float],
        copy_features: dict[str, float],
    ) -> AxisFeedback:
        reference_amount = self._axis_amount(rules, reference_features)
        copy_amount = self._axis_amount(rules, copy_features)
        difference = copy_amount - reference_amount
        similarity = max(0.0, 1.0 - abs(difference) / 100.0)
        action = self._action_from_difference(difference)
        message = self._message(axis, action)
        feature_rows = [self._compare_feature(rule, reference_features, copy_features) for rule in rules]
        feature_rows.sort(key=lambda feature: abs(feature.percent_change), reverse=True)

        return AxisFeedback(
            axis=axis,
            reference_amount=reference_amount,
            copy_amount=copy_amount,
            difference=difference,
            similarity=round(similarity, 3),
            action=action,
            message=message,
            features=feature_rows[:5],
        )

    def _axis_amount(self, rules: list[FeatureRule], features: dict[str, float]) -> int:
        score = 0.0
        weight = 0.0
        for rule in rules:
            score += self._scale(features.get(rule.name, 0.0), rule.low, rule.high, rule.invert) * rule.weight
            weight += rule.weight
        return int(round(np.clip(score / max(weight, EPSILON), 0.0, 1.0) * 100))

    def _compare_feature(
        self,
        rule: FeatureRule,
        reference_features: dict[str, float],
        copy_features: dict[str, float],
    ) -> FeatureComparison:
        reference_value = float(reference_features.get(rule.name, 0.0))
        copy_value = float(copy_features.get(rule.name, 0.0))
        delta = copy_value - reference_value
        percent_change = delta / max(abs(reference_value), EPSILON) * 100.0
        return FeatureComparison(
            feature=rule.name,
            reference_value=round(reference_value, 6),
            copy_value=round(copy_value, 6),
            delta=round(delta, 6),
            percent_change=round(percent_change, 2),
            expected_direction=rule.direction,
            meaning=rule.meaning,
        )

    def _drive_rules(self) -> list[FeatureRule]:
        return [
            FeatureRule("amplitude_modulation", "increase", "distortion raises amplitude variation", 1.5, 0.25, 1.20),
            FeatureRule("rms", "increase", "drive raises average energy", 1.5, 0.025, 0.18),
            FeatureRule("crest_factor", "decrease", "clipping lowers peak-to-RMS ratio", 1.4, 9.0, 2.2, True),
            FeatureRule("spectral_bandwidth", "increase", "harmonics widen frequency bandwidth", 1.1, 0.22, 0.42),
            FeatureRule("spectral_flatness", "increase", "distortion increases noise-like content", 0.8, 0.01, 0.12),
            FeatureRule("autocorrelation", "decrease", "distortion lowers waveform repetition", 0.7, 0.55, 0.18, True),
            FeatureRule("harmonic_ratio", "decrease", "drive destabilizes harmonic structure", 0.65, 0.82, 0.30, True),
            FeatureRule("zcr", "increase", "clipping raises high-frequency crossings", 0.55, 0.04, 0.14),
        ]

    def _space_rules(self) -> list[FeatureRule]:
        return [
            FeatureRule("spectral_flux_variance", "decrease", "reverb/delay smooths spectral changes", 1.5, 35.0, 16.0, True),
            FeatureRule("rms", "increase", "tail/repeats raise average energy", 1.1, 0.025, 0.18),
            FeatureRule("amplitude_modulation", "increase", "repeats/tails create amplitude structure", 1.1, 0.25, 1.20),
            FeatureRule("energy_decay", "increase", "space effects keep energy in the tail", 1.1, 0.25, 1.10),
            FeatureRule("sustain_ratio", "increase", "reverb extends sustain", 1.0, 0.18, 0.72),
            FeatureRule("inter_peak_interval_variance", "decrease", "delay repeats stabilize peak spacing", 0.85, 0.9, 0.45, True),
            FeatureRule("spectral_contrast", "decrease", "space effects soften band contrast", 0.75, 24.0, 8.0, True),
            FeatureRule("spectral_flatness", "decrease", "space effects can reduce noise-like sharpness", 0.65, 0.08, 0.05, True),
            FeatureRule("secondary_peak_count", "increase", "echo/repeat peaks increase", 0.65, 0.5, 4.0),
            FeatureRule("spectral_flux", "decrease", "space effects reduce abrupt spectral movement", 0.55, 22.0, 12.0, True),
        ]

    def _phase_rules(self) -> list[FeatureRule]:
        return [
            FeatureRule("modulation_energy", "increase", "LFO modulation energy rises", 1.5, 0.02, 0.20),
            FeatureRule("spectral_flux", "increase", "chorus/phaser keeps the spectrum moving", 1.1, 10.0, 34.0),
            FeatureRule("harmonic_ratio", "decrease", "comb filtering lowers stable harmonic ratio", 1.1, 0.82, 0.30, True),
            FeatureRule("delta_mfcc_std", "increase", "tone color changes frame by frame", 1.0, 1.4, 7.0),
            FeatureRule("amplitude_modulation", "increase", "chorus/phaser adds amplitude wobble", 0.7, 0.25, 1.20),
            FeatureRule("centroid_modulation_depth", "increase", "brightness modulation increases", 0.65, 0.08, 0.45),
            FeatureRule("mfcc_modulation_variance", "increase", "cepstral modulation increases", 0.65, 8.0, 90.0),
        ]

    def _action_from_difference(self, difference: int) -> str:
        if difference > self.action_threshold:
            return "lower"
        if difference < -self.action_threshold:
            return "raise"
        return "keep"

    def _message(self, axis: str, action: str) -> str:
        if action == "raise":
            return f"{axis} is weaker than the reference. Raise {axis}."
        if action == "lower":
            return f"{axis} is stronger than the reference. Lower {axis}."
        return f"{axis} is close to the reference. Keep {axis}."

    def _scale(self, value: float, low: float, high: float, invert: bool = False) -> float:
        if high == low:
            return 0.0
        raw = (value - low) / (high - low)
        if invert:
            raw = 1.0 - raw
        return float(np.clip(raw, 0.0, 1.0))

    def _autocorrelation_repetition(self, audio: np.ndarray) -> float:
        if audio.size < 4:
            return 0.0
        sample = audio[: min(audio.size, 65536)]
        centered = sample - float(np.mean(sample))
        fft_size = 1 << (2 * centered.size - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size)
        corr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: centered.size]
        corr = corr / (corr[0] + EPSILON)
        search = corr[min(64, len(corr) - 1) : min(len(corr), 4096)]
        return float(np.max(search)) if search.size else 0.0

    def _modulation_energy(self, rms: np.ndarray, hop_length: int) -> float:
        if rms.size < 4:
            return 0.0
        centered = rms - float(np.mean(rms))
        spectrum = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(rms.size, d=hop_length / self.sample_rate)
        band = (freqs >= 0.3) & (freqs <= 8.0)
        return float(np.sum(spectrum[band]) / (np.sum(spectrum) + EPSILON))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reference and copied guitar effector tones.")
    parser.add_argument("reference_path")
    parser.add_argument("copy_path")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--hop-length", type=int, default=256)
    args = parser.parse_args()

    model = ToneMatchModel(sample_rate=args.sample_rate, hop_length=args.hop_length)
    result = model.compare(args.reference_path, args.copy_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
