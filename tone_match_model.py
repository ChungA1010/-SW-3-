from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks


EPSILON = 1e-8

# Controls how quickly tanh saturates as a function of the delta ratio.
# delta_ratio = (copy - ref) / |ref|; tanh(delta_ratio * sensitivity) → [-1, 1].
# At sensitivity=2.0: a 50% change gives tanh(1.0) ≈ 0.76 (strong signal).
DEFAULT_SENSITIVITY = 2.0


@dataclass(frozen=True)
class FeatureRule:
    """Maps one audio feature to an effect axis.

    direction   : expected change direction when the effect is present.
    weight      : relative importance (higher = more influential on axis score).
    sensitivity : tanh scale factor; tune per-feature when typical delta magnitudes differ.
                  Replaces the old hard-coded low/high/invert absolute bounds.
    """

    name: str
    direction: str        # "increase" | "decrease"
    meaning: str
    weight: float
    sensitivity: float = DEFAULT_SENSITIVITY


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
    # reference_amount is always 50 — the reference audio anchors the midpoint.
    # copy_amount = clip(50 + difference, 0, 100).
    reference_amount: int
    copy_amount: int
    difference: int      # positive = copy has more of the effect than reference
    similarity: float    # 1 - |difference| / 100
    action: str          # "raise" | "lower" | "keep"
    message: str
    features: list[FeatureComparison]


class ToneMatchModel:
    """Feature-based tone matcher for reference/copy effector comparison.

    Computes how much Drive, Space, and Phase effect is present in the copy
    *relative to the reference* using tanh-scaled delta-ratio scoring.
    Hard-coded absolute bounds are replaced with data-adaptive relative-change
    measurements so scores are invariant to recording level.

    Cross-validation guards prevent shared features from double-counting:
    - Space axis : rms is gated by spectral_flux_variance confirmation.
    - Phase axis : amplitude_modulation is gated by core phase indicators
                   (modulation_energy, spectral_flux, harmonic_ratio).
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, reference_path: str | Path, copy_path: str | Path) -> dict:
        ref_audio = self._load_audio(reference_path)
        copy_audio = self._load_audio(copy_path)
        length = min(ref_audio.size, copy_audio.size)

        ref_feats = self._extract_features(ref_audio[:length])
        copy_feats = self._extract_features(copy_audio[:length])

        axes = [
            self._build_drive_axis(ref_feats, copy_feats),
            self._build_space_axis(ref_feats, copy_feats),
            self._build_phase_axis(ref_feats, copy_feats),
        ]

        total_similarity = float(np.mean([ax.similarity for ax in axes])) if axes else 0.0
        return {
            "reference_path": str(reference_path),
            "copy_path": str(copy_path),
            "overall_similarity": round(total_similarity, 3),
            "axes": [asdict(ax) for ax in axes],
        }

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        audio, _ = librosa.load(Path(audio_path), sr=self.sample_rate, mono=True)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        max_samples = int(self.sample_rate * self.max_analysis_seconds)
        return audio[:max_samples] if audio.size > max_samples else audio

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, audio: np.ndarray) -> dict[str, float]:
        n_fft = 2048
        hop = max(128, self.hop_length)
        frame_length = min(n_fft, max(512, int(self.sample_rate * 0.064)))

        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop)[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop
        )[0]
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop)[0]
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop
        )
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop
        )[0]
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop
        )
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop, n_mels=48
        )
        mel_db = librosa.power_to_db(mel + EPSILON)

        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = float(np.mean(np.square(harmonic)) + EPSILON)
        percussive_energy = float(np.mean(np.square(percussive)) + EPSILON)
        signal_rms = float(np.sqrt(np.mean(np.square(audio)) + EPSILON))
        peak = float(np.max(np.abs(audio)) + EPSILON)
        flux_frames = (
            np.sqrt(np.mean(np.diff(mel_db, axis=1) ** 2, axis=0))
            if mel_db.shape[1] > 1
            else np.zeros(1)
        )

        onset = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=hop)
        peaks, _ = find_peaks(onset, distance=max(1, int(self.sample_rate / hop * 0.12)))
        peak_intervals = (
            np.diff(peaks) * hop / self.sample_rate
            if peaks.size > 1
            else np.asarray([], dtype=np.float32)
        )

        early = rms[: max(1, len(rms) // 4)]
        tail = rms[max(0, len(rms) * 3 // 4) :]
        voiced_threshold = max(float(np.max(rms)) * 0.25, EPSILON)
        centroid_mean = float(np.mean(centroid) + EPSILON)

        return {
            # Drive: increase — am, rms, zcr, bandwidth, flatness
            # Drive: decrease — crest_factor, autocorrelation, harmonic_ratio
            "amplitude_modulation": float(np.std(rms) / (np.mean(rms) + EPSILON)),
            "rms": signal_rms,
            "zcr": float(np.mean(zcr)),
            "spectral_bandwidth": float(np.mean(bandwidth) / max(self.sample_rate / 2, EPSILON)),
            "spectral_flatness": float(np.mean(flatness)),
            "crest_factor": float(peak / signal_rms),
            "autocorrelation": self._autocorrelation_repetition(audio),
            "harmonic_ratio": float(harmonic_energy / (harmonic_energy + percussive_energy)),
            # Space: decrease — sfv, peak_interval_var, contrast, flatness, flux
            # Space: increase — rms*, am, energy_decay, sustain, secondary_peaks
            # (* rms is cross-validated against sfv in _build_space_axis)
            "spectral_flux_variance": float(np.var(flux_frames)),
            "inter_peak_interval_variance": (
                float(np.std(peak_intervals) / (np.mean(peak_intervals) + EPSILON))
                if peak_intervals.size
                else 0.0
            ),
            "spectral_contrast": float(np.mean(contrast)),
            "spectral_flux": float(np.mean(flux_frames)),
            "energy_decay": float(np.mean(tail) / (np.mean(early) + EPSILON)),
            "sustain_ratio": float(np.mean(rms > voiced_threshold)),
            "secondary_peak_count": float(
                max(0, len(peaks) - 1)
                / max(librosa.get_duration(y=audio, sr=self.sample_rate), 1.0)
            ),
            # Phase: increase — mod_energy, flux, delta_mfcc, am*, centroid_mod, mfcc_var
            # Phase: decrease — harmonic_ratio
            # (* am is cross-validated against core phase indicators in _build_phase_axis)
            "modulation_energy": self._modulation_energy(rms, hop),
            "delta_mfcc_std": float(np.mean(np.std(librosa.feature.delta(mfcc), axis=1))),
            "centroid_modulation_depth": float(np.std(centroid) / centroid_mean),
            "mfcc_modulation_variance": float(np.mean(np.var(mfcc, axis=1))),
        }

    # ------------------------------------------------------------------
    # Axis builders
    # ------------------------------------------------------------------

    def _build_drive_axis(
        self, ref: dict[str, float], copy: dict[str, float]
    ) -> AxisFeedback:
        return self._build_axis("drive", self._drive_rules(), ref, copy)

    def _build_space_axis(
        self, ref: dict[str, float], copy: dict[str, float]
    ) -> AxisFeedback:
        """Build space axis with rms cross-validation.

        Problem  : rms rises strongly under drive too, so a naïve space scorer
                   would misread a heavy-drive copy as having more space.
        Solution : gate the rms contribution by how much spectral_flux_variance
                   moves in the space direction (decreasing). sfv is largely
                   insensitive to drive and therefore acts as a reliable
                   space-specific discriminator.

        rms_scale = clip(tanh(sfv_signed * sensitivity), 0, 1)
          - sfv unchanged or wrong direction → rms_scale ≈ 0  → rms contributes 0
          - sfv clearly decreasing (space)   → rms_scale → 1  → rms gets full weight
        """
        sfv_signed = self._signed_delta(ref, copy, "spectral_flux_variance", "decrease")
        rms_scale = float(np.clip(np.tanh(sfv_signed * DEFAULT_SENSITIVITY), 0.0, 1.0))
        return self._build_axis(
            "space", self._space_rules(), ref, copy, overrides={"rms": rms_scale}
        )

    def _build_phase_axis(
        self, ref: dict[str, float], copy: dict[str, float]
    ) -> AxisFeedback:
        """Build phase axis with amplitude_modulation cross-validation.

        Problem  : amplitude_modulation reacts to drive and space as well.
                   Counting it unconditionally inflates phase scores for
                   non-phase effects.
        Solution : compute a 'core phase score' from the three indicators that
                   are most specific to chorus/phaser/flanger — modulation_energy,
                   spectral_flux (phase direction), and harmonic_ratio (decrease).
                   amplitude_modulation is only credited when those core signals
                   are collectively positive.

        am_scale = clip(mean(tanh(core_signals)), 0, 1)
          - core indicators not firing (mean ≤ 0) → am_scale = 0  → am contributes 0
          - core indicators strongly positive      → am_scale → 1  → am gets full weight
        """
        core_signals = [
            np.tanh(self._signed_delta(ref, copy, "modulation_energy", "increase") * DEFAULT_SENSITIVITY),
            np.tanh(self._signed_delta(ref, copy, "spectral_flux", "increase") * DEFAULT_SENSITIVITY),
            np.tanh(self._signed_delta(ref, copy, "harmonic_ratio", "decrease") * DEFAULT_SENSITIVITY),
        ]
        core_mean = float(np.mean(core_signals))  # ∈ [-1, 1]
        am_scale = float(np.clip(core_mean, 0.0, 1.0))
        return self._build_axis(
            "phase", self._phase_rules(), ref, copy, overrides={"amplitude_modulation": am_scale}
        )

    def _build_axis(
        self,
        axis: str,
        rules: list[FeatureRule],
        ref: dict[str, float],
        copy: dict[str, float],
        overrides: dict[str, float] | None = None,
    ) -> AxisFeedback:
        """Score one axis using weighted tanh-scaled delta ratios.

        Replaces the old absolute-bound _scale approach. Instead of asking
        "where does this feature's raw value fall between low and high?",
        we ask "how much did this feature change relative to the reference,
        and is that change in the expected direction?"

        Scoring per feature:
            signed_delta   = (copy - ref) / |ref|, then sign-flipped if "decrease"
            contribution   = tanh(signed_delta * sensitivity)  ∈ [-1, 1]
            effective_weight = rule.weight * overrides.get(name, 1.0)

        axis_delta = Σ(contribution * effective_weight) / Σ(effective_weight)  ∈ [-1, 1]
        difference = round(axis_delta * 100)  — positive means copy has more effect.

        overrides: weight multipliers in [0, 1] applied to cross-validated features.
        """
        if overrides is None:
            overrides = {}

        total_weight = 0.0
        weighted_score = 0.0

        for rule in rules:
            signed = self._signed_delta(ref, copy, rule.name, rule.direction)
            contribution = float(np.tanh(signed * rule.sensitivity))
            effective_weight = rule.weight * overrides.get(rule.name, 1.0)
            weighted_score += contribution * effective_weight
            total_weight += effective_weight

        axis_delta = weighted_score / max(total_weight, EPSILON)
        difference = int(round(axis_delta * 100))
        copy_amount = int(np.clip(50 + difference, 0, 100))
        similarity = round(max(0.0, 1.0 - abs(difference) / 100.0), 3)
        action = self._action_from_difference(difference)

        feature_rows = [self._compare_feature(rule, ref, copy) for rule in rules]
        feature_rows.sort(key=lambda f: abs(f.percent_change), reverse=True)

        return AxisFeedback(
            axis=axis,
            reference_amount=50,
            copy_amount=copy_amount,
            difference=difference,
            similarity=similarity,
            action=action,
            message=self._message(axis, action),
            features=feature_rows[:5],
        )

    # ------------------------------------------------------------------
    # Delta-ratio helper
    # ------------------------------------------------------------------

    def _signed_delta(
        self,
        ref: dict[str, float],
        copy: dict[str, float],
        feature: str,
        direction: str,
    ) -> float:
        """Return the delta ratio signed so that positive = copy moved in expected direction.

        delta_ratio = (copy_val - ref_val) / (|ref_val| + ε)

        direction "increase": returned as-is   (positive delta → positive score).
        direction "decrease": sign is inverted  (negative delta → positive score).
        """
        ref_val = ref.get(feature, 0.0)
        copy_val = copy.get(feature, 0.0)
        delta_ratio = (copy_val - ref_val) / (abs(ref_val) + EPSILON)
        return delta_ratio if direction == "increase" else -delta_ratio

    # ------------------------------------------------------------------
    # Feature comparison for report output
    # ------------------------------------------------------------------

    def _compare_feature(
        self,
        rule: FeatureRule,
        ref: dict[str, float],
        copy: dict[str, float],
    ) -> FeatureComparison:
        ref_val = float(ref.get(rule.name, 0.0))
        copy_val = float(copy.get(rule.name, 0.0))
        delta = copy_val - ref_val
        pct = delta / max(abs(ref_val), EPSILON) * 100.0
        return FeatureComparison(
            feature=rule.name,
            reference_value=round(ref_val, 6),
            copy_value=round(copy_val, 6),
            delta=round(delta, 6),
            percent_change=round(pct, 2),
            expected_direction=rule.direction,
            meaning=rule.meaning,
        )

    # ------------------------------------------------------------------
    # Rules  (ordered highest-weight first = largest observed change first)
    # ------------------------------------------------------------------

    def _drive_rules(self) -> list[FeatureRule]:
        return [
            # --- increase with drive ---
            FeatureRule("amplitude_modulation", "increase", "distortion raises RMS envelope variation", weight=2.0),
            FeatureRule("rms",                  "increase", "drive raises average signal energy",        weight=1.8),
            FeatureRule("zcr",                  "increase", "clipping creates high-frequency zero crossings", weight=1.5),
            FeatureRule("spectral_bandwidth",   "increase", "harmonics widen the frequency spread",     weight=1.3),
            FeatureRule("spectral_flatness",    "increase", "distortion makes the spectrum noise-like",  weight=1.0),
            # --- decrease with drive ---
            FeatureRule("crest_factor",         "decrease", "clipping reduces peak-to-RMS ratio",       weight=1.8),
            FeatureRule("autocorrelation",      "decrease", "distortion lowers waveform periodicity",   weight=1.2),
            FeatureRule("harmonic_ratio",       "decrease", "drive adds percussive transient content",   weight=1.0),
        ]

    def _space_rules(self) -> list[FeatureRule]:
        # rms weight is overridden at runtime by sfv cross-validation (see _build_space_axis).
        return [
            # --- decrease with space ---
            FeatureRule("spectral_flux_variance",       "decrease", "reverb/delay smooths spectral-change variance",  weight=2.0),
            FeatureRule("inter_peak_interval_variance", "decrease", "delay echoes regularize onset spacing",           weight=1.5),
            FeatureRule("spectral_contrast",            "decrease", "space effects soften inter-band contrast",        weight=1.3),
            FeatureRule("spectral_flatness",            "decrease", "reverb tail levels out spectral peaks",           weight=1.0),
            FeatureRule("spectral_flux",                "decrease", "space effects reduce abrupt spectral change",     weight=1.0),
            # --- increase with space ---
            FeatureRule("energy_decay",         "increase", "space effects sustain energy in the tail",               weight=1.5),
            FeatureRule("sustain_ratio",        "increase", "reverb extends the voiced portion of the signal",        weight=1.3),
            FeatureRule("rms",                  "increase", "reverb/delay tail raises average energy (sfv-gated)",    weight=1.2),
            FeatureRule("amplitude_modulation", "increase", "echo repeats modulate the RMS envelope",                 weight=1.2),
            FeatureRule("secondary_peak_count", "increase", "echo/repeat events add onset peaks",                     weight=1.0),
        ]

    def _phase_rules(self) -> list[FeatureRule]:
        # amplitude_modulation weight is overridden at runtime by core-phase cross-validation.
        return [
            # --- increase with phase ---
            FeatureRule("modulation_energy",        "increase", "LFO creates RMS energy in 0.3–8 Hz band",      weight=2.0),
            FeatureRule("spectral_flux",            "increase", "chorus/phaser continuously sweeps the spectrum", weight=1.8),
            FeatureRule("delta_mfcc_std",           "increase", "tone colour shifts frame by frame",              weight=1.5),
            FeatureRule("centroid_modulation_depth","increase", "brightness oscillates with LFO sweep",           weight=1.3),
            FeatureRule("mfcc_modulation_variance", "increase", "cepstral coefficients vary with modulation",     weight=1.0),
            FeatureRule("amplitude_modulation",     "increase", "LFO adds amplitude wobble (core-gated)",         weight=1.0),
            # --- decrease with phase ---
            FeatureRule("harmonic_ratio",           "decrease", "comb filtering disrupts stable harmonic structure", weight=1.8),
        ]

    # ------------------------------------------------------------------
    # Action / message helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Signal-processing helpers (unchanged)
    # ------------------------------------------------------------------

    def _autocorrelation_repetition(self, audio: np.ndarray) -> float:
        if audio.size < 4:
            return 0.0
        sample = audio[: min(audio.size, 65536)]
        centered = sample - float(np.mean(sample))
        fft_size = 1 << (2 * centered.size - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size)
        corr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: centered.size]
        corr /= corr[0] + EPSILON
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
