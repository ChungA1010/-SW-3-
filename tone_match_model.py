from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks


EPSILON = 1e-8

# Effects supported by the upstream classifier
_ALL_EFFECTS: list[str] = ["dist", "reverb", "chorus"]


@dataclass(frozen=True)
class FeatureRule:
    """Maps one audio feature to an effect axis.

    direction   : expected change direction when the effect intensifies.
    weight      : relative importance (higher = more influence on axis score).
    sensitivity : tanh scale factor — larger values make small deltas score strongly.
                  Tune per-feature based on typical observed delta magnitudes.
    """

    name: str
    direction: str   # "increase" | "decrease"
    meaning: str
    weight: float
    sensitivity: float = 3.0


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
    # reference_amount is always 50 — anchors the midpoint.
    # copy_amount = clip(50 + difference, 0, 100).
    reference_amount: int
    copy_amount: int
    difference: int      # positive = copy has MORE of the effect than reference
    similarity: float    # 1 - |difference| / 100
    action: str          # "turn_on" | "raise" | "keep" | "lower" | "turn_off"
    message: str
    features: list[FeatureComparison]


class ToneMatchModel:
    """Feature-based tone matcher with classifier-guided axis gating.

    The upstream effect classifier provides active_effects (e.g. ["dist"]).
    Only the axis corresponding to the active effect is scored; all other
    axes immediately return difference=0 without feature computation.

    This eliminates cross-axis contamination entirely: no inter-axis penalty
    logic is needed because the classifier already resolves ambiguity.

    Backward-compatible mode (active_effects=None or []): all axes are scored
    with pure feature-rule tanh scoring, no cross-axis gates.
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

    def compare(
        self,
        reference_path: str | Path,
        copy_path: str | Path,
        active_effects: list[str] | None = None,
    ) -> dict:
        """Compare reference vs copy audio across Drive / Space / Phase axes.

        active_effects — effect labels from the upstream classifier.
            Supported labels: "dist" | "reverb" | "chorus"
            If None or empty, all axes are scored (backward-compatible).
            Example: ["dist"] → only Drive axis is scored; Space/Phase return 0.
        """
        effects: list[str] = (
            [e.lower().strip() for e in active_effects]
            if active_effects
            else _ALL_EFFECTS
        )

        ref_audio  = self._load_audio(reference_path)
        copy_audio = self._load_audio(copy_path)
        length     = min(ref_audio.size, copy_audio.size)

        ref_feats  = self._extract_features(ref_audio[:length])
        copy_feats = self._extract_features(copy_audio[:length])

        axes = [
            self._build_drive_axis(ref_feats, copy_feats, effects),
            self._build_space_axis(ref_feats, copy_feats, effects),
            self._build_phase_axis(ref_feats, copy_feats, effects),
        ]

        total_similarity = float(np.mean([ax.similarity for ax in axes])) if axes else 0.0
        return {
            "reference_path":     str(reference_path),
            "copy_path":          str(copy_path),
            "overall_similarity": round(total_similarity, 3),
            "axes":               [asdict(ax) for ax in axes],
        }

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        audio, _ = librosa.load(Path(audio_path), sr=self.sample_rate, mono=True)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        # Peak normalization: equalizes recording gain so that rms-dependent
        # features reflect effector character, not the user's volume knob.
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0:
            audio = audio / peak

        max_samples = int(self.sample_rate * self.max_analysis_seconds)
        return audio[:max_samples] if audio.size > max_samples else audio

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, audio: np.ndarray) -> dict[str, float]:
        n_fft        = 2048
        hop          = max(128, self.hop_length)
        frame_length = min(n_fft, max(512, int(self.sample_rate * 0.064)))

        rms       = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
        zcr       = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop)[0]
        flatness  = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop)[0]
        contrast  = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop)
        mfcc      = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop)
        mel       = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop, n_mels=48)
        mel_db    = librosa.power_to_db(mel + EPSILON)

        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy   = float(np.mean(np.square(harmonic)) + EPSILON)
        percussive_energy = float(np.mean(np.square(percussive)) + EPSILON)
        signal_rms        = float(np.sqrt(np.mean(np.square(audio)) + EPSILON))
        peak              = float(np.max(np.abs(audio)) + EPSILON)

        flux_frames = (
            np.sqrt(np.mean(np.diff(mel_db, axis=1) ** 2, axis=0))
            if mel_db.shape[1] > 1 else np.zeros(1)
        )

        early            = rms[: max(1, len(rms) // 4)]
        tail             = rms[max(0, len(rms) * 3 // 4):]
        voiced_threshold = max(float(np.max(rms)) * 0.25, EPSILON)

        return {
            # Distortion (Drive axis)
            "crest_factor":           float(peak / signal_rms),
            "rms":                    signal_rms,
            "zcr":                    float(np.mean(zcr)),
            "spectral_flatness":      float(np.mean(flatness)),
            "spectral_bandwidth":     float(np.mean(bandwidth) / max(self.sample_rate / 2, EPSILON)),
            "autocorrelation":        self._autocorrelation_repetition(audio),
            "amplitude_modulation":   float(np.std(rms) / (np.mean(rms) + EPSILON)),
            # Shared (Drive + Phase)
            "harmonic_ratio":         float(harmonic_energy / (harmonic_energy + percussive_energy)),
            # Reverb (Space axis)
            "energy_decay":           float(np.mean(tail) / (np.mean(early) + EPSILON)),
            "sustain_ratio":          float(np.mean(rms > voiced_threshold)),
            "spectral_flux_variance": float(np.var(flux_frames)),
            "spectral_flux":          float(np.mean(flux_frames)),
            "spectral_contrast":      float(np.mean(contrast)),
            # Chorus (Phase axis)
            "modulation_energy":      self._modulation_energy(rms, hop),
            "delta_mfcc_std":         float(np.mean(np.std(librosa.feature.delta(mfcc), axis=1))),
            "mfcc_modulation_variance": float(np.mean(np.var(mfcc, axis=1))),
        }

    # ------------------------------------------------------------------
    # Axis builders
    # ------------------------------------------------------------------

    def _keep_feedback(self, axis: str) -> AxisFeedback:
        """Immediate 'keep' response — effect not active on this axis."""
        return AxisFeedback(
            axis=axis, reference_amount=50, copy_amount=50,
            difference=0, similarity=1.0, action="keep",
            message=self._message(axis, "keep"), features=[],
        )

    def _build_drive_axis(
        self, ref: dict[str, float], copy: dict[str, float], active_effects: list[str]
    ) -> AxisFeedback:
        """Score distortion (Drive) axis.
        Early-returns difference=0 if 'dist' is not in active_effects.
        """
        if "dist" not in active_effects:
            return self._keep_feedback("drive")
        return self._build_axis("drive", self._drive_rules(), ref, copy)

    def _build_space_axis(
        self, ref: dict[str, float], copy: dict[str, float], active_effects: list[str]
    ) -> AxisFeedback:
        """Score reverb (Space) axis.
        Early-returns difference=0 if 'reverb' is not in active_effects.
        """
        if "reverb" not in active_effects:
            return self._keep_feedback("space")
        return self._build_axis("space", self._space_rules(), ref, copy)

    def _build_phase_axis(
        self, ref: dict[str, float], copy: dict[str, float], active_effects: list[str]
    ) -> AxisFeedback:
        """Score chorus/phaser (Phase) axis.
        Early-returns difference=0 if 'chorus' is not in active_effects.
        Uses threshold=5 (half of default) because chorus changes are more subtle.
        """
        if "chorus" not in active_effects:
            return self._keep_feedback("phase")
        return self._build_axis("phase", self._phase_rules(), ref, copy, threshold=5)

    def _build_axis(
        self,
        axis: str,
        rules: list[FeatureRule],
        ref: dict[str, float],
        copy: dict[str, float],
        overrides: dict[str, float] | None = None,
        threshold: int | None = None,
    ) -> AxisFeedback:
        """Score one axis using weighted tanh-scaled delta ratios.

        signed_delta = (copy - ref) / |ref|   (sign-flipped for 'decrease' direction)
        contribution = tanh(signed_delta * sensitivity)   ∈ [-1, 1]
        axis_delta   = Σ(contribution × weight) / Σ(weight)   ∈ [-1, 1]
        difference   = round(axis_delta × 100)
        """
        if overrides is None:
            overrides = {}

        total_weight   = 0.0
        weighted_score = 0.0

        for rule in rules:
            signed       = self._signed_delta(ref, copy, rule.name, rule.direction)
            contribution = float(np.tanh(signed * rule.sensitivity))
            eff_weight   = rule.weight * overrides.get(rule.name, 1.0)
            weighted_score += contribution * eff_weight
            total_weight   += eff_weight

        axis_delta  = weighted_score / max(total_weight, EPSILON)
        difference  = int(round(axis_delta * 100))
        copy_amount = int(np.clip(50 + difference, 0, 100))
        similarity  = round(max(0.0, 1.0 - abs(difference) / 100.0), 3)
        action      = self._action_from_difference(difference, threshold)

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
        self, ref: dict[str, float], copy: dict[str, float],
        feature: str, direction: str,
    ) -> float:
        """Return delta ratio signed so that positive = copy moved in expected direction.

        delta_ratio = (copy_val - ref_val) / (|ref_val| + ε)
        direction "increase": returned as-is    (positive delta → positive score).
        direction "decrease": sign is inverted  (negative delta → positive score).
        """
        ref_val  = ref.get(feature, 0.0)
        copy_val = copy.get(feature, 0.0)
        delta    = (copy_val - ref_val) / (abs(ref_val) + EPSILON)
        return delta if direction == "increase" else -delta

    # ------------------------------------------------------------------
    # Feature comparison for report output
    # ------------------------------------------------------------------

    def _compare_feature(
        self, rule: FeatureRule, ref: dict[str, float], copy: dict[str, float],
    ) -> FeatureComparison:
        ref_val  = float(ref.get(rule.name, 0.0))
        copy_val = float(copy.get(rule.name, 0.0))
        delta    = copy_val - ref_val
        pct      = delta / max(abs(ref_val), EPSILON) * 100.0
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
    # Rules (tuned for single-effect pure scoring, no cross-axis suppression)
    # ------------------------------------------------------------------

    def _drive_rules(self) -> list[FeatureRule]:
        """Distortion fingerprints: clipping collapses crest_factor, raises RMS/ZCR/flatness."""
        return [
            # crest_factor is the single most reliable distortion indicator.
            # Clipping hard-limits peaks → RMS rises while peak stays near 1.0.
            FeatureRule("crest_factor",       "decrease", "clipping collapses peak-to-RMS ratio",           weight=3.5, sensitivity=4.0),
            FeatureRule("rms",                "increase", "drive raises average signal energy",             weight=2.0, sensitivity=4.0),
            FeatureRule("zcr",                "increase", "clipping creates high-frequency zero crossings", weight=2.0, sensitivity=4.0),
            FeatureRule("spectral_flatness",  "increase", "distortion makes the spectrum noise-like",       weight=1.5, sensitivity=4.0),
            FeatureRule("harmonic_ratio",     "decrease", "drive adds percussive transient content",        weight=1.5, sensitivity=4.0),
            FeatureRule("spectral_bandwidth", "increase", "harmonics widen the frequency spread",           weight=1.0, sensitivity=4.0),
            FeatureRule("autocorrelation",    "decrease", "distortion lowers waveform periodicity",         weight=1.0, sensitivity=4.0),
            FeatureRule("amplitude_modulation","increase","distortion raises RMS envelope variation",       weight=0.5, sensitivity=4.0),
        ]

    def _space_rules(self) -> list[FeatureRule]:
        """Reverb fingerprints: tail energy rises, spectral variation smooths out."""
        return [
            # energy_decay and sustain_ratio are the most reverb-specific features —
            # they capture the characteristic decay tail that only reverb creates.
            FeatureRule("energy_decay",          "increase", "reverb sustains energy deep in the tail",        weight=3.0, sensitivity=3.0),
            FeatureRule("sustain_ratio",         "increase", "reverb extends the voiced portion of the signal", weight=2.5, sensitivity=4.0),
            FeatureRule("spectral_flux_variance","decrease", "reverb smooths spectral-change variance over time",weight=2.0, sensitivity=3.0),
            FeatureRule("spectral_flux",         "decrease", "space effects reduce abrupt spectral change",     weight=2.0, sensitivity=4.0),
            FeatureRule("spectral_contrast",     "decrease", "space effects soften inter-band contrast",        weight=1.5, sensitivity=3.0),
            FeatureRule("rms",                   "increase", "reverb tail raises average energy",               weight=1.0, sensitivity=4.0),
        ]

    def _phase_rules(self) -> list[FeatureRule]:
        """Chorus/phaser fingerprints: LFO creates periodic modulation and spectral sweep.

        High sensitivity values (8–12) because chorus changes tend to be subtle
        (5–20% per intensity step). threshold=5 in _build_phase_axis gives enough
        headroom to detect even mild intensity transitions (e.g. 50→25).
        """
        return [
            # spectral_flux is the strongest chorus indicator — LFO continuously
            # sweeps the comb filter or all-pass delay, changing the spectrum each frame.
            FeatureRule("spectral_flux",           "increase", "chorus continuously sweeps the spectrum",        weight=5.0, sensitivity=12.0),
            FeatureRule("mfcc_modulation_variance","increase", "cepstral coefficients vary with LFO modulation", weight=3.0, sensitivity=12.0),
            FeatureRule("modulation_energy",       "increase", "LFO creates RMS energy in the 0.3–8 Hz band",    weight=3.0, sensitivity=10.0),
            FeatureRule("delta_mfcc_std",          "increase", "tone colour shifts frame by frame with the LFO", weight=2.0, sensitivity=10.0),
            FeatureRule("harmonic_ratio",          "decrease", "comb filtering disrupts stable harmonic structure",weight=2.0, sensitivity=8.0),
            FeatureRule("amplitude_modulation",    "increase", "LFO adds periodic amplitude wobble",             weight=1.0, sensitivity=8.0),
        ]

    # ------------------------------------------------------------------
    # Action / message helpers
    # ------------------------------------------------------------------

    def _action_from_difference(self, difference: int, threshold: int | None = None) -> str:
        t = threshold if threshold is not None else self.action_threshold
        if difference > 30:   return "much_lower"
        if difference < -30:  return "much_raise"
        if difference > t:    return "lower"
        if difference < -t:   return "raise"
        return "keep"

    def _message(self, axis: str, action: str) -> str:
        kr = {"drive": "드라이브", "space": "공간계", "phase": "위상계"}.get(axis, axis)
        if action == "much_raise": return f"[{kr}] 강도를 대폭 올려주세요! 🔼"
        if action == "raise":      return f"[{kr}] 강도를 살짝 올려주세요. 🔼"
        if action == "much_lower": return f"[{kr}] 강도를 대폭 줄여주세요! 🔽"
        if action == "lower":      return f"[{kr}] 강도를 살짝 줄여주세요. 🔽"
        return f"[{kr}] 아주 좋습니다. 지금 강도를 유지하세요. ✅"

    # ------------------------------------------------------------------
    # Signal-processing helpers
    # ------------------------------------------------------------------

    def _autocorrelation_repetition(self, audio: np.ndarray) -> float:
        if audio.size < 4:
            return 0.0
        sample   = audio[: min(audio.size, 65536)]
        centered = sample - float(np.mean(sample))
        fft_size = 1 << (2 * centered.size - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size)
        corr     = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: centered.size]
        corr    /= corr[0] + EPSILON
        search   = corr[min(64, len(corr) - 1) : min(len(corr), 4096)]
        return float(np.max(search)) if search.size else 0.0

    def _modulation_energy(self, rms: np.ndarray, hop_length: int) -> float:
        if rms.size < 4:
            return 0.0
        centered = rms - float(np.mean(rms))
        spectrum = np.abs(np.fft.rfft(centered))
        freqs    = np.fft.rfftfreq(rms.size, d=hop_length / self.sample_rate)
        band     = (freqs >= 0.3) & (freqs <= 8.0)
        return float(np.sum(spectrum[band]) / (np.sum(spectrum) + EPSILON))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reference and copied guitar effector tones.")
    parser.add_argument("reference_path")
    parser.add_argument("copy_path")
    parser.add_argument("--sample-rate",    type=int,  default=32000)
    parser.add_argument("--hop-length",     type=int,  default=256)
    parser.add_argument("--active-effects", nargs="+", default=None,
                        help="Active effect labels from classifier (e.g. --active-effects dist)")
    args = parser.parse_args()

    model  = ToneMatchModel(sample_rate=args.sample_rate, hop_length=args.hop_length)
    result = model.compare(args.reference_path, args.copy_path,
                           active_effects=args.active_effects)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
