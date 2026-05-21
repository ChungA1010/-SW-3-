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
    action: str          # "turn_on" | "raise" | "keep" | "lower" | "turn_off"
    message: str
    features: list[FeatureComparison]


class ToneMatchModel:
    """Feature-based tone matcher for reference/copy effector comparison.

    Computes how much Drive, Space, and Phase effect is present in the copy
    *relative to the reference* using tanh-scaled delta-ratio scoring.
    Hard-coded absolute bounds are replaced with data-adaptive relative-change
    measurements so scores are invariant to recording level.

    Cross-validation guards prevent shared features from double-counting:

    - Drive axis [FIX 3] : rms / spectral_flatness / zcr are gated by phase_factor
                   (derived from spectral_flux increase) to prevent phase
                   cancellation energy loss from being misread as "drive too weak".

    - Space axis [FIX 1] : rms is gated by spectral_flux_variance confirmation.
                   spectral_flux_variance / spectral_flatness / spectral_flux
                   are additionally penalised when drive core confidence is high,
                   preventing distortion noise-fill from being misread as reverb.

    - Phase axis           : amplitude_modulation is gated by core phase indicators
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
        """Build drive axis with Phase → Drive cross-validation.  [FIX 3]

        문제: 코러스/페이저의 위상 상쇄(Phase Cancellation)로 카피본의 rms와
        spectral_flatness가 낮아지고 zcr이 변한다. 단순 drive 스코어러는
        이를 "드라이브가 원본보다 약함 → Raise drive"로 오판한다.

        해결: 위상계의 핵심 지문인 spectral_flux 증가에서 phase_factor를 계산.
        spectral_flux는 코러스/페이저의 스펙트럼 스위핑에 가장 예민하게
        반응하면서, 드라이브 효과와는 혼동되지 않는 최적의 판별자다.
        phase_factor가 높을수록 아래 세 피처의 가중치를 비율적으로 억제:
          rms, spectral_flatness, zcr → weight *= (1.0 - phase_factor)

        phase_factor = clip(tanh(sf_signed * sensitivity), 0, 1)
          - spectral_flux 증가 없음 → phase_factor ≈ 0 → 억제 없음, 전체 가중치 유지
          - spectral_flux 명확히 증가 → phase_factor → 1 → 세 피처 가중치 → 0
        """
        # ── Space → Drive 누설 방어 (리버브/딜레이 추가·제거 양방향) ─────────────────
        # sfv(spectral_flux_variance)만 사용: 리버브는 sfv를 급격히 낮추지만
        # 드라이브/코러스는 sfv를 거의 바꾸지 않는다는 물리적 차이를 이용.
        # sensitivity=4.0: sfv 20% 변화(인접 단계 reverb_25→50)도 factor≈1.0으로 완전 차단.
        # energy_decay·sustain은 드라이브 압축 효과로도 변하므로 제외.
        sfv_increase = self._signed_delta(ref, copy, "spectral_flux_variance", "increase")
        sfv_decrease = self._signed_delta(ref, copy, "spectral_flux_variance", "decrease")

        # 리버브 걷힘: sfv↑
        space_unsmear_factor = float(np.clip(
            np.tanh(sfv_increase * 4.0) * 1.5, 0.0, 1.0
        ))
        # 리버브 추가: sfv↓
        space_smear_factor = float(np.clip(
            np.tanh(sfv_decrease * 4.0) * 1.5, 0.0, 1.0
        ))

        # ── Phase → Drive 누설 방어 (코러스 추가·제거 양방향) ─────────────────────
        # 코러스 정방향: mfcc_var 증가 = LFO 변조 깊어짐.
        # sensitivity × 3.0 = 6.0: mfcc_var 13.4% 이상 변화 시 factor=1.0 → 완전 차단.
        # (인접 단계 chorus 50→75에서 mfcc 15% 변화도 차단)
        mfcc_mod_increase = self._signed_delta(ref, copy, "mfcc_modulation_variance", "increase")
        phase_interference_factor = float(np.clip(
            np.tanh(mfcc_mod_increase * DEFAULT_SENSITIVITY * 3.0) * 1.5, 0.0, 1.0
        ))

        # 코러스 역방향: mfcc_var 감소 = 코러스 LFO 변조 약해짐.
        # space_unsmear_factor로 게이팅: 코러스 역방향은 sfv↑, 드라이브 감소는 sfv 불변.
        mfcc_mod_decrease = self._signed_delta(ref, copy, "mfcc_modulation_variance", "decrease")
        phase_leaving_mfcc = float(np.clip(
            np.tanh(mfcc_mod_decrease * DEFAULT_SENSITIVITY * 3.0) * 1.5, 0.0, 1.0
        ))
        phase_leaving_factor = phase_leaving_mfcc * float(np.clip(space_unsmear_factor * 3.0, 0.0, 1.0))

        # 네 요인 중 하나라도 강하면 드라이브 피처 전체 억제 (균일 scale=0 → axis_delta=0)
        drive_penalty_scale = 1.0 - max(
            space_smear_factor, space_unsmear_factor,
            phase_interference_factor, phase_leaving_factor,
        )

        return self._build_axis(
            "drive", self._drive_rules(), ref, copy,
            overrides={
                "rms":                  drive_penalty_scale,
                "spectral_flatness":    drive_penalty_scale,
                "zcr":                  drive_penalty_scale,
                "amplitude_modulation": drive_penalty_scale,
                "spectral_bandwidth":   drive_penalty_scale,
                "autocorrelation":      drive_penalty_scale,
                "harmonic_ratio":       drive_penalty_scale,
                # 코러스 제거 시 crest_factor(다이내믹 복원)도 드라이브 감소로 오판하므로 차단
                "crest_factor":         drive_penalty_scale,
            },
        )

    def _build_space_axis(
        self, ref: dict[str, float], copy: dict[str, float]
    ) -> AxisFeedback:
        """Build space axis with rms cross-validation and Drive → Space cross-validation.

        [기존] rms 교차 검증:
        문제: rms는 드라이브가 걸려도 올라가므로, 단순 스코어러는 헤비 드라이브
        카피본을 "스페이스가 더 많다"고 오판한다.
        해결: sfv(spectral_flux_variance)가 space 방향(감소)으로 움직일 때만
        rms 기여를 허용. sfv는 드라이브에 무관한 신뢰할 수 있는 space 판별자다.

        rms_scale = clip(tanh(sfv_signed * sensitivity), 0, 1)
          - sfv 변화 없거나 반대 방향 → rms_scale ≈ 0 → rms 기여 없음
          - sfv 명확히 감소(space)    → rms_scale → 1 → rms 전체 가중치 적용

        [FIX 1] Drive → Space 교차 검증:
        문제: 강한 디스토션은 파형을 압축하고 조용한 구간을 광대역 노이즈로
        채운다. 이로 인해 spectral_flux_variance가 감소(노이즈는 스펙트럼적으로
        균일하고 시간적으로 안정적)한다. space 스코어러는 이 압축 아티팩트를
        리버브의 스무딩으로 착각해 "space를 줄여라"는 오판을 내린다.

        해결: 리버브/딜레이가 만들어낼 수 없는 드라이브 전용 피처 3종으로
        drive_core_factor를 계산:
          crest_factor (decrease) — 클리핑이 peak-to-RMS 비율을 낮춤
          autocorrelation (decrease) — 노이즈가 파형 주기성을 낮춤
          zcr (increase) — 노이즈가 고주파 영교차를 증가시킴
        drive 확신도가 높으면, 드라이브가 흉내 낼 수 있는 space 피처들을 억제:
          spectral_flux_variance, spectral_flatness, spectral_flux

        drive_core_factor = clip(mean(tanh(drive_signals)), 0, 1)
          - drive 지표 조용함         → factor ≈ 0 → 억제 없음, 전체 가중치
          - drive 지표 명확히 반응    → factor → 1 → sfv/flatness/flux 가중치 → 0
        """
        # --- 기존: rms 게이트 (로직 동일) ---
        sfv_signed = self._signed_delta(ref, copy, "spectral_flux_variance", "decrease")
        rms_scale = float(np.clip(np.tanh(sfv_signed * DEFAULT_SENSITIVITY), 0.0, 1.0))

        # --- FIX 1: drive 교차 검증 ---
        # 리버브/딜레이가 만들어낼 수 없는 드라이브 전용 피처 3종으로 확신도 계산.
        drive_core_signals = [
            np.tanh(self._signed_delta(ref, copy, "crest_factor",    "decrease") * DEFAULT_SENSITIVITY),
            np.tanh(self._signed_delta(ref, copy, "autocorrelation", "decrease") * DEFAULT_SENSITIVITY),
            np.tanh(self._signed_delta(ref, copy, "zcr",             "increase") * DEFAULT_SENSITIVITY),
        ]
        drive_core_factor = float(np.clip(np.mean(drive_core_signals), 0.0, 1.0))

        # drive 확신도가 높을수록 "drive가 흉내 낼 수 있는" space 피처를 억제.
        drive_penalty_scale = 1.0 - drive_core_factor

        # 1. 코러스가 폭주할 때 (정방향 방어): 배음 파괴 + 모듈레이션 증가
        hr_decrease = self._signed_delta(ref, copy, "harmonic_ratio", "decrease")
        mfcc_mod_increase = self._signed_delta(ref, copy, "mfcc_modulation_variance", "increase")
        phase_explosion_factor = float(np.clip(np.tanh(max(hr_decrease, mfcc_mod_increase) * 2.0) * 1.5, 0.0, 1.0))

        # 2. 코러스가 확 빠질 때 (역방향 방어): 배음 복원 + 모듈레이션 감소
        hr_increase = self._signed_delta(ref, copy, "harmonic_ratio", "increase")
        mfcc_mod_decrease = self._signed_delta(ref, copy, "mfcc_modulation_variance", "decrease")
        phase_leaving_factor = float(np.clip(np.tanh(max(hr_increase, mfcc_mod_decrease) * 2.0) * 1.5, 0.0, 1.0))

        # 코러스가 켜지든 꺼지든 공간계가 간섭받지 않도록 양방향 제어
        space_penalty_scale = 1.0 - max(phase_explosion_factor, phase_leaving_factor)

        return self._build_axis(
            "space", self._space_rules(), ref, copy,
            overrides={
                "spectral_flux_variance": space_penalty_scale,
                "spectral_contrast":      space_penalty_scale,
                "spectral_flatness":      space_penalty_scale,
                "spectral_flux":          space_penalty_scale,
            },
        )

    def _build_phase_axis(
        self, ref: dict[str, float], copy: dict[str, float]
    ) -> AxisFeedback:
        """Build phase axis with amplitude_modulation cross-validation.

        문제: amplitude_modulation은 드라이브와 스페이스에도 반응한다.
        무조건 계산에 포함하면 비위상계 이펙트의 phase 점수가 부풀어 오른다.

        해결: 코러스/페이저/플랜저에 가장 특화된 3개 지표로 'core phase score'를 계산:
          modulation_energy (LFO 주파수 대역 에너지 상승)
          spectral_flux (스펙트럼 스위핑에 의한 변화량 상승)
          harmonic_ratio (콤 필터링에 의한 배음 안정성 하락)
        이 core 신호들이 집합적으로 양수일 때만 amplitude_modulation을 인정.

        am_scale = clip(mean(tanh(core_signals)), 0, 1)
          - core 지표 반응 없음 (mean ≤ 0) → am_scale = 0 → am 기여 없음
          - core 지표 강하게 반응           → am_scale → 1 → am 전체 가중치 적용

        [FIX 2b] spectral_flux 가중치 1.8 → 2.5로 상향.
        코러스의 핵심 지문인 지속적 스펙트럼 스위핑에 훨씬 예민하게 반응하도록 튜닝.
        """
        core_signals = [
            np.tanh(self._signed_delta(ref, copy, "modulation_energy", "increase") * DEFAULT_SENSITIVITY),
            np.tanh(self._signed_delta(ref, copy, "spectral_flux",     "increase") * DEFAULT_SENSITIVITY),
            np.tanh(self._signed_delta(ref, copy, "harmonic_ratio",    "decrease") * DEFAULT_SENSITIVITY),
        ]
        core_mean = float(np.mean(core_signals))  # ∈ [-1, 1]
        am_scale = float(np.clip(core_mean, 0.0, 1.0))

        # ── Space → Phase 누설 방어 (리버브 추가·제거 양방향) ─────────────────────
        # OR 게이트: sfv + energy_decay + sustain_ratio → 리버브(꼬리)와 코러스(LFO) 구분.
        # energy_decay·sustain은 리버브 전용 지문: 코러스 테스트에서 오탐 없음.
        # sensitivity=4.0: sfv 20% 변화(인접 단계 reverb)도 factor≈1.0 → 완전 차단.
        sfv_decrease = self._signed_delta(ref, copy, "spectral_flux_variance", "decrease")
        sfv_increase = self._signed_delta(ref, copy, "spectral_flux_variance", "increase")
        ed_increase  = self._signed_delta(ref, copy, "energy_decay",           "increase")
        ed_decrease  = self._signed_delta(ref, copy, "energy_decay",           "decrease")
        sr_increase  = self._signed_delta(ref, copy, "sustain_ratio",          "increase")
        sr_decrease  = self._signed_delta(ref, copy, "sustain_ratio",          "decrease")

        # 리버브 추가: sfv↓ OR energy_decay↑ OR sustain↑
        space_smear_factor   = float(np.clip(
            np.tanh(max(sfv_decrease, ed_increase, sr_increase) * 4.0) * 1.5, 0.0, 1.0
        ))
        # 리버브 걷힘: sfv↑ OR energy_decay↓ OR sustain↓
        space_unsmear_factor = float(np.clip(
            np.tanh(max(sfv_increase, ed_decrease, sr_decrease) * 4.0) * 1.5, 0.0, 1.0
        ))

        space_penalty_scale = 1.0 - max(space_smear_factor, space_unsmear_factor)

        # --- FIX 5: 드라이브 변화 → 위상계 오판 방어막 ---
        # crest_factor = 드라이브 클리핑의 직접 지문. 코러스/리버브는 crest_factor에 거의 영향 없음.
        # 양방향 감지: drive 추가(crest↓), drive 제거(crest↑) 모두 차단.
        crest_forward = self._signed_delta(ref, copy, "crest_factor", "decrease")  # drive 추가
        crest_reverse = self._signed_delta(ref, copy, "crest_factor", "increase")  # drive 제거
        drive_intrusion_factor = float(np.clip(
            np.tanh(max(crest_forward, crest_reverse) * DEFAULT_SENSITIVITY * 3.0) * 1.5,
            0.0, 1.0,
        ))
        # 선택적 억제: drive에 오염되는 피처만 억제, mfcc/delta/centroid는 앵커로 유지.
        # 코러스 고유 피처(mfcc_modulation_variance 등)는 full weight 유지 → 코러스 감지 보존.
        drive_penalty_phase = 1.0 - drive_intrusion_factor

        # drive_penalty_phase를 모든 피처에 균일 적용:
        #   drive_penalty_phase=0  → 전체 weight=0 → axis_delta=0 → diff=0 (완전 차단)
        #   drive_penalty_phase=k>0 → 분자/분모 모두 k 곱 → 상쇄, 방향 보존 (코러스 감지 무손실)
        return self._build_axis(
            "phase", self._phase_rules(), ref, copy,
            overrides={
                "spectral_flux":             space_penalty_scale * drive_penalty_phase,
                "modulation_energy":         space_penalty_scale * drive_penalty_phase,
                "harmonic_ratio":            space_penalty_scale * drive_penalty_phase,
                "amplitude_modulation":      am_scale * space_penalty_scale * drive_penalty_phase,
                "delta_mfcc_std":            space_penalty_scale * drive_penalty_phase,
                "centroid_modulation_depth": space_penalty_scale * drive_penalty_phase,
                "mfcc_modulation_variance":  space_penalty_scale * drive_penalty_phase,
            },
            threshold=5,
        )

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
        action = self._action_from_difference(difference, threshold)

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
        # rms / spectral_flatness / zcr: phase 교차 검증으로 런타임에 오버라이드됨
        # (위상 상쇄 감지 시 phase_suppressed_scale = 1 - phase_factor 적용)
        return [
            # 리버브가 걷힐 때 발생하는 선명함에 속지 않도록 sensitivity를 4.0으로 상향
            FeatureRule("amplitude_modulation", "increase", "distortion raises RMS envelope variation", weight=2.0, sensitivity=4.0),
            FeatureRule("rms",                  "increase", "drive raises average signal energy",        weight=1.8, sensitivity=4.0),
            FeatureRule("zcr",                  "increase", "clipping creates high-frequency zero crossings", weight=1.5, sensitivity=4.0),
            FeatureRule("spectral_bandwidth",   "increase", "harmonics widen the frequency spread",      weight=1.3, sensitivity=4.0),
            FeatureRule("spectral_flatness",    "increase", "distortion makes the spectrum noise-like",  weight=1.0, sensitivity=4.0),
            # --- decrease with drive ---
            FeatureRule("crest_factor",         "decrease", "clipping reduces peak-to-RMS ratio",        weight=1.8, sensitivity=4.0),
            FeatureRule("autocorrelation",      "decrease", "distortion lowers waveform periodicity",   weight=1.2, sensitivity=4.0),
            FeatureRule("harmonic_ratio",       "decrease", "drive adds percussive transient content",   weight=1.0, sensitivity=4.0),
        ]

    def _space_rules(self) -> list[FeatureRule]:
        # rms: sfv 교차 검증 × drive 교차 검증 (곱산 AND 게이트)으로 런타임 오버라이드됨.
        # spectral_flux_variance / spectral_flatness / spectral_flux:
        #   drive 교차 검증으로 runtime에 drive_penalty_scale 오버라이드됨.
        return [
            # --- 확실하게 점수를 밀어줄 핵심 지표들 (가중치 상향) ---
            FeatureRule("spectral_flux_variance", "decrease", "reverb/delay smooths spectral-change variance", weight=3.0),
            FeatureRule("spectral_flux",          "decrease", "space effects reduce abrupt spectral change",    weight=2.5),
            
            # --- 나머지 피처들은 보조 역할 (가중치 하향) ---
            FeatureRule("spectral_contrast",      "decrease", "space effects soften inter-band contrast",       weight=1.0),
            FeatureRule("spectral_flatness",      "decrease", "reverb tail levels out spectral peaks",          weight=0.8),
            FeatureRule("energy_decay",           "increase", "space effects sustain energy in the tail",       weight=1.0),
            FeatureRule("sustain_ratio",          "increase", "reverb extends the voiced portion of the signal", weight=0.8),
            FeatureRule("rms",                    "increase", "reverb/delay tail raises average energy",        weight=0.5), # rms는 변동성이 심하니 약하게
            FeatureRule("amplitude_modulation",   "increase", "echo repeats modulate the RMS envelope",         weight=0.5),
        ]

    def _phase_rules(self) -> list[FeatureRule]:
        # amplitude_modulation: core-phase 교차 검증으로 런타임에 오버라이드됨.
        # [FIX 2b] spectral_flux 가중치 1.8 → 2.5 상향: 코러스의 핵심 지문인
        # 지속적 스펙트럼 스위핑에 훨씬 예민하게 반응하도록 튜닝.
        return [
            # --- increase with phase ---
            # sensitivity 상향: 코러스 25% 같은 미세 변화(delta 5~12%)에 확실히 반응하도록
            FeatureRule("spectral_flux",             "increase", "chorus/phaser continuously sweeps the spectrum",   weight=6.0, sensitivity=6.0),
            FeatureRule("modulation_energy",         "increase", "LFO creates RMS energy in 0.3–8 Hz band",         weight=4.0, sensitivity=5.0),
            FeatureRule("delta_mfcc_std",            "increase", "tone colour shifts frame by frame",                weight=1.5),
            FeatureRule("centroid_modulation_depth", "increase", "brightness oscillates with LFO sweep",             weight=1.3),
            FeatureRule("mfcc_modulation_variance",  "increase", "cepstral coefficients vary with modulation",       weight=1.0),
            FeatureRule("amplitude_modulation",      "increase", "LFO adds amplitude wobble (core-gated)",           weight=1.0),
            # --- decrease with phase ---
            FeatureRule("harmonic_ratio",            "decrease", "comb filtering disrupts stable harmonic structure", weight=1.8),
        ]

    # ------------------------------------------------------------------
    # Action / message helpers
    # ------------------------------------------------------------------

    def _action_from_difference(self, difference: int, threshold: int | None = None) -> str:
        t = threshold if threshold is not None else self.action_threshold
        if difference > 30:   return "turn_off"
        if difference < -30:  return "turn_on"
        if difference > t:    return "lower"
        if difference < -t:   return "raise"
        return "keep"

    def _message(self, axis: str, action: str) -> str:
        axis_kr = {"drive": "드라이브", "space": "공간계", "phase": "위상계"}.get(axis, axis)
        if action == "turn_on":  return f"[{axis_kr}] 원본에 비해 너무 약하거나 꺼져 있습니다. 켜주세요! 🟢"
        if action == "raise":    return f"[{axis_kr}] 원본보다 약합니다. 강도를 더 올려주세요. 🔼"
        if action == "lower":    return f"[{axis_kr}] 원본보다 셉니다. 강도를 줄여주세요. 🔽"
        if action == "turn_off": return f"[{axis_kr}] 원본에 비해 너무 과하게 걸려 있습니다. 끄거나 대폭 줄여주세요! 🔴"
        return f"[{axis_kr}] 완벽합니다! 지금 톤을 그대로 유지하세요. 🎵"

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
