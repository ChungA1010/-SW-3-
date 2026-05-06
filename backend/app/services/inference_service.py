from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.core.config import Settings
from app.core.errors import AppError
from app.models.gru_model import SequenceGRU
from app.schemas.analyze import ScoreItem, TimelineSegment
from app.services.audio_preprocess import (
    load_mono_audio,
    sliding_window_waveforms,
    waveform_to_sequence_tensor,
)


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cpu")
        self.model, self.training_config = self._load_model()

    def _load_model(self) -> tuple[SequenceGRU, dict]:
        weights_path = self.settings.weights_path
        if not weights_path.exists():
            raise AppError(
                f"GRU weight 파일을 찾을 수 없습니다: {weights_path}",
                status_code=500,
            )

        artifact = torch.load(weights_path, map_location=self.device)
        model = SequenceGRU(**artifact["model_config"])
        model.load_state_dict(artifact["model_state_dict"])
        model.eval()
        return model, artifact["training_config"]

    def predict(self, audio_path: Path) -> dict:
        # NOTE: These defaults fall back to .env values, but the checkpoint's
        # training_config is preferred so inference stays aligned with training.
        sample_rate = int(self.training_config.get("sample_rate", self.settings.model_sample_rate))
        segment_seconds = float(self.training_config.get("segment_seconds", self.settings.model_segment_seconds))
        n_mfcc = int(self.training_config.get("n_mfcc", self.settings.model_n_mfcc))
        hop_length = int(self.training_config.get("hop_length", self.settings.model_hop_length))

        waveform = load_mono_audio(audio_path, sample_rate=sample_rate)
        audio_duration_sec = float(len(waveform) / sample_rate)
        window_size = int(segment_seconds * sample_rate)
        hop_size = max(window_size // 2, 1)
        windows = sliding_window_waveforms(waveform, window_size=window_size, hop_size=hop_size)

        all_scores: list[np.ndarray] = []
        timeline_windows: list[dict] = []
        with torch.inference_mode():
            for index, window in enumerate(windows):
                sequence = waveform_to_sequence_tensor(
                    waveform=window,
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    hop_length=hop_length,
                ).unsqueeze(0).to(self.device)
                logits = self.model(sequence)
                scores = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                all_scores.append(scores)

                start_sample = min(index * hop_size, max(len(waveform) - window_size, 0))
                end_sample = min(start_sample + window_size, len(waveform))
                timeline_windows.append(
                    self._build_window_prediction(
                        fine_scores=scores,
                        start_sec=start_sample / sample_rate,
                        end_sec=end_sample / sample_rate,
                    )
                )

        mean_scores = np.mean(np.asarray(all_scores, dtype=np.float32), axis=0)
        prediction = self._prediction_dict_from_fine_scores(mean_scores)
        prediction["timeline_segments"] = self._merge_timeline_windows(timeline_windows)
        prediction["audio_duration_sec"] = round(audio_duration_sec, 3)
        prediction["analysis_window_sec"] = round(window_size / sample_rate, 3)
        prediction["analysis_hop_sec"] = round(hop_size / sample_rate, 3)
        return prediction

    def _prediction_dict_from_fine_scores(self, fine_scores: np.ndarray) -> dict:
        fine_index = int(np.argmax(fine_scores))
        predicted_fine_label = self.settings.fine_class_names[fine_index]
        coarse_scores = {label: 0.0 for label in self.settings.coarse_class_names}

        fine_score_items: list[ScoreItem] = []
        for idx, label in enumerate(self.settings.fine_class_names):
            score = float(fine_scores[idx])
            coarse_scores[self.settings.fine_to_coarse[label]] += score
            fine_score_items.append(
                ScoreItem(
                    label=label,
                    display_name=self.settings.label_display_names.get(label, label),
                    score=round(score, 6),
                )
            )

        family_score_items = [
            ScoreItem(
                label=label,
                display_name=self.settings.label_display_names.get(label, label),
                score=round(score, 6),
            )
            for label, score in sorted(coarse_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        fine_score_items.sort(key=lambda item: item.score, reverse=True)
        predicted_family = max(coarse_scores, key=coarse_scores.get)

        return {
            "predicted_effect": predicted_fine_label,
            "predicted_effect_display_name": self.settings.label_display_names.get(predicted_fine_label, predicted_fine_label),
            "predicted_family": predicted_family,
            "predicted_family_display_name": self.settings.label_display_names.get(predicted_family, predicted_family),
            "fine_scores": fine_score_items,
            "family_scores": family_score_items,
        }

    def _build_window_prediction(self, fine_scores: np.ndarray, start_sec: float, end_sec: float) -> dict:
        prediction = self._prediction_dict_from_fine_scores(fine_scores)
        top_score = next(
            (item.score for item in prediction["fine_scores"] if item.label == prediction["predicted_effect"]),
            0.0,
        )
        return {
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "effect_label": prediction["predicted_effect"],
            "effect_display_name": prediction["predicted_effect_display_name"],
            "family_label": prediction["predicted_family"],
            "family_display_name": prediction["predicted_family_display_name"],
            "confidence": round(float(top_score), 6),
        }

    def _merge_timeline_windows(self, windows: list[dict]) -> list[TimelineSegment]:
        if not windows:
            return []

        merged: list[dict] = [windows[0].copy()]
        for current in windows[1:]:
            previous = merged[-1]
            same_effect = previous["effect_label"] == current["effect_label"]
            same_family = previous["family_label"] == current["family_label"]
            adjacent = current["start_sec"] <= previous["end_sec"] + 0.001

            if same_effect and same_family and adjacent:
                previous["end_sec"] = current["end_sec"]
                previous["confidence"] = round(max(previous["confidence"], current["confidence"]), 6)
            else:
                merged.append(current.copy())

        return [
            TimelineSegment(
                start_sec=round(item["start_sec"], 3),
                end_sec=round(item["end_sec"], 3),
                duration_sec=round(item["end_sec"] - item["start_sec"], 3),
                effect_label=item["effect_label"],
                effect_display_name=item["effect_display_name"],
                family_label=item["family_label"],
                family_display_name=item["family_display_name"],
                confidence=round(item["confidence"], 6),
            )
            for item in merged
        ]
