from __future__ import annotations

import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from app.core.config import Settings
from app.core.errors import AppError


class DemucsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device(settings.demucs_device)
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = get_model(self.settings.demucs_model)
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            raise AppError(
                "Demucs 모델을 불러오지 못했습니다. 설치 상태와 모델 이름을 확인해주세요.",
                status_code=500,
            ) from exc

    def separate_guitar_stem(self, source_audio_path: Path, job_dir: Path) -> tuple[Path, str]:
        output_root = job_dir / "demucs_output"
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            waveform = self._load_audio_for_demucs(source_audio_path)
            with torch.inference_mode():
                estimates = apply_model(
                    self.model,
                    waveform,
                    device=self.device,
                    progress=False,
                    split=True,
                )
            model_dir = output_root / self.settings.demucs_model / source_audio_path.stem
            model_dir.mkdir(parents=True, exist_ok=True)
            self._save_sources(model_dir, estimates)
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                f"Demucs 기타 분리에 실패했습니다. 입력 오디오 또는 모델 환경을 확인해주세요. 상세: {str(exc)[:300]}",
                status_code=500,
            ) from exc

        for stem_name in self.settings.resolved_demucs_stem_candidates:
            candidate = model_dir / f"{stem_name}.wav"
            if candidate.exists():
                final_name = f"{job_dir.name}_{stem_name}.wav"
                final_path = self.settings.separated_dir / final_name
                shutil.copy2(candidate, final_path)
                return final_path, stem_name

        available = ", ".join(sorted(path.name for path in model_dir.glob("*.wav")))
        raise AppError(
            "Demucs 결과에서 사용할 기타 stem을 찾지 못했습니다. "
            f"현재 stem 후보 설정: {self.settings.resolved_demucs_stem_candidates}, 실제 결과: {available}",
            status_code=500,
        )

    def _load_audio_for_demucs(self, audio_path: Path) -> torch.Tensor:
        # NOTE: We load audio ourselves to avoid Windows-specific torchaudio/torchcodec issues.
        waveform, _ = librosa.load(
            audio_path,
            sr=int(self.model.samplerate),
            mono=False,
        )

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform], axis=0)
        elif waveform.shape[0] == 1 and int(self.model.audio_channels) == 2:
            waveform = np.repeat(waveform, 2, axis=0)

        if waveform.shape[0] > int(self.model.audio_channels):
            waveform = waveform[: int(self.model.audio_channels), :]

        if waveform.shape[0] < int(self.model.audio_channels):
            missing_channels = int(self.model.audio_channels) - waveform.shape[0]
            pad = np.repeat(waveform[-1:, :], missing_channels, axis=0)
            waveform = np.concatenate([waveform, pad], axis=0)

        return torch.from_numpy(waveform).unsqueeze(0)

    def _save_sources(self, model_dir: Path, estimates: torch.Tensor) -> None:
        estimate_tensor = estimates[0].detach().cpu().numpy()
        for stem_name, stem_audio in zip(self.model.sources, estimate_tensor):
            output_path = model_dir / f"{stem_name}.wav"
            sf.write(output_path, stem_audio.T, int(self.model.samplerate))
