from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.schemas.analyze import AnalysisResponse
from app.services.demucs_service import DemucsService
from app.services.inference_service import InferenceService
from app.services.youtube_service import YouTubeService
from app.utils.file_utils import copy_upload_file, relative_media_url
from app.utils.validators import validate_youtube_url


class AnalysisPipelineService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.youtube_service = YouTubeService()
        self.demucs_service = DemucsService(settings)
        self.inference_service = InferenceService(settings)

    async def analyze_uploaded_file(self, file: UploadFile) -> AnalysisResponse:
        job_id = uuid4().hex
        source_path = await copy_upload_file(
            file=file,
            target_dir=self.settings.uploads_dir / job_id,
            max_upload_mb=self.settings.max_upload_mb,
        )
        return self._run_pipeline(
            source_audio_path=source_path,
            source_type="file",
            source_name=file.filename or source_path.name,
            job_id=job_id,
        )

    async def analyze_youtube_url(self, youtube_url: str) -> AnalysisResponse:
        validate_youtube_url(str(youtube_url))
        job_id = uuid4().hex
        source_path, title = self.youtube_service.download_audio(
            youtube_url=str(youtube_url),
            output_dir=self.settings.downloads_dir / job_id,
        )
        return self._run_pipeline(
            source_audio_path=source_path,
            source_type="youtube",
            source_name=title,
            job_id=job_id,
        )

    def _run_pipeline(
        self,
        source_audio_path: Path,
        source_type: str,
        source_name: str,
        job_id: str,
    ) -> AnalysisResponse:
        job_dir = self.settings.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        guitar_stem_path, stem_name = self.demucs_service.separate_guitar_stem(source_audio_path, job_dir)
        prediction = self.inference_service.predict(guitar_stem_path)

        return AnalysisResponse(
            success=True,
            source_type=source_type,
            source_name=source_name,
            predicted_effect=prediction["predicted_effect"],
            predicted_effect_display_name=prediction["predicted_effect_display_name"],
            predicted_family=prediction["predicted_family"],
            predicted_family_display_name=prediction["predicted_family_display_name"],
            fine_scores=prediction["fine_scores"],
            family_scores=prediction["family_scores"],
            timeline_segments=prediction["timeline_segments"],
            audio_duration_sec=prediction["audio_duration_sec"],
            analysis_window_sec=prediction["analysis_window_sec"],
            analysis_hop_sec=prediction["analysis_hop_sec"],
            guitar_stem_url=relative_media_url(self.settings.project_root, guitar_stem_path),
            original_audio_url=relative_media_url(self.settings.project_root, source_audio_path),
            processing_time_sec=0.0,
            demucs_stem_used=stem_name,
            model_name="gru",
        )
