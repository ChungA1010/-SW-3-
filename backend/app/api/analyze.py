from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.errors import AppError
from app.schemas.analyze import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
)
from app.services.pipeline_service import AnalysisPipelineService
from app.utils.file_utils import cleanup_expired_directories
from app.utils.validators import validate_uploaded_audio

router = APIRouter()
settings = get_settings()
pipeline = AnalysisPipelineService(settings=settings)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        success=True,
        status="ok",
        model_weights_exists=settings.weights_path.exists(),
        demucs_model=settings.demucs_model,
    )


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_uploaded_file(file: UploadFile = File(...)) -> AnalysisResponse:
    cleanup_expired_directories(settings.transient_roots, settings.temp_file_ttl_hours)

    try:
        validate_uploaded_audio(file.filename, file.content_type, settings.max_upload_mb)
        start = perf_counter()
        result = await pipeline.analyze_uploaded_file(file)
        result.processing_time_sec = round(perf_counter() - start, 2)
        return result
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.user_message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="파일 분석 중 예기치 못한 오류가 발생했습니다.") from exc


@router.post("/analyze/youtube", response_model=AnalysisResponse)
async def analyze_youtube(payload: AnalysisRequest) -> AnalysisResponse:
    cleanup_expired_directories(settings.transient_roots, settings.temp_file_ttl_hours)

    try:
        start = perf_counter()
        result = await pipeline.analyze_youtube_url(payload.youtube_url)
        result.processing_time_sec = round(perf_counter() - start, 2)
        return result
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.user_message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="유튜브 링크 분석 중 예기치 못한 오류가 발생했습니다.") from exc


@router.get("/files/{relative_path:path}")
def get_media_hint(relative_path: str) -> dict[str, str]:
    path = Path(relative_path)
    return {"url": f"/media/{path.as_posix()}"}

