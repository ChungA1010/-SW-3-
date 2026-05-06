from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError

YOUTUBE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)[\w\-]{6,}$",
    re.IGNORECASE,
)


def validate_uploaded_audio(filename: str | None, content_type: str | None, max_upload_mb: int) -> None:
    settings = get_settings()
    if not filename:
        raise AppError("업로드 파일명이 비어 있습니다.", status_code=400)

    extension = Path(filename).suffix.lower()
    if extension not in settings.supported_extensions:
        raise AppError(
            "지원하지 않는 오디오 형식입니다. wav, mp3, flac, m4a, ogg, aac 파일만 업로드해주세요.",
            status_code=400,
        )

    if content_type and content_type not in settings.allowed_content_types:
        raise AppError("지원하지 않는 파일 타입입니다.", status_code=400)

    if max_upload_mb <= 0:
        raise AppError("서버 업로드 설정이 올바르지 않습니다.", status_code=500)


def validate_youtube_url(url: str) -> None:
    if not YOUTUBE_PATTERN.match(url.strip()):
        raise AppError("유효한 유튜브 링크 형식이 아닙니다.", status_code=400)

