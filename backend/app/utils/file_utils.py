from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.errors import AppError


async def copy_upload_file(file: UploadFile, target_dir: Path, max_upload_mb: int) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.wav").suffix.lower()
    target_path = target_dir / f"{uuid4().hex}{suffix}"

    written_bytes = 0
    max_bytes = max_upload_mb * 1024 * 1024

    with target_path.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written_bytes += len(chunk)
            if written_bytes > max_bytes:
                output.close()
                target_path.unlink(missing_ok=True)
                raise AppError(f"업로드 가능한 최대 파일 크기는 {max_upload_mb}MB 입니다.", status_code=413)
            output.write(chunk)

    await file.close()
    return target_path


def relative_media_url(project_root: Path, file_path: Path) -> str:
    relative = file_path.resolve().relative_to(project_root.resolve())
    return f"/media/{relative.as_posix()}"


def cleanup_expired_directories(paths: list[Path], ttl_hours: int) -> None:
    threshold = datetime.now() - timedelta(hours=ttl_hours)
    for root in paths:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.name == ".gitkeep":
                continue
            modified_time = datetime.fromtimestamp(child.stat().st_mtime)
            if modified_time < threshold:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)

