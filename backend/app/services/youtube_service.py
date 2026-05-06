from __future__ import annotations

from pathlib import Path

from yt_dlp import DownloadError, YoutubeDL

from app.core.errors import AppError


class YouTubeService:
    def download_audio(self, youtube_url: str, output_dir: Path) -> tuple[Path, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "%(title)s.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "paths": {"home": str(output_dir)},
            "noplaylist": True,
            "extractaudio": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                title = info.get("title") or "youtube_audio"
                requested = ydl.prepare_filename(info)
                downloaded_path = Path(requested).with_suffix(".wav")
        except DownloadError as exc:
            message = str(exc)
            if "Private video" in message:
                raise AppError("비공개 영상은 분석할 수 없습니다.", status_code=400) from exc
            if "Sign in to confirm your age" in message:
                raise AppError("연령 제한 영상은 현재 분석할 수 없습니다.", status_code=400) from exc
            if "This video is unavailable" in message:
                raise AppError("현재 접근할 수 없는 유튜브 영상입니다.", status_code=400) from exc
            raise AppError(
                "유튜브 오디오 다운로드에 실패했습니다. 링크가 유효한지, 공개 접근 가능한지 확인해주세요.",
                status_code=400,
            ) from exc
        except Exception as exc:
            raise AppError("유튜브 링크 처리 중 오류가 발생했습니다.", status_code=400) from exc

        if not downloaded_path.exists():
            candidates = sorted(output_dir.glob("*.wav"))
            if not candidates:
                raise AppError("유튜브 오디오 추출 결과를 찾지 못했습니다.", status_code=500)
            downloaded_path = candidates[-1]

        return downloaded_path, title

