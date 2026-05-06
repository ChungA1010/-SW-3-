from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Effect Analyzer API"
    app_env: str = "development"
    api_prefix: str = "/api"
    frontend_origin: str = "http://localhost:3000"
    max_upload_mb: int = 80
    temp_file_ttl_hours: int = 12

    project_root: Path | None = None
    weights_path: Path | None = None

    demucs_model: str = "htdemucs"
    demucs_stem_candidates: str = "guitar,other"
    demucs_device: str = "cpu"

    model_sample_rate: int = 32000
    model_segment_seconds: float = 3.0
    model_n_mfcc: int = 20
    model_hop_length: int = 256

    supported_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")
    allowed_content_types: tuple[str, ...] = (
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",
        "audio/aac",
        "audio/ogg",
        "video/mp4",
        "application/octet-stream",
    )

    fine_class_names: tuple[str, ...] = (
        "clean",
        "Blues_Driver",
        "Tube_Screamer",
        "RAT",
        "Chorus",
        "Flanger",
        "Phaser",
        "Tape_Echo",
        "Digital_Delay",
        "Sweep_Echo",
        "Plate_Reverb",
        "Hall_Reverb",
        "Spring_Reverb",
    )
    coarse_class_names: tuple[str, ...] = ("clean", "drive", "space", "phase")
    fine_to_coarse: dict[str, str] = Field(
        default_factory=lambda: {
            "clean": "clean",
            "Blues_Driver": "drive",
            "Tube_Screamer": "drive",
            "RAT": "drive",
            "Chorus": "phase",
            "Flanger": "phase",
            "Phaser": "phase",
            "Tape_Echo": "space",
            "Digital_Delay": "space",
            "Sweep_Echo": "space",
            "Plate_Reverb": "space",
            "Hall_Reverb": "space",
            "Spring_Reverb": "space",
        }
    )
    label_display_names: dict[str, str] = Field(
        default_factory=lambda: {
            "clean": "Clean",
            "Blues_Driver": "Blues Driver",
            "Tube_Screamer": "Tube Screamer",
            "RAT": "RAT",
            "Chorus": "Chorus",
            "Flanger": "Flanger",
            "Phaser": "Phaser",
            "Tape_Echo": "Tape Echo",
            "Digital_Delay": "Digital Delay",
            "Sweep_Echo": "Sweep Echo",
            "Plate_Reverb": "Plate Reverb",
            "Hall_Reverb": "Hall Reverb",
            "Spring_Reverb": "Spring Reverb",
            "drive": "Drive",
            "space": "Space",
            "phase": "Phase",
        }
    )

    @field_validator("project_root", "weights_path", mode="before")
    @classmethod
    def empty_path_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def resolved_project_root(self) -> Path:
        if self.project_root is not None:
            return self.project_root.resolve()
        return Path(__file__).resolve().parents[2]

    @property
    def media_root(self) -> Path:
        return self.resolved_project_root

    @property
    def uploads_dir(self) -> Path:
        return self.resolved_project_root / "uploads"

    @property
    def downloads_dir(self) -> Path:
        return self.resolved_project_root / "downloads"

    @property
    def separated_dir(self) -> Path:
        return self.resolved_project_root / "separated"

    @property
    def temp_dir(self) -> Path:
        return self.resolved_project_root / "temp"

    @property
    def resolved_demucs_stem_candidates(self) -> list[str]:
        return [item.strip() for item in self.demucs_stem_candidates.split(",") if item.strip()]

    @property
    def resolved_weights_path(self) -> Path:
        # NOTE: Change WEIGHTS_PATH in backend/.env if you replace the GRU checkpoint later.
        if self.weights_path is not None:
            return self.weights_path.resolve()
        return self.resolved_project_root / "weights" / "gru_model.pt"

    @property
    def transient_roots(self) -> list[Path]:
        return [self.uploads_dir, self.downloads_dir, self.separated_dir, self.temp_dir]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.project_root = settings.resolved_project_root
    settings.weights_path = settings.resolved_weights_path
    for path in settings.transient_roots:
        path.mkdir(parents=True, exist_ok=True)
    return settings
