from pydantic import BaseModel, HttpUrl


class AnalysisRequest(BaseModel):
    youtube_url: HttpUrl


class ScoreItem(BaseModel):
    label: str
    display_name: str
    score: float


class TimelineSegment(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float
    effect_label: str
    effect_display_name: str
    family_label: str
    family_display_name: str
    confidence: float


class AnalysisResponse(BaseModel):
    success: bool
    source_type: str
    source_name: str
    predicted_effect: str
    predicted_effect_display_name: str
    predicted_family: str
    predicted_family_display_name: str
    fine_scores: list[ScoreItem]
    family_scores: list[ScoreItem]
    timeline_segments: list[TimelineSegment]
    audio_duration_sec: float
    analysis_window_sec: float
    analysis_hop_sec: float
    guitar_stem_url: str
    original_audio_url: str
    processing_time_sec: float
    demucs_stem_used: str
    model_name: str


class HealthResponse(BaseModel):
    success: bool
    status: str
    model_weights_exists: bool
    demucs_model: str
