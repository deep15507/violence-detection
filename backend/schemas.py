"""Pydantic models describing every request / response of the API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["Violence", "NonViolence"]


class Probabilities(BaseModel):
    NonViolence: float = Field(..., ge=0, le=1)
    Violence: float = Field(..., ge=0, le=1)


class VideoInfo(BaseModel):
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    fps: float = Field(..., description="Frames per second")
    total_frames: int = Field(..., description="Number of frames in the video")
    duration_seconds: float = Field(..., description="Video length in seconds")
    size_mb: float = Field(..., description="Uploaded file size in MB")


class SegmentResult(BaseModel):
    """Prediction for one time-slice of the video (used for the timeline chart)."""

    index: int
    start_seconds: float
    end_seconds: float
    label: Label
    violence_probability: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "fight_clip.mp4",
                "label": "Violence",
                "is_violent": True,
                "confidence": 0.967,
                "probabilities": {"NonViolence": 0.033, "Violence": 0.967},
                "threshold": 0.5,
                "video": {
                    "width": 1280, "height": 720, "fps": 30.0,
                    "total_frames": 150, "duration_seconds": 5.0, "size_mb": 2.4,
                },
                "segments": [
                    {"index": 0, "start_seconds": 0.0, "end_seconds": 2.5,
                     "label": "Violence", "violence_probability": 0.94},
                    {"index": 1, "start_seconds": 2.5, "end_seconds": 5.0,
                     "label": "Violence", "violence_probability": 0.97},
                ],
                "processing_time_ms": 812.4,
            }
        }
    )

    filename: str
    label: Label = Field(..., description="Final verdict for the whole video")
    is_violent: bool
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted label")
    probabilities: Probabilities
    threshold: float = Field(..., description="Violence probability threshold that was applied")
    video: VideoInfo
    segments: list[SegmentResult] = Field(
        default_factory=list, description="Per-segment predictions (empty for very short videos)"
    )
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_loaded: bool
    model_path: str
    classes: list[str]
    input_shape: str


class ErrorResponse(BaseModel):
    detail: str
