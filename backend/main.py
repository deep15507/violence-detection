"""FastAPI service for the Real-Life Violence Detection model.

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_EXTENSIONS, CLASSES, MAX_UPLOAD_MB, MODEL_PATH
from .model_service import ViolenceModel
from .schemas import (
    ErrorResponse, HealthResponse, PredictionResponse, Probabilities, SegmentResult, VideoInfo,
)
from .video_utils import VideoError, build_sequences

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("violence-api")

VIOLENCE_IDX = CLASSES.index("Violence")
CHUNK = 1024 * 1024  # 1 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ViolenceModel(MODEL_PATH)
    try:
        app.state.model.load()
    except Exception:  # keep the server up so /health can report what went wrong
        log.exception("Failed to load model")
    yield


app = FastAPI(
    title="Real-Life Violence Detection API",
    description="Upload a video and get a Violence / NonViolence prediction from a MobileNetV2 + BiLSTM model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Violence Detection API is running. See /docs for the interactive documentation."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    m: ViolenceModel = app.state.model
    return HealthResponse(
        status="ok" if m.loaded else "model_not_loaded",
        model_loaded=m.loaded,
        model_path=str(m.path),
        classes=CLASSES,
        input_shape=m.input_shape,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="Upload a video and classify it as Violence / NonViolence",
)
def predict(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
    threshold: float = Query(0.5, ge=0.05, le=0.95,
                             description="Video is 'Violence' if P(Violence) >= threshold"),
):
    model: ViolenceModel = app.state.model
    if not model.loaded:
        raise HTTPException(503, "Model is not loaded. Check the server logs and /health.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    t0 = time.perf_counter()
    tmp_path = None
    try:
        # Stream the upload to disk, enforcing the size limit as we go
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        written = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            while chunk := file.file.read(CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.")
                tmp.write(chunk)
        if written == 0:
            raise HTTPException(400, "The uploaded file is empty.")

        try:
            meta, whole_seq, segs = build_sequences(tmp_path)
        except VideoError as e:
            raise HTTPException(422, str(e))

        # One batched forward pass: index 0 = whole video, the rest = timeline segments
        batch = [whole_seq] + [s[2] for s in segs]
        probs = model.predict(np.stack(batch))

        p_viol = float(probs[0][VIOLENCE_IDX])
        is_violent = p_viol >= threshold
        label = "Violence" if is_violent else "NonViolence"

        segments = []
        for i, (start, end, _) in enumerate(segs):
            pv = float(probs[i + 1][VIOLENCE_IDX])
            segments.append(SegmentResult(
                index=i,
                start_seconds=round(start / meta.fps, 2),
                end_seconds=round(end / meta.fps, 2),
                label="Violence" if pv >= threshold else "NonViolence",
                violence_probability=round(pv, 4),
            ))

        return PredictionResponse(
            filename=file.filename or "video",
            label=label,
            is_violent=is_violent,
            confidence=round(p_viol if is_violent else 1 - p_viol, 4),
            probabilities=Probabilities(NonViolence=round(1 - p_viol, 4), Violence=round(p_viol, 4)),
            threshold=threshold,
            video=VideoInfo(
                width=meta.width, height=meta.height, fps=round(meta.fps, 2),
                total_frames=meta.total_frames, duration_seconds=round(meta.duration, 2),
                size_mb=round(written / (1024 * 1024), 2),
            ),
            segments=segments,
            processing_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    finally:
        file.file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
