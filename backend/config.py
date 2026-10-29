"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", BASE_DIR / "models" / "Real_Life_Violence_Detation_MoBiLSTM_model.h5")
)

# --- Must match the values used in the training notebook ---
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 16
CLASSES = ["NonViolence", "Violence"]  # index 0 / index 1 (same order as CLASSES_LIST in training)

# --- Upload limits ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# --- Timeline analysis ---
MAX_SEGMENTS = int(os.getenv("MAX_SEGMENTS", "8"))
MIN_FRAMES_PER_SEGMENT = 32  # a segment needs enough frames to spread 16 samples across it
