"""Loads the trained MoBiLSTM model once and exposes a tiny predict() helper."""
import logging
from pathlib import Path

import numpy as np

from .config import CLASSES

log = logging.getLogger("violence-api")


class ViolenceModel:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def input_shape(self) -> str:
        return str(self.model.input_shape) if self.loaded else "n/a"

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}")
        import tensorflow as tf  # imported lazily so the module can be imported without TF

        log.info("Loading model from %s ...", self.path)
        # compile=False -> we only need inference, avoids optimizer/loss deserialisation issues
        self.model = tf.keras.models.load_model(str(self.path), compile=False)
        # warm-up so the first real request isn't slow
        self.model.predict(np.zeros((1, *self.model.input_shape[1:]), dtype="float32"), verbose=0)
        log.info("Model ready. Input shape: %s", self.model.input_shape)

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """sequences: uint8 array (N, 16, 64, 64, 3) in RGB. Returns probabilities (N, 2)."""
        x = sequences.astype("float32") / 255.0  # same normalisation as training
        probs = self.model.predict(x, batch_size=8, verbose=0)
        assert probs.shape[1] == len(CLASSES)
        return probs
