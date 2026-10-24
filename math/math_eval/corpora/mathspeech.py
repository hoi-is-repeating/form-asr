"""MathSpeech benchmark (Hugging Face `AAAI2025/MathSpeech`, split `train`: 1,101 clips with transcription, LaTeX and Source)."""

from __future__ import annotations

import io
import re
from typing import Iterator

import numpy as np

DATASET = "AAAI2025/MathSpeech"
SR = 16000
_YT_RE = re.compile(r"(?:v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")


def video_id(source: str | None) -> str | None:
    m = _YT_RE.search(source or "")
    return m.group(1) if m else None


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(x, dtype=np.float32)


def load(with_audio: bool = True) -> Iterator[dict]:
    """Rows in dataset order: idx, transcription, latex, source, video_id and (with_audio) the 16 kHz mono waveform."""
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET, split="train")
    if with_audio:
        ds = ds.cast_column("audio", Audio(decode=False))
    else:
        ds = ds.remove_columns(["audio"])
    for idx, row in enumerate(ds):
        out = {
            "idx": idx,
            "transcription": row["transcription"],
            "latex": row["LaTeX"],
            "source": row["Source"],
            "video_id": video_id(row["Source"]),
        }
        if with_audio:
            import soundfile as sf

            au = row["audio"]
            data, sr = sf.read(
                io.BytesIO(au["bytes"]) if au.get("bytes") else au["path"], dtype="float32", always_2d=False
            )
            out["audio"] = to_16k(data, int(sr))
        yield out
