"""Audio sanity checks and content hashes (16 kHz mono float32)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass(frozen=True)
class AudioStats:
    duration_s: float
    rms: float
    clip_ratio: float
    sha256: str
    n_samples: int


def stats(x: np.ndarray, sr: int = SR) -> AudioStats:
    if x.ndim != 1:
        raise ValueError("mono expected")
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    clip = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    h = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    return AudioStats(x.size / sr, rms, clip, h, int(x.size))


def check(x: np.ndarray, sr: int = SR, max_s: float = 30.0, min_rms: float = 1e-3, max_clip: float = 0.01) -> list[str]:
    """Return a list of problems (empty means the clip is acceptable for short-form decoding)."""
    s = stats(x, sr)
    problems = []
    if s.n_samples == 0:
        problems.append("empty")
    if s.duration_s > max_s:
        problems.append(f"long_form:{s.duration_s:.1f}s")
    if s.rms < min_rms:
        problems.append(f"silent:rms={s.rms:.2e}")
    if s.clip_ratio > max_clip:
        problems.append(f"clipping:{s.clip_ratio:.3f}")
    return problems
