"""Audio spans and prompt text of the in-discourse conditions.

  D0  expression span [math_start, math_end] cut from the lecture audio
  D1  right-aligned window of at most WINDOW seconds ending at math_end + PAD
  D2  D0 audio + the preceding caption text as the prompt, truncated from the left to Whisper's prompt limit
D0 and D2 decode the same array.
"""

from __future__ import annotations

import hashlib

import numpy as np

from math_eval.corpora.indiscourse import PAD, WINDOW, Clip

SR = 16000
WHISPER_PROMPT_TOKENS = (
    223  # Whisper's 224-token prompt window minus <|startofprev|>; the same text for every recognizer
)


def cut(audio_16k: np.ndarray, start_s: float, end_s: float, sr: int = SR) -> np.ndarray:
    a, b = max(0, int(round(start_s * sr))), min(len(audio_16k), int(round(end_s * sr)))
    if b <= a:
        raise ValueError(f"empty cut [{start_s}, {end_s}] on {len(audio_16k) / sr:.1f} s of audio")
    return np.ascontiguousarray(audio_16k[a:b], dtype=np.float32)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def spans(clip: Clip) -> dict[str, tuple[float, float]]:
    d1 = clip.d1
    assert d1[1] - d1[0] <= WINDOW + 1e-9 and d1[1] == clip.math_end + PAD
    return {"d0": clip.d0, "d1": d1}


def truncate_left_by_tokens(text: str, tokenizer, max_tokens: int = WHISPER_PROMPT_TOKENS) -> tuple[str, int]:
    """Keep the longest word suffix whose Whisper prompt-token count is at most max_tokens (binary search on word boundaries)."""
    words = text.split()
    if not words:
        return "", 0

    def n_tok(ws):
        s = " ".join(ws)
        return (
            (len(tokenizer.get_prompt_ids(s)) - 1)
            if hasattr(tokenizer, "get_prompt_ids")
            else len(tokenizer.encode(s, add_special_tokens=False))
        )

    if n_tok(words) <= max_tokens:
        return " ".join(words), n_tok(words)
    lo, hi = 1, len(words)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if n_tok(words[-mid:]) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    kept = words[-best:] if best else []
    return " ".join(kept), n_tok(kept) if kept else 0
