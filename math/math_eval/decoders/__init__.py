"""Recognizer registry: the seven loanword_eval wrappers, and the Whisper size series (base, small, large-v2) through the same Whisper
wrapper with another checkpoint."""

from __future__ import annotations

from loanword_eval.decoders import BATCHED as _BATCHED
from loanword_eval.decoders import PROMPT_CAPABLE as _PROMPT_CAPABLE
from loanword_eval.decoders import RECOGNIZERS as _RECOGNIZERS
from loanword_eval.decoders import load_decoder as _load

from math_eval.paths import load_yaml

WHISPER_SIZES = ("whisper_base", "whisper_small", "whisper_large_v2")
RECOGNIZERS = _RECOGNIZERS + WHISPER_SIZES
PROMPT_CAPABLE = _PROMPT_CAPABLE + WHISPER_SIZES
BATCHED = _BATCHED + WHISPER_SIZES


def load_decoder(recognizer_id: str, cfg: dict | None = None, **kw):
    cfg = cfg or load_yaml(f"decode_{recognizer_id}.yaml")
    if recognizer_id in _RECOGNIZERS:
        return _load(recognizer_id, cfg, **kw)
    if recognizer_id in WHISPER_SIZES:
        from loanword_eval.decoders.whisper import WhisperDecoder

        dec = WhisperDecoder(cfg, **kw)
        dec.id = recognizer_id
        return dec
    raise KeyError(f"no wrapper for recognizer {recognizer_id!r}")
