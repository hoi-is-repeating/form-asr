"""Recognizer wrappers. Each wrapper imports its framework lazily, so the package imports without every toolkit installed."""

from __future__ import annotations

from loanword_eval.paths import load_yaml


def load_decoder(recognizer_id: str, cfg: dict | None = None, **kw):
    cfg = cfg or load_yaml(f"decode_{recognizer_id}.yaml")
    mods = {
        "whisper": ("whisper", "WhisperDecoder"),
        "owsm": ("owsm", "OwsmDecoder"),
        "qwen3asr": ("qwen3asr", "Qwen3AsrDecoder"),
        "raon": ("raon", "RaonDecoder"),
        "cohere": ("cohere", "CohereDecoder"),
        "voxtral": ("voxtral", "VoxtralDecoder"),
        "omni": ("omni", "OmniDecoder"),
    }
    if recognizer_id not in mods:
        raise KeyError(f"no wrapper for recognizer {recognizer_id!r}")
    import importlib

    mod, cls = mods[recognizer_id]
    return getattr(importlib.import_module(f"loanword_eval.decoders.{mod}"), cls)(cfg, **kw)


RECOGNIZERS = ("whisper", "owsm", "qwen3asr", "raon", "cohere", "voxtral", "omni")
PROMPT_CAPABLE = ("whisper", "owsm", "qwen3asr", "raon")
# Wrappers whose transcribe_batch decodes a batch in one call; the others loop over utterances.
BATCHED = ("whisper", "owsm", "qwen3asr", "omni")
