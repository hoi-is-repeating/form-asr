"""Decoder wrapper contract. Model-specific code lives only in the wrappers; scoring never branches on the recognizer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

SLOTS = ("prefix", "system", "user_text", "unsupported")


class PromptUnsupported(ValueError):
    """Raised when a prompt is passed to a recognizer whose interface has no text-conditioning slot."""


@dataclass
class DecodeResult:
    text: str  # transcript with the prompt stripped, no post-processing
    raw: str  # model output before stripping
    prompt_slot: str
    language_forced: bool
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Decoder(Protocol):
    id: str
    checkpoint: str
    revision: str | None
    prompt_slot: str

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult: ...


def generation_hash(gen_kwargs: dict) -> str:
    return hashlib.sha256(json.dumps(gen_kwargs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def frame_prompt(prompt: str, slot_frames: dict, recognizer_id: str) -> str:
    """Wrap the prompt in the recognizer's slot frame from labels/precedents.v1.json."""
    fr = slot_frames.get(recognizer_id, {"frame": "{prompt}"})["frame"]
    return fr.replace("{prompt}", prompt)
