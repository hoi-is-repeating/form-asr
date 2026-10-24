"""Decoding conditions: no prompt, or a preceding-text prompt from labels/precedents.v1.json.

N0 carries no content words; H1/H2 write their generic loanwords in Hangul, L1/L2 the same words in Latin script, and R1/R2 in
romanized Hangul. The prompt text is identical for every recognizer; only its slot frame differs. All conditions force Korean.
"""

from __future__ import annotations

import json
from pathlib import Path

from loanword_eval.decoders.base import frame_prompt
from loanword_eval.paths import LABELS

PROMPTED = ("N0", "H1", "H2", "L1", "L2", "R1", "R2")
ALL = ("none",) + PROMPTED
PAIRS = (("H1", "L1"), ("H2", "L2"))


def cond_slug(cond: str) -> str:
    return cond.replace("/", "-")


def load_precedents(labels: Path = LABELS) -> tuple[dict, Path]:
    path = labels / "precedents.v1.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def resolve(cond: str, precedents: dict | None, recognizer_id: str) -> tuple[str | None, str, str | None]:
    """Return (framed prompt or None, language, raw prompt text)."""
    if cond == "none":
        return None, "ko", None
    if cond not in PROMPTED:
        raise KeyError(cond)
    text = precedents["texts"][cond]
    return frame_prompt(text, precedents.get("slot_frames", {}), recognizer_id), "ko", text
