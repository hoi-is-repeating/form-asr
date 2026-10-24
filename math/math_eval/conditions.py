"""Decoding conditions. English is forced in every condition.

Benchmark clips: `none` (no prompt), `N0` (a prompt without mathematics), `W1`/`W2` (generic expressions written in words) and
`S1`/`S2` (the same expressions in notation), from labels/precedents.v3.json; the prompt text is identical for every recognizer and
only its slot frame differs.
In-discourse clips: `D0` (expression span), `D1` (window of at most 30 s ending just after the expression), `D2` (D0 audio with the
preceding caption text as the prompt).
"""

from __future__ import annotations

import json
from pathlib import Path

from loanword_eval.decoders.base import frame_prompt

from math_eval.paths import LABELS

LANGUAGE = "en"
PROMPTED = ("N0", "W1", "W2", "S1", "S2")
BENCHMARK = ("none",) + PROMPTED
DISCOURSE = ("D0", "D1", "D2")
ALL = BENCHMARK + DISCOURSE
PAIRS = (("W1", "S1"), ("W2", "S2"))


def corpus_of(cond: str) -> str:
    return "discourse" if cond in DISCOURSE else "benchmark"


def audio_key(cond: str) -> str:
    """The audio an in-discourse condition decodes: D1 the window, D0 and D2 the expression span."""
    return "d1" if cond == "D1" else "d0"


def load_precedents(labels: Path = LABELS) -> dict:
    return json.loads((labels / "precedents.v3.json").read_text(encoding="utf-8"))


def resolve(
    cond: str, precedents: dict | None, recognizer_id: str, item: dict | None = None
) -> tuple[str | None, str, str | None]:
    """Return (framed prompt or None, language, raw prompt text). D2 needs the item row (its caption text)."""
    if cond in ("none", "D0", "D1"):
        return None, LANGUAGE, None
    frames = (precedents or {}).get("slot_frames", {})
    if cond in PROMPTED:
        text = precedents["texts"][cond]
        return frame_prompt(text, frames, recognizer_id), LANGUAGE, text
    if cond == "D2":
        text = item["d2_prompt"]
        if not text.strip():
            return None, LANGUAGE, ""
        return frame_prompt(text, frames, recognizer_id), LANGUAGE, text
    raise KeyError(cond)
