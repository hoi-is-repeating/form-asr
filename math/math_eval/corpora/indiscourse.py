"""In-discourse set: MathSpeech expressions located in the caption track of their lecture.

labels/discourse_clips.v1.jsonl holds one row per clip (video id, expression span, caption text before and after it); the expression's
transcription and LaTeX are restored from the MathSpeech benchmark by `original_index`.

Spans (seconds in the lecture audio):
  D0  = [math_start, math_end]                                      expression only
  D1  = [max(context_before_start, math_end + PAD - WINDOW), math_end + PAD]
        right-aligned window of at most WINDOW = 30 s (the full preceding discourse often exceeds Whisper's 30-s input)
  D2  = D0 audio + `context_before` text as the prompt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from math_eval.paths import LABELS

CLIPS = LABELS / "discourse_clips.v1.jsonl"
WINDOW = 30.0
PAD = 0.5
LOW_BAND_MAX = 85.0


@dataclass(frozen=True)
class Clip:
    sample_id: str
    original_index: int
    video_id: str
    transcription: str
    latex: str
    match_score: float
    math_start: float
    math_end: float
    context_before: str
    context_after: str
    context_before_start: float
    lead_in: str = ""
    lead_out: str = ""
    realigned: bool = False

    @property
    def d0(self) -> tuple[float, float]:
        return (self.math_start, self.math_end)

    @property
    def d1(self) -> tuple[float, float]:
        end = self.math_end + PAD
        start = max(self.context_before_start, end - WINDOW)
        return (start, end)

    @property
    def low_band(self) -> bool:
        return self.match_score < LOW_BAND_MAX


def load(texts: dict[int, tuple[str, str]], path: Path | str = CLIPS) -> list[Clip]:
    """Clips of the label file; `texts` maps a MathSpeech index to its (transcription, LaTeX). Clips of other indices are skipped."""
    clips = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["original_index"]) not in texts:
            continue
        tr, tex = texts[int(r["original_index"])]
        clips.append(
            Clip(
                sample_id=r["sample_id"],
                original_index=int(r["original_index"]),
                video_id=r["video_id"],
                transcription=tr.strip(),
                latex=tex,
                match_score=float(r["match_score"]),
                math_start=float(r["math_start"]),
                math_end=float(r["math_end"]),
                context_before=r["context_before"],
                context_after=r["context_after"],
                context_before_start=float(r["context_before_start"]),
                lead_in=r.get("lead_in") or "",
                lead_out=r.get("lead_out") or "",
                realigned=bool(r["realigned"]),
            )
        )
    return clips


def context(path: Path | str = CLIPS) -> dict[str, dict]:
    """sample_id -> caption text around the expression (used for the flag reference of in-discourse decodes)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["sample_id"]] = {
                "context_before": r["context_before"],
                "lead_out": r.get("lead_out") or "",
                "context_after": r["context_after"],
            }
    return out
