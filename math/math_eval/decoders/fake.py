"""Test decoder (no model), keyed by the audio sha256 of the fixture items (benchmark clips and the in-discourse d0/d1 cuts).

modes
  words    returns the reference spoken form
  steer    under a prompt containing a notation symbol, the fixture's notation surface; otherwise items with an odd idx are
           misrecognized (the fixture's wrong hypothesis) and the others are returned in words
  partial  items with idx % 3 == 0 are misrecognized unless the prompt contains a notation symbol
For d1 audio the output is "<last six caption words> <expression>", so the in-context scorer has material around the item.
"""

from __future__ import annotations

import hashlib

import numpy as np
from loanword_eval.decoders.base import DecodeResult

_SYMBOLS = set("^=+/*<>√∫∑×−-_")


class MathFakeDecoder:
    id = "fake"
    checkpoint = "fake"
    revision = "0"

    def __init__(self, items: list[dict], mode: str = "words", surfaces: dict[str, dict] | None = None):
        """`surfaces`: utt_id -> fixture dict (notation / wrong) for in-discourse items, whose rows carry no `fixture` field."""
        self.mode = mode
        self.surfaces = surfaces or {}
        self.by_sha: dict[str, tuple[dict, str]] = {}
        for it in items:
            if it.get("audio_sha256"):
                self.by_sha[it["audio_sha256"]] = (it, "d0")
            for key, blob in (it.get("audio") or {}).items():
                if blob and blob.get("sha256"):
                    self.by_sha[blob["sha256"]] = (it, key)

    def _text(self, it: dict, variant: str, prompt: str | None) -> str:
        fx = it.get("fixture") or self.surfaces.get(it.get("utt_id"), {})
        ref = it["text"]
        if self.mode in ("steer", "partial") and prompt and any(ch in _SYMBOLS for ch in prompt):
            body = fx.get("notation", ref)
        elif self.mode == "steer" and int(it.get("idx") or 0) % 2 == 1:
            body = fx.get("wrong", ref)
        elif self.mode == "partial" and int(it.get("idx") or 0) % 3 == 0:
            body = fx.get("wrong", ref)
        else:
            body = ref
        if variant == "d1":
            ctx = " ".join((it.get("d2_prompt") or "").split()[-6:])
            body = (ctx + " " + body).strip()
        return body

    def transcribe(self, audio_16k: np.ndarray, *, language, prompt):
        sha = hashlib.sha256(np.asarray(audio_16k, dtype=np.float32).tobytes()).hexdigest()
        it, variant = self.by_sha.get(sha, ({"text": "", "fixture": {}}, "d0"))
        text = self._text(it, variant, prompt)
        return DecodeResult(
            text=text,
            raw=text,
            prompt_slot="prefix",
            language_forced=bool(language),
            meta={
                "recognizer_id": self.id,
                "checkpoint": "fake",
                "revision": "0",
                "prompt_given": int(bool(prompt)),
                "strip_failed": 0,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
            },
        )

    def transcribe_batch(self, audios, *, language, prompt):
        return [self.transcribe(a, language=language, prompt=prompt) for a in audios]
