"""Fake decoders for tests of the decode driver and the injection test (no model). They honour the Decoder contract."""

from __future__ import annotations

import hashlib

import numpy as np

from loanword_eval.decoders.base import DecodeResult, PromptUnsupported


class FlipDecoder:
    """Test double that behaves like a steerable recognizer: returns the reference text; under a Latin-script prompt (any Latin
    letter in the prompt) it writes every item in its alternate form; `echo` prepends the prompt; `inert` ignores prompts."""

    id = "fake"
    checkpoint = "fake"
    revision = "0"

    def __init__(
        self,
        items: list[dict],
        tokens: list[dict],
        prompt_slot="prefix",
        echo=False,
        inert=False,
        nondeterministic=False,
    ):
        self.prompt_slot, self.echo, self.inert, self.nondeterministic = prompt_slot, echo, inert, nondeterministic
        self.by_sha = {}
        toks_by_utt = {}
        for t in tokens:
            toks_by_utt.setdefault(t["utt_id"], []).append(t)
        for it in items:
            if it.get("audio_sha256"):
                self.by_sha[it["audio_sha256"]] = (it["text"], toks_by_utt.get(it["utt_id"], []))
        self.calls = 0

    def transcribe(self, audio_16k, *, language, prompt):
        if prompt and self.prompt_slot == "unsupported":
            raise PromptUnsupported("fake: unsupported")
        self.calls += 1
        sha = hashlib.sha256(np.asarray(audio_16k, dtype=np.float32).tobytes()).hexdigest()
        text, toks = self.by_sha.get(sha, ("", []))
        use_prompt = prompt if not self.inert else None
        if use_prompt and any(ch.isascii() and ch.isalpha() for ch in use_prompt):
            for t in sorted(toks, key=lambda t: -t["start"]):
                alt = t["hangul"] if t["ref_form"] != "hangul" else t["latin"]
                if alt is None:  # token with a single written form
                    continue
                text = text[: t["start"]] + alt + text[t["end"] :]
        if self.echo and prompt:
            text = prompt + " " + text
        if self.nondeterministic:
            self.seen = getattr(self, "seen", {})
            self.seen[sha] = self.seen.get(sha, 0) + 1
            text = text + ("" if self.seen[sha] % 2 else " 음")
        return DecodeResult(
            text=text,
            raw=text,
            prompt_slot=self.prompt_slot,
            language_forced=bool(language),
            meta={
                "recognizer_id": self.id,
                "checkpoint": "fake",
                "revision": "0",
                "generation_config_hash": "fake",
                "generation_hash": "fake",
                "prompt_given": int(bool(prompt)),
                "prompt_chars": len(prompt) if prompt else 0,
                "strip_failed": 0,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
            },
        )

    def transcribe_batch(self, audios, *, language, prompt):
        return [self.transcribe(a, language=language, prompt=prompt) for a in audios]
