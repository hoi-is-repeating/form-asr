"""Omnilingual ASR LLM 7B v2, following the omnilingual-asr README example (`ASRInferencePipeline.transcribe`).
Inputs are waveform dicts of at most 40 s; conditioning is a language-script code only, so there is no prompt slot."""

from __future__ import annotations

import numpy as np

from loanword_eval.decoders.base import DecodeResult, PromptUnsupported, generation_hash


class OmniDecoder:
    id = "omni"
    prompt_slot = "unsupported"

    def __init__(self, cfg: dict, device: str | None = None, model=None):
        self.cfg = cfg
        self.checkpoint = cfg["checkpoint"]  # model card name
        self.revision = cfg.get("revision")  # package version; there is no HF revision
        self.lang = cfg.get("lang", "kor_Hang")
        self.batch_size = int(cfg.get("batch_size", 2))
        if model is not None:
            self.pipe = model
        else:
            from omnilingual_asr.models.inference.pipeline import (
                ASRInferencePipeline,
            )

            self.pipe = ASRInferencePipeline(model_card=self.checkpoint)
        self.gen_hash = generation_hash(
            {"lang": self.lang, "toolkit": "omnilingual-asr==0.2.0", "batch_size": self.batch_size}
        )

    def transcribe_batch(self, audios, *, language, prompt):
        if prompt:
            raise PromptUnsupported("omni: conditioning is a language-script code only; no text prompt")
        lang = {"ko": "kor_Hang", "en": "eng_Latn", None: self.lang}.get(language, language)
        inputs = [{"waveform": np.asarray(a, dtype=np.float32), "sample_rate": 16000} for a in audios]
        texts = self.pipe.transcribe(inputs, lang=[lang] * len(inputs), batch_size=self.batch_size)
        return [
            DecodeResult(
                text=t,
                raw=t,
                prompt_slot=self.prompt_slot,
                language_forced=True,
                meta={
                    "recognizer_id": self.id,
                    "checkpoint": self.checkpoint,
                    "revision": self.revision,
                    "lang": lang,
                    "generation_config_hash": self.gen_hash,
                    "generation_hash": self.gen_hash,
                    "prompt_given": 0,
                    "prompt_chars": 0,
                },
            )
            for t in texts
        ]

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        return self.transcribe_batch([audio_16k], language=language, prompt=prompt)[0]
