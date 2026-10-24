"""Qwen3-ASR-1.7B via the authors' toolkit (qwen-asr 0.0.6), following the README example (`Qwen3ASRModel.from_pretrained`,
`model.transcribe`). The prompt is passed as `context`, which the toolkit places in the system message before the audio turn
(arXiv 2601.21337); the unprompted condition passes an empty context through the same code path."""

from __future__ import annotations

import numpy as np

from loanword_eval.decoders.base import DecodeResult, PromptUnsupported, generation_hash


class Qwen3AsrDecoder:
    id = "qwen3asr"
    prompt_slot = "system"
    LANG = {"ko": "Korean", "en": "English", None: None}

    def __init__(self, cfg: dict, device: str | None = None, model=None):
        self.cfg = cfg
        self.checkpoint = cfg["checkpoint"]
        self.revision = cfg.get("revision")
        g = cfg.get("generation", {})
        self.max_new_tokens = int(g.get("max_new_tokens", 256))
        self.inject = "context"  # toolkit argument that fills the system message
        if model is not None:
            self.model = model
        else:
            import torch
            from qwen_asr import Qwen3ASRModel

            self.model = Qwen3ASRModel.from_pretrained(
                self.checkpoint,
                dtype=torch.bfloat16,
                device_map=device or "cuda:0",
                max_inference_batch_size=int(cfg.get("batch_size", 8)),
                max_new_tokens=self.max_new_tokens,
            )
        self.gen_hash = generation_hash(
            {"max_new_tokens": self.max_new_tokens, "toolkit": "qwen-asr==0.0.6", "greedy": True}
        )

    def _result(self, r0, prompt: str | None, language: str | None) -> DecodeResult:
        text = getattr(r0, "text", None) if not isinstance(r0, dict) else r0.get("text")
        lang = getattr(r0, "language", None) if not isinstance(r0, dict) else r0.get("language")
        return DecodeResult(
            text=text or "",
            raw=text or "",
            prompt_slot=self.prompt_slot,
            language_forced=bool(language),
            meta={
                "recognizer_id": self.id,
                "checkpoint": self.checkpoint,
                "revision": self.revision,
                "generation_config_hash": self.gen_hash,
                "generation_hash": generation_hash({"g": self.gen_hash, "prompt": prompt or ""}),
                "detected_language": lang,
                "prompt_given": int(bool(prompt)),
                "prompt_chars": len(prompt) if prompt else 0,
                "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest() if prompt else None,
                "inject": self.inject,
            },
        )

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        return self.transcribe_batch([audio_16k], language=language, prompt=prompt)[0]

    def transcribe_batch(self, audios, *, language, prompt):
        if prompt and self.inject is None:
            raise PromptUnsupported("qwen3asr: system-message injection path not located")
        res = self.model.transcribe(
            audio=[(np.asarray(a, dtype=np.float32), 16000) for a in audios],
            context=prompt or "",
            language=self.LANG.get(language, language),
            return_time_stamps=False,
        )
        return [self._result(r, prompt, language) for r in res]
