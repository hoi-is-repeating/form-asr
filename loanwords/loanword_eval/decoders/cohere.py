"""Cohere Transcribe (03-2026), following the transformers model doc example (AutoProcessor,
CohereAsrForConditionalGeneration, generate). The processor builds the decoder prompt from the language code; there is no
text-conditioning slot."""

from __future__ import annotations

import numpy as np

from loanword_eval.decoders.base import DecodeResult, PromptUnsupported, generation_hash


class CohereDecoder:
    id = "cohere"
    prompt_slot = "unsupported"

    def __init__(self, cfg: dict, device: str | None = None, model=None, processor=None):
        self.cfg = cfg
        self.checkpoint = cfg["checkpoint"]
        self.revision = cfg.get("revision")
        g = cfg.get("generation", {})
        self.gen_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": int(g.get("max_new_tokens", 256))}
        self.punctuation = bool(cfg.get("punctuation", False))
        if model is not None:
            self.model, self.processor = model, processor
        else:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
            import torch

            # dtype from the config when set, else the checkpoint dtype
            _dt = (
                getattr(torch, str(self.cfg.get("dtype")), None)
                if isinstance(getattr(self, "cfg", None), dict) and self.cfg.get("dtype")
                else None
            )
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                self.checkpoint,
                revision=self.revision,
                device_map=device or "auto",
                **({"dtype": _dt} if _dt is not None else {}),
            ).eval()
        self.gen_hash = generation_hash({**self.gen_kwargs, "punctuation": self.punctuation})

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        if prompt:
            raise PromptUnsupported("cohere: no text-conditioning slot")
        import torch

        inputs = self.processor(
            audio_16k, sampling_rate=16000, return_tensors="pt", language=language or "ko", punctuation=self.punctuation
        ).to(self.model.device)
        inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs)
        raw = self.processor.decode(out, skip_special_tokens=False)
        text = self.processor.decode(out, skip_special_tokens=True)
        raw = raw if isinstance(raw, str) else raw[0]
        text = text if isinstance(text, str) else text[0]
        return DecodeResult(
            text=text,
            raw=raw,
            prompt_slot=self.prompt_slot,
            language_forced=bool(language),
            meta={
                "recognizer_id": self.id,
                "checkpoint": self.checkpoint,
                "revision": self.revision,
                "generation_config_hash": self.gen_hash,
                "generation_hash": self.gen_hash,
                "prompt_given": 0,
                "prompt_chars": 0,
                "dtype": str(getattr(self.model, "dtype", None)),
                "batched": False,
            },
        )

    def transcribe_batch(self, audios, *, language, prompt):
        return [self.transcribe(a, language=language, prompt=prompt) for a in audios]
