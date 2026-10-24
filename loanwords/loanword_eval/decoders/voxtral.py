"""Voxtral Mini 4B Realtime, following the model card transformers example. Decoding is greedy (temperature 0.0, as the card
requires); transcription_delay_ms comes from tekken.json. There is no text-conditioning slot."""

from __future__ import annotations

import numpy as np

from loanword_eval.decoders.base import DecodeResult, PromptUnsupported, generation_hash


class VoxtralDecoder:
    id = "voxtral"
    prompt_slot = "unsupported"

    def __init__(self, cfg: dict, device: str | None = None, model=None, processor=None):
        self.cfg = cfg
        self.checkpoint = cfg["checkpoint"]
        self.revision = cfg.get("revision")
        g = cfg.get("generation", {})
        self.gen_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": int(g.get("max_new_tokens", 256))}
        if model is not None:
            self.model, self.processor = model, processor
        else:
            from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
            import torch

            # dtype from the config when set, else the checkpoint dtype
            _dt = (
                getattr(torch, str(self.cfg.get("dtype")), None)
                if isinstance(getattr(self, "cfg", None), dict) and self.cfg.get("dtype")
                else None
            )
            self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                self.checkpoint,
                revision=self.revision,
                device_map=device or "auto",
                **({"dtype": _dt} if _dt is not None else {}),
            ).eval()
        self.gen_hash = generation_hash(self.gen_kwargs)

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        if prompt:
            raise PromptUnsupported("voxtral: no text-conditioning slot for transcription")
        import torch

        inputs = self.processor(audio_16k, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs)
        raw = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return DecodeResult(
            text=text,
            raw=raw,
            prompt_slot=self.prompt_slot,
            language_forced=False,
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
                # delay applied by the loaded processor; the config key is documentation only
                "transcription_delay_ms_loaded": getattr(
                    getattr(self.processor, "tokenizer", None), "transcription_delay_ms", None
                )
                if hasattr(getattr(self.processor, "tokenizer", None), "transcription_delay_ms")
                else getattr(self.processor, "transcription_delay_ms", None),
            },
        )

    def transcribe_batch(self, audios, *, language, prompt):
        return [self.transcribe(a, language=language, prompt=prompt) for a in audios]
