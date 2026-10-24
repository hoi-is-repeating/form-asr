"""Raon-Speech-9B, following the model card example (`RaonPipeline.chat` with an audio and a text part in the user turn).

The text part is the framed prompt from the driver, or the bare instruction "Transcribe the audio into text." (the STT format of
the technical report) for the unprompted condition. `RaonPipeline.chat` hard-codes sampling, so `_chat_greedy` repeats its
text-output path with do_sample=False as the only change. The remote code is pinned to `revision`. Weights CC BY-NC 4.0.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from loanword_eval.decoders.base import DecodeResult, generation_hash

INSTRUCTION = "Transcribe the audio into text."


def _sha(s):
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest() if s else None


class RaonDecoder:
    id = "raon"
    prompt_slot = "user_text"

    def __init__(self, cfg: dict, device: str | None = None, model=None):
        self.cfg = cfg
        self.checkpoint = cfg["checkpoint"]
        self.revision = cfg.get("revision")
        self.instruction = cfg.get("instruction", INSTRUCTION)
        self.max_new_tokens = int(
            (cfg.get("generation") or {}).get("max_new_tokens", 1024)
        )  # RaonPipeline.chat default for text output
        if model is not None:
            self.pipe = model
        else:
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            config = AutoConfig.from_pretrained(self.checkpoint, trust_remote_code=True, revision=self.revision)
            RaonPipeline = get_class_from_dynamic_module(
                "modeling_raon.RaonPipeline",
                self.checkpoint,
                revision=self.revision or getattr(config, "_commit_hash", None),
            )
            self.pipe = RaonPipeline(self.checkpoint, device=device or "cuda", dtype=cfg.get("dtype", "bfloat16"))
        self.gen_hash = generation_hash(
            {
                "instruction": self.instruction,
                "do_sample": False,
                "dtype": cfg.get("dtype", "bfloat16"),
                "max_new_tokens": self.max_new_tokens,
                "api_path": "chat_greedy",
            }
        )

    def _audio_path(self, audio_16k: np.ndarray) -> str:
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio_16k, 16000)
        return path

    def _chat_greedy(self, messages: list[dict]) -> str:
        """RaonPipeline.chat's text-output path with do_sample=False."""
        pipe = self.pipe
        if hasattr(pipe, "chat_greedy"):  # test double
            return pipe.chat_greedy(messages)
        inputs = pipe.processor(
            messages,
            add_generation_prompt=True,
            force_audio_output=False,
            device=pipe.device,
            dtype=pipe.dtype,
            max_audio_chunk_length=192000,
        )
        input_length = int(inputs["attention_mask"].sum().item())
        output = pipe.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            audio_input=inputs.get("audio_input"),
            audio_input_lengths=inputs.get("audio_input_lengths"),
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            do_sample=False,
            force_audio_output=False,
            force_text_output=True,
            speaker_audio=None,
        )
        return pipe.processor.tokenizer.decode(output["sequences"][0, input_length:], skip_special_tokens=True)

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        path = self._audio_path(audio_16k)
        try:
            text_part = prompt if prompt else self.instruction  # prompt is already framed by the driver
            messages = [
                {"role": "user", "content": [{"type": "audio", "audio": path}, {"type": "text", "text": text_part}]}
            ]
            out = self._chat_greedy(messages)
        finally:
            os.unlink(path)
        text = out if isinstance(out, str) else str(out)
        return DecodeResult(
            text=text,
            raw=text,
            prompt_slot=self.prompt_slot,
            language_forced=False,
            meta={
                "recognizer_id": self.id,
                "checkpoint": self.checkpoint,
                "revision": self.revision,
                "generation_config_hash": self.gen_hash,
                "generation_hash": generation_hash({"g": self.gen_hash, "prompt": prompt or ""}),
                "prompt_given": int(bool(prompt)),
                "prompt_chars": len(prompt) if prompt else 0,
                "prompt_sha256": _sha(prompt),
                "api_path": "chat_greedy",
                "user_text": text_part,
                "trust_remote_code": True,
            },
        )

    def transcribe_batch(self, audios, *, language, prompt):
        return [self.transcribe(a, language=language, prompt=prompt) for a in audios]
