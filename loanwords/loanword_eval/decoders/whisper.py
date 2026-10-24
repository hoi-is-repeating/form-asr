"""Whisper via Hugging Face transformers, following the model card example (WhisperProcessor; generate with language,
task="transcribe" and prompt_ids). Decoding is greedy; for short-form audio the prompt conditions the first segment.
"""

from __future__ import annotations

import numpy as np

from loanword_eval.decoders.base import DecodeResult, generation_hash


def multi_language_prefix(processor, model, *, prompt: str | None, languages: list[str]) -> list[int]:
    """Decoder prefix with one language token per language (Peng et al., Interspeech 2023):
    [<|startofprev|> prompt] <|startoftranscript|> <|lang|>... <|transcribe|><|notimestamps|>."""
    gc = model.generation_config
    ids: list[int] = []
    if prompt:
        ids += [int(x) for x in processor.get_prompt_ids(prompt)]  # begins with <|startofprev|>
    ids.append(int(gc.decoder_start_token_id))
    for lang in languages:
        tok = f"<|{lang}|>"
        if tok not in gc.lang_to_id:
            raise ValueError(f"language {lang!r} not in the model's lang_to_id")
        ids.append(int(gc.lang_to_id[tok]))
    ids += [int(gc.task_to_id["transcribe"]), int(gc.no_timestamps_token_id)]
    return ids


class WhisperDecoder:
    id = "whisper"
    prompt_slot = "prefix"

    def __init__(self, cfg: dict, device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.cfg = cfg
        self.checkpoint = cfg["checkpoint"]
        self.revision = cfg.get("revision")
        # device: explicit argument, else $CH3LOAN_DEVICE, else CUDA, else CPU
        import os

        self.device = device or os.environ.get("CH3LOAN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("CH3LOAN_DEVICE=mps but Metal is not available")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            cfg.get("dtype", "float32")
        ]
        if self.device == "cpu" and dtype == torch.float16:
            dtype = torch.float32  # fp16 is unsupported on CPU
        self.dtype = dtype
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(self.checkpoint, revision=self.revision, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        g = cfg.get("generation", {})
        self.gen_kwargs = {
            "task": g.get("task", "transcribe"),
            "num_beams": int(g.get("num_beams", 1)),
            "do_sample": bool(g.get("do_sample", False)),
            "max_new_tokens": int(g.get("max_new_tokens", 200)),
            "return_timestamps": bool(g.get("return_timestamps", False)),
        }
        assert self.gen_kwargs["num_beams"] == 1 and not self.gen_kwargs["do_sample"], "pinned decode is greedy"

    def transcribe_batch(
        self, audios: list[np.ndarray], *, language: str | None, prompt: str | None
    ) -> list[DecodeResult]:
        import torch

        feats = self.processor(
            audios, sampling_rate=int(self.cfg.get("sampling_rate", 16000)), return_tensors="pt"
        ).input_features
        feats = feats.to(self.device, dtype=self.dtype)
        gen = dict(self.gen_kwargs)
        multi = bool(language) and "+" in language  # e.g. "ko+en": one language token each
        if language and not multi:
            gen["language"] = language
        if prompt and not multi:
            gen["prompt_ids"] = self.processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device)
            gen["prompt_condition_type"] = "first-segment"
        dec_in = None
        if multi:
            # prompt_ids and language are not passed beside decoder_input_ids: transformers would drop prompt_ids and
            # rebuild its own prefix
            ids = multi_language_prefix(self.processor, self.model, prompt=prompt, languages=language.split("+"))
            cap = int(gen.get("max_new_tokens", 200))
            gen["max_new_tokens"] = max(1, min(cap, int(self.model.config.max_target_positions) - len(ids) - 1))
            dec_in = torch.tensor([ids] * len(audios), device=self.device)
            gen["language_tokens"] = language  # hashed with the configuration, removed before generate
        with torch.inference_mode():
            if dec_in is not None:
                g = {k: v for k, v in gen.items() if k != "language_tokens"}
                out = self.model.generate(feats, return_dict_in_generate=True, decoder_input_ids=dec_in, **g)
            else:
                out = self.model.generate(feats, return_dict_in_generate=True, **gen)
        seqs = out.sequences if hasattr(out, "sequences") else out
        # With prompt_ids, `sequences` starts with the prompt and batch_decode keeps it; everything before the first
        # <|startoftranscript|> is conditioning and is cut by token position.
        sot = self.model.generation_config.decoder_start_token_id
        raws = self.processor.batch_decode(seqs, skip_special_tokens=False)
        gen_seqs, strip_failed = [], []
        for seq in seqs:
            ids = seq.tolist()
            if sot in ids:
                cut = ids.index(sot)
                strip_failed.append(0)
            else:  # no <|startoftranscript|>: keep everything and flag it
                cut = 0
                strip_failed.append(1)
            gen_seqs.append([int(x) for x in ids[cut:]])
        texts = self.processor.batch_decode(gen_seqs, skip_special_tokens=True)
        ghash = generation_hash({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in gen.items()})
        cfg_hash = generation_hash(
            {k: v for k, v in gen.items() if k not in ("prompt_ids", "prompt_condition_type")}
        )  # prompt-independent
        res = []
        for raw, text, seq, gseq, sf in zip(raws, texts, seqs, gen_seqs, strip_failed):
            res.append(
                DecodeResult(
                    text=text,
                    raw=raw,
                    prompt_slot=self.prompt_slot,
                    language_forced=bool(language),
                    meta={
                        "recognizer_id": self.id,
                        "checkpoint": self.checkpoint,
                        "revision": self.revision,
                        "device": self.device,
                        "dtype": str(self.dtype).replace("torch.", ""),
                        "batch_size": len(audios),
                        "generation_hash": ghash,
                        "generation_config_hash": cfg_hash,
                        "strip_failed": sf,
                        "prompt_given": int(bool(prompt)),
                        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest() if prompt else None,
                        "n_tokens": int(seq.shape[-1]),
                        "n_generated_tokens": len(gseq),
                        "prompt_in_sequence": int(seq.shape[-1] != len(gseq)),
                        "prompt_chars": len(prompt) if prompt else 0,
                        # generated ids from <|startoftranscript|> on
                        "token_ids": gseq,
                    },
                )
            )
        return res

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        return self.transcribe_batch([audio_16k], language=language, prompt=prompt)[0]
