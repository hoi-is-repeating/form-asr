"""OWSM v4 medium via ESPnet, following the model card example (`Speech2Text.from_pretrained`, `s2t(speech)`).

The prompt is passed as `text_prev`, which ESPnet places before the task tokens (`[sop] + text_prev + [sos, lang, task]`): the
same preceding-text slot as Whisper's prompt. Decoding is greedy. The pinned revision is loaded as `<repo>@<revision>` and the
checkpoint sha256 is recorded per row. With `language=None`, ESPnet's `Speech2Language` picks the language token first.
"""

from __future__ import annotations

import numpy as np

from loanword_eval.decoders.base import DecodeResult, generation_hash


class OwsmDecoder:
    id = "owsm"
    prompt_slot = "prefix"

    def __init__(self, cfg: dict, device: str | None = None, model=None, lid=None):
        self.cfg = cfg
        self.lid = lid  # Speech2Language, built lazily on the first `language=None`
        self._device = device
        self.checkpoint = cfg["checkpoint"]
        self.revision = cfg.get("revision")
        g = cfg.get("generation", {})
        self.lang_sym = cfg.get("lang_sym", "<kor>")
        self.task_sym = cfg.get("task_sym", "<asr>")
        self.beam_size = int(g.get("beam_size", 1))
        assert self.beam_size == 1, "pinned decode is greedy"
        self.resolved_files: dict = {}
        self.model_file_sha256: str | None = None
        if model is not None:  # injected in tests
            self.model = model
        else:
            import hashlib

            import torch
            from espnet2.bin.s2t_inference import Speech2Text
            from espnet2.utils.pretrained import download_pretrained

            # Equivalent to Speech2Text.from_pretrained; espnet_model_zoo passes the revision in "<repo>@<revision>" to
            # snapshot_download.
            tag = f"{self.checkpoint}@{self.revision}" if self.revision else self.checkpoint
            files = download_pretrained(tag)  # s2t_train_config, s2t_model_file
            self.resolved_files = {k: str(v) for k, v in files.items()}
            mf = files.get("s2t_model_file")
            if self.revision and (mf is None or self.revision not in str(mf)):
                raise RuntimeError(f"owsm: resolved checkpoint {mf} is not the pinned revision {self.revision}")
            h = hashlib.sha256()
            with open(mf, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            self.model_file_sha256 = h.hexdigest()
            self.model = Speech2Text(
                **files,
                device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
                lang_sym=self.lang_sym,
                task_sym=self.task_sym,
                beam_size=self.beam_size,
                maxlenratio=float(g.get("maxlenratio", 1.0)),
                predict_time=False,
            )
        self.gen_hash = generation_hash(
            {
                "beam_size": self.beam_size,
                "lang_sym": self.lang_sym,
                "task_sym": self.task_sym,
                "maxlenratio": g.get("maxlenratio", 1.0),
            }
        )

    def transcribe(self, audio_16k: np.ndarray, *, language: str | None, prompt: str | None) -> DecodeResult:
        detected = None
        if language is None:
            lang_sym, detected = self._detect_language(audio_16k)
        else:
            lang_sym = self._lang_sym(language)
        out = self.model(audio_16k, text_prev=prompt, lang_sym=lang_sym, task_sym=self.task_sym)
        return self._result(out[0], language, prompt, lang_sym, detected)

    def _lang_sym(self, language: str) -> str:
        return {"ko": "<kor>", "en": "<eng>"}.get(language, language)

    def _detect_language(self, audio_16k: np.ndarray) -> tuple[str, dict]:
        """Top language token and probabilities from ESPnet's Speech2Language."""
        if self.lid is None:
            if not self.resolved_files:
                raise ValueError(
                    "owsm: automatic language identification needs ESPnet's Speech2Language; none was injected and no checkpoint "
                    "files are resolved (test double without `lid`)"
                )
            import torch
            from espnet2.bin.s2t_inference_language import Speech2Language

            self.lid = Speech2Language(
                s2t_train_config=self.resolved_files["s2t_train_config"],
                s2t_model_file=self.resolved_files["s2t_model_file"],
                device=self._device or ("cuda" if torch.cuda.is_available() else "cpu"),
                nbest=3,
            )
        nbest = self.lid(audio_16k)
        lang, prob = nbest[0]
        return str(lang), {
            "language_detected": str(lang),
            "lid_prob": float(prob),
            "lid_nbest": [[str(l), float(p)] for l, p in nbest],
        }

    def _result(self, out_tuple, language, prompt, lang_sym, detected=None) -> DecodeResult:
        text, token, token_int, text_nospecial, hyp = out_tuple
        return DecodeResult(
            text=text_nospecial if text_nospecial is not None else text,
            raw=text,
            prompt_slot=self.prompt_slot,
            language_forced=language is not None,
            meta={
                "recognizer_id": self.id,
                "checkpoint": self.checkpoint,
                "revision": self.revision,
                "generation_config_hash": self.gen_hash,
                "generation_hash": generation_hash({"g": self.gen_hash, "prompt": prompt or ""}),
                "n_tokens": len(token_int),
                "prompt_given": int(bool(prompt)),
                "prompt_chars": len(prompt) if prompt else 0,
                "lang_sym": lang_sym,
                **(detected or {}),
                "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest() if prompt else None,
                "model_file_sha256": self.model_file_sha256,
            },
        )

    def transcribe_batch(self, audios, *, language, prompt):
        """Batched decoding through `Speech2Text.batch_decode`; loops over utterances for a batch of one, a model without
        `batch_decode`, or language identification."""
        if len(audios) <= 1 or not hasattr(self.model, "batch_decode") or language is None:
            return [self.transcribe(a, language=language, prompt=prompt) for a in audios]
        import torch

        lang_sym = self._lang_sym(language)
        n = max(len(a) for a in audios)
        speech = torch.zeros((len(audios), n), dtype=torch.float32)
        for i, a in enumerate(audios):
            speech[i, : len(a)] = torch.as_tensor(np.asarray(a, dtype=np.float32))
        text_prev = text_prev_lengths = None
        if prompt:
            ids = self.model.converter.tokens2ids(self.model.tokenizer.text2tokens(prompt))
            text_prev = torch.tensor([ids] * len(audios), dtype=torch.long)
            text_prev_lengths = torch.full((len(audios),), len(ids), dtype=torch.long)
        outs = self.model.batch_decode(speech, None, text_prev, text_prev_lengths, lang_sym, self.task_sym)
        return [self._result(o[0], language, prompt, lang_sym) for o in outs]
