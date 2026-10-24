"""Wrapper stubs: contract, prompt slot, PromptUnsupported and config pins, with each toolkit monkeypatched."""

import numpy as np
import pytest

from loanword_eval.decoders import PROMPT_CAPABLE, RECOGNIZERS
from loanword_eval.decoders.base import Decoder, DecodeResult, PromptUnsupported
from loanword_eval.paths import load_yaml

WAV = np.zeros(16000, dtype=np.float32)


class _OwsmModel:
    def __call__(self, speech, text_prev=None, lang_sym=None, task_sym=None):
        self.last = dict(text_prev=text_prev, lang_sym=lang_sym, task_sym=task_sym)
        return [("<kor><asr> 안녕", ["안", "녕"], [1, 2], "안녕", None)]


class _QwenModel:
    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        class R:
            text, language = "안녕", "Korean"

        self.last = dict(language=language, context=context, n=len(audio) if isinstance(audio, list) else 1)
        return [R() for _ in (audio if isinstance(audio, list) else [audio])]


class _RaonPipe:
    def stt(self, path):
        return "안녕"

    def chat(self, messages):  # the sampling path; the wrapper must never call it
        raise AssertionError(
            "RaonPipeline.chat samples (do_sample=True hard-coded); the wrapper must use its greedy copy"
        )

    def chat_greedy(self, messages):
        self.last = messages
        return "안녕"


class _HFModel:
    device, dtype = "cpu", None

    def generate(self, **kw):
        return np.array([[1, 2, 3]])


class _HFProc:
    def __call__(self, audio, **kw):
        class B(dict):
            def to(self, *a, **k):
                return self

        self.last = kw
        return B()

    def batch_decode(self, out, skip_special_tokens=False):
        return ["<s> 안녕" if not skip_special_tokens else "안녕"]

    def decode(self, out, skip_special_tokens=False):
        return "<s> 안녕" if not skip_special_tokens else "안녕"


class _OmniPipe:
    def transcribe(self, inputs, lang, batch_size):
        self.last = dict(lang=lang, n=len(inputs))
        return ["안녕"] * len(inputs)


def _make(rid):
    cfg = load_yaml(f"decode_{rid}.yaml")
    if rid == "owsm":
        from loanword_eval.decoders.owsm import OwsmDecoder

        return OwsmDecoder(cfg, model=_OwsmModel()), cfg
    if rid == "qwen3asr":
        from loanword_eval.decoders.qwen3asr import Qwen3AsrDecoder

        return Qwen3AsrDecoder(cfg, model=_QwenModel()), cfg
    if rid == "raon":
        from loanword_eval.decoders.raon import RaonDecoder

        return RaonDecoder(cfg, model=_RaonPipe()), cfg
    if rid == "cohere":
        from loanword_eval.decoders.cohere import CohereDecoder

        return CohereDecoder(cfg, model=_HFModel(), processor=_HFProc()), cfg
    if rid == "voxtral":
        from loanword_eval.decoders.voxtral import VoxtralDecoder

        return VoxtralDecoder(cfg, model=_HFModel(), processor=_HFProc()), cfg
    if rid == "omni":
        from loanword_eval.decoders.omni import OmniDecoder

        return OmniDecoder(cfg, model=_OmniPipe()), cfg
    raise KeyError(rid)


@pytest.mark.parametrize("rid", [r for r in RECOGNIZERS if r != "whisper"])
def test_contract_slot_and_pins(rid):
    dec, cfg = _make(rid)
    assert isinstance(dec, Decoder) and dec.id == rid == cfg["id"]
    assert dec.prompt_slot == cfg["prompt_slot"] and cfg.get("revision") and cfg.get("env_name")
    r = dec.transcribe(WAV, language="ko", prompt=None)
    assert isinstance(r, DecodeResult) and r.text == "안녕" and r.prompt_slot == dec.prompt_slot
    assert r.meta["recognizer_id"] == rid and r.meta["revision"] == cfg["revision"] and r.meta["generation_config_hash"]
    if rid in PROMPT_CAPABLE:
        r2 = dec.transcribe(WAV, language="ko", prompt="택시 커피")
        assert r2.meta["prompt_given"] == 1 and r2.meta["prompt_chars"] == 5
    else:
        with pytest.raises(PromptUnsupported):
            dec.transcribe(WAV, language="ko", prompt="택시 커피")


def test_owsm_passes_text_prev_and_kor_symbol():
    dec, _ = _make("owsm")
    dec.transcribe(WAV, language="ko", prompt="택시")
    assert dec.model.last == {"text_prev": "택시", "lang_sym": "<kor>", "task_sym": "<asr>"}


def test_raon_sends_framed_text_verbatim_and_one_api_path():
    dec, _ = _make("raon")
    framed = "택시 커피\nTranscribe the audio into text."
    r = dec.transcribe(WAV, language=None, prompt=framed)
    text_part = [c for c in dec.pipe.last[0]["content"] if c["type"] == "text"][0]["text"]
    assert text_part == framed and r.meta["api_path"] == "chat_greedy" and r.meta["prompt_sha256"]
    r0 = dec.transcribe(WAV, language=None, prompt=None)
    assert r0.meta["user_text"] == "Transcribe the audio into text." and r0.meta["api_path"] == "chat_greedy"


class _RaonProc:
    class _Tok:
        def decode(self, ids, skip_special_tokens=False):
            self.last = (list(ids), skip_special_tokens)
            return "안녕"

    def __init__(self):
        self.tokenizer = self._Tok()

    def __call__(self, messages, **kw):
        self.last = kw

        class _T:
            def __init__(self, v):
                self.v = v

            def sum(self):
                return self

            def item(self):
                return self.v

        return {"input_ids": "ids", "attention_mask": _T(3), "audio_input": "a", "audio_input_lengths": "al"}


class _RaonModel:
    def generate(self, **kw):
        self.last = kw
        return {"sequences": np.array([[9, 9, 9, 5, 6]])}


class _RaonRealShape:
    """The attributes RaonPipeline exposes (model, processor, device, dtype) without `chat_greedy`: exercises the copied path."""

    device, dtype = "cpu", "bf16"

    def __init__(self):
        self.model, self.processor = _RaonModel(), _RaonProc()

    def chat(self, messages, **kw):
        raise AssertionError("chat samples; must not be called")


def test_raon_greedy_copy_of_chat_passes_do_sample_false_and_slices_the_prompt():
    """The copy of RaonPipeline.chat's text path calls generate with do_sample=False and force_text_output=True and decodes
    only the tokens after the prompt."""
    from loanword_eval.decoders.raon import RaonDecoder
    from loanword_eval.paths import load_yaml

    pipe = _RaonRealShape()
    dec = RaonDecoder(load_yaml("decode_raon.yaml"), model=pipe)
    r = dec.transcribe(WAV, language=None, prompt=None)
    kw = pipe.model.last
    assert (
        kw["do_sample"] is False
        and kw["force_text_output"] is True
        and kw["force_audio_output"] is False
        and kw["speaker_audio"] is None
    )
    assert kw["max_new_tokens"] == 1024 and kw["audio_input"] == "a" and kw["audio_input_lengths"] == "al"
    assert pipe.processor.last == {
        "add_generation_prompt": True,
        "force_audio_output": False,
        "device": "cpu",
        "dtype": "bf16",
        "max_audio_chunk_length": 192000,
    }
    assert pipe.processor.tokenizer.last == ([5, 6], True) and r.text == "안녕"


def test_qwen_context_is_the_system_text_and_none_is_the_empty_context():
    """qwen-asr `transcribe(context=...)` fills the system message; no prompt passes "" through the same call."""
    dec, _ = _make("qwen3asr")
    dec.transcribe(WAV, language="ko", prompt=None)
    assert dec.model.last["context"] == "" and dec.model.last["n"] == 1
    r = dec.transcribe(WAV, language="ko", prompt="택시 커피")
    assert dec.model.last["context"] == "택시 커피" and r.meta["inject"] == "context"
    rs = dec.transcribe_batch([WAV, WAV, WAV], language="ko", prompt="택시")
    assert len(rs) == 3 and dec.model.last["n"] == 3 and dec.model.last["context"] == "택시"


def test_owsm_auto_language_refused_without_lid():
    dec, _ = _make("owsm")
    with pytest.raises(ValueError):
        dec.transcribe(WAV, language=None, prompt=None)


class _FakeLid:
    def __init__(self):
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        return [("<kor>", 0.91), ("<jpn>", 0.05), ("<eng>", 0.02)]


def test_owsm_auto_language_runs_lid_then_decodes_with_the_detected_token():
    """OWSM without a language: Speech2Language top token, then Speech2Text primed with it; the batch path loops."""
    from loanword_eval.decoders.owsm import OwsmDecoder

    lid = _FakeLid()
    dec = OwsmDecoder(load_yaml("decode_owsm.yaml"), model=_OwsmModel(), lid=lid)
    r = dec.transcribe(WAV, language=None, prompt=None)
    assert lid.calls == 1 and dec.model.last["lang_sym"] == "<kor>" and r.language_forced is False
    assert (
        r.meta["language_detected"] == "<kor>"
        and r.meta["lid_prob"] == 0.91
        and r.meta["lid_nbest"][1] == ["<jpn>", 0.05]
        and r.meta["lang_sym"] == "<kor>"
    )
    rs = dec.transcribe_batch([WAV, WAV], language=None, prompt=None)
    assert len(rs) == 2 and lid.calls == 3
    r_ko = dec.transcribe(WAV, language="ko", prompt=None)
    assert r_ko.language_forced is True and "language_detected" not in r_ko.meta and lid.calls == 3


def test_qwen_language_mapping_and_omni_lang_code():
    dec, _ = _make("qwen3asr")
    dec.transcribe(WAV, language="ko", prompt=None)
    assert dec.model.last["language"] == "Korean"
    dec, _ = _make("omni")
    dec.transcribe(WAV, language="ko", prompt=None)
    assert dec.pipe.last["lang"] == ["kor_Hang"]
    dec2, _ = _make("cohere")
    dec2.transcribe(WAV, language="ko", prompt=None)
    assert dec2.processor.last["language"] == "ko" and dec2.processor.last["punctuation"] is False


class _OwsmBatchModel(_OwsmModel):
    """Test double with ESPnet's batch_decode signature: records the call and returns one n-best list per utterance."""

    class _Conv:
        def tokens2ids(self, toks):
            return [100 + i for i, _ in enumerate(toks)]

    class _Tok:
        def text2tokens(self, text):
            return list(text)

    converter, tokenizer = _Conv(), _Tok()

    def batch_decode(
        self,
        speech,
        speech_lengths=None,
        text_prev=None,
        text_prev_lengths=None,
        lang_sym=None,
        task_sym=None,
        predict_time=None,
    ):
        self.batch = dict(
            n=int(speech.shape[0]),
            nsamples=int(speech.shape[1]),
            text_prev=None if text_prev is None else text_prev.tolist(),
            lengths=None if text_prev_lengths is None else text_prev_lengths.tolist(),
            lang_sym=lang_sym,
            task_sym=task_sym,
        )
        return [[(f"<kor><asr> 안녕{i}", ["안", "녕"], [1, 2], f"안녕{i}", None)] for i in range(int(speech.shape[0]))]


def test_owsm_batch_path_uses_batch_decode_with_equal_length_prompt_ids():
    from loanword_eval.decoders.owsm import OwsmDecoder

    dec = OwsmDecoder(load_yaml("decode_owsm.yaml"), model=_OwsmBatchModel())
    auds = [np.zeros(16000, dtype=np.float32), np.zeros(8000, dtype=np.float32), np.zeros(12000, dtype=np.float32)]
    res = dec.transcribe_batch(auds, language="ko", prompt="택시")
    b = dec.model.batch
    assert b["n"] == 3 and b["nsamples"] == 16000 and b["lang_sym"] == "<kor>" and b["task_sym"] == "<asr>"
    assert b["text_prev"] == [[100, 101]] * 3 and b["lengths"] == [
        2,
        2,
        2,
    ]
    assert [r.text for r in res] == ["안녕0", "안녕1", "안녕2"] and all(
        r.meta["prompt_given"] == 1 and r.meta["lang_sym"] == "<kor>" for r in res
    )
    res0 = dec.transcribe_batch(auds, language="ko", prompt=None)
    assert dec.model.batch["text_prev"] is None and all(r.meta["prompt_given"] == 0 for r in res0)
    r1 = dec.transcribe_batch(auds[:1], language="ko", prompt="택시")
    assert r1[0].text == "안녕" and dec.model.last["text_prev"] == "택시"


def test_batched_registry_names_only_wrappers_that_decode_a_batch_in_one_call():
    from loanword_eval import decoders

    assert decoders.BATCHED == ("whisper", "owsm", "qwen3asr", "omni")
    assert "cohere" not in decoders.BATCHED and "voxtral" not in decoders.BATCHED
