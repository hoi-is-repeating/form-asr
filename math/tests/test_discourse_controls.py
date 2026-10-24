"""In-discourse clips: label file, D0/D1 spans, D2 truncation, and comparison words."""

import json

import numpy as np
import pytest

from math_eval import discourse
from math_eval.corpora import indiscourse as ind


def _clip(cbs=10.0, ms=40.0, me=43.2):
    return ind.Clip("f01", 0, "v1", "two over the square root of pi", "$2/\\sqrt{\\pi}$", 95.6, ms, me, "ctx", "", cbs)


def test_label_file_loads_the_decoded_set():
    rows = [json.loads(l) for l in ind.CLIPS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert not any(k in r for r in rows for k in ("transcription", "latex", "text"))
    clips = ind.load({r["original_index"]: ("x", "$x$") for r in rows})
    assert len(clips) == 317 and len({c.video_id for c in clips}) == 69 and not any(c.low_band for c in clips)
    for c in clips:
        a, b = c.d1
        assert b == c.math_end + ind.PAD and b - a <= ind.WINDOW + 1e-9 and a >= c.context_before_start - 1e-9
    assert (
        len(ind.load({rows[0]["original_index"]: ("x", "$x$")})) >= 1
        and ind.context()[rows[0]["sample_id"]]["context_before"]
    )


def test_d1_window_right_aligned_and_cuts_hash_stably():
    c = _clip()
    sp = discourse.spans(c)
    assert sp["d0"] == (40.0, 43.2) and sp["d1"] == pytest.approx((13.7, 43.7)) and sp["d1"][1] - sp["d1"][0] <= 30.0
    assert discourse.spans(_clip(cbs=35.0, ms=50.0, me=53.6))["d1"] == pytest.approx((35.0, 54.1))
    wav = np.random.default_rng(0).standard_normal(16000 * 60).astype(np.float32)
    a, b = discourse.cut(wav, 40.0, 43.2), discourse.cut(wav, 40.0, 43.2)
    assert discourse.sha256_array(a) == discourse.sha256_array(b) and len(a) == int(3.2 * 16000)
    with pytest.raises(ValueError):
        discourse.cut(wav, 50.0, 50.0)


def test_d2_truncation_from_the_left():
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    text = " ".join(f"word{i}" for i in range(400))
    t, n = discourse.truncate_left_by_tokens(text, tok, 224)
    assert n <= 224 and t.endswith("word399") and not t.startswith("word0")
    short, n2 = discourse.truncate_left_by_tokens("a few words only", tok, 224)
    assert short == "a few words only" and n2 < 10


def test_controls_from_caption_text_exclude_math_and_window_edge():
    spacy = pytest.importorskip("spacy")
    from math_eval import controls as ctl

    nlp = spacy.load("en_core_web_sm")
    text = "the error function was normalized by putting two over three and the denominators are these factorials"
    cs, wf = ctl.extract_controls("u", text, 10.0, 40.0, 13.7, nlp, "2 OVER SQRT PI")
    words = {c.word for c in cs}
    assert "normalized" in words and "denominators" in words and not {"two", "over", "three"} & words
    assert all(c.est_start >= 13.7 + ctl.BOUNDARY_GUARD for c in cs) and wf["caption_words"] == len(text.split())
    cov = ctl.covariates(text="square root of pi", tokenizer=None, g2p=None, min_over_words=True)
    assert cov["log_freq"] > 0 and cov["n_words"] == 4 and cov["n_phones"] is None and cov["n_phones_missing"] == 1
