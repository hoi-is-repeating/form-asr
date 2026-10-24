"""Ordinary-word controls and covariates.

Controls are kiwipiepy tokens of the same utterances tagged NNG/NNP/VV/VA, at least 2 Hangul characters, no Latin letters or
digits, not overlapping an item or dual-notation span, and not in any acceptance set. Covariates: log_freq (wordfreq Zipf,
0.0 = out of vocabulary), n_syll, n_phones (jamo count of the g2pk2 output), n_bpe_<id> (recognizer tokenizer length).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from loanword_eval.items import HANGUL_RE, LATIN_RE
from loanword_eval.normalize import KoreanPolicy
from loanword_eval.phon import to_jamo

CONTROL_TAGS = {"NNG", "NNP", "VV", "VA"}
SYLLABLE_RE = re.compile(r"[가-힣]")
DIGIT_RE = re.compile(r"\d")
_POLICY = KoreanPolicy()


@dataclass
class Control:
    utt_id: str
    start: int
    end: int
    surface: str
    tag: str
    lemma: str
    covariates: dict = field(default_factory=dict)


def _overlaps(s: int, e: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(s < e2 and s2 < e for s2, e2 in spans)


def extract_controls(
    utt_id: str,
    text: str,
    item_spans: Sequence[tuple[int, int]],
    kiwi,
    accept_members: set[str],
    dual_spans: Sequence[tuple[int, int]] = (),
) -> tuple[list[Control], dict]:
    wf = {
        "kiwi_tokens": 0,
        "tag_ok": 0,
        "dropped_short": 0,
        "dropped_latin_or_digit": 0,
        "dropped_overlap_item": 0,
        "dropped_dual_span": 0,
        "dropped_in_accept_set": 0,
        "kept": 0,
    }
    out = []
    for t in kiwi.tokenize(text):
        wf["kiwi_tokens"] += 1
        if t.tag not in CONTROL_TAGS:
            continue
        wf["tag_ok"] += 1
        s, e = t.start, t.start + t.len
        surface = text[s:e]
        if len(surface) < 2:
            wf["dropped_short"] += 1
            continue
        if LATIN_RE.search(surface) or DIGIT_RE.search(surface) or not HANGUL_RE.search(surface):
            wf["dropped_latin_or_digit"] += 1
            continue
        if _overlaps(s, e, item_spans):
            wf["dropped_overlap_item"] += 1
            continue
        if _overlaps(s, e, dual_spans):
            wf["dropped_dual_span"] += 1
            continue
        if _POLICY.apply(surface).text in accept_members or _POLICY.apply(t.form).text in accept_members:
            wf["dropped_in_accept_set"] += 1
            continue
        wf["kept"] += 1
        out.append(Control(utt_id, s, e, surface, t.tag, t.form))
    return out, wf


def n_syllables(hangul_form: str) -> int:
    return len(SYLLABLE_RE.findall(hangul_form))


def n_phones(hangul_form: str, g2p: Callable[[str], str] | None) -> int | None:
    if g2p is None:
        return None
    pron = g2p(hangul_form)
    return len([j for j in to_jamo(pron) if j.strip()])


def covariates(
    *,
    ref_surface: str,
    hangul_form: str,
    tokenizer,
    g2p: Callable[[str], str] | None = None,
    freq_form: str | None = None,
    recognizer_id: str = "whisper",
) -> dict:
    """Covariates of one token; `freq_form` overrides the form looked up in wordfreq (sets freq_proxy=1)."""
    from wordfreq import zipf_frequency

    look = freq_form or ref_surface
    z = zipf_frequency(look, "ko")
    return {
        "log_freq": z,
        "freq_oov": int(z == 0.0),
        "freq_proxy": int(look != ref_surface),
        "freq_form": look,
        "n_syll": n_syllables(hangul_form),
        "n_phones": n_phones(hangul_form, g2p),
        "n_phones_missing": int(g2p is None),
        f"n_bpe_{recognizer_id}": len(tokenizer.encode(ref_surface, add_special_tokens=False))
        if tokenizer is not None
        else None,
    }
