"""Comparison words and covariates for the in-discourse set.

Comparison words are content words (spaCy en_core_web_sm NOUN / VERB / ADJ / ADV; alphabetic; at least 3 letters; not a number word or
a mathematical surface; not a word of the item) of the caption text before the expression, inside the D1 window. Captions give no word
timing, so word times are interpolated linearly over `context_before` between `context_before_start` and `math_start`, and words
within BOUNDARY_GUARD seconds of the window start are dropped, since they may fall outside the audio.
Covariates for items and comparison words alike: log_freq (wordfreq `en` Zipf; an expression takes the minimum over its words),
n_syll and n_phones (CMUdict through g2p_en: stress digits count syllables), n_bpe_<id> (tokenizer length).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from math_eval.mathforms import grammar
from math_eval.normalize_en import normalize

CONTROL_POS = {"NOUN", "VERB", "ADJ", "ADV"}
BOUNDARY_GUARD = 2.0
MIN_LETTERS = 3
_ALPHA_RE = re.compile(r"^[a-z]+$")


@dataclass
class Control:
    utt_id: str
    word: str
    index: int  # whitespace-token index inside context_before
    est_start: float
    pos: str
    lemma: str
    covariates: dict = field(default_factory=dict)


def interpolated_times(n_words: int, start: float, end: float) -> list[float]:
    if n_words <= 0:
        return []
    step = (end - start) / n_words
    return [start + i * step for i in range(n_words)]


def math_surfaces(grammar_obj=None) -> set[str]:
    """Words that are mathematical surfaces in the grammar (operators, numbers, functions, Greek names): never controls."""
    g = grammar_obj or grammar()
    words = {w for surf, _, side, _ in g.table for w in surf.split() if side == "word"}
    words |= set(g.units) | set(g.tens) | set(g.scales) | set(g.fractions) | {"point"}
    return words


def extract_controls(
    utt_id: str,
    context_before: str,
    context_before_start: float,
    math_start: float,
    d1_start: float,
    nlp,
    item_canonical: str,
    guard: float = BOUNDARY_GUARD,
) -> tuple[list[Control], dict]:
    wf = {
        "caption_words": 0,
        "inside_window": 0,
        "tag_ok": 0,
        "dropped_short_or_nonalpha": 0,
        "dropped_math_surface": 0,
        "dropped_item_surface": 0,
        "kept": 0,
    }
    words = context_before.split()
    wf["caption_words"] = len(words)
    times = interpolated_times(len(words), context_before_start, math_start)
    doc = nlp(" ".join(words))
    # align spaCy tokens to whitespace words by character offsets
    offs, pos = [], 0
    for w in words:
        offs.append(pos)
        pos += len(w) + 1
    tok_by_word: dict[int, list] = {}
    for t in doc:
        k = max(i for i, o in enumerate(offs) if o <= t.idx) if offs else 0
        tok_by_word.setdefault(k, []).append(t)
    surfaces = math_surfaces()
    item_toks = set(item_canonical.lower().split())
    out = []
    for k, w in enumerate(words):
        if times[k] < d1_start + guard:
            continue
        wf["inside_window"] += 1
        toks = tok_by_word.get(k, [])
        head = next((t for t in toks if t.pos_ in CONTROL_POS), None)
        if head is None:
            continue
        wf["tag_ok"] += 1
        nw = normalize(w)
        if not _ALPHA_RE.match(nw) or len(nw) < MIN_LETTERS:
            wf["dropped_short_or_nonalpha"] += 1
            continue
        if nw in surfaces:
            wf["dropped_math_surface"] += 1
            continue
        if nw in item_toks:
            wf["dropped_item_surface"] += 1
            continue
        wf["kept"] += 1
        out.append(Control(utt_id, nw, k, round(times[k], 2), head.pos_, head.lemma_.lower()))
    return out, wf


def _phones(text: str, g2p) -> list[str]:
    if g2p is None:
        return []
    return [p for p in g2p(text) if p.strip()]


def n_syllables(phones: Sequence[str]) -> int | None:
    return sum(1 for p in phones if p[-1:].isdigit()) if phones else None


def covariates(*, text: str, tokenizer, g2p, recognizer_id: str = "whisper", min_over_words: bool = False) -> dict:
    """log_freq (Zipf, en), freq_oov, n_syll, n_phones, n_bpe_<id>. For an expression, `min_over_words` takes the minimum Zipf over
    its alphabetic words."""
    from wordfreq import zipf_frequency

    words = [w for w in normalize(text).split() if _ALPHA_RE.match(w)]
    if min_over_words and words:
        z = min(zipf_frequency(w, "en") for w in words)
    else:
        z = zipf_frequency(normalize(text), "en")
    ph = _phones(text, g2p)
    return {
        "log_freq": z,
        "freq_oov": int(z == 0.0),
        "n_syll": n_syllables(ph),
        "n_phones": len(ph) if ph else None,
        "n_phones_missing": int(g2p is None),
        "n_words": len(normalize(text).split()),
        f"n_bpe_{recognizer_id}": len(tokenizer.encode(text, add_special_tokens=False))
        if tokenizer is not None
        else None,
    }
