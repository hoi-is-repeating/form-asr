"""Utterance-level flags: translated, hallucinated, empty, off_task, prompt_copy. Identical for every recognizer."""

from __future__ import annotations

import re

from loanword_eval.items import hangul_letter_share
from loanword_eval.normalize import KoreanPolicy
from loanword_eval.paths import CONFIGS, load_yaml, sha256_of

POLICY = KoreanPolicy()
LETTER_RE = re.compile(r"[가-힣A-Za-z]")


def load_off_task_patterns(name: str = "off_task_patterns.yaml") -> tuple[list[re.Pattern], str]:
    cfg = load_yaml(name)
    return [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]], sha256_of(CONFIGS / name)


def _max_ngram_repeat(text: str, n: int = 4) -> int:
    t = POLICY.apply(text).text
    if len(t) < n:
        return 0
    from collections import Counter

    return max(Counter(t[i : i + n] for i in range(len(t) - n + 1)).values())


def hallucinated(ref: str, hyp: str, ratio: float = 1.5, ngram: int = 4, repeats: int = 3) -> int:
    """Length-ratio rule on normalized text, or a repeated n-gram in the hypothesis that the reference does not contain."""
    r, h = POLICY.apply(ref).text, POLICY.apply(hyp).text
    if len(h) > ratio * max(1, len(r)):
        return 1
    hr = _max_ngram_repeat(h, ngram)
    return int(hr >= repeats and hr > _max_ngram_repeat(r, ngram))


def translated(ref: str, hyp: str) -> int:
    """Reference Hangul-letter share > 0.5 and hypothesis Hangul-letter share < 0.2."""
    return int(hangul_letter_share(ref) > 0.5 and hangul_letter_share(hyp) < 0.2 and bool(LETTER_RE.search(hyp)))


def empty(hyp: str) -> int:
    return int(not LETTER_RE.search(hyp or ""))


def prompt_copy(hyp: str, prompt: str | None, n: int = 6) -> int:
    """At least n consecutive prompt characters verbatim (spaces included) in the hypothesis."""
    if not prompt:
        return 0
    p = prompt.strip()
    if len(p) < n:
        return int(p in hyp)
    return int(any(p[i : i + n] in hyp for i in range(len(p) - n + 1)))


def off_task(hyp: str, patterns: list[re.Pattern], frame_text: str | None = None, ref: str | None = None) -> int:
    """Frame text or an off-task pattern in the hypothesis; a pattern hit is ignored when the reference matches it too."""
    if frame_text and frame_text.strip() and frame_text.strip().lower() in hyp.lower():
        return 1
    for p in patterns:
        if p.search(hyp) and not (ref is not None and p.search(ref)):
            return 1
    return 0


def utterance_flags(
    ref: str, hyp: str, prompt: str | None, patterns: list[re.Pattern], frame_text: str | None = None
) -> dict:
    return {
        "translated": translated(ref, hyp),
        "hallucinated": hallucinated(ref, hyp),
        "empty": empty(hyp),
        "off_task": off_task(hyp, patterns, frame_text, ref=ref),
        "prompt_copy": prompt_copy(hyp, prompt),
        "prompt_copy_ref": prompt_copy(ref, prompt),
    }


def any_flag(f: dict) -> int:
    return int(any(f[k] for k in ("translated", "hallucinated", "empty", "off_task", "prompt_copy")))
