"""Utterance flags for mathematical speech: hallucination, empty output, off-task text, prompt copy, and a solved instead of
transcribed expression. Off-task matching and the character-level prompt-copy rule come from loanword_eval.flags; `empty`
accepts an all-notation output, `hallucinated` allows an absolute slack for short references, and prompt copy is counted on
canonical tokens."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from loanword_eval import flags as _lw

from math_eval.normalize_en import normalize

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_WORD_RE = re.compile(r"\w", re.UNICODE)
HALLUCINATION_SLACK = 20  # characters allowed beyond ratio * reference length before the length rule fires
FLAG_KEYS = ("hallucinated", "empty", "off_task", "prompt_copy", "solved")


def empty(hyp: str) -> int:
    """1 iff the hypothesis has no word character at all (letters, digits, or marks in any script)."""
    return int(not _WORD_RE.search(hyp or ""))


def hallucinated(
    ref: str, hyp: str, ratio: float = 1.5, slack: int = HALLUCINATION_SLACK, ngram: int = 4, repeats: int = 3
) -> int:
    """Length rule on normalized text with an absolute slack, or a repeated word n-gram the reference does not contain."""
    r, h = normalize(ref or "").replace(" ", ""), normalize(hyp or "").replace(" ", "")
    if len(h) > ratio * len(r) + slack:
        return 1
    # word n-grams: character 4-grams repeat naturally in English mathematics ("theta", "squared"); a looping decoder repeats phrases
    hr = _max_word_ngram_repeat(hyp, ngram)
    return int(hr >= repeats and hr > _max_word_ngram_repeat(ref, ngram))


def _max_word_ngram_repeat(text: str, n: int = 4) -> int:
    w = normalize(text or "").split()
    if len(w) < n:
        return 0
    from collections import Counter

    return max(Counter(tuple(w[i : i + n]) for i in range(len(w) - n + 1)).values())


def prompt_copy(
    hyp: str, prompt: str | None, ref: str | None = None, n_tokens: int = 3, unprompted_hyp: str | None = None
) -> int:
    """A run of at least n_tokens consecutive canonical prompt tokens that occurs in the hypothesis but not in the reference.
    Canonical tokens neutralize the written form, so sharing one symbol or operator word with the prompt does not count."""
    if not prompt:
        return 0
    from math_eval.mathforms import grammar

    g = grammar()
    p, h, r = (
        g.canonical(prompt).compared_tokens,
        g.canonical(hyp or "").compared_tokens,
        g.canonical(ref or "").compared_tokens,
    )

    def _has(seq, sub):
        m = len(sub)
        return any(seq[i : i + m] == sub for i in range(len(seq) - m + 1))

    if len(p) < n_tokens:
        return int(bool(p) and _has(h, p) and not _has(r, p))
    u = g.canonical(unprompted_hyp).compared_tokens if unprompted_hyp is not None else None
    # given the unprompted transcript of the same audio, a prompt n-gram counts only when that transcript lacks it
    return int(
        any(
            _has(h, p[i : i + n_tokens])
            and not _has(r, p[i : i + n_tokens])
            and (u is None or not _has(u, p[i : i + n_tokens]))
            for i in range(len(p) - n_tokens + 1)
        )
    )


def load_off_task_patterns(name: str = "off_task_patterns.yaml") -> tuple[list[re.Pattern], str]:
    import hashlib

    p = CONFIGS / name
    cfg = yaml.safe_load(p.read_text())
    return [re.compile(x, re.IGNORECASE) for x in cfg["patterns"]], hashlib.sha256(p.read_bytes()).hexdigest()


def load_solved_patterns(name: str = "solved_patterns.yaml") -> tuple[list[re.Pattern], str]:
    p = CONFIGS / name
    cfg = yaml.safe_load(p.read_text())
    pats = [re.compile(x, re.IGNORECASE) for x in cfg["patterns"]]
    return pats, cfg["version"]


def solved(ref: str, hyp: str, patterns: list[re.Pattern]) -> int:
    """1 iff the hypothesis states a result the reference does not: an answer frame from the pattern list, or an 'equals <number>'
    where the reference contains no equality."""
    h = hyp or ""
    # an answer frame counts only when the reference does not contain the same phrase
    for p in patterns:
        if p.search(h) and not re.search(
            p.pattern.lstrip("^").replace("\\s*", "", 1) if p.pattern.startswith("^") else p.pattern,
            ref or "",
            re.IGNORECASE,
        ):
            return 1
    from math_eval.mathforms import grammar

    g = grammar()
    ht, rt = g.canonical(h).compared_tokens, g.canonical(ref or "").compared_tokens
    # an equality followed by a number that the reference does not contain ("is", "equals" and "=" share one canonical token)
    ref_pairs = {(rt[i], rt[i + 1]) for i in range(len(rt) - 1)}
    if any(
        ht[i] == "EQ"
        and i + 1 < len(ht)
        and ht[i + 1].replace(".", "").isdigit()
        and (ht[i], ht[i + 1]) not in ref_pairs
        and "EQ" not in rt
        for i in range(len(ht))
    ):
        return 1
    return 0


DISCOURSE_CLIP_WORDS = (12, 12)  # caption words before / after the expression in the flag reference of D0 / D2 clips
DISCOURSE_WINDOW_AFTER = 5  # D1 windows end 0.5 s after the expression, so only a few following words fit


def discourse_flag_reference(item_text: str, context_before: str, lead_out: str, context_after: str, cond: str) -> str:
    """Flag reference of an in-discourse decode: the caption text its audio can contain. That is the last 12 words before the
    expression (D1: the whole context), the expression, and the first 12 caption words after it (D1: 5)."""
    before = (context_before or "").split()
    after = ((lead_out or "") + " " + (context_after or "")).split()
    if cond == "D1":
        return " ".join(before + [item_text] + after[:DISCOURSE_WINDOW_AFTER])
    nb, na = DISCOURSE_CLIP_WORDS
    return " ".join(before[-nb:] + [item_text] + after[:na])


def utterance_flags(
    ref: str,
    hyp: str,
    prompt: str | None,
    off_task_patterns: list[re.Pattern],
    solved_patterns: list[re.Pattern],
    frame_text: str | None = None,
    unprompted_hyp: str | None = None,
) -> dict:
    """Flags of one transcript; `prompt_copy_chars` and `prompt_copy_ref` are descriptive."""
    return {
        "hallucinated": hallucinated(ref, hyp),
        "empty": empty(hyp),
        "off_task": _lw.off_task(hyp, off_task_patterns, frame_text=frame_text, ref=ref),
        "prompt_copy": prompt_copy(hyp, prompt, ref, unprompted_hyp=unprompted_hyp),
        "prompt_copy_chars": _lw.prompt_copy(
            normalize(hyp or ""), normalize(prompt) if prompt else None
        ),  # 6-character rule
        "prompt_copy_ref": _lw.prompt_copy(
            normalize(ref or ""), normalize(prompt) if prompt else None
        ),  # prompt material in the reference itself
        "solved": solved(ref, hyp, solved_patterns),
    }


def any_flag(f: dict) -> int:
    return int(any(f.get(k) for k in FLAG_KEYS))
