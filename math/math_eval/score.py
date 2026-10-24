"""Item scoring for spoken mathematics on canonical token sequences. Identical for every recognizer.
norm(x) is the normalized string and canon(x) its canonical token sequence.

whole clip (benchmark clips): the item is the entire reference
  R        = 1 iff canon(h) == canon(r)
  R_strict = 1 iff norm(h) == norm(r)
  F        = words | notation | mixed from the sides that fired on the transcript (when R = 1); `latex` is set by the caller
  outcome  = CORRECT (strict) | FORM (R, not strict) | WRONG | EMPTY
in context (in-discourse clips): the item is the reference's canonical token sequence
  R        = 1 iff that sequence occurs contiguously in canon(h)
  R_tail   = the same within the last len(ref) + TAIL tokens
  F        from the sides of the matched span
token_cer: normalized Levenshtein distance between the reference tokens and the best-matching transcript span (descriptive)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rapidfuzz.distance import Levenshtein

from math_eval.mathforms import form_of_sides, grammar
from math_eval.normalize_en import normalize

TAIL = 8
LATEX_HINT = ("\\", "^{", "_{", "$")


@dataclass
class ItemScore:
    R: int
    R_strict: int
    F: str | None  # words | notation | mixed | latex | None
    outcome: str  # CORRECT | FORM | WRONG | EMPTY
    canonical_ref: str
    canonical_hyp: str  # whole hypothesis (clip) or the matched span (context)
    n_tokens_ref: int
    token_cer: float
    R_tail: int | None = None  # in-context only
    match_start: int | None = None
    match_end: int | None = None
    n_tokens_hyp: int | None = None

    def row(self) -> dict:
        return asdict(self)


def looks_like_latex(hyp: str) -> bool:
    return any(h in (hyp or "") for h in LATEX_HINT)


def _cer(a: list[str], b: list[str]) -> float:
    if not a:
        return 1.0
    return min(1.0, Levenshtein.distance(a, b) / len(a))


def score_clip(ref_text: str, hyp_text: str) -> ItemScore:
    g = grammar()
    cr, ch = g.canonical(ref_text), g.canonical(hyp_text or "")
    rt, ht = cr.compared_tokens, ch.compared_tokens
    strict = int(normalize(ref_text) == normalize(hyp_text or "") and bool(normalize(ref_text)))
    R = int(rt == ht and bool(rt))
    if not (hyp_text or "").strip():
        outcome = "EMPTY"
    elif strict:
        outcome = "CORRECT"
    elif R:
        outcome = "FORM"
    else:
        outcome = "WRONG"
    return ItemScore(
        R=R,
        R_strict=strict,
        F=(form_of_sides(ch.compared_sides) if R else None),
        outcome=outcome,
        canonical_ref=" ".join(rt),
        canonical_hyp=" ".join(ht),
        n_tokens_ref=len(rt),
        token_cer=_cer(rt, ht),
        n_tokens_hyp=len(ht),
    )


def _find(seq: list[str], sub: list[str], lo: int = 0) -> int | None:
    n, m = len(seq), len(sub)
    if m == 0:
        return None
    for i in range(max(lo, 0), n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return None


def _best_span(ref: list[str], hyp: list[str]) -> tuple[int, int, float]:
    """Hypothesis token span closest to the reference tokens (rapidfuzz partial alignment on sequences); (start, end, cer)."""
    from rapidfuzz import fuzz

    if not hyp or not ref:
        return 0, 0, 1.0
    al = fuzz.partial_ratio_alignment(ref, hyp)
    if al is None:
        return 0, 0, 1.0
    s, e = al.dest_start, al.dest_end
    return s, e, _cer(ref, hyp[s:e])


def score_in_context(ref_text: str, hyp_text: str, tail: int = TAIL) -> ItemScore:
    g = grammar()
    cr, ch = g.canonical(ref_text), g.canonical(hyp_text or "")
    rt, ht, hs = cr.compared_tokens, ch.compared_tokens, ch.compared_sides
    if not (hyp_text or "").strip():
        return ItemScore(0, 0, None, "EMPTY", " ".join(rt), "", len(rt), 1.0, 0, None, None, 0)
    pos = _find(ht, rt)
    pos_tail = _find(ht, rt, lo=len(ht) - len(rt) - tail)
    if pos is not None:
        span = (pos, pos + len(rt))
        F = form_of_sides(hs[span[0] : span[1]])
        # strict: approximated as the reference's normalized token sequence occurring in norm(h)
        strict = int(normalize(ref_text) in normalize(hyp_text) and bool(normalize(ref_text)))
        return ItemScore(
            1,
            strict,
            F,
            "CORRECT" if strict else "FORM",
            " ".join(rt),
            " ".join(ht[span[0] : span[1]]),
            len(rt),
            0.0,
            int(pos_tail is not None),
            span[0],
            span[1],
            len(ht),
        )
    s, e, cer = _best_span(rt, ht)
    return ItemScore(0, 0, None, "WRONG", " ".join(rt), " ".join(ht[s:e]), len(rt), cer, 0, s, e, len(ht))


def item_surface_in_context(ref_text: str, hyp_text: str, slack: int = 4) -> str:
    """Transcript words that carry the item inside a D1 window: the window ends 0.5 s after the expression, so the shortest word
    suffix whose canonical tokens contain the reference's; when the reference is not found, the suffix as long as the best-aligned
    span plus `slack` tokens."""
    g = grammar()
    words = (hyp_text or "").split()
    rt = g.canonical(ref_text).compared_tokens
    if not words or not rt:
        return ""
    for k in range(1, len(words) + 1):
        if _find(g.canonical(" ".join(words[-k:])).compared_tokens, rt) is not None:
            return " ".join(words[-k:])
    ht = g.canonical(hyp_text).compared_tokens
    s_, e_, _ = _best_span(rt, ht)
    want = max(1, len(ht) - s_ + slack)
    for k in range(1, len(words) + 1):
        if len(g.canonical(" ".join(words[-k:])).compared_tokens) >= want:
            return " ".join(words[-k:])
    return " ".join(words)


def item_surface_window(ref_text: str, hyp_text: str) -> str:
    """Transcript words that carry the item inside a D0 / D2 clip, the input of the converter: the shortest word window whose canonical
    tokens contain the reference's; when the reference is not found, the whole transcript."""
    g = grammar()
    words = (hyp_text or "").split()
    rt = g.canonical(ref_text).compared_tokens
    if not words or not rt:
        return hyp_text or ""
    for n in range(1, len(words) + 1):
        for s in range(0, len(words) - n + 1):
            if _find(g.canonical(" ".join(words[s : s + n])).compared_tokens, rt) is not None:
                return " ".join(words[s : s + n])
    return " ".join(words)


def control_strict(word: str, hyp_text: str) -> int:
    """Comparison word scored strictly: its normalized form occurs as a whole token of norm(h), anywhere."""
    w = normalize(word).strip()
    return int(bool(w) and w in normalize(hyp_text or "").split())
