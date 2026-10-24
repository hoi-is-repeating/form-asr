r"""Conversion-stage normalizer for LaTeX.

Speech-to-LaTeX (Korzh et al., ICLR 2026) normalizes predictions and references with `ProcessLaTeXFormulaTools` before computing CER;
that tool is the PyPI package `process_formula` (KaTeX-based; requires Node.js). Strings are compared after `NormalizeFormula` and a
light post-step: strip `$` delimiters and code fences, remove `\displaystyle`, `\left` and `\right`, and collapse whitespace, so that
`$\dfrac{1}{2}$` and `\frac{1}{2}` agree. Exact match and character error rate (jiwer) are the two measures.
"""

from __future__ import annotations

import re
from functools import lru_cache

import jiwer

_DOLLAR_RE = re.compile(r"^\$+|\$+$")
_WS_RE = re.compile(r"\s+")
_NOOP_RE = re.compile(r"\\displaystyle\b|\\left(?=\s*[\(\[\|\\{.])|\\right(?=\s*[\)\]\|\\}.])")


def strip_delims(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:latex)?\s*|```\s*$", "", s, flags=re.MULTILINE).strip()  # code fences from LLM outputs
    s = _DOLLAR_RE.sub("", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    return s


@lru_cache(maxsize=1)
def _nf():
    from process_formula import NormalizeFormula  # imported lazily: needs Node.js at construction

    return NormalizeFormula()


def normalize_many(latex: list[str]) -> list[str]:
    """Batch normalization (one Node call per batch). Empty strings stay empty."""
    cleaned = [strip_delims(x) for x in latex]
    idx = [i for i, x in enumerate(cleaned) if x]
    out = list(cleaned)
    if idx:
        res = _nf()([cleaned[i] for i in idx])
        for i, r in zip(idx, res):
            out[i] = _WS_RE.sub(" ", _NOOP_RE.sub("", r or "")).strip()
    return out


def exact_match(pred_norm: str, gold_norm: str) -> int:
    return int(pred_norm.replace(" ", "") == gold_norm.replace(" ", "") and gold_norm != "")


def cer(pred_norm: str, gold_norm: str) -> float:
    """Character error rate on the normalized strings, spaces removed (as Speech-to-LaTeX does); 1.0 when gold is empty."""
    g = gold_norm.replace(" ", "")
    p = pred_norm.replace(" ", "")
    if not g:
        return 1.0
    if not p:
        return 1.0
    return float(jiwer.cer(g, p))
