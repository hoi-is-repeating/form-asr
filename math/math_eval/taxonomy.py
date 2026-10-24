"""Expression category by rule on the reference LaTeX: number, arithmetic, algebra, calculus, set_logic.
Plus the LaTeX token count (covariate). Rules are ordered: the first that matches wins."""

from __future__ import annotations

import re

_CALC = re.compile(r"\\(int|iint|oint|sum|prod|lim|partial|nabla|infty)\b|\bd[a-z]\b|\\frac\{\s*d|\\dot|\\prime|'")
_SET = re.compile(
    r"\\(in|notin|cup|cap|subset|subseteq|supset|forall|exists|neg|lor|land|emptyset|setminus|to|mapsto|implies|iff|Rightarrow|Leftrightarrow)\b|\\mathbb"
)
_LETTER = re.compile(r"[A-Za-z]|\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|omega)\b")
_NUMBER_ONLY = re.compile(r"^[\s\d.,/]+$")  # digits, decimal point, a fraction bar: a number (21/7, 3.5, \frac{1}{2})
_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z]|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def _strip(latex: str) -> str:
    s = (latex or "").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return s


def expression_category(latex: str) -> str:
    s = _strip(latex)
    if _CALC.search(s):
        return "calculus"
    if _SET.search(s):
        return "set_logic"
    body = re.sub(r"\\(frac|sqrt|cdot|times|div|pm|left|right|,|;|!)\b", " ", s)
    if _NUMBER_ONLY.match(re.sub(r"[{}]", "", body)) and _strip(s):
        return "number"
    if _LETTER.search(body):
        return "algebra"
    return "arithmetic"


def latex_tokens(latex: str) -> int:
    return len(_TOKEN.findall(_strip(latex)))
