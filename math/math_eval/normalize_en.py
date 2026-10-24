"""Recognition-stage normalizer for English mathematical speech.

It keeps everything that can carry written form. Whisper's BasicTextNormalizer is not used: it maps every symbol and punctuation
character to a space and deletes parenthesized text, which would erase notation.

Steps: superscript/subscript runs -> "^..." / "_...", fold look-alike symbols, NFKC, fold again, lower-case, listed sentence
punctuation -> space, apostrophe inside a word removed (a prime stays), whitespace collapsed. Digits, operator symbols, Greek
letters, brackets, the percent sign and the degree sign are kept.
"""

from __future__ import annotations

import re
import unicodedata

# Sentence punctuation that never carries form. Handled separately: the apostrophe (a prime unless inside a word), "!" (factorial
# unless sentence punctuation) and "." (kept inside numbers); the hyphen is kept.
_SENTENCE_PUNCT = ',;:?"“”‘…'
_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_PUNCT_RE = re.compile("[" + re.escape(_SENTENCE_PUNCT) + "]")
_BANG_RE = re.compile(
    r"(?<![0-9)\]a-zA-Z])!|(?<=[a-zA-Z]{2})!"
)  # "!" after a word of >= 2 letters or after nothing mathematical is punctuation; "n!", "5!", ")!" stay
_APOS_IN_WORD_RE = re.compile(r"(?<=[a-z]{2})'(?=[a-z])")  # "that's" -> "thats"; "f'" and "x's" keep the prime
_WS_RE = re.compile(r"\s+")

# A run of superscript or subscript characters becomes "^" or "_" followed by the base characters.
_SUP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "n",
    "ⁱ": "i",
}
_SUB = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
    "ₙ": "n",
    "ᵢ": "i",
    "ₖ": "k",
    "ₘ": "m",
    "ₜ": "t",
    "ₓ": "x",
}
_SUP_RE = re.compile("[" + "".join(map(re.escape, _SUP)) + "]+")
_SUB_RE = re.compile("[" + "".join(map(re.escape, _SUB)) + "]+")

# Unicode minus and similar look-alikes are folded to their ASCII/common form so that the canonicaliser sees one surface per symbol.
_FOLD = str.maketrans(
    {
        "−": "-",  # minus sign
        "–": "-",  # en dash
        "—": "-",  # em dash
        "⋅": "·",  # dot operator -> middle dot
        "∙": "·",  # bullet operator -> middle dot
        "∕": "/",  # division slash
        "⁄": "/",  # fraction slash (also produced by NFKC)
        "′": "'",  # prime
        "″": "''",  # double prime
        "’": "'",  # right single quotation mark used as an apostrophe / prime
        "Σ": "∑",  # capital sigma written for the sum (before lower-casing)
        "Π": "∏",  # capital pi written for the product
    }
)


def normalize(text: str) -> str:
    """Apply the steps listed in the module docstring."""
    if not text:
        return ""
    s = _SUP_RE.sub(lambda m: "^" + "".join(_SUP[c] for c in m.group(0)), text)
    s = _SUB_RE.sub(lambda m: "_" + "".join(_SUB[c] for c in m.group(0)), s)
    s = s.translate(_FOLD)  # fold before NFKC
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_FOLD)  # and after: NFKC produces look-alikes of its own
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _APOS_IN_WORD_RE.sub("", s)
    s = _BANG_RE.sub(" ", s)
    s = _DOT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
