"""Jamo-level, feature-weighted edit distance for Korean, plus a G2P wrapper.

Hangul syllables are decomposed into (initial, medial, final) jamo. The substitution cost between two jamo is 1 minus the
fraction of shared articulatory features, so pairs that loanword adaptation often conflates cost less than unrelated pairs.
An unweighted variant is available.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from jamo import h2j, j2hcj

# Consonants: (place, manner, laryngeal). Vowels: (height, backness, rounding[, glide]).
_CONS = {
    "ㄱ": ("velar", "stop", "lax"),
    "ㄲ": ("velar", "stop", "tense"),
    "ㅋ": ("velar", "stop", "asp"),
    "ㄷ": ("alveolar", "stop", "lax"),
    "ㄸ": ("alveolar", "stop", "tense"),
    "ㅌ": ("alveolar", "stop", "asp"),
    "ㅂ": ("labial", "stop", "lax"),
    "ㅃ": ("labial", "stop", "tense"),
    "ㅍ": ("labial", "stop", "asp"),
    "ㅈ": ("palatal", "affricate", "lax"),
    "ㅉ": ("palatal", "affricate", "tense"),
    "ㅊ": ("palatal", "affricate", "asp"),
    "ㅅ": ("alveolar", "fricative", "lax"),
    "ㅆ": ("alveolar", "fricative", "tense"),
    "ㅎ": ("glottal", "fricative", "asp"),
    "ㄴ": ("alveolar", "nasal", "voiced"),
    "ㅁ": ("labial", "nasal", "voiced"),
    "ㅇ": ("velar", "nasal", "voiced"),
    "ㄹ": ("alveolar", "liquid", "voiced"),
}
_VOW = {
    "ㅏ": ("low", "central", "unr"),
    "ㅓ": ("mid", "back", "unr"),
    "ㅗ": ("mid", "back", "rnd"),
    "ㅜ": ("high", "back", "rnd"),
    "ㅡ": ("high", "central", "unr"),
    "ㅣ": ("high", "front", "unr"),
    "ㅔ": ("mid", "front", "unr"),
    "ㅐ": ("mid", "front", "unr"),
    "ㅑ": ("low", "central", "unr", "y"),
    "ㅕ": ("mid", "back", "unr", "y"),
    "ㅛ": ("mid", "back", "rnd", "y"),
    "ㅠ": ("high", "back", "rnd", "y"),
    "ㅖ": ("mid", "front", "unr", "y"),
    "ㅒ": ("mid", "front", "unr", "y"),
    "ㅘ": ("low", "central", "unr", "w"),
    "ㅙ": ("mid", "front", "unr", "w"),
    "ㅚ": ("mid", "front", "rnd", "w"),
    "ㅝ": ("mid", "back", "unr", "w"),
    "ㅞ": ("mid", "front", "unr", "w"),
    "ㅟ": ("high", "front", "unr", "w"),
    "ㅢ": ("high", "central", "unr", "j"),
}
_COMPOUND_FINAL = {
    "ㄳ": "ㄱㅅ",
    "ㄵ": "ㄴㅈ",
    "ㄶ": "ㄴㅎ",
    "ㄺ": "ㄹㄱ",
    "ㄻ": "ㄹㅁ",
    "ㄼ": "ㄹㅂ",
    "ㄽ": "ㄹㅅ",
    "ㄾ": "ㄹㅌ",
    "ㄿ": "ㄹㅍ",
    "ㅀ": "ㄹㅎ",
    "ㅄ": "ㅂㅅ",
}


def to_jamo(text: str) -> str:
    """Decompose Hangul syllables into compatibility jamo; leave other characters as they are."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if "가" <= ch <= "힣":
            j = j2hcj(h2j(ch))
            for c in j:
                out.append(_COMPOUND_FINAL.get(c, c))
        else:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=None)
def sub_cost(a: str, b: str, weighted: bool = True) -> float:
    if a == b:
        return 0.0
    if not weighted:
        return 1.0
    fa, fb = _CONS.get(a) or _VOW.get(a), _CONS.get(b) or _VOW.get(b)
    if fa is None or fb is None:
        return 1.0
    if (a in _CONS) != (b in _CONS):
        return 1.0
    n = max(len(fa), len(fb))
    fa = tuple(fa) + ("",) * (n - len(fa))
    fb = tuple(fb) + ("",) * (n - len(fb))
    shared = sum(1 for x, y in zip(fa, fb) if x == y)
    return 1.0 - shared / n


def jamo_distance(a: str, b: str, weighted: bool = True, normalize: bool = True) -> float:
    """Levenshtein distance over jamo with feature-weighted substitution; normalized by max length."""
    ja, jb = to_jamo(a), to_jamo(b)
    n, m = len(ja), len(jb)
    if n == 0 and m == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub_cost(ja[i - 1], jb[j - 1], weighted))
        prev = cur
    d = prev[m]
    return d / max(n, m) if normalize else d


class G2P:
    """Thin wrapper over g2pk2/g2pK (not required for the rest of the module)."""

    def __init__(self):
        self._impl = None
        for mod in ("g2pk2", "g2pk"):
            try:
                self._impl = __import__(mod).G2p()
                self.backend = mod
                break
            except Exception:
                continue
        if self._impl is None:
            raise RuntimeError("no Korean G2P backend installed (pip install g2pk2)")

    def __call__(self, text: str, descriptive: bool = False) -> str:
        return self._impl(text, descriptive=descriptive, group_vowels=True)

    def divergence(self, text: str) -> float:
        """Jamo distance between prescriptive and descriptive pronunciations."""
        return jamo_distance(self(text, False), self(text, True))
