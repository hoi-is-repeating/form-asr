"""Text normalization policies with an index map back to the original string.

KoreanPolicy: NFC -> strip punctuation -> lower-case Latin -> remove all whitespace.
Every kept character records the index it came from in the original string, so a span
located in the original can be mapped into the normalized string and back.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def _is_punct(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


@dataclass(frozen=True)
class Normalized:
    text: str
    index_map: tuple[int, ...]  # normalized position -> position in the NFC original
    original: str

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open span in normalized coordinates to original coordinates."""
        if start >= end:
            raise ValueError("empty span")
        return self.index_map[start], self.index_map[end - 1] + 1

    def to_normalized_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open span in original coordinates to normalized coordinates.

        Characters dropped by normalization inside the span are skipped; the result is the
        smallest normalized span covering every kept original character in [start, end).
        """
        kept = [i for i, orig in enumerate(self.index_map) if start <= orig < end]
        if not kept:
            raise ValueError("span contains no kept characters")
        return kept[0], kept[-1] + 1


class KoreanPolicy:
    name = "korean_v1"

    def __init__(self, strip_whitespace: bool = True, strip_punct: bool = True, lower_latin: bool = True):
        self.strip_whitespace = strip_whitespace
        self.strip_punct = strip_punct
        self.lower_latin = lower_latin

    def apply(self, text: str) -> Normalized:
        original = unicodedata.normalize("NFC", text)
        out: list[str] = []
        idx: list[int] = []
        for i, ch in enumerate(original):
            if self.strip_whitespace and ch.isspace():
                continue
            if self.strip_punct and _is_punct(ch):
                continue
            out.append(ch.lower() if self.lower_latin else ch)
            idx.append(i)
        return Normalized("".join(out), tuple(idx), original)


def contains_latin(text: str) -> bool:
    return any(c.isalpha() and "LATIN" in unicodedata.name(c, "") for c in text)
