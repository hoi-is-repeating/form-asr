"""Locate a labeled surface string inside a reference and validate offsets.

Labels store surfaces and an occurrence index; offsets are recomputed here and re-checked with `validate`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


class SpanNotFound(ValueError):
    pass


class AmbiguousSpan(ValueError):
    pass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    surface: str


def locate(text: str, surface: str, occurrence: int | None = 1) -> Span:
    """Return the half-open span of the k-th non-overlapping occurrence of `surface` in `text`.

    Both strings are compared in NFC. `occurrence` is 1-based. If `occurrence` is None and the
    surface occurs more than once, raise AmbiguousSpan.
    """
    text_n = unicodedata.normalize("NFC", text)
    surf_n = unicodedata.normalize("NFC", surface)
    if not surf_n:
        raise SpanNotFound("empty surface")
    hits: list[int] = []
    pos = 0
    while True:
        i = text_n.find(surf_n, pos)
        if i < 0:
            break
        hits.append(i)
        pos = i + len(surf_n)
    if not hits:
        raise SpanNotFound(f"{surface!r} not in text")
    if occurrence is None:
        if len(hits) > 1:
            raise AmbiguousSpan(f"{surface!r} occurs {len(hits)} times and no occurrence index given")
        occurrence = 1
    if occurrence < 1 or occurrence > len(hits):
        raise SpanNotFound(f"occurrence {occurrence} of {surface!r} does not exist ({len(hits)} found)")
    start = hits[occurrence - 1]
    return Span(start, start + len(surf_n), surf_n)


def validate(text: str, start: int, end: int, surface: str) -> bool:
    """True iff text[start:end] equals surface after NFC."""
    text_n = unicodedata.normalize("NFC", text)
    return 0 <= start < end <= len(text_n) and text_n[start:end] == unicodedata.normalize("NFC", surface)
