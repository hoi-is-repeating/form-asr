"""Character-level alignment and span outcomes on top of jiwer.

`span_outcome` walks the jiwer alignment for a reference span (in normalized coordinates)
and returns what the hypothesis did with it. Insertions adjacent to the span are attributed
with the Le et al. (2021) rule: attributed to the span if the inserted text is Latin or in a
caller-supplied lexicon, otherwise split half to each neighbour (reported, not scored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import jiwer

Outcome = Literal["CORRECT", "SUB", "DEL", "PARTIAL"]


@dataclass
class CharAlignment:
    ref: str
    hyp: str
    # per reference char: (op, aligned hyp char or None)
    ref_ops: list[tuple[str, str | None]]
    # insertions keyed by the reference index they precede; len(ref) means trailing
    insertions: dict[int, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def char_align(ref: str, hyp: str) -> CharAlignment:
    """Align two already-normalized strings at character level."""
    if not ref:
        raise ValueError("empty reference")
    out = jiwer.process_characters(ref, hyp)
    chunks = out.alignments[0]
    ref_ops: list[tuple[str, str | None]] = [("delete", None)] * len(ref)
    insertions: dict[int, str] = {}
    for ch in chunks:
        if ch.type == "equal":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ref_ops[ch.ref_start_idx + k] = ("equal", hyp[ch.hyp_start_idx + k])
        elif ch.type == "substitute":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ref_ops[ch.ref_start_idx + k] = ("substitute", hyp[ch.hyp_start_idx + k])
        elif ch.type == "delete":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ref_ops[ch.ref_start_idx + k] = ("delete", None)
        elif ch.type == "insert":
            insertions[ch.ref_start_idx] = insertions.get(ch.ref_start_idx, "") + hyp[ch.hyp_start_idx : ch.hyp_end_idx]
    counts = {"S": out.substitutions, "D": out.deletions, "I": out.insertions, "H": out.hits, "N": len(ref)}
    return CharAlignment(ref, hyp, ref_ops, insertions, counts)


@dataclass(frozen=True)
class SpanResult:
    outcome: Outcome
    ref_span: str
    hyp_span: str
    hyp_span_with_ins: str  # hyp_span plus insertions inside or attributed adjacent to the span
    n_equal: int
    n_sub: int
    n_del: int
    n_ins_inside: int
    n_ins_adjacent: int
    span_cer: float  # (sub + del + ins_inside) / len(ref_span)


def span_outcome(
    al: CharAlignment, start: int, end: int, attribute_adjacent: Callable[[str], bool] | None = None
) -> SpanResult:
    """Classify what happened to ref[start:end].

    `attribute_adjacent(text) -> bool` decides whether an insertion immediately before or after
    the span is attributed to it (Le et al. 2021: yes if Latin or in the biasing lexicon).
    """
    if not (0 <= start < end <= len(al.ref)):
        raise ValueError("span out of range")
    ops = al.ref_ops[start:end]
    n_equal = sum(1 for o, _ in ops if o == "equal")
    n_sub = sum(1 for o, _ in ops if o == "substitute")
    n_del = sum(1 for o, _ in ops if o == "delete")
    hyp_chars = [h for o, h in ops if h is not None]
    # insertions strictly inside the span (preceding ref index in (start, end))
    inside = "".join(al.insertions.get(i, "") for i in range(start + 1, end))
    adjacent_before = al.insertions.get(start, "")
    adjacent_after = al.insertions.get(end, "")
    attributed = ""
    n_adj = 0
    for txt in (adjacent_before, adjacent_after):
        if txt and (attribute_adjacent is None or attribute_adjacent(txt)):
            attributed += txt
            n_adj += len(txt)
    hyp_span = "".join(hyp_chars)
    pieces: list[str] = []
    if adjacent_before and (attribute_adjacent is None or attribute_adjacent(adjacent_before)):
        pieces.append(adjacent_before)
    for i in range(start, end):
        h = al.ref_ops[i][1]
        if h is not None:
            pieces.append(h)
        if i + 1 < end:
            pieces.append(al.insertions.get(i + 1, ""))
    if adjacent_after and (attribute_adjacent is None or attribute_adjacent(adjacent_after)):
        pieces.append(adjacent_after)
    hyp_with_ins = "".join(pieces)
    n = end - start
    span_cer = (n_sub + n_del + len(inside)) / n
    if n_equal == n and not inside:
        outcome: Outcome = "CORRECT"
    elif n_del == n:
        outcome = "DEL"
    elif n_equal == 0:
        outcome = "SUB"
    else:
        outcome = "PARTIAL"
    return SpanResult(
        outcome, al.ref[start:end], hyp_span, hyp_with_ins, n_equal, n_sub, n_del, len(inside), n_adj, span_cer
    )
