"""Dominant written form per form category, from the surfaces that fired on the recognized items of the no-prompt decode.

Categories: numbers, operators (relations and arithmetic), roots_exponents, greek, functions. Greek names and function names do not
decide the written form, so their rows are descriptive. A category with fewer than MIN_TOKENS recognized tokens, or a tie, falls back
to the a-priori form (`notation` for numbers, `words` otherwise) and is flagged. An item's category is that of its first form-deciding
canonical token.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from math_eval.mathforms import grammar

MIN_TOKENS = 10
ROOT_EXP = {"SQRT", "CBRT", "POW"}
APRIORI = {
    "numbers": "notation",
    "operators": "words",
    "roots_exponents": "words",
    "greek": "words",
    "functions": "words",
}
DECIDING = ("numbers", "operators", "roots_exponents")


def category_of(cat: str, canon: str) -> str:
    if cat == "numbers":
        return "numbers"
    if cat == "greek":
        return "greek"
    if cat == "functions":
        return "functions"
    return "roots_exponents" if canon.split()[0] in ROOT_EXP else "operators"


def fired_by_category(text: str) -> dict[str, Counter]:
    """Counts of word-side and symbol-side surfaces that fired on `text`, per category."""
    g = grammar()
    c = g.canonical(text)
    out: dict[str, Counter] = defaultdict(Counter)
    canon_of = {}
    for surf, canon, side, cat in g.table:
        canon_of[(surf, cat)] = canon
    for cat, side, surf in c.fired:
        canon = canon_of.get((surf, cat), surf)
        if canon.split()[0] in g.grouping:
            continue  # grouping is recorded but not compared and must not count as notation
        out[category_of(cat, canon)][side] += 1
    return out


def item_category(reference: str) -> str | None:
    """Category of the first form-deciding canonical token of the reference (the item's dominant-form cell)."""
    g = grammar()
    c = g.canonical(reference)
    canon_of = {(surf, cat): canon for surf, canon, side, cat in g.table}
    for cat, side, surf in c.fired:
        canon = canon_of.get((surf, cat), surf)
        if canon.split()[0] in g.grouping:
            continue
        k = category_of(cat, canon)
        if k in DECIDING:
            return k
    return None


def direction_rows(span_rows: Iterable[dict], recognizer_id: str) -> tuple[list[dict], dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    n_items = 0
    for r in span_rows:
        n_items += 1
        if not r.get("itt_R", r.get("R")):
            continue
        for cat, cnt in (r.get("fired_by_category") or {}).items():
            counts[cat].update({k: int(v) for k, v in cnt.items()})
    rows, wf = [], Counter()
    for cat in ("numbers", "operators", "roots_exponents", "greek", "functions"):
        c = counts.get(cat, Counter())
        n_w, n_s = c.get("word", 0), c.get("symbol", 0)
        if cat not in DECIDING:
            dom, fb, reason = None, 0, "non-deciding category (descriptive)"
        elif n_w + n_s < MIN_TOKENS:
            dom, fb, reason = APRIORI[cat], 1, f"fewer than {MIN_TOKENS} recognized tokens"
        elif n_w == n_s:
            dom, fb, reason = APRIORI[cat], 1, "tie"
        else:
            dom, fb, reason = ("words" if n_w > n_s else "notation"), 0, None
        wf["categories"] += 1
        wf["fallback"] += fb
        if dom:
            wf[f"dominant_{dom}"] += 1
        rows.append(
            {
                "category": cat,
                "recognizer_id": recognizer_id,
                "n_word_side": n_w,
                "n_symbol_side": n_s,
                "deciding": cat in DECIDING,
                "dominant": dom,
                "fallback": fb,
                "fallback_reason": reason,
            }
        )
    wf["items_seen"] = n_items
    return rows, dict(wf)


def direction_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def write_direction(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(direction_bytes(rows))


def load_direction(path: str | Path) -> dict[str, dict]:
    return {r["category"]: r for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}
