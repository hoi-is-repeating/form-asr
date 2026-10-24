"""Per-type predicted direction from the unprompted decode.

For each item type (Hangul|Latin pair) the direction is the form produced most often among recognized tokens. Ties,
mixed-dominant types and types with no recognized token fall back to Latin and are flagged `fallback=1`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

APRIORI = "latin"


def direction_rows(span_rows: Iterable[dict], recognizer_id: str) -> tuple[list[dict], dict]:
    forms: dict[str, Counter] = defaultdict(Counter)
    n_tokens: Counter = Counter()
    corpus: dict[str, set] = defaultdict(set)
    for r in span_rows:
        n_tokens[r["type_key"]] += 1
        corpus[r["type_key"]].add(r["corpus"])
        R, F = (
            r.get("itt_R", r["R"]),
            r.get("itt_F", r["F"]),
        )
        if R and F:
            forms[r["type_key"]][F] += 1
    rows, wf = [], Counter()
    for tk in sorted(n_tokens):
        c = forms.get(tk, Counter())
        n_rec = sum(c.values())
        if n_rec == 0:
            dom, fb, reason = APRIORI, 1, "no_recognized_token"
        else:
            top = c.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                dom, fb, reason = APRIORI, 1, "tie"
            elif top[0][0] == "mixed":
                dom, fb, reason = APRIORI, 1, "mixed_dominant"
            else:
                dom, fb, reason = top[0][0], 0, None
        wf["types"] += 1
        wf["fallback"] += fb
        wf[f"dominant_{dom}"] += 1
        rows.append(
            {
                "type_key": tk,
                "recognizer_id": recognizer_id,
                "n_tokens": n_tokens[tk],
                "n_recognized": n_rec,
                "n_latin": c.get("latin", 0),
                "n_hangul": c.get("hangul", 0),
                "n_mixed": c.get("mixed", 0),
                "dominant": dom,
                "fallback": fb,
                "fallback_reason": reason,
                "corpora": sorted(corpus[tk]),
            }
        )
    return rows, dict(wf)


def direction_bytes(rows: list[dict]) -> bytes:
    """The exact bytes `write_direction` writes (one JSON object per line, UTF-8, unescaped Hangul)."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def write_direction(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(direction_bytes(rows))
