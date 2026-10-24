"""Item table: Korean-matrix rule, loanword token location and the item waterfall.

`ref_form` is the script of a token in the reference ("hangul", "latin" or "mixed").
"""

from __future__ import annotations

import re
from typing import Iterable

from loanword_eval.corpora.hike import HikeItem, tokens
from loanword_eval.paths import Item

HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
LATIN_RE = re.compile(r"[A-Za-z]")


def hangul_token_share(text: str) -> tuple[int, int]:
    """(number of whitespace tokens containing at least one Hangul character, number of whitespace tokens)."""
    toks = text.split()
    return sum(1 for t in toks if HANGUL_RE.search(t)), len(toks)


def is_korean_matrix(text: str) -> bool | None:
    """Literal rule: a majority of whitespace tokens contain Hangul (strict majority). None on an exact tie."""
    h, n = hangul_token_share(text)
    if n == 0:
        return None
    if 2 * h == n:
        return None
    return 2 * h > n


def hangul_letter_share(text: str) -> float:
    """Hangul characters over Hangul + Latin letters."""
    h = len(HANGUL_RE.findall(text))
    l = len(LATIN_RE.findall(text))
    return h / (h + l) if (h + l) else 0.0


def ref_form_of(surface: str) -> str:
    has_h = bool(HANGUL_RE.search(surface))
    has_l = bool(LATIN_RE.search(surface))
    return "mixed" if (has_h and has_l) else ("hangul" if has_h else "latin")


def build_items(hike_items: Iterable[HikeItem]) -> tuple[list[Item], list[dict], dict]:
    """Return (items, token rows, waterfall); every listed pair is either located or counted as unlocatable."""
    items: list[Item] = []
    tok_rows: list[dict] = []
    wf = {
        "utterances_loaded": 0,
        "korean_matrix": 0,
        "english_matrix": 0,
        "matrix_tie": 0,
        "pairs_listed": 0,
        "pairs_located": 0,
        "pairs_unlocatable": 0,
        "tokens_total": 0,
        "tokens_korean_matrix": 0,
        "tokens_english_matrix": 0,
        "tokens_matrix_tie": 0,
        "tokens_ko_ref_latin": 0,
        "tokens_ko_ref_hangul": 0,
        "tokens_ko_ref_mixed": 0,
        "utterances_ko_with_items": 0,
    }
    for hi in hike_items:
        wf["utterances_loaded"] += 1
        km = is_korean_matrix(hi.text)
        h, n = hangul_token_share(hi.text)
        if km is None:
            wf["matrix_tie"] += 1
        elif km:
            wf["korean_matrix"] += 1
        else:
            wf["english_matrix"] += 1
        toks = tokens(hi)
        located_pairs = {(t.hangul, t.latin) for t in toks}
        listed = {(p.hangul, p.latin) for p in hi.pairs}
        wf["pairs_listed"] += len(listed)
        wf["pairs_located"] += len(listed & located_pairs)
        wf["pairs_unlocatable"] += len(listed - located_pairs)
        item = Item(
            utt_id=hi.utt_id,
            corpus=hi.corpus,
            split="test",
            speaker=hi.speaker,
            text=hi.text,
            pron=None,
            audio_path="",
            extra={
                "cs_level": hi.cs_level,
                "category": hi.category,
                "pairs": [[p.hangul, p.latin] for p in hi.pairs],
                "korean_matrix": km,
                "hangul_tokens": h,
                "n_tokens": n,
                "hangul_letter_share": round(hangul_letter_share(hi.text), 4),
                "unlocatable_pairs": [list(p) for p in sorted(listed - located_pairs)],
                "audio_problems": [],
            },
        )
        items.append(item)
        if km and toks:
            wf["utterances_ko_with_items"] += 1
        for t in toks:
            surface = hi.text[t.start : t.end]
            rf = ref_form_of(surface)
            wf["tokens_total"] += 1
            if km:
                wf["tokens_korean_matrix"] += 1
                wf[f"tokens_ko_ref_{rf}"] += 1
            elif km is None:
                wf["tokens_matrix_tie"] += 1
            else:
                wf["tokens_english_matrix"] += 1
            tok_rows.append(
                {
                    "item_id": f"{hi.utt_id}:{t.start}:{t.end}",
                    "utt_id": hi.utt_id,
                    "corpus": hi.corpus,
                    "start": t.start,
                    "end": t.end,
                    "surface": surface,
                    "hangul": t.hangul,
                    "latin": t.latin,
                    "ref_form": rf,
                    "occurrence": t.occurrence,
                    "korean_matrix": km,
                    "cs_level": hi.cs_level,
                    "type_key": f"{t.hangul}|{t.latin.lower()}",
                }
            )
    return items, tok_rows, wf


def conservation_ok(wf: dict) -> bool:
    return (
        wf["pairs_listed"] == wf["pairs_located"] + wf["pairs_unlocatable"]
        and wf["utterances_loaded"] == wf["korean_matrix"] + wf["english_matrix"] + wf["matrix_tie"]
        and wf["tokens_total"] == wf["tokens_korean_matrix"] + wf["tokens_english_matrix"] + wf["tokens_matrix_tie"]
    )
