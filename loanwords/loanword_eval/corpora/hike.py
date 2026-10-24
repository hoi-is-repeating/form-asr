"""HiKE loader (Paik et al., Findings of EACL 2026): HF dataset `thetaone-ai/HiKE`, split `test`.

Fields: audio, text, text_normalized, text_pier_labeled, cs_level, cs_levels_all, category, sample_id, and `loanwords`, a JSON
list of {"Korean", "English"} pairs that may be written in either script. There is no speaker column. Both spellings of each
pair are located in the reference text.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from loanword_eval.spans import SpanNotFound, locate

HF_ID = "thetaone-ai/HiKE"


@dataclass(frozen=True)
class LoanwordPair:
    hangul: str
    latin: str


@dataclass
class HikeItem:
    utt_id: str
    text: str
    text_normalized: str
    cs_level: str
    cs_levels_all: tuple[str, ...]
    category: str
    pairs: tuple[LoanwordPair, ...]
    audio: dict | None = None
    speaker: str | None = None  # not provided by the dataset
    corpus: str = "hike"
    notes: list[str] = field(default_factory=list)


def parse_pairs(raw: str | list | None) -> tuple[LoanwordPair, ...]:
    if raw is None or raw == "":
        return ()
    obj = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for e in obj:
        k = unicodedata.normalize("NFC", str(e.get("Korean", ""))).strip()
        en = unicodedata.normalize("NFC", str(e.get("English", ""))).strip()
        if k or en:
            out.append(LoanwordPair(k, en))
    return tuple(out)


def _from_row(row: dict) -> HikeItem:
    return HikeItem(
        utt_id=row["sample_id"],
        text=unicodedata.normalize("NFC", row["text"]),
        text_normalized=unicodedata.normalize("NFC", row.get("text_normalized", "")),
        cs_level=row.get("cs_level", ""),
        cs_levels_all=tuple(x for x in str(row.get("cs_levels_all", "")).split(",") if x),
        category=row.get("category", ""),
        pairs=parse_pairs(row.get("loanwords")),
        audio=row.get("audio"),
    )


def _decode_audio(raw: dict, target_sr: int = 16000) -> dict:
    """Decode audio bytes with soundfile and resample to 16 kHz (no torchcodec dependency)."""
    import io

    import numpy as np
    import soundfile as sf

    if raw is None:
        return None
    if "array" in raw and raw.get("array") is not None:
        arr, sr = np.asarray(raw["array"], dtype=np.float32), int(raw.get("sampling_rate", target_sr))
    else:
        data = raw.get("bytes")
        if data is None and raw.get("path"):
            arr, sr = sf.read(raw["path"], dtype="float32")
        else:
            arr, sr = sf.read(io.BytesIO(data), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, target_sr)
        arr = resample_poly(arr, target_sr // g, sr // g).astype(np.float32)
    return {"array": arr, "sampling_rate": target_sr}


def load(split: str = "test", with_audio: bool = True) -> Iterator[HikeItem]:
    """Stream HiKE from the HF hub (cached after the first download). Audio is decoded with soundfile."""
    from datasets import Audio, load_dataset

    ds = load_dataset(HF_ID, split=split)
    if not with_audio:
        ds = ds.remove_columns(["audio"])
    else:
        ds = ds.cast_column("audio", Audio(decode=False))
    for row in ds:
        if with_audio:
            row = dict(row)
            row["audio"] = _decode_audio(row["audio"])
        yield _from_row(row)


def load_rows(rows: Iterable[dict]) -> list[HikeItem]:
    """Build items from plain dict rows (used by tests and by the datasets-server sample)."""
    return [_from_row(r) for r in rows]


_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _occurrences(text: str, surface: str) -> int:
    if not surface:
        return 0
    # Latin surfaces are matched case-insensitively at word boundaries; Hangul surfaces as substrings
    if any("LATIN" in unicodedata.name(c, "") for c in surface):
        pat = _WORD_RE_CACHE.get(surface)
        if pat is None:
            pat = re.compile(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])", re.IGNORECASE)
            _WORD_RE_CACHE[surface] = pat
        return len(pat.findall(text))
    return text.count(surface)


@dataclass(frozen=True)
class LoanwordToken:
    utt_id: str
    hangul: str
    latin: str
    realization: str  # "hangul" | "latin"
    start: int
    end: int
    occurrence: int


def tokens(item: HikeItem) -> list[LoanwordToken]:
    """Every located occurrence of every pair in the reference text, in both spellings.

    Tokens = occurrences in `text`, not the length of the pairs list (a pair can occur twice
    or not at all as written).
    """
    out: list[LoanwordToken] = []
    for p in item.pairs:
        for realization, surf in (("hangul", p.hangul), ("latin", p.latin)):
            n = _occurrences(item.text, surf)
            for k in range(1, n + 1):
                try:
                    sp = locate(item.text, surf, k) if realization == "hangul" else _locate_latin(item.text, surf, k)
                except SpanNotFound:
                    continue
                out.append(LoanwordToken(item.utt_id, p.hangul, p.latin, realization, sp.start, sp.end, k))
    # drop overlapping tokens, keeping the earlier (then longer) one
    out.sort(key=lambda t: (t.start, -(t.end - t.start)))
    kept: list[LoanwordToken] = []
    last_end = -1
    for t in out:
        if t.start >= last_end:
            kept.append(t)
            last_end = t.end
    return kept


def _locate_latin(text: str, surface: str, k: int):
    from loanword_eval.spans import Span

    pat = re.compile(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])", re.IGNORECASE)
    hits = list(pat.finditer(text))
    if k < 1 or k > len(hits):
        raise SpanNotFound(f"latin occurrence {k} of {surface!r} not found")
    m = hits[k - 1]
    return Span(m.start(), m.end(), text[m.start() : m.end()])
