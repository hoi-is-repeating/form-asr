"""Acceptance-set construction and the NIKL loanword-orthography example lookups (spelling variants, `VariantLexicon`).

Every lookup is cached on disk; the HTTP client is injectable so tests run offline."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Callable

Fetch = Callable[[str, dict], dict]  # (url, params) -> parsed JSON


def build_accept_set(
    hangul: str,
    *,
    hike_latin: str | None,
    nikl_original: str | None,
    pron_form: str | None,
    g2p: Callable[[str], str] | None,
) -> set[str]:
    """Hangul form, HiKE Latin pair, NIKL original, pronunciation form and g2p outputs of the Latin form (NFC, Latin
    lower-cased)."""
    acc = {unicodedata.normalize("NFC", hangul)}
    for x in (hike_latin, nikl_original, pron_form):
        if x:
            acc.add(unicodedata.normalize("NFC", x).lower())
    if g2p is not None:
        for latin in (hike_latin, nikl_original):
            if latin:
                try:
                    nat = g2p(latin)
                    if nat:
                        acc.add(unicodedata.normalize("NFC", nat))
                except Exception:
                    pass
    return {a for a in acc if a}


NIKL_EXAMPLE_URL = "https://korean.go.kr/kornorms/exampleReqList.do"


_MARK_ANNOTATION = re.compile(r"\([^)]*\)")  # NIKL appends "(X)" to incorrect notations


def _clean_mark(x: str) -> str:
    """Bare form of one NIKL notation: parenthesized annotations removed, NFC, stripped."""
    return unicodedata.normalize("NFC", _MARK_ANNOTATION.sub("", str(x)).strip())


def _split_marks(v) -> list[str]:
    """Split a `relate_mark_*` string on commas, slashes or semicolons (the middle dot joins one name and is kept).
    Entries containing ':' or Latin letters are annotations and are dropped."""
    if v is None:
        return []
    if isinstance(v, list):
        return [x for y in v for x in _split_marks(y)]
    parts = [_clean_mark(x) for x in str(v).replace("/", ",").replace(";", ",").split(",")]
    return [x for x in parts if x and ":" not in x and not re.search(r"[A-Za-z]", x)]


def _result_code(raw) -> tuple:
    """(resultCode, resultMsg) found anywhere in the payload, or (None, None)."""
    found = [None, None]

    def walk(x):
        if isinstance(x, dict):
            if "resultCode" in x and found[0] is None:
                found[0], found[1] = x.get("resultCode"), x.get("resultMsg")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(raw)
    return tuple(found)


def parse_nikl_examples(raw) -> list[dict]:
    """Records of the NIKL loanword example API (korean.go.kr/kornorms): any dict with a `korean_mark` field.

    Fields: korean_mark (standard Hangul notation), relate_mark_e (variant notations), relate_mark_o (incorrect notations),
    srclang_mark (original), lang_nm (language), source."""
    out: list[dict] = []

    def walk(x):
        if isinstance(x, dict):
            if "korean_mark" in x:
                out.append(
                    {
                        "korean_mark": unicodedata.normalize("NFC", str(x.get("korean_mark") or "").strip()),
                        "variants": _split_marks(x.get("relate_mark_e")),
                        "misspellings": _split_marks(x.get("relate_mark_o")),
                        "original": (x.get("srclang_mark") or None),
                        "language": (x.get("lang_nm") or None),
                        "source": (x.get("source") or None),
                    }
                )
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(raw)
    return out


class VariantLexicon:
    """Spelling variants of a Hangul loanword form from the NIKL loanword example API (key in NIKL_EXAMPLE_API_KEY).
    Cached on disk; the HTTP client is injectable."""

    def __init__(
        self,
        cache_path: str | Path,
        fetch: Fetch | None = None,
        api_key: str | None = None,
        url: str = NIKL_EXAMPLE_URL,
    ):
        self.cache_path, self.fetch, self.api_key, self.url = Path(cache_path), fetch, api_key, url
        self._cache: dict[str, list[dict]] = (
            json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        )

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    def lookup(self, hangul: str, condition: str = "korean_mark") -> list[dict]:
        """Records whose `condition` field (korean_mark, relate_mark_e or relate_mark_o) contains `hangul`."""
        hangul = unicodedata.normalize("NFC", hangul)
        ck = hangul if condition == "korean_mark" else f"{condition}:{hangul}"
        if ck in self._cache:
            return self._cache[ck]
        if self.fetch is None or not self.api_key:
            raise RuntimeError("NIKL example lookup needs a fetch function and NIKL_EXAMPLE_API_KEY")
        raw = self.fetch(
            self.url,
            {
                "serviceKey": self.api_key,
                "langType": "0003",
                "resultType": "json",
                "searchKeyword": hangul,
                "searchCondition": condition,
                "searchEquals": "like",
                "pageNo": "1",
                "numOfRows": "100",
            },
        )
        code, msg = _result_code(raw)
        if code is not None and str(code) not in ("00", "0"):
            raise RuntimeError(f"NIKL example API resultCode {code} {msg or ''}".strip())
        recs = parse_nikl_examples(raw)
        self._cache[ck] = recs
        self._save()
        return recs

    def variants_for(self, hangul: str) -> dict:
        """Variants of `hangul`: variant notations of its own records plus the standard form of any record listing `hangul` as a
        variant or misspelling. The form itself is never returned."""
        h = unicodedata.normalize("NFC", hangul)
        recs = self.lookup(h)
        exact = [r for r in recs if r["korean_mark"] == h]
        cl = lambda xs: [
            c for c in (_clean_mark(x) for x in xs) if c and ":" not in c and not re.search(r"[A-Za-z]", c)
        ]  # noqa: E731
        variants = {v for r in exact for v in cl(r["variants"])}
        misspell = {v for r in exact for v in cl(r["misspellings"])}
        # Records that list the form as their variant or misspelling are only found by searching those fields.
        reverse = self.lookup(h, "relate_mark_e") + self.lookup(h, "relate_mark_o") + recs
        standard_records = [
            (r["korean_mark"], r.get("original"))
            for r in reverse
            if r["korean_mark"] != h and (h in cl(r["variants"]) or h in cl(r["misspellings"]))
        ]
        standard = {k for k, _ in standard_records}
        variants |= standard
        variants.discard(h)
        misspell.discard(h)
        return {
            "found": bool(exact) or bool(standard),
            "standard": sorted(standard),
            "variants": sorted(variants),
            "misspellings": sorted(misspell),
            "standard_records": sorted(set(standard_records), key=str),
            "n_records": len(recs),
            "n_reverse_records": len(reverse) - len(recs),
        }
