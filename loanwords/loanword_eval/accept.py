"""Acceptance sets A_i: the forms counted as recognizing an item.

Extends lexicon.build_accept_set with the space-removed Latin variant, normalization of every member, per-member sources,
and a g2p guard: a g2pk2 output is kept only if it is Hangul-only and is not a reference form of another item type."""

from __future__ import annotations

import re
from typing import Callable, Iterable

from loanword_eval.lexicon import build_accept_set
from loanword_eval.normalize import KoreanPolicy

POLICY = KoreanPolicy()
HANGUL_ONLY = re.compile(r"^[가-힣]+$")
SOURCES = ("hangul", "latin", "latin_nospace", "nikl", "dual_spelling", "pron_form", "g2p")


def nu(s: str) -> str:
    return POLICY.apply(s).text


def build_with_sources(
    hangul: str,
    *,
    latin: str | None = None,
    nikl_variants: Iterable[str] = (),
    pron_form: str | None = None,
    g2p: Callable[[str], str] | None = None,
    dual_spelling: str | None = None,
) -> dict[str, set[str]]:
    """Map each normalized member to the set of sources that produced it."""
    base = build_accept_set(hangul, hike_latin=latin, nikl_original=None, pron_form=pron_form, g2p=None)
    src: dict[str, set[str]] = {}

    def add(m, s):
        m = nu(m)
        if m:
            src.setdefault(m, set()).add(s)

    add(hangul, "hangul")
    if latin:
        add(latin, "latin")
        if " " in latin.strip():
            add(latin.replace(" ", ""), "latin_nospace")
    if pron_form:
        add(pron_form, "pron_form")
    for v in nikl_variants:
        if v:
            add(v, "nikl")
    if dual_spelling:
        add(dual_spelling, "dual_spelling")
    for m in base:  # members from build_accept_set not attributed above
        add(m, "hangul" if nu(m) == nu(hangul) else "latin")
    if g2p is not None and latin:
        try:
            out = g2p(latin)
        except Exception:
            out = None
        if out:
            cand = nu(out)
            if HANGUL_ONLY.match(cand):  # cross-type collisions are removed later by guard_g2p
                add(cand, "g2p")
    for m in src:
        assert nu(m) == m, m
    return src


def build(hangul: str, **kw) -> set[str]:
    return set(build_with_sources(hangul, **kw))


def guard_g2p(rows: list[dict]) -> dict:
    """Remove g2p-only members that coincide with another type's non-g2p forms; return the collision report.
    Rows must carry `member_sources` and `type_key`."""
    forms_by_type: dict[str, set[str]] = {}
    for r in rows:
        forms_by_type.setdefault(r["type_key"], set()).update(
            m for m, s in r["member_sources"].items() if "g2p" not in s or len(s) > 1
        )
    owner: dict[str, set[str]] = {}
    for t, ms in forms_by_type.items():
        for m in ms:
            owner.setdefault(m, set()).add(t)
    removed, shared = [], {}
    for r in rows:
        for m, s in list(r["member_sources"].items()):
            if s == {"g2p"} and any(t != r["type_key"] for t in owner.get(m, ())):
                removed.append(
                    {
                        "item_id": r["item_id"],
                        "type_key": r["type_key"],
                        "member": m,
                        "collides_with": sorted(t for t in owner[m] if t != r["type_key"]),
                    }
                )
                del r["member_sources"][m]
        r["members"] = sorted(r["member_sources"])
        r["members_nog2p"] = sorted(m for m, s in r["member_sources"].items() if s != {"g2p"})
        r["member_sources"] = {m: sorted(s) for m, s in r["member_sources"].items()}
    for m, ts in owner.items():
        if len(ts) > 1:
            shared[m] = sorted(ts)
    return {
        "g2p_members_removed": len(removed),
        "removed": removed[:50],
        "members_shared_across_types": len(shared),
        "shared_examples": dict(list(shared.items())[:20]),
        "one_char_members": sorted({m for r in rows for m in r["members"] if len(m) == 1}),
    }


def _row(item_id, corpus, utt_id, start, end, ref_form, type_key, src, dual, extra=None):
    r = {
        "item_id": item_id,
        "corpus": corpus,
        "utt_id": utt_id,
        "start": start,
        "end": end,
        "ref_form": ref_form,
        "type_key": type_key,
        "member_sources": src,
        "members": sorted(src),
        "dual_notation": int(dual),
    }
    r.update(extra or {})
    return r


def rows_for_hike(tokens: Iterable[dict], g2p=None) -> list[dict]:
    out = []
    for t in tokens:
        if not t.get("korean_matrix"):
            continue
        src = build_with_sources(t["hangul"], latin=t["latin"], g2p=g2p)
        out.append(
            _row(
                t["item_id"],
                t["corpus"],
                t["utt_id"],
                t["start"],
                t["end"],
                t["ref_form"],
                t["type_key"],
                src,
                0,
                {"g2p_used": g2p is not None},
            )
        )
    return out
