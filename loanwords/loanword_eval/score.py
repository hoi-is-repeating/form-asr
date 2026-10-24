"""Item scoring: recognition R, strict recognition R_strict and the form produced F. Identical for every recognizer.

For item i with reference span r_i and acceptance set A_i:
  - reference and hypothesis are normalized (KoreanPolicy) and character-aligned (jiwer);
  - h_i is the aligned span plus insertions inside it, plus adjacent insertions attributed by the rule of Le et al. (2021),
    extended to Latin script (attributed iff Latin or in A_i, and never to a span that is already CORRECT);
  - R = 1 iff h_i is in A_i, or a word-bounded member of A_i with >= 2 characters occurs within +-window characters of the
    aligned span (`R_nowindow` drops this tolerance);
  - R_strict = 1 iff h_i equals the normalized reference span;
  - F is the script of the matched member (hangul, latin or mixed) when R = 1, else None.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from loanword_eval.align import CharAlignment, char_align, span_outcome
from loanword_eval.items import ref_form_of
from loanword_eval.normalize import KoreanPolicy, contains_latin

POLICY = KoreanPolicy()


@dataclass
class ItemScore:
    R: int
    R_strict: int
    R_nowindow: int
    F: str | None
    matched_member: str | None
    match_kind: str | None
    outcome: str  # CORRECT | SUB | DEL | PARTIAL | NO_REF (span empty after normalization)
    hyp_material: str
    ref_norm: str
    n_ins_adjacent: int

    def row(self) -> dict:
        return asdict(self)


def _hyp_bounds(al: CharAlignment, start: int, end: int) -> tuple[int, int]:
    """Hypothesis offsets (normalized coordinates) of the material aligned to ref[start:end], excluding adjacent insertions."""
    consumed_before = sum(1 for o, h in al.ref_ops[:start] if h is not None) + sum(
        len(v) for k, v in al.insertions.items() if k <= start
    )
    consumed_to_end = sum(1 for o, h in al.ref_ops[:end] if h is not None) + sum(
        len(v) for k, v in al.insertions.items() if k < end
    )
    return consumed_before, max(consumed_before, consumed_to_end)


def _no_ref(ref_span_norm: str) -> ItemScore:
    return ItemScore(0, 0, 0, None, None, None, "NO_REF", "", ref_span_norm, 0)


# Particles and copula endings that may attach to a noun without making it another word; anything else attached without a
# gap makes a different token.
PARTICLES = (
    "을",
    "를",
    "이",
    "가",
    "은",
    "는",
    "도",
    "에",
    "에서",
    "에게",
    "에는",
    "에서는",
    "로",
    "으로",
    "로는",
    "으로는",
    "와",
    "과",
    "의",
    "만",
    "까지",
    "부터",
    "처럼",
    "보다",
    "랑",
    "이랑",
    "나",
    "이나",
    "든지",
    "마다",
    "밖에",
    "한테",
    "께",
    "께서",
    "요",
    "야",
    "예요",
    "이에요",
    "입니다",
    "이다",
    "다",
    "이고",
    "고",
    "이라고",
    "라고",
    "이며",
    "며",
    "이면",
    "면",
    "이라",
    "라",
    "인데",
    "은데",
    "인",
    "일",
)
_PARTICLE_RE = re.compile("^(?:" + "|".join(sorted(map(re.escape, PARTICLES), key=len, reverse=True)) + ")+$")
_LATIN_RE = re.compile(r"[A-Za-z0-9]")
_HANGUL_RE = re.compile(r"[가-힣]")


def is_particle_string(s: str) -> bool:
    return bool(s) and bool(_PARTICLE_RE.match(s))


def _script(ch: str) -> str:
    return "latin" if _LATIN_RE.match(ch) else ("hangul" if _HANGUL_RE.match(ch) else "other")


def _gap(index_map, a: int, b: int) -> bool:
    """True if raw whitespace/punctuation lies between normalized positions a and b (b = a + 1)."""
    return index_map[b] - index_map[a] > 1


def _core_material(al: CharAlignment, s: int, e: int) -> str:
    """Hypothesis characters aligned to ref[s:e] plus the insertions strictly inside the span, in order."""
    pieces = []
    for i in range(s, e):
        h = al.ref_ops[i][1]
        if h is not None:
            pieces.append(h)
        if i + 1 < e:
            pieces.append(al.insertions.get(i + 1, ""))
    return "".join(pieces)


def _run(text, im, claimed, anchor, start: int, step: int) -> str:
    """Unclaimed Hangul stretch adjacent to `start` (step +1 forward, -1 backward), up to the next gap or script change.
    Anchors inside the stretch are included so a whole ending is compared with the reference affix."""
    out = []
    k = start
    while 0 <= k < len(text) and not claimed[k] and _script(text[k]) == "hangul":
        if out and _gap(im, min(k, k - step), max(k, k - step)):
            break
        out.append(text[k])
        k += step
    return "".join(out if step > 0 else reversed(out))


NOUN_LIKE_COPULA = (
    "인",
    "일",
)  # copula syllables that also end common nouns


_GLIDE_PAIRS = {
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (8, 12),
    (13, 17),
}  # medial-vowel indices differing only by a y-glide: a/ya, ae/yae, eo/yeo, e/ye, o/yo, u/yu


def _i_contraction(a: str, b: str) -> bool:
    """True when two syllables differ only by the y-glide of their vowel (the contraction i + eo -> yeo)."""
    if len(a) != len(b) or not a:
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if not diff:
        return True
    for x, y in diff:
        cx, cy = ord(x) - 0xAC00, ord(y) - 0xAC00
        if not (0 <= cx < 11172 and 0 <= cy < 11172):
            return False
        if cx // 588 != cy // 588 or cx % 28 != cy % 28:
            return False
        vx, vy = (cx % 588) // 28, (cy % 588) // 28
        if (min(vx, vy), max(vx, vy)) not in _GLIDE_PAIRS:
            return False
    return True


def _starts_with_bare_copula(run: str) -> bool:
    """True when the run starts with a bare copula syllable (in, il), read as noun-forming rather than as a particle."""
    return run[:1] in NOUN_LIKE_COPULA and not any(
        run.startswith(pp) for pp in PARTICLES if len(pp) > 1 and pp[0] in NOUN_LIKE_COPULA
    )


def _affix_ok(run: str, ref: str, side: str) -> bool:
    """Whether `run`, an unclaimed Hangul stretch attached to a member, may replace the reference affix `ref` on this side.

    Accepted: empty or equal to the affix; a particle string other than a bare copula syllable; a truncation of the affix;
    the same attachment character and at most one character longer; or an i-contraction of the affix. Derivational or
    compounding syllables that form another word are refused."""
    if not run or run == ref:
        return True
    if is_particle_string(run) and not _starts_with_bare_copula(run):
        return True
    if not ref:
        return False
    if side == "suffix":
        if ref.startswith(run):
            return True
        if len(run) <= len(ref) + 1 and run[0] == ref[0]:
            return True
        return _i_contraction(run, ref)
    if ref.endswith(run):
        return True
    if len(run) <= len(ref) + 1 and run[-1] == ref[-1]:
        return True
    return _i_contraction(run, ref)


def _bounded(
    hyp_n,
    claimed: list[bool],
    pos: int,
    m: str,
    neighbours: frozenset[str] = frozenset(),
    anchor: list[bool] | None = None,
    ref_affix: tuple[str, str] = ("", ""),
) -> bool:
    """Word-boundary rule for a member occurrence at normalized [pos, pos+len(m)), the same for Hangul and Latin members.

    Boundaries are a gap (raw whitespace or punctuation), a claimed character, an anchor (hypothesis character aligned equal
    to reference context) or a neighbouring item's member. Otherwise Latin letters touching a Latin member make another token
    ("bugs"), Latin letters next to a Hangul member are a separate token, and an attached Hangul stretch must pass `_affix_ok`
    against the reference affix on that side (`ref_affix` = (prefix, suffix)), possibly after an inserted particle."""
    text, im = hyp_n.text, hyp_n.index_map
    anchor = anchor or [False] * len(text)
    end = pos + len(m)
    pre_txt, suf_txt = ref_affix

    def _suffix_ok(run: str) -> bool:
        if _affix_ok(run, suf_txt, "suffix"):
            return True
        return any(
            is_particle_string(run[:k])
            and not _starts_with_bare_copula(run[:k])
            and _affix_ok(run[k:], suf_txt, "suffix")
            for k in range(1, len(run))
        )

    if pos > 0 and not claimed[pos - 1] and not anchor[pos - 1] and not _gap(im, pos - 1, pos):
        if not any(text.endswith(m2, 0, pos) for m2 in neighbours):
            prv = _script(text[pos - 1])
            if prv == "latin" and _script(m[0]) == "latin":
                return False
            if prv == "hangul" and not _affix_ok(_run(text, im, claimed, anchor, pos - 1, -1), pre_txt, "prefix"):
                return False
    if end < len(text) and not claimed[end] and not anchor[end] and not _gap(im, end - 1, end):
        if any(text.startswith(m2, end) for m2 in neighbours):
            return True
        nxt = _script(text[end])
        if nxt == "latin" and _script(m[-1]) == "latin":
            return False
        if nxt == "hangul" and not _suffix_ok(_run(text, im, claimed, anchor, end, +1)):
            return False
    return True


def _find_bounded(
    hyp_n,
    claimed: list[bool],
    lo: int,
    hi: int,
    m: str,
    neighbours: frozenset[str] = frozenset(),
    anchor: list[bool] | None = None,
    ref_affix: tuple[str, str] = ("", ""),
) -> int | None:
    """First occurrence of m inside [lo, hi) that is unclaimed and word-bounded, or None."""
    region = hyp_n.text
    start = lo
    while True:
        pos = region.find(m, start, hi)
        if pos < 0:
            return None
        if not any(claimed[pos : pos + len(m)]) and _bounded(hyp_n, claimed, pos, m, neighbours, anchor, ref_affix):
            return pos
        start = pos + 1


def score_utterance(
    ref_text: str, hyp_text: str, items: list[tuple[int, int, set[str]]], window: int = 4
) -> list[ItemScore]:
    """Score every item of one utterance jointly. `items` = (start, end, members) in reference offsets; output in the same order.

    Normalization removes spaces, so the aligner may assign a whole hypothesis run to the first of two adjacent items. Hence
    (i) every item first tries an exact match on its aligned material and claims those hypothesis characters; (ii) items
    without one search between their nearest equal-aligned context anchors +- window, in reference order, with only claimed
    characters masked (one-character members search inside the anchor segment only, after longer members); (iii) an adjacent
    insertion with no gap is glued to the material unless it is a particle string, and a separated one is attributed by the
    extended Le et al. (2021) rule. Window matches must be word-bounded, so "bug" is not found inside "debugging"."""
    ref_n = POLICY.apply(ref_text)
    hyp_n = POLICY.apply(hyp_text)
    n = len(items)
    out: list[ItemScore | None] = [None] * n
    spans_norm: list[tuple[int, int] | None] = [None] * n
    ref_norms: list[str] = [POLICY.apply(ref_text[s0:e0]).text for s0, e0, _ in items]
    for i, (s0, e0, _) in enumerate(items):
        try:
            spans_norm[i] = ref_n.to_normalized_span(s0, e0)
        except ValueError:
            out[i] = _no_ref(ref_norms[i])
    if not hyp_n.text:
        return [
            o if o is not None else ItemScore(0, 0, 0, None, None, None, "DEL", "", ref_norms[i], 0)
            for i, o in enumerate(out)
        ]
    al = char_align(ref_n.text, hyp_n.text)
    claimed = [False] * len(hyp_n.text)
    im = hyp_n.index_map
    # hypothesis index aligned to each reference character (None for deletions)
    hpos: list[int | None] = []
    consumed = 0
    for j, (op, h) in enumerate(al.ref_ops):
        consumed += len(al.insertions.get(j, ""))
        hpos.append(consumed if h is not None else None)
        if h is not None:
            consumed += 1
    in_item = [False] * len(al.ref)
    for sp_ in spans_norm:
        if sp_ is not None:
            for j in range(sp_[0], sp_[1]):
                in_item[j] = True
    # Anchors: hypothesis characters aligned equal to reference context. Substituted context characters are placed
    # arbitrarily by the aligner and are not anchors.
    anchor = [False] * len(hyp_n.text)
    for j, (op, h) in enumerate(al.ref_ops):
        if op == "equal" and not in_item[j] and hpos[j] is not None:
            anchor[hpos[j]] = True
    order = sorted((i for i in range(n) if spans_norm[i] is not None), key=lambda i: spans_norm[i])
    ref_orig = ref_n.original

    def _affix(i: int) -> tuple[str, str]:
        s0, e0 = items[i][0], items[i][1]
        pre, suf = [], []
        k = s0 - 1
        while k >= 0 and _script(ref_orig[k]) == "hangul":
            pre.append(ref_orig[k])
            k -= 1
        k = e0
        while k < len(ref_orig) and _script(ref_orig[k]) == "hangul":
            suf.append(ref_orig[k])
            k += 1
        return "".join(reversed(pre)), "".join(suf)

    affix = {i: _affix(i) for i in order}
    pending: list[tuple[int, str, object, tuple[int, int]]] = []
    # Pass 1: exact match on the item's material; a match claims its characters.
    for i in order:
        s, e = spans_norm[i]
        members = items[i][2]
        sp = span_outcome(al, s, e, attribute_adjacent=lambda ins: False)
        core = _core_material(al, s, e)
        hs, he = _hyp_bounds(al, s, e)
        before, after = al.insertions.get(s, ""), al.insertions.get(e, "")
        glued_b = glued_a = attr_b = attr_a = False
        if core:
            if before and hs > 0 and not _gap(im, hs - 1, hs) and not is_particle_string(before):
                glued_b = True
            if after and he < len(hyp_n.text) and not _gap(im, he - 1, he) and not is_particle_string(after):
                glued_a = True
        if sp.outcome != "CORRECT":
            attr_b = bool(before) and not glued_b and (contains_latin(before) or POLICY.apply(before).text in members)
            attr_a = bool(after) and not glued_a and (contains_latin(after) or POLICY.apply(after).text in members)
        material = (before if (glued_b or attr_b) else "") + core + (after if (glued_a or attr_a) else "")
        h = POLICY.apply(material).text
        outcome = sp.outcome
        if (glued_b or glued_a) and outcome == "CORRECT":
            outcome = "PARTIAL"  # reference characters present, but inside another token
        n_adj = len(before) * (glued_b or attr_b) + len(after) * (glued_a or attr_a)
        lo_c = hs - (len(before) if (glued_b or attr_b) else 0)
        hi_c = he + (len(after) if (glued_a or attr_a) else 0)
        neighbours = frozenset(m for j, (_, _, ms) in enumerate(items) if j != i for m in ms)
        bounded = (
            lo_c < hi_c and _bounded(hyp_n, claimed, max(0, lo_c), h, neighbours, anchor, affix[i]) if h else False
        )
        if h in members and bounded:
            for k in range(max(0, lo_c), min(hi_c, len(claimed))):
                claimed[k] = True
            out[i] = ItemScore(
                1, int(h == ref_norms[i]), 1, ref_form_of(h), h, "exact", outcome, h, ref_norms[i], n_adj
            )
        else:
            # material equal to the reference but not word-bounded is not strict either, so R_strict implies R
            strict = int(h == ref_norms[i] and bounded)
            pending.append((i, h, (outcome, n_adj, strict), (hs, he)))
    # Pass 2a: members of >= 2 characters in the anchor-bounded region +- window, in reference order.
    segments: dict[int, tuple[int, int]] = {}
    still: list[tuple[int, str, object, tuple[int, int]]] = []
    for i, h, meta, (hs, he) in pending:
        members = items[i][2]
        neighbours = frozenset(m for j, (_, _, ms) in enumerate(items) if j != i for m in ms)
        matched = None
        if window > 0:
            s_i, e_i = spans_norm[i]
            left = next(
                (hpos[j] + 1 for j in range(s_i - 1, -1, -1) if not in_item[j] and al.ref_ops[j][0] == "equal"), 0
            )
            right = next(
                (hpos[j] for j in range(e_i, len(al.ref)) if not in_item[j] and al.ref_ops[j][0] == "equal"),
                len(hyp_n.text),
            )
            seg_lo, seg_hi = max(0, min(left, hs)), min(len(hyp_n.text), max(right, he))
            segments[i] = (seg_lo, seg_hi)
            lo = max(0, seg_lo - window)
            hi = min(len(hyp_n.text), seg_hi + window)
            for m in sorted((m for m in members if len(m) >= 2), key=lambda m: (-len(m), m)):
                pos = _find_bounded(hyp_n, claimed, lo, hi, m, neighbours, anchor, affix[i])
                if pos is not None:
                    matched = m
                    for k in range(pos, pos + len(m)):
                        claimed[k] = True
                    break
        if matched is None:
            still.append((i, h, meta, (hs, he)))
            continue
        out[i] = ItemScore(1, meta[2], 0, ref_form_of(matched), matched, "window", meta[0], h, ref_norms[i], meta[1])
    # Pass 2b: one-character members inside the anchor segment only.
    for i, h, meta, (hs, he) in still:
        members = items[i][2]
        neighbours = frozenset(m for j, (_, _, ms) in enumerate(items) if j != i for m in ms)
        matched = None
        if window > 0 and i in segments:
            seg_lo, seg_hi = segments[i]
            for m in sorted(m for m in members if len(m) == 1):
                pos = _find_bounded(hyp_n, claimed, seg_lo, seg_hi, m, neighbours, anchor, affix[i])
                if pos is not None:
                    matched = m
                    claimed[pos] = True
                    break
        R = int(matched is not None)
        out[i] = ItemScore(
            R,
            meta[2] if R else 0,
            0,
            ref_form_of(matched) if matched else None,
            matched,
            "window-1char" if matched else None,
            meta[0],
            h,
            ref_norms[i],
            meta[1],
        )
    return [o for o in out]  # type: ignore[misc]


def score_item(ref_text: str, hyp_text: str, start: int, end: int, members: set[str], window: int = 4) -> ItemScore:
    """One item on its own (controls; single-item utterances). Items that share an utterance must be scored together with
    `score_utterance`, otherwise a neighbour's output can be borrowed."""
    return score_utterance(ref_text, hyp_text, [(start, end, members)], window=window)[0]
