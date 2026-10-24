"""Canonicalizer for spoken mathematics and the form detector.

The canonicalizer maps a normalized string to a canonical token sequence by the rules in ``configs/mathforms.yaml``, so that
surfaces of the same expression ("two over the square root of pi", "2 / sqrt(pi)") give the same string; recognition is agreement
of canonical strings. The canonical string is a recognition key, not a parse: it keeps every spoken word that changes which
expression was said ("f of x" vs "f x") and drops only words that never do.

Each rewrite records whether a word-side or a symbol-side surface fired; the form produced is
  words    - only word-side surfaces fired (or none fired but the strings agree),
  notation - only symbol-side surfaces fired,
  mixed    - both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from math_eval.normalize_en import normalize

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mathforms.yaml"

_NUM_RE = re.compile(r"^\d+(\.\d+)?$")
_LETTER_ORD_RE = re.compile(r"^([a-z])th$")  # "nth", "kth": the letter as an exponent after POW
# English two-letter words that stay whole in a transcript; every other two-letter run is two single-letter symbols written together
_TWO_LETTER_WORDS = {
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "hi",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "oh",
    "ok",
    "on",
    "or",
    "so",
    "to",
    "up",
    "us",
    "we",
    "am",
    "ah",
    "um",
    "uh",
    "ex",
    "pm",
    "id",
    "ie",
    "eg",
    "mr",
    "ms",
    "dr",
}


@dataclass
class Canon:
    tokens: list[str]
    word_fires: int = 0
    symbol_fires: int = 0
    fired: list[tuple[str, str, str]] = field(default_factory=list)  # (category, side, surface)
    # side that produced each output token, parallel to `tokens`: "word", "symbol" or None (non-deciding tokens);
    # a span's form can be read from its tokens
    sides: list[str | None] = field(default_factory=list)

    grouping: frozenset = frozenset()
    diff_vars: set = field(
        default_factory=set
    )  # token indices of variables of differentiation; "of" after them is not application

    @property
    def text(self) -> str:
        """Compared canonical string: grouping tokens excluded (spoken forms rarely verbalise them); see configs/mathforms.yaml."""
        return " ".join(t for t in self.tokens if t not in self.grouping)

    @property
    def compared_tokens(self) -> list[str]:
        return [t for t in self.tokens if t not in self.grouping]

    @property
    def compared_sides(self) -> list[str | None]:
        return [s for t, s in zip(self.tokens, self.sides) if t not in self.grouping]

    @property
    def text_with_grouping(self) -> str:
        return " ".join(self.tokens)


def form_from_counts(word_fires: int, symbol_fires: int) -> str:
    if word_fires and symbol_fires:
        return "mixed"
    if symbol_fires:
        return "notation"
    return "words"


def form_of_sides(sides: list[str | None]) -> str:
    """Form of a token span from its per-token sides (None tokens do not decide)."""
    return form_from_counts(sum(1 for s in sides if s == "word"), sum(1 for s in sides if s == "symbol"))


class Grammar:
    MAX_PHRASE = 7  # longest alphabetic surface in tokens ("is greater than or equal to")
    BOUND_TAIL = 4  # an upper bound is at most this many tokens ("pi over 2", "2 pi", "infinity"); the "of" after it is the integrand marker

    def __init__(self, path: Path | str = CONFIG_PATH):
        self.path = Path(path)
        self.cfg = yaml.safe_load(self.path.read_text())
        self.version = self.cfg["version"]
        # phrase table: surface -> (canon, side, category), longest surfaces first
        table: list[tuple[str, str, str, str]] = []
        for rule in self.cfg["phrases"]:
            for w in rule.get("words", []):
                table.append((w, rule["canon"], "word", "phrases"))
            for sy in rule.get("symbols", []):
                table.append((sy, rule["canon"], "symbol", "phrases"))
        for name, letter in self.cfg["greek"].items():
            table.append((name, name.upper(), "word", "greek"))
            table.append((letter, name.upper(), "symbol", "greek"))
        for canon, sides in self.cfg["functions"].items():
            for w in sides.get("words", []):
                table.append((w, canon, "word", "functions"))
            for sy in sides.get("symbols", []):
                table.append((sy, canon, "symbol", "functions"))
        # A symbol surface that is also a word surface of the same canon (e.g. "sin", "log") is a word-side spelling, not notation.
        seen_word = {(s, c) for s, c, side, _ in table if side == "word"}
        table = [(s, c, ("word" if (s, c) in seen_word else side), cat) for s, c, side, cat in table]
        self.table = sorted(
            set(table), key=lambda t: (-len(t[0]), t[0], t[1], t[2], t[3])
        )  # total order: deterministic across hash seeds
        dup = {}
        for surf, canon, side, cat in self.table:
            dup.setdefault(surf, set()).add(canon)
        bad = {k: v for k, v in dup.items() if len(v) > 1}
        assert not bad, f"grammar surfaces mapping to two canons: {bad}"
        self.alpha_surfaces = [t for t in self.table if re.match(r"^[a-z][a-z ]*$", t[0])]
        self.symbol_surfaces = [t for t in self.table if not re.match(r"^[a-z][a-z ]*$", t[0])]
        assert all(len(s.split()) <= self.MAX_PHRASE for s, _, _, _ in self.alpha_surfaces)
        nums = self.cfg["numbers"]
        self.units = {k: int(v) for k, v in nums["units"].items()}
        self.tens = {k: int(v) for k, v in nums["tens"].items()}
        self.scales = {k: int(v) for k, v in nums["scales"].items()}
        self.fractions = dict(nums["fractions"])
        self.articles = set(nums.get("articles_as_one", []))
        self.ordinals = {k: int(v) for k, v in nums.get("ordinals", {}).items()}
        lr = self.cfg.get("letter_run_split", {})
        self.split_max_len, self.split_max_zipf = int(lr.get("max_len", 0)), float(lr.get("max_zipf", 0.0))
        self.fillers = set(self.cfg.get("fillers", []))
        self.grouping = frozenset(self.cfg.get("grouping_tokens", []))
        app = self.cfg.get("application") or {}
        self.app_canon, self.app_word, self.app_symbol = (
            app.get("canon", "OF"),
            app.get("word", "of"),
            app.get("symbol", "("),
        )
        self.diff_surfaces = set(app.get("differentiation_surfaces", []))
        pp = app.get("partial_patterns") or ([app["partial_pattern"]] if app.get("partial_pattern") else [])
        self.partial_patterns = [tuple(x) for x in pp]
        b = self.cfg.get("bounds") or {}
        self.bounds_after, self.bounds_from, self.bounds_to = (
            set(b.get("after", [])),
            b.get("from", "SUB"),
            b.get("to", "POW"),
        )
        ef = self.cfg.get("exponent_follows") or {}
        ef_cats, ef_canons, ef_except = (
            set(ef.get("categories", [])),
            set(ef.get("canons", [])),
            set(ef.get("except_canons", [])),
        )
        self._exp_starts = {
            surf.split()[0]
            for surf, canon, _, cat in self.alpha_surfaces
            if (cat in ef_cats or canon in ef_canons) and canon not in ef_except
        }
        sufs = self.cfg.get("ordinal_suffixes", [])
        self._ord_re = re.compile(r"^(\d+)(" + "|".join(map(re.escape, sufs)) + r")s?$") if sufs else None
        self.post_rewrites = [(tuple(r["from"]), list(r["to"])) for r in self.cfg.get("post_rewrites", [])]
        self._known_words = (
            {w for surf, _, _, _ in self.table for w in surf.split()}
            | set(self.units)
            | set(self.tens)
            | set(self.scales)
            | set(self.fractions)
            | self.fillers
            | self.articles
            | set(self.ordinals)
            | {"point", "oh", "to", "the", "of", "from", "power"}
        )

    def _tokenize(self, s: str) -> list[str]:
        """Split a normalized string into words, numbers (including digit ordinals such as "3rd") and single symbol characters."""
        out: list[str] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit():
                j = i
                while j < len(s) and (s[j].isdigit() or (s[j] == "." and j + 1 < len(s) and s[j + 1].isdigit())):
                    j += 1
                if self._ord_re is not None:  # "3rd", "9th", "3rds": one token when the letters end the run
                    k = j
                    while k < len(s) and s[k].isalpha():
                        k += 1
                    if k > j and self._ord_re.match(s[i:k]):
                        j = k
                out.append(s[i:j])
                i = j
                continue
            if ch.isalpha():
                if not ch.isascii():  # a Greek letter is its own token even when glued to a letter
                    out.append(ch)
                    i += 1
                    continue
                j = i
                while j < len(s) and s[j].isalpha() and s[j].isascii():
                    j += 1
                out.extend(self._split_run(s[i:j]))
                i = j
                continue
            # multi-character symbol surfaces ("**", "<=", ">=", "!=", "->")
            for sy, _, side, _ in self.symbol_surfaces:
                if len(sy) > 1 and s.startswith(sy, i):
                    out.append(sy)
                    i += len(sy)
                    break
            else:
                out.append(ch)
                i += 1
        return out

    def _split_run(self, run: str) -> list[str]:
        """A short alphabetic run that is not a known word and has (almost) no English frequency is letters written together
        ("ax", "dx", "cz" -> a x / d x / c z); known surfaces and English words ("by", "sin", "the") stay whole."""
        if len(run) <= 1 or len(run) > self.split_max_len or run in self._known_words or not run.isascii():
            return [run]
        if len(run) == 2:
            return (
                [run] if run in _TWO_LETTER_WORDS else list(run)
            )  # "ax", "dx", "ab" are letters; "by", "in", "is" are words
        if _LETTER_ORD_RE.match(run):
            return [run]  # "nth", "kth": handled as ordinals
        try:
            from wordfreq import zipf_frequency

            z = zipf_frequency(run, "en")
        except Exception:  # noqa: BLE001
            return [run]
        return list(run) if z < self.split_max_zipf else [run]

    def _is_variable(self, c: Canon, k: int = -1) -> bool:
        """Token k of the output is a single-letter variable (an ordinary token of one letter)."""
        return (
            bool(c.tokens)
            and len(c.tokens) >= -k
            and len(c.tokens[k]) == 1
            and c.tokens[k].isalpha()
            and c.sides[k] is None
        )

    def _application_here(self, c: Canon) -> bool:
        """'of' / '(' at this position is function application: after a variable, or a variable followed by PRIME / DOT, unless that variable
        is the variable of differentiation ("d by d x of f" = "d/dx f")."""
        if not c.tokens:
            return False
        k = -1
        while len(c.tokens) >= -k and c.tokens[k] in ("PRIME", "DOT"):
            k -= 1
        if not self._is_variable(c, k):
            return False
        return (len(c.tokens) + k) not in c.diff_vars

    def canonical(self, text: str) -> Canon:
        s = normalize(text)
        toks = self._tokenize(s)
        c = Canon(tokens=[], grouping=self.grouping)
        i = 0
        bound_open = False  # after "from" of an INT/SUM/PROD bound: the next "to" is the upper bound
        bound_tail = (
            -1
        )  # output length when the upper bound opened; the first "of" within BOUND_TAIL tokens is the integrand marker
        while i < len(toks):
            tok = toks[i]
            # 0. context rules: function application, bounds of big operators, "power" after an exponent
            if tok == self.app_word:
                if (
                    0 <= bound_tail
                    and len(c.tokens) - bound_tail <= self.BOUND_TAIL
                    and not (
                        len(c.tokens) >= bound_tail + 2
                        and self._is_variable(c)
                        and (self._is_variable(c, -2) or _NUM_RE.match(c.tokens[-2]))
                    )
                ):
                    bound_tail = -1  # "from a to b OF x": the integrand marker, nothing on the notation side (not "from 0 to 1 f OF x")
                elif self._application_here(c):
                    bound_tail = -1
                    self._emit(c, [self.app_canon], "phrases", "word", tok)
                else:
                    bound_tail = -1
                i += 1
                continue  # elsewhere "of" is dropped (absorbed by the operator surface)
            if tok == self.app_symbol and self._application_here(c):
                self._emit(c, [self.app_canon], "phrases", "symbol", tok)
                i += 1
                continue
            if tok == "from" and c.tokens and c.tokens[-1] in self.bounds_after:
                self._emit(c, [self.bounds_from], "phrases", "word", tok)
                bound_open = True
                i += 1
                continue
            if tok == "to" and bound_open:
                self._emit(c, [self.bounds_to], "phrases", "word", tok)
                bound_open = False
                bound_tail = len(c.tokens)
                i += 1
                continue
            if tok == "power" and len(c.tokens) >= 2 and c.tokens[-2] == "POW":
                i += 1
                continue  # "to the 9th power", "to the nth power": POW n already emitted
            # 1. longest alphabetic phrase match (word side)
            matched = False
            for n in range(self.MAX_PHRASE, 0, -1):
                if i + n > len(toks):
                    continue
                phrase = " ".join(toks[i : i + n])
                for surf, canon, side, cat in self.alpha_surfaces:
                    if surf == phrase:
                        if surf == "to the" and not self._exponent_follows(toks, i + n):
                            continue  # "x goes to the limit": not an exponent
                        self._emit(c, canon.split(), cat, side, surf)
                        i += n
                        matched = True
                        break
                if matched:
                    break
            if matched:
                continue
            # 1b. ordinal exponent: "to the fourth" -> POW 4; "to the 9th" -> POW 9; "to the nth" -> POW n
            if tok in self.ordinals and c.tokens and c.tokens[-1] == "POW":
                self._emit(c, [str(self.ordinals[tok])], "numbers", "word", tok)
                i += 1
                continue
            m_ord = self._ord_re.match(tok) if self._ord_re else None
            if m_ord:
                if c.tokens and c.tokens[-1] == "POW":
                    self._emit(c, [m_ord.group(1)], "numbers", "word", tok)
                    i += 1
                    continue
                if c.tokens and _NUM_RE.match(c.tokens[-1]):
                    self._emit(c, ["OVER", m_ord.group(1)], "numbers", "word", tok)
                    i += 1
                    continue  # "1 3rd" -> 1 OVER 3
                c.tokens.append(tok)
                c.sides.append(None)
                i += 1
                continue
            m_lord = _LETTER_ORD_RE.match(tok)
            if m_lord and c.tokens and c.tokens[-1] == "POW":
                c.tokens.append(m_lord.group(1))
                c.sides.append(None)
                i += 1
                continue
            if tok in self.articles and i + 1 < len(toks) and self._ord_re and self._ord_re.match(toks[i + 1]):
                self._emit(
                    c,
                    ["1", "OVER", self._ord_re.match(toks[i + 1]).group(1)],
                    "numbers",
                    "word",
                    " ".join(toks[i : i + 2]),
                )
                i += 2
                continue
            # 2. symbol surfaces
            sym = next(((canon, side, cat) for surf, canon, side, cat in self.symbol_surfaces if surf == tok), None)
            if sym:
                canon, side, cat = sym
                self._emit(c, canon.split(), cat, side, tok)
                i += 1
                continue
            # 3. numbers: digits (symbol side) or number words (word side), incl. "twenty one", "one hundred", "a half"
            if _NUM_RE.match(tok):
                self._emit(c, [tok.lstrip("0") or "0" if "." not in tok else tok], "numbers", "symbol", tok)
                i += 1
                continue
            if (
                tok in self.units
                or tok in self.tens
                or tok in self.articles
                and i + 1 < len(toks)
                and toks[i + 1] in self.fractions
            ):
                value, j = self._number_words(toks, i)
                if value is not None:
                    num = str(value)
                    # "three point five" -> 3.5: decimal digits spoken one by one after "point"
                    if j < len(toks) and toks[j] == "point":
                        k = j + 1
                        digits = ""
                        while k < len(toks) and (toks[k] in self.units and self.units[toks[k]] < 10 or toks[k] == "oh"):
                            digits += str(self.units.get(toks[k], 0))
                            k += 1
                        if digits:
                            num = f"{value}.{digits}"
                            j = k
                    self._emit(c, [num], "numbers", "word", " ".join(toks[i:j]))
                    i = j
                    if i + 1 < len(toks) and toks[i] == "-" and toks[i + 1] in self.fractions:
                        i += 1  # "one-half", "five-halves": the hyphen is not a minus here
                    if i < len(toks) and toks[i] in self.fractions:  # "one half", "three quarters"
                        self._emit(c, self.fractions[toks[i]].split(), "numbers", "word", toks[i])
                        i += 1
                    continue
            if (
                tok in self.fractions and c.tokens and _NUM_RE.match(c.tokens[-1])
            ):  # bare "half" only after a number ("the third derivative" stays words)
                self._emit(c, self.fractions[tok].split(), "numbers", "word", tok)
                i += 1
                continue
            # 4. fillers dropped; everything else (variables, ordinary words) kept verbatim
            if tok in self.fillers:
                i += 1
                continue
            if (
                len(tok) == 1
                and tok.isalpha()
                and c.tokens
                and (
                    (c.tokens[-1] == "DERIV" and c.fired and c.fired[-1][2] in self.diff_surfaces)
                    or any(tuple(c.tokens[-len(pp) :]) == pp for pp in self.partial_patterns)
                )
            ):
                c.diff_vars.add(
                    len(c.tokens)
                )  # "d by d x", "partial over partial x": x is the variable of differentiation
            c.tokens.append(tok)
            c.sides.append(None)
            i += 1
        self._post_rewrite(c)
        assert len(c.tokens) == len(c.sides)
        return c

    def _post_rewrite(self, c: Canon) -> None:
        """Token-sequence rewrites for notation whose word side is a single surface ("d/dx" -> DERIV, "sin^-1" -> ARCSIN); the merged
        token takes the side of the operator token."""
        for src, dst in self.post_rewrites:
            i = 0
            while i < len(c.tokens):
                # match src against the tokens from i, skipping grouping tokens between the pattern's elements (sin^{-1} = sin^-1)
                j, k = i, 0
                while j < len(c.tokens) and k < len(src):
                    if c.tokens[j] == src[k]:
                        j += 1
                        k += 1
                    elif k > 0 and c.tokens[j] in self.grouping:
                        j += 1
                    else:
                        break
                if k == len(src):
                    side = next((s for s in c.sides[i:j] if s is not None), None)
                    c.tokens[i:j] = dst
                    c.sides[i:j] = [side] * len(dst)
                    c.fired.append(("post", side or "word", " ".join(src)))
                    if (
                        dst == ["DERIV"]
                        and i + 2 < len(c.tokens)
                        and len(c.tokens[i + 1]) == 1
                        and c.tokens[i + 1].isalpha()
                        and c.tokens[i + 2] == self.app_canon
                    ):
                        del c.tokens[i + 2]
                        del c.sides[
                            i + 2
                        ]  # "d/dx (f)": the "(" after the variable of differentiation is not application
                    i += len(dst)
                else:
                    i += 1

    def _emit(self, c: Canon, canon_tokens: list[str], cat: str, side: str, surf: str) -> None:
        """Append canonical tokens with their deciding side (None when the category does not decide F_i) and record the firing."""
        decides = cat in self.cfg.get("form_categories", [])
        c.tokens.extend(canon_tokens)
        c.sides.extend([side if decides else None] * len(canon_tokens))
        self._fire(c, cat, side, surf)

    def _fire(self, c: Canon, cat: str, side: str, surf: str) -> None:
        c.fired.append((cat, side, surf))
        if cat in self.cfg.get("form_categories", []):
            if side == "word":
                c.word_fires += 1
            else:
                c.symbol_fires += 1

    def _exponent_follows(self, toks: list[str], j: int) -> bool:
        """After a bare "to the": an exponent is a number word or digit, an ordinal, a single-letter variable, a sign word, "power", or
        the start of a function / Greek / root / grouping surface ("e to the log x"). Anything else ("to the limit", "to the
        left") keeps "to the" as ordinary words."""
        if j >= len(toks):
            return False
        t = toks[j]
        return (
            bool(_NUM_RE.match(t))
            or t in self.units
            or t in self.tens
            or t in self.ordinals
            or t in ("minus", "negative", "power")
            or (len(t) == 1 and t.isalpha())
            or bool(self._ord_re and self._ord_re.match(t))
            or bool(_LETTER_ORD_RE.match(t))
            or t in self._exp_starts
        )

    def _number_words(self, toks: list[str], i: int) -> tuple[int | None, int]:
        """Parse a run of number words starting at i; return (value, next_index) or (None, i). A units word after a units word, or a tens
        word after a units/tens word, ends the number ("one five ten" is a list, not sixteen); "and" continues a
        number only after a scale word ("one hundred and five"), never between plain numbers ("zero and one")."""
        total = 0
        current = 0
        j = i
        consumed = False
        last = None
        if toks[i] in self.articles:
            return 1, i + 1
        while j < len(toks):
            t = toks[j]
            if t in self.units:
                if last in ("units", "tens") and not (last == "tens" and self.units[t] < 10):
                    break
                current += self.units[t]
                consumed = True
                last = "units"
            elif t in self.tens:
                if last in ("units", "tens"):
                    break
                current += self.tens[t]
                consumed = True
                last = "tens"
            elif t in self.scales and consumed:
                sc = self.scales[t]
                if sc == 100:
                    current *= 100
                else:
                    total += current * sc
                    current = 0
                last = "scale"
            elif (
                t == "and"
                and last == "scale"
                and j + 1 < len(toks)
                and (toks[j + 1] in self.units or toks[j + 1] in self.tens)
            ):
                pass
            elif (
                t == "-"
                and last == "tens"
                and j + 1 < len(toks)
                and toks[j + 1] in self.units
                and self.units[toks[j + 1]] < 10
            ):
                pass  # "twenty-one"
            else:
                break
            j += 1
        if not consumed:
            return None, i
        return total + current, j


_DEFAULT: Grammar | None = None


def grammar() -> Grammar:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Grammar()
    return _DEFAULT


def canonical(text: str) -> str:
    return grammar().canonical(text).text
