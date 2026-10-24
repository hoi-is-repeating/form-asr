"""Dominant form per category, expression taxonomy, attribution estimators, recognizer registry and configs."""

import pytest

from math_eval import attribution as attr
from math_eval.decoders import PROMPT_CAPABLE, RECOGNIZERS, load_decoder
from math_eval.direction import direction_rows, fired_by_category, item_category
from math_eval.paths import CONFIGS, load_yaml
from math_eval.taxonomy import expression_category, latex_tokens


def test_fired_by_category_and_item_category():
    f = fired_by_category("x^2 + 1 = 0")
    assert f["operators"]["symbol"] == 2 and f["numbers"]["symbol"] == 3 and f["roots_exponents"]["symbol"] == 1
    f = fired_by_category("sine squared theta plus cosine squared theta equals one")
    assert f["functions"]["word"] == 2 and f["greek"]["word"] == 2 and f["roots_exponents"]["word"] == 2
    assert item_category("x squared plus one equals zero") == "roots_exponents"
    assert item_category("twenty one over seven") == "numbers"
    assert item_category("a x plus b y") == "operators"
    assert item_category("sine theta") is None


def test_direction_rows_fallback_rules():
    spans = [
        {
            "itt_R": 1,
            "fired_by_category": {
                "numbers": {"symbol": 12},
                "operators": {"word": 6, "symbol": 6},
                "roots_exponents": {"word": 2},
            },
        },
        {"itt_R": 0, "fired_by_category": {"numbers": {"word": 50}}},
    ]
    rows, wf = direction_rows(spans, "whisper")
    by = {r["category"]: r for r in rows}
    assert by["numbers"]["dominant"] == "notation" and by["numbers"]["fallback"] == 0
    assert by["operators"]["dominant"] == "words" and by["operators"]["fallback_reason"] == "tie"
    assert by["roots_exponents"]["fallback"] == 1 and by["roots_exponents"]["dominant"] == "words"
    assert by["greek"]["dominant"] is None and len(rows) == 5


def test_expression_taxonomy():
    assert expression_category(r"$\int_0^1 f(x)\,dx$") == "calculus"
    assert expression_category(r"$x \in A \cup B$") == "set_logic"
    assert expression_category(r"$ax + by = d$") == "algebra"
    assert expression_category(r"$21/7$") == "number"
    assert expression_category(r"$\frac{1}{2}$") == "number"
    assert latex_tokens(r"$x^2 + 1 = 0$") == 7


def _rows():
    out = []
    for i in range(40):
        R = int(i % 4 != 0)
        C = int(R and i % 5 != 0)
        Cref = int(i % 10 != 0)
        out.append(
            {
                "item_id": f"i{i}",
                "video_id": f"v{i % 6}",
                "R": R,
                "C": C,
                "C_ref": Cref,
                "F": "words",
                "category": "algebra" if i % 2 else "number",
            }
        )
    return out


def test_attribution_identity_and_h3():
    rows = _rows()
    d = attr.decomposition(rows, B=100, seed=1)
    assert d["identity_ok"] and abs(d["term_recognition"] + d["term_conversion"] - d["P_C0"]) < 1e-9
    h = attr.h3a(rows, B=100, seed=1, n_perm=200)
    assert h["larger_term"] in ("recognition", "conversion") and h["delta_rec"]["estimate"] is not None
    d2 = [dict(r, R=1, C=int(r["C_ref"])) for r in rows]
    b = attr.h3b(rows, d2, B=100, n_perm=200, seed=2)
    assert b["recovered_0_to_1"] == 10 and b["difference_D2_minus_D0"]["estimate"] < 0


def test_registry_and_pinned_configs():
    assert RECOGNIZERS[0] == "whisper" and len(RECOGNIZERS) == 10
    assert {"whisper", "whisper_base", "whisper_small", "whisper_large_v2", "owsm", "qwen3asr", "raon"} <= set(
        PROMPT_CAPABLE
    )
    hashes = set()
    for rid in RECOGNIZERS:
        cfg = load_yaml(f"decode_{rid}.yaml")
        assert cfg["id"] == rid and cfg.get("revision")
        hashes.add(hash((CONFIGS / f"decode_{rid}.yaml").read_text()))
    assert len(hashes) == len(RECOGNIZERS)
    assert (
        load_yaml("decode_whisper_small.yaml")["checkpoint"] == "openai/whisper-small"
        and load_yaml("decode_owsm.yaml")["lang_sym"] == "<eng>"
    )
    with pytest.raises(KeyError):
        load_decoder("nope", {"id": "nope"})


def test_converter_config_and_fake():
    from math_eval import converter as conv

    cfg = load_yaml("converter.yaml")
    assert (
        cfg["default"] == "legacy_two_stage"
        and cfg["converters"]["s2l_equations"]["prompts"]["system"] == "You are a helpful assistant."
    )
    fake = conv.load_converter("fake", cfg)
    out = fake.convert(["two over the square root of pi", "2 / sqrt(pi)", "nonsense words here"])
    assert out[0] == out[1] == "$2/\\sqrt{\\pi}$" and out[2] == ""
    assert (
        conv._s2l_latex_line("Fixed transcribation: two over pi\nLaTex: \\frac{2}{\\pi}<|im_end|>") == "\\frac{2}{\\pi}"
    )
    assert conv._first_line(" $x^2$ \nmore") == "$x^2$"
