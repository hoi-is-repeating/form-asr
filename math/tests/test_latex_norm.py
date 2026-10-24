"""Conversion-stage normalizer. The KaTeX tests need Node.js and process_formula; they are skipped where absent so the
pure-Python parts still run on any machine."""

import shutil

import pytest

from math_eval import latex_norm

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or pytest.importorskip("process_formula", reason="process_formula missing") is None,
    reason="Node.js / process_formula not available",
)


def test_strip_delims_and_fences():
    assert latex_norm.strip_delims("$ax + b$") == "ax + b"
    assert latex_norm.strip_delims("```latex\n\\frac{1}{2}\n```") == "\\frac{1}{2}"
    assert latex_norm.strip_delims("\\[x\\]") == "x"


def test_cer_and_exact_match_pure_python():
    assert latex_norm.exact_match("\\frac{1}{2}", "\\frac{1}{2}") == 1
    assert latex_norm.exact_match("\\frac {1}{2}", "\\frac{1}{2}") == 1
    assert latex_norm.exact_match("", "x") == 0
    assert latex_norm.cer("x+1", "x+1") == 0.0
    assert latex_norm.cer("", "x+1") == 1.0
    assert 0 < latex_norm.cer("x+2", "x+1") < 1


@needs_node
def test_katex_normalizer_merges_variants():
    out = latex_norm.normalize_many([r"$\dfrac{1}{2}$", r"\frac{1}{2}", r"$ax + by + cz = d$", r"$ ax + by + cz = d $"])
    assert out[0].replace(" ", "") == out[1].replace(" ", ""), out[:2]
    assert out[2].replace(" ", "") == out[3].replace(" ", ""), out[2:]
    assert latex_norm.exact_match(out[0], out[1]) == 1


@needs_node
def test_semantics_free_commands_removed_after_katex():
    out = latex_norm.normalize_many([r"\displaystyle \left( x \right)^2", r"(x)^2"])
    assert "displaystyle" not in out[0] and "left" not in out[0]
    assert latex_norm.exact_match(out[0], out[1]) == 1, out
