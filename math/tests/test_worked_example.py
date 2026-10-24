"""Worked example: the item "rho cubed over three" (MathSpeech idx 960, reference LaTeX `$\\rho^{3}/3$`) under constructed renderings."""

import shutil

import pytest

from math_eval import latex_norm, score

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")

REF = "rho cubed over three"
REF_LATEX = r"$\rho^{3}/3$"

# rendering, R_strict, R, F  (the LaTeX row is decided by the latex outcome, not by the key)
ROWS = [
    ("rho cubed over three", 1, 1, "words"),
    ("rho to the third over three", 0, 1, "words"),
    ("ρ³/3", 0, 1, "notation"),
    ("rho^3 over 3", 0, 1, "mixed"),
    ("row cubed over three", 0, 0, None),
    ("rho squared over three", 0, 0, None),
]


def test_key_rows():
    for hyp, strict, r, f in ROWS:
        s = score.score_clip(REF, hyp)
        assert s.R_strict == strict, (hyp, s)
        assert s.R == r, (hyp, s)
        assert s.F == f, (hyp, s)


@needs_node
def test_latex_row_equal_under_norm():
    a, b = latex_norm.normalize_many([r"\rho^{3}/3", REF_LATEX])
    assert a == b
    assert score.looks_like_latex(r"\rho^{3}/3")


@needs_node
def test_frac_row_not_equal_under_norm():
    a, b = latex_norm.normalize_many([r"\frac{\rho^3}{3}", REF_LATEX])
    assert a != b
    assert score.score_clip(REF, r"\frac{\rho^3}{3}").R == 0
