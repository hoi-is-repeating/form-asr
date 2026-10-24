"""Canonicalizer and form detector: paired surfaces must agree; forms are read from the side that fired."""

import pytest

from math_eval.mathforms import canonical, form_from_counts, grammar


def recognized(reference: str, hypothesis: str) -> bool:
    return canonical(reference) == canonical(hypothesis)


def form_of(text: str) -> str:
    c = grammar().canonical(text)
    return form_from_counts(c.word_fires, c.symbol_fires)


PAIRS = [
    # words form (as MathSpeech spoken-form references read)        notation form a recognizer might write
    ("two over the square root of pi", "2 / sqrt(pi)"),
    ("two over the square root of pi", "2/√π"),
    ("x squared plus one equals zero", "x^2 + 1 = 0"),
    ("x squared plus one equals zero", "x² + 1 = 0"),
    (
        "a x plus b y plus c z equals d",
        "ax plus by plus cz equals d".replace("ax", "a x").replace("by", "b y").replace("cz", "c z"),
    ),
    ("the integral from zero to one of f of x d x", "∫ from 0 to 1 of f of x dx".replace("dx", "d x")),
    ("sine squared theta plus cosine squared theta equals one", "sin^2 θ + cos^2 θ = 1"),
    ("one half", "1/2"),
    ("twenty one over seven", "21/7"),
    ("x sub i", "x_i".replace("_i", "_ i")),
    ("e to the power of minus t squared", "e^-t^2".replace("^-", "^ -")),
    ("three point five percent", "3.5%"),
    ("n factorial", "n!"),
    ("the limit as x goes to infinity", "lim as x → ∞"),
    ("f of x plus delta x", "f(x + Δx)"),
    ("f prime of x is e to the x", "f'(x) = e^x"),
    ("the integral from zero to one of f of x d x", "∫_0^1 f(x) dx"),
    ("e to the log x is equal to x", "e^(log x) = x"),
    ("dy is 1 3rd times 64 to the minus 2 3rds dx", "dy = 1/3 × 64^(-2/3) dx"),
    ("10 times sine of t raised to the 9th power", "10 × sin(t)^9"),
    ("one minus y is less than square root of one minus x squared", "1 − y < √(1 − x²)"),
    ("sine inverse of a", "sin^-1(a)"),
    ("tan inverse x is equal to pi over two", "arctan x = π/2"),
    ("d by dx of tan inverse of x", "d/dx arctan(x)"),
    ("the double integral of P sub x plus Q sub y dA", "∬ P_x + Q_y dA"),
]


@pytest.mark.parametrize("words,notation", PAIRS)
def test_paired_surfaces_canonicalise_equal(words, notation):
    assert canonical(words) == canonical(notation), (canonical(words), canonical(notation))
    assert recognized(words, notation)


def test_form_detection():
    assert form_of("two over the square root of pi") == "words"
    assert form_of("2 / sqrt(pi)") == "notation"
    assert form_of("2 over the square root of pi") == "mixed"
    assert form_of("x^2 + 1 = 0") == "notation"
    assert form_of("x squared plus one equals zero") == "words"


def test_wrong_operator_is_not_recognized():
    assert not recognized("two over the square root of pi", "2 * sqrt(pi)")
    assert not recognized("x squared plus one", "x cubed plus one")
    assert not recognized("twenty one over seven", "twenty over seven")


def test_parentheses_survive_normalisation_and_are_recorded_but_not_compared():
    g = grammar()
    c = g.canonical("sqrt(pi)")
    assert "SQRT" in c.text and "PI" in c.text
    assert "LPAREN" in c.text_with_grouping and "RPAREN" in c.text_with_grouping
    assert "LPAREN" not in c.text


def test_greek_names_and_function_abbreviations_do_not_decide_form():
    assert form_of("sin theta") == "words"
    assert form_of("sin θ") == "words"
    assert form_of("sine of pi over two") == "words"
    assert form_of("sin(π/2)") == "notation"


def test_decimal_and_factorial():
    assert canonical("three point five percent") == "3.5 PCT"
    assert canonical("n factorial") == "n FACT"


def test_grammar_version():
    assert grammar().version.startswith("mathforms.v4")


def test_letter_runs_primes_ordinals_and_bang():
    """Determinism (sigma is Greek), notation surfaces (letter runs, differentials, primes, ordinal exponents, braces), bang."""
    assert canonical("sigma of x") == canonical("σ x") == "SIGMA x" and canonical("the sum of x") == "SUM x"
    assert canonical("ax + cz") == canonical("a x plus c z") == "a x PLUS c z"
    assert canonical("ax + by") == "a x PLUS by"
    assert canonical("∫ f(x) dx") == canonical("the integral of f of x d x")
    assert canonical("f'(x)") == canonical("f prime of x") == "f PRIME OF x"
    assert canonical("x to the fourth") == canonical("x^4") and canonical("x to the third") == canonical("x cubed")
    assert canonical("10^{-6}") == canonical("ten to the minus six") == "10 POW MINUS 6"
    assert canonical("wow!") == "wow" and canonical("n!") == "n FACT" and canonical("5!") == "5 FACT"
    g = grammar()
    surf = {}
    for s_, c, _, _ in g.table:
        surf.setdefault(s_, set()).add(c)
    assert all(len(v) == 1 for v in surf.values())


def test_application_bounds_exponents_and_ordinals():
    """ "of" is kept where it changes the expression; bounds, exponents before functions, digit ordinals."""
    assert (
        canonical("f of x") == canonical("f(x)") == "f OF x"
        and canonical("f x") == "f x"
        and not recognized("f of x", "f x")
    )
    assert canonical("z of t") == canonical("z(t)") == "z OF t" and not recognized("z of t", "z t")
    assert (
        form_of("f(x) = 2") == "notation"
        and form_of("f of x equals two") == "words"
        and form_of("f of x = 2") == "mixed"
    )
    assert canonical("sine of x") == canonical("sine x") == canonical("sin(x)") == "SIN x"
    assert canonical("the integral from zero to one of x") == canonical("∫_0^1 x") == "INT SUB 0 POW 1 x"
    assert canonical("d by d x of f of x") == canonical("d/dx f(x)") == "DERIV x f OF x"
    assert canonical("2(x+1)") == "2 x PLUS 1"
    assert (
        canonical("the sum from i equals 1 to n of x sub i")
        == canonical("∑_{i=1}^{n} x_i")
        == "SUM SUB i EQ 1 POW n x SUB i"
    )
    assert canonical("integral of x dx from 0 to 1") == "INT x d x from 0 to 1"
    assert canonical("e to the log x") == canonical("e^(log x)") == "e POW LOG x"
    assert canonical("two to the pi") == "2 POW PI" and canonical("x goes to the limit") == "x ARROW LIM"
    assert canonical("x to the 1 3rd") == canonical("x^(1/3)") == "x POW 1 OVER 3"
    assert canonical("a 3rd pi") == "1 OVER 3 PI" and canonical("2 5ths") == "2 OVER 5"
    assert (
        canonical("to the 9th power") == canonical("to the ninth power") == "POW 9"
        and canonical("to the nth power") == "POW n"
    )
    assert (
        canonical("the 3rd derivative") == "3rd derivative" and canonical("the third derivative") == "third derivative"
    )
    assert canonical("x is less than y") == canonical("x < y") == "x LT y" and canonical("x is at least y") == "x GEQ y"
    assert canonical("x equals to y") == "x EQ y"
    assert canonical("inverse sine x") == canonical("arcsin x") == canonical("sin^-1 x") == "ARCSIN x"
    assert (
        canonical("Δx") == canonical("delta x") == "DELTA x"
        and canonical("x → ∞") == canonical("x goes to infinity") == "x ARROW INF"
    )
    assert canonical("f of x plus delta x and y") == "f OF x PLUS DELTA x y"
    assert (
        canonical("the limit as delta x goes to 0 of f of x plus delta x minus f of x divided by delta x")
        == "LIM as DELTA x ARROW 0 f OF x PLUS DELTA x MINUS f OF x OVER DELTA x"
    )
    assert canonical("v is equal to a 3rd pi times 2 5ths") == "v EQ 1 OVER 3 PI TIMES 2 OVER 5"
    assert canonical("x dot of 0 equals to 0") == "x DOT OF 0 EQ 0"
    assert canonical("one-half m v squared") == canonical("1/2 m v^2") == "1 OVER 2 m v POW 2"
    assert canonical("x to the one-half") == "x POW 1 OVER 2" and canonical("twenty-one over seven") == "21 OVER 7"
    assert canonical("x-axis") == "x MINUS axis"


def test_rewrites_through_grouping_and_normalizer_symbols():
    """Post-rewrites through grouping, the integrand marker vs application, double prime and "raised to the power", the variable
    of differentiation, and the normalizer's super/subscripts, fraction slash, prime, capital sigma and in-word apostrophe."""
    assert canonical("sin^{-1} x") == canonical("sin⁻¹(x)") == canonical("inverse sine of x") == "ARCSIN x"
    assert (
        canonical("integral from 0 to 1 f of x dx")
        == canonical("integral from 0 to 1 of f of x dx")
        == canonical("∫₀¹ f(x) dx")
        == "INT SUB 0 POW 1 f OF x d x"
    )
    assert (
        canonical("integral from 0 to pi over 2 of f of x") == "INT SUB 0 POW PI OVER 2 f OF x"
        and canonical("integral from a to b of x") == "INT SUB a POW b x"
    )
    assert canonical("f double prime of x") == canonical("f''(x)") == canonical("f″(x)") == "f PRIME PRIME OF x"
    assert canonical("x raised to the power 3") == canonical("x to the power 3") == "x POW 3"
    assert canonical("the derivative of f of x") == "DERIV f OF x" and not recognized(
        "the derivative of f of x", "the derivative of f x"
    )
    assert canonical("d by d x of f of x") == canonical("d/dx f(x)") == canonical("d/dx (f(x))") == "DERIV x f OF x"
    assert canonical("partial over partial x of f") == canonical("∂/∂x f") == "PARTIAL OVER PARTIAL x f"
    assert (
        canonical("10⁻⁶") == canonical("10^{-6}") == "10 POW MINUS 6"
        and canonical("x₁") == canonical("x_1") == "x SUB 1"
    )
    assert (
        canonical("½") == canonical("one half") == "1 OVER 2"
        and canonical("f′(x)") == canonical("f prime of x") == "f PRIME OF x"
    )
    assert (
        canonical("Σ_{i=1}^n x_i")
        == canonical("the sum from i equals 1 to n of x sub i")
        == "SUM SUB i EQ 1 POW n x SUB i"
    )
    assert canonical("σ(x)") == canonical("sigma of x") == "SIGMA x"
    assert canonical("that's x squared") == "thats x POW 2" and canonical("f'(x)") == "f PRIME OF x"
