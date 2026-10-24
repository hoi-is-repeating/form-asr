"""The normalizer keeps every character that can carry written form."""

from math_eval.normalize_en import normalize


def test_keeps_operators_digits_greek_and_parentheses():
    assert normalize("2 / sqrt(pi)") == "2 / sqrt(pi)"
    assert normalize("x^2 + 1 = 0") == "x^2 + 1 = 0"
    assert normalize("π/4") == "π/4"
    assert normalize("f(x) = e^{-t^2}") == "f(x) = e^{-t^2}"


def test_sentence_punctuation_to_space_and_lowercase():
    assert normalize("Ax plus By plus Cz equals D.") == "ax plus by plus cz equals d"
    assert normalize("So, two over three; right?") == "so two over three right"
    assert normalize("3.5 percent.") == "3.5 percent"
    assert normalize("n!") == "n!"


def test_folds_lookalikes_but_not_meaning():
    assert normalize("x − 1") == "x - 1"
    assert normalize("x²") == "x^2"
    assert normalize("a ⋅ b") == "a · b"


def test_number_words_are_not_converted():
    assert normalize("two") == "two"
    assert normalize("2") == "2"


def test_idempotent():
    for s in ["2 / sqrt(pi)", "Ax plus By.", "x − 1", "π/4"]:
        assert normalize(normalize(s)) == normalize(s)


def test_super_subscripts_fold_twice_prime_sigma_apostrophe():
    assert (
        normalize("sin⁻¹(x)") == "sin^-1(x)"
        and normalize("10⁻⁶") == "10^-6"
        and normalize("x₁") == "x_1"
        and normalize("∫₀¹") == "∫_0^1"
    )
    assert normalize("½") == "1/2"
    assert normalize("f′(x)") == "f'(x)" and normalize("f″") == "f''"
    assert normalize("Σ x") == "∑ x" and normalize("σ x") == "σ x" and normalize("Π x") == "∏ x"
    assert (
        normalize("that's it, don't") == "thats it dont" and normalize("f'(x)") == "f'(x)" and normalize("x's") == "x's"
    )
