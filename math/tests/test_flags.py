"""Flags redefined for mathematics: empty output and solved expressions."""

from math_eval import flags


def test_empty_flag_accepts_all_notation():
    assert flags.empty("π/4") == 0
    assert flags.empty("2 + 2 = 4") == 0
    assert flags.empty("...") == 1
    assert flags.empty("") == 1


def test_solved_flag():
    pats, ver = flags.load_solved_patterns()
    assert ver == "solved_patterns.v1"
    assert flags.solved("two plus two", "The answer is 4.", pats) == 1
    assert flags.solved("two plus two", "two plus two equals 4", pats) == 1
    assert flags.solved("x squared plus one equals zero", "x^2 + 1 = 0", pats) == 0
    assert flags.solved("two plus two", "two plus two", pats) == 0
