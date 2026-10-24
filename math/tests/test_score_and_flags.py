"""Item scoring: whole-clip and in-context rules, LaTeX hint, control words; flags redefined for mathematics."""

from math_eval import flags
from math_eval.score import control_strict, item_surface_in_context, looks_like_latex, score_clip, score_in_context


def test_clip_words_notation_wrong_empty():
    ref = "two over the square root of pi"
    assert (
        score_clip(ref, "2 / sqrt(pi)").R,
        score_clip(ref, "2 / sqrt(pi)").F,
        score_clip(ref, "2 / sqrt(pi)").outcome,
    ) == (1, "notation", "FORM")
    s = score_clip(ref, "Two over the square root of pi.")
    assert (s.R, s.R_strict, s.F, s.outcome) == (1, 1, "words", "CORRECT")
    assert score_clip(ref, "2 over the square root of pi").F == "mixed"
    w = score_clip(ref, "two over the square root of e")
    assert (w.R, w.outcome) == (0, "WRONG") and 0 < w.token_cer < 1
    assert score_clip(ref, "").outcome == "EMPTY" and score_clip(ref, "").R == 0


def test_strict_implies_recognized_and_wrong_operator_rejected():
    assert score_clip("x squared plus one", "x cubed plus one").R == 0
    assert score_clip("twenty one over seven", "twenty over seven").R == 0
    for h in ["x^2 + 1", "x squared plus one"]:
        s = score_clip("x squared plus one", h)
        assert s.R >= s.R_strict


def test_in_context_finds_item_anywhere_and_reads_form_from_span():
    ref = "two over the square root of pi"
    s = score_in_context(ref, "the error function was normalized by putting 2 over the square root of pi")
    assert (s.R, s.F, s.R_tail) == (1, "mixed", 1) and s.match_end == s.n_tokens_hyp
    early = score_in_context(
        ref,
        "two over the square root of pi and then many other words follow here in the window "
        + " ".join(f"w{i}" for i in range(20)),
    )
    assert early.R == 1 and early.R_tail == 0
    miss = score_in_context(ref, "the error function was normalized by putting two over the cube root of pi")
    assert miss.R == 0 and 0 < miss.token_cer < 1 and miss.match_start is not None


def test_item_surface_in_context_returns_the_raw_tail():
    assert item_surface_in_context("one half", "which we write as one half") == "one half"
    assert item_surface_in_context("one half", "which we write as 1/2") == "1/2"
    t = item_surface_in_context("one half", "which we write as one third")
    assert t.endswith("one third")


def test_latex_hint_and_controls():
    assert looks_like_latex(r"\frac{1}{2}") and looks_like_latex("x^{2}") and not looks_like_latex("x squared")
    assert control_strict("normalized", "the error function was Normalized, by putting") == 1
    assert control_strict("normalized", "the error function was normalizing") == 0


def test_flags_for_mathematics():
    pats, _ = flags.load_off_task_patterns()
    sp, ver = flags.load_solved_patterns()
    assert ver == "solved_patterns.v1"
    f = flags.utterance_flags("x", "x squared plus one", None, pats, sp)
    assert f["hallucinated"] == 0
    assert flags.hallucinated("x squared", "x squared " * 8) == 1
    f = flags.utterance_flags("x squared", "I'm sorry, I cannot transcribe this.", None, pats, sp)
    assert f["off_task"] == 1 and f["solved"] == 1
    assert flags.empty("π/4") == 0 and flags.empty("...") == 1
    W = "suppose y cubed minus eight equals zero and take the square root of minus one"
    assert flags.prompt_copy("2 / sqrt(pi)", "take sqrt(-1) and call it i", "two over the square root of pi") == 0
    assert flags.prompt_copy("y cubed minus eight equals zero", W, "x squared") == 1
    assert flags.prompt_copy("x squared plus one equals zero", W, "x squared plus one equals zero") == 0


def test_discourse_flag_reference_and_copy_cross_check():
    """In-discourse flag reference; word-level repeat rule; answer frames only when not spoken; prompt copy cross-checked
    against the same clip's unprompted transcript."""
    pats, _ = flags.load_solved_patterns()
    ref = flags.discourse_flag_reference(
        "x equals a inverse b", "so the left side a inverse a x simplifies to", "", "and that is the solution", "D0"
    )
    assert ref.startswith("so the left side") and "x equals a inverse b" in ref and ref.endswith("the solution")
    assert flags.solved(ref, "simplifies to x equals a inverse b", pats) == 0
    assert flags.solved("x plus y", "the answer is 5", pats) == 1
    assert flags.hallucinated("theta minus theta cubed over six", "theta minus theta minus theta cubed over six") == 0
    assert flags.hallucinated("x plus one", "x plus one thank you thank you thank you thank you thank you") == 1
    w = flags.discourse_flag_reference(
        "item", " ".join(f"w{i}" for i in range(40)), "", " ".join(f"a{i}" for i in range(20)), "D1"
    )
    assert w.split()[0] == "w0" and w.split()[-1] == "a4"
    hyp, prompt = "dr dt dot r and the derivative of r dot r is", "the derivative of r dot r is what we want"
    assert flags.prompt_copy(hyp, prompt, "dr dt dot r") == 1
    assert flags.prompt_copy(hyp, prompt, "dr dt dot r", unprompted_hyp=hyp) == 0
