from loanword_eval.align import char_align, span_outcome
from loanword_eval.normalize import KoreanPolicy, contains_latin


def _norm(s):
    return KoreanPolicy().apply(s).text


def test_correct_span():
    ref = _norm("그리고 우리 피시방 가자.")
    hyp = _norm("그리고 우리 피시방 가자")
    al = char_align(ref, hyp)
    r = span_outcome(al, ref.index("피시방"), ref.index("피시방") + 3)
    assert r.outcome == "CORRECT" and r.span_cer == 0.0


def test_full_deletion():
    ref = _norm("그리고 우리 피시방 가자")
    hyp = _norm("그리고 우리 가자")
    al = char_align(ref, hyp)
    r = span_outcome(al, ref.index("피시방"), ref.index("피시방") + 3)
    assert r.outcome == "DEL" and r.n_del == 3


def test_substitution_to_native_competitor():
    ref = _norm("나는 버스를 탔다")
    hyp = _norm("나는 차를 탔다")
    al = char_align(ref, hyp)
    s = ref.index("버스")
    r = span_outcome(al, s, s + 2)
    assert r.outcome in ("SUB", "PARTIAL") and r.n_equal < 2


def test_latin_longer_than_hangul_uses_adjacent_insertions():
    ref = _norm("우리 피시방 가자")
    hyp = _norm("우리 PC방 가자")
    al = char_align(ref, hyp)
    s = ref.index("피시방")
    r = span_outcome(al, s, s + 3, attribute_adjacent=contains_latin)
    assert "pc" in r.hyp_span_with_ins and r.hyp_span_with_ins.endswith("방")
    assert r.outcome != "CORRECT"


def test_partial_and_inside_insertion():
    ref = _norm("컴퓨터 켰어")
    hyp = _norm("컴퓨우터 켰어")
    al = char_align(ref, hyp)
    r = span_outcome(al, 0, 3)
    assert r.n_ins_inside >= 1 or r.outcome == "PARTIAL"
