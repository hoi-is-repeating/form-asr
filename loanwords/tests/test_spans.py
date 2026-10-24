import pytest

from loanword_eval.spans import AmbiguousSpan, SpanNotFound, locate, validate


def test_locate_first_and_second_occurrence():
    t = "버스 타고 또 버스 갈아탔어"
    s1 = locate(t, "버스", 1)
    s2 = locate(t, "버스", 2)
    assert (s1.start, s1.end) == (0, 2)
    assert t[s2.start : s2.end] == "버스" and s2.start > s1.end


def test_locate_none_occurrence_requires_uniqueness():
    t = "버스 타고 또 버스 갈아탔어"
    with pytest.raises(AmbiguousSpan):
        locate(t, "버스", None)
    assert locate("게임을 했지", "게임", None).start == 0


def test_locate_missing_and_out_of_range():
    with pytest.raises(SpanNotFound):
        locate("게임을 했지", "호텔")
    with pytest.raises(SpanNotFound):
        locate("게임을 했지", "게임", 2)


def test_validate_catches_bad_offsets():
    t = "그치 서로 다른 게임을 했지."
    good = locate(t, "게임")
    assert validate(t, good.start, good.end, "게임")
    assert not validate(t, 10, 12, "게임")
    assert not validate(t, 0, 0, "게임")
