import random
import unicodedata

from loanword_eval.normalize import KoreanPolicy


def test_index_map_round_trip_random():
    pol = KoreanPolicy()
    alphabet = "가나다라마바사 abcXYZ .,!?/*+()[]0123456789"
    rng = random.Random(0)
    for _ in range(200):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        n = pol.apply(s)
        assert len(n.text) == len(n.index_map)
        for i, ch in enumerate(n.text):
            orig = n.original[n.index_map[i]]
            assert ch == orig.lower()


def test_nfd_input_is_normalized():
    pol = KoreanPolicy()
    nfd = unicodedata.normalize("NFD", "버스를 탔다.")
    n = pol.apply(nfd)
    assert n.text == "버스를탔다"
    assert unicodedata.normalize("NFC", n.text) == n.text


def test_punctuation_and_space_removed_latin_lowered():
    n = KoreanPolicy().apply("그리고 우리 PC방 가자.")
    assert n.text == "그리고우리pc방가자"


def test_span_maps_both_ways():
    s = "그리고 우리 피시방 가자."
    n = KoreanPolicy().apply(s)
    o_start = s.index("피시방")
    o_end = o_start + 3
    ns, ne = n.to_normalized_span(o_start, o_end)
    assert n.text[ns:ne] == "피시방"
    assert n.to_original_span(ns, ne) == (o_start, o_end)
