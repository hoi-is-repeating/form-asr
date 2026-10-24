from loanword_eval.phon import jamo_distance, sub_cost, to_jamo


def test_decomposition():
    assert to_jamo("버스") == "ㅂㅓㅅㅡ"
    assert to_jamo("닭") == "ㄷㅏㄹㄱ"
    assert to_jamo("bus") == "bus"


def test_feature_weights_order():
    assert sub_cost("ㅍ", "ㅂ") < sub_cost("ㅍ", "ㄴ") == 1.0
    assert sub_cost("ㅏ", "ㅏ") == 0.0
    assert sub_cost("ㅏ", "ㄱ") == 1.0


def test_distance_normalized_and_symmetric():
    d1 = jamo_distance("버스", "뻐쓰")
    d2 = jamo_distance("뻐쓰", "버스")
    assert abs(d1 - d2) < 1e-12 and 0 < d1 < 0.5
    assert jamo_distance("버스", "버스") == 0.0
    assert jamo_distance("버스", "호텔") > jamo_distance("버스", "버쓰")
    assert jamo_distance("버스", "뻐쓰", weighted=False) >= d1
