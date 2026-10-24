import json

from loanword_eval.corpora.hike import load_rows, parse_pairs, tokens

ROWS = [
    {
        "sample_id": "a",
        "text": "이번 bug는 session management logic에 문제가 있었어.",
        "text_normalized": "",
        "cs_level": "phrase",
        "cs_levels_all": "word,phrase",
        "category": "software development",
        "loanwords": json.dumps(
            [
                {"Korean": "버그", "English": "bug"},
                {"Korean": "세션", "English": "session"},
                {"Korean": "로직", "English": "logic"},
            ]
        ),
    },
    {
        "sample_id": "b",
        "text": "버스 타고 또 버스 타고 bus 정류장까지.",
        "text_normalized": "",
        "cs_level": "word",
        "cs_levels_all": "word",
        "category": "everyday",
        "loanwords": json.dumps([{"Korean": "버스", "English": "bus"}]),
    },
    {
        "sample_id": "c",
        "text": "Debugging is fun.",
        "text_normalized": "",
        "cs_level": "sentence",
        "cs_levels_all": "sentence",
        "category": "software development",
        "loanwords": json.dumps([{"Korean": "버그", "English": "bug"}]),
    },
]


def test_parse_pairs_nfc_and_shape():
    ps = parse_pairs(ROWS[0]["loanwords"])
    assert [p.hangul for p in ps] == ["버그", "세션", "로직"] and ps[0].latin == "bug"


def test_tokens_count_both_realizations_and_occurrences():
    items = load_rows(ROWS)
    t_a = tokens(items[0])
    assert {t.realization for t in t_a} == {"latin"} and len(t_a) == 3
    t_b = tokens(items[1])
    assert sum(1 for t in t_b if t.realization == "hangul") == 2
    assert sum(1 for t in t_b if t.realization == "latin") == 1
    assert [t.occurrence for t in t_b if t.realization == "hangul"] == [1, 2]
    for t in t_a + t_b:
        assert (
            items[0 if t.utt_id == "a" else 1].text[t.start : t.end].lower()
            == (t.hangul if t.realization == "hangul" else t.latin).lower()
        )


def test_latin_match_is_word_bounded():
    items = load_rows(ROWS)
    assert tokens(items[2]) == []
