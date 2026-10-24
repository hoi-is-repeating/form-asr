from loanword_eval import accept


def test_g2p_member_colliding_with_another_type_is_removed_and_sources_kept():
    toks = [
        {
            "item_id": "a:0:5",
            "corpus": "hike",
            "utt_id": "a",
            "start": 0,
            "end": 5,
            "hangul": "피자",
            "latin": "pizza",
            "ref_form": "latin",
            "type_key": "피자|pizza",
            "korean_matrix": True,
        },
        {
            "item_id": "b:0:7",
            "corpus": "hike",
            "utt_id": "b",
            "start": 0,
            "end": 7,
            "hangul": "피처",
            "latin": "feature",
            "ref_form": "latin",
            "type_key": "피처|feature",
            "korean_matrix": True,
        },
        {
            "item_id": "c:0:3",
            "corpus": "hike",
            "utt_id": "c",
            "start": 0,
            "end": 3,
            "hangul": "버스",
            "latin": "bus",
            "ref_form": "latin",
            "type_key": "버스|bus",
            "korean_matrix": True,
        },
    ]
    fake = {"pizza": "피처", "feature": "피처", "bus": "버스"}
    rows = accept.rows_for_hike(toks, g2p=lambda s: fake[s])
    report = accept.guard_g2p(rows)
    pizza = next(r for r in rows if r["type_key"] == "피자|pizza")
    assert "피처" not in pizza["members"] and report["g2p_members_removed"] == 1
    assert pizza["members_nog2p"] == ["pizza", "피자"] and pizza["member_sources"]["피자"] == ["hangul"]
    bus = next(r for r in rows if r["type_key"] == "버스|bus")
    assert set(bus["member_sources"]["버스"]) == {
        "g2p",
        "hangul",
    }


def test_g2p_output_with_latin_is_rejected():
    src = accept.build_with_sources("온오프라인", latin="on·offline", g2p=lambda s: "안offline")
    assert not any(s == {"g2p"} for s in src.values())
