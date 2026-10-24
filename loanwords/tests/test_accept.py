from loanword_eval import accept


def test_spaces_removed_variant_and_normalization():
    s = accept.build("아이스크림", latin="ice cream")
    assert s == {"아이스크림", "icecream"}
    assert accept.build("피시방", latin="PC방", pron_form="피씨방") == {"피시방", "pc방", "피씨방"}


def test_members_idempotent_under_nu():
    for m in accept.build("카페", latin="Café!"):
        assert accept.nu(m) == m


def test_g2p_and_dual_spelling_members():
    s = accept.build("컴퓨터", latin="computer", g2p=lambda x: "컴퓨터", dual_spelling="Computer")
    assert {"컴퓨터", "computer"} <= s
