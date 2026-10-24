from loanword_eval.score import score_item, score_utterance

A_BUS = {"버스", "bus"}
A_PC = {"피시방", "pc방", "피씨방"}


def test_exact_hangul_and_latin_forms():
    ref = "어제 버스 타고 학교에 갔어."
    s = score_item(ref, "어제 버스 타고 학교에 갔어", 3, 5, A_BUS)
    assert (s.R, s.R_strict, s.F, s.match_kind, s.outcome) == (1, 1, "hangul", "exact", "CORRECT")
    s = score_item(ref, "어제 bus 타고 학교에 갔어", 3, 5, A_BUS)
    assert (s.R, s.R_strict, s.F, s.outcome) == (1, 0, "latin", "SUB")


def test_mixed_form_and_case():
    ref = "주말에 PC방 가서 게임 했어."
    s = score_item(ref, "주말에 피시방 가서 게임 했어", 4, 7, A_PC)
    assert s.R == 1 and s.F == "hangul" and s.R_strict == 0
    s = score_item(ref, "주말에 pc방 가서 게임 했어", 4, 7, A_PC)
    assert s.R == 1 and s.F == "mixed" and s.R_strict == 1


def test_not_recognized_outcomes():
    ref = "어제 버스 타고 학교에 갔어."
    s = score_item(ref, "어제 택시 타고 학교에 갔어", 3, 5, A_BUS)
    assert s.R == 0 and s.F is None and s.outcome == "SUB"
    s = score_item(ref, "어제 타고 학교에 갔어", 3, 5, A_BUS)
    assert s.R == 0 and s.outcome == "DEL"
    s = score_item(ref, "", 3, 5, A_BUS)
    assert s.R == 0 and s.outcome == "DEL"


def test_window_tolerance_and_sensitivity_value():
    ref = "어제 버스 타고 학교에 갔어."
    hyp = "어제는 그 bus 타고 학교에 갔어"
    s = score_item(ref, hyp, 3, 5, A_BUS, window=4)
    assert s.R == 1 and s.F == "latin"
    assert s.R_nowindow == 0 and s.match_kind == "window"
    far = "어제 타고 학교에 갔어 아주 멀리 bus"
    s2 = score_item(ref, far, 3, 5, A_BUS, window=4)
    assert s2.R == 0


def test_adjacent_latin_insertion_attributed_le_rule():
    ref = "그 카페 옆에"
    s = score_item(ref, "그 cafe카페 옆에", 2, 4, {"카페", "cafe"})
    assert s.n_ins_adjacent > 0 and s.R == 1


def test_window_cannot_borrow_a_neighbouring_item():
    ref, hyp, A = "AI AI 좋아", "AI 좋아", {"ai", "에이아이"}
    first, second = score_utterance(ref, hyp, [(0, 2, A), (3, 5, A)])
    assert first.R == 1 and second.R == 0
    assert score_item(ref, hyp, 3, 5, A).R == 1


def test_adjacent_items_both_hangul_and_both_latin():
    """Normalization removes spaces and the aligner assigns the run to the first item; both must still be recognized."""
    ref = "그 brunch cafe 발견했는데"
    A_br, A_cf = {"brunch", "브런치"}, {"cafe", "카페"}
    br, cf = score_utterance(ref, "그 브런치카페 발견했는데", [(2, 8, A_br), (9, 13, A_cf)])
    assert (br.R, br.F) == (1, "hangul") and (cf.R, cf.F) == (1, "hangul"), (br, cf)
    br, cf = score_utterance(ref, "그 브런치 카페 발견했는데", [(2, 8, A_br), (9, 13, A_cf)])
    assert br.R_strict == 0 and cf.R == 1 and cf.F == "hangul"
    ref2 = "다음 주 팀 미팅에서"
    tm, mt = score_utterance(ref2, "다음 주 team meeting에서", [(5, 6, {"팀", "team"}), (7, 9, {"미팅", "meeting"})])
    assert (tm.R, tm.F) == (1, "latin") and (mt.R, mt.F) == (1, "latin"), (tm, mt)
    tm, mt = score_utterance(ref2, ref2, [(5, 6, {"팀", "team"}), (7, 9, {"미팅", "meeting"})])
    assert tm.R_strict == 1 and mt.R_strict == 1
    tm, mt = score_utterance(ref2, "다음 주 팀 에서", [(5, 6, {"팀", "team"}), (7, 9, {"미팅", "meeting"})])
    assert tm.R == 1 and mt.R == 0


def test_correct_span_does_not_absorb_neighbouring_latin_insertion():
    """A control that is correct must stay strict-correct when the neighbouring item comes out Latin."""
    ref = "효과 모니터링 했어"
    ctrl = score_item(ref, "효과 monitoring 했어", 0, 2, {"효과"})
    assert ctrl.R_strict == 1 and ctrl.outcome == "CORRECT", ctrl
    item = score_item(ref, "효과 monitoring 했어", 3, 7, {"모니터링", "monitoring"})
    assert item.R == 1 and item.F == "latin"


def test_window_ties_are_deterministic():
    outs = {
        score_item("주말에 X 가서", "주말에 피시방 pc방 가서", 4, 5, {"피시방", "pc방", "피씨방"}).matched_member
        for _ in range(30)
    }
    assert len(outs) == 1


def test_one_character_member_matches_only_inside_its_anchor_segment():
    """One-character members match an unclaimed occurrence inside the anchor segment, after longer words have claimed theirs."""
    ref = "완료하려면 team member한테 물어봐"
    A_team, A_member = {"team", "팀"}, {"member", "멤버"}
    tm, mb = score_utterance(ref, "완료하려면 팀 멤버한테 물어봐", [(6, 10, A_team), (11, 17, A_member)])
    assert (tm.R, tm.F, tm.match_kind) == (1, "hangul", "window-1char") and (mb.R, mb.F) == (1, "hangul"), (tm, mb)
    ref2 = "우리 team teamwork 좋아"
    tm, tw = score_utterance(ref2, "우리 팀워크 좋아", [(3, 7, A_team), (8, 16, {"teamwork", "팀워크"})])
    assert tw.R == 1 and tm.R == 0, (tm, tw)
    tm = score_item("회의 team 있어", "회의 있어 팀", 3, 7, A_team)
    assert tm.R == 0, tm
    tm = score_item("팀 회의 있어", "팀장 회의 있어", 0, 1, A_team)
    assert (tm.R, tm.R_strict, tm.outcome, tm.hyp_material) == (0, 0, "PARTIAL", "팀장")


def test_material_is_the_whole_token_not_the_substring():
    """Strict recognition must not credit an output that merely contains the reference."""
    A_bug, A_bus = {"bug", "버그"}, {"bus", "버스"}
    assert score_item("이번 bug 수정했어", "이번 bugs 수정했어", 3, 6, A_bug).R_strict == 0
    assert score_item("이번 bug 수정했어", "이번 bugs 수정했어", 3, 6, A_bug).R == 0
    assert score_item("어제 bus 타고 갔어", "어제 business 타고 갔어", 3, 6, A_bus).R == 0
    assert score_item("이번 bug 수정했어", "이번 debugging 수정했어", 3, 6, A_bug).R == 0
    ctrl = score_item("효과 확인했어", "효과적 확인했어", 0, 2, {"효과"})
    assert (ctrl.R_strict, ctrl.hyp_material) == (0, "효과적")
    assert score_item("코드 리뷰 했어", "바코드 리뷰 했어", 0, 2, {"code", "코드"}).R == 0
    ok = score_item("카페 갔어", "카페에 갔어", 0, 2, {"cafe", "카페"})
    assert (ok.R, ok.R_strict, ok.F) == (1, 1, "hangul")
    assert score_item("이번 bug 수정했어", "이번 bug fix 수정했어", 3, 6, A_bug).R_strict == 1
    assert score_item("효과 monitoring 했어", "효과 monitoring 했어", 0, 2, {"효과"}).R_strict == 1
    br, cf = score_utterance(
        "그 brunch cafe 발견했는데",
        "그 브런치카페 발견했는데",
        [(2, 8, {"brunch", "브런치"}), (9, 13, {"cafe", "카페"})],
    )
    assert (br.R, cf.R) == (1, 1), (br, cf)


def test_strict_implies_recognized_and_boundaries_apply_to_exact_material():
    A = {"team", "팀"}
    r = score_item("인프라 팀은 그거", "인프라 바팀은 그거", 4, 5, A)
    assert (r.R, r.R_strict) == (0, 0), r
    for hyp in ("인프라 팀은 그거", "인프라 team은 그거", "인프라 팀을 그거"):
        r = score_item("인프라 팀은 그거", hyp, 4, 5, A)
        assert r.R == 1 and (r.R_strict <= r.R)


def test_hangul_boundary_is_symmetric_with_latin_under_context_errors():
    """A mis-transcribed ending after the item is judged by the same affix rule for
    both scripts. Particle-like replacements keep recognition in both scripts; a compounding syllable costs it in both."""
    A = {"bug", "버그"}
    for ref, hyp_h, hyp_l, expect in (
        ("이번 bug는 고쳤어", "이번 버그도 고쳤어", "이번 bug도 고쳤어", 1),
        ("그건 bug였어", "그건 버그이었어", "그건 bug이었어", 1),
        ("이번 bug하고 고쳤어", "이번 버그하고 고쳤어", "이번 bug하고 고쳤어", 1),
        ("이번 bug는 고쳤어", "이번 버그들 고쳤어", "이번 bug들 고쳤어", 0),
    ):
        rh, rl = score_item(ref, hyp_h, 3, 6, A), score_item(ref, hyp_l, 3, 6, A)
        assert rh.R == rl.R == expect, (ref, hyp_h, hyp_l, rh, rl)
    assert score_item("이번 bug는 고쳤어", "이번 버그장는 고쳤어", 3, 6, A).R == 0
    assert score_item("이번 bug는 고쳤어", "이번 bug장는 고쳤어", 3, 6, A).R == 0
    assert score_item("그건 bug였어", "그건 버그장였어", 3, 6, A).R == 0
    assert score_item("이번 bug 고쳤어", "이번 버그들 고쳤어", 3, 6, A).R == 0
    assert score_item("이번 bug하고 고쳤어", "이번 버그를하고 고쳤어", 3, 6, A).R == 1


def test_spacing_errors_do_not_cost_recognition_in_either_script():
    """Normalization removes spaces, so a missing space next to a correctly produced item is not an item error.
    Equal-aligned context is a boundary whatever its script."""
    M = {"카페", "cafe"}
    for ref, hyp, exp_R, exp_strict in (
        ("그 카페 옆에", "그카페 옆에", 1, 1),
        ("그 카페 옆에", "그 카페옆에", 1, 1),
        ("그 카페 옆에", "그카페옆에", 1, 1),
        ("그 카페 옆에", "그cafe옆에", 1, 0),
        ("그 cafe 옆에", "그카페 옆에", 1, 0),
        ("그 cafe 옆에", "그 cafe옆에", 1, 1),
    ):
        start = ref.index("카페") if "카페" in ref else ref.index("cafe")
        end = start + (2 if "카페" in ref else 4)
        r = score_item(ref, hyp, start, end, M)
        assert (r.R, r.R_strict) == (exp_R, exp_strict), (ref, hyp, r)
    B = {"bug", "버그"}
    assert score_item("이번 bug 고쳤어", "이번버그 고쳤어", 3, 6, B).R == 1
    assert score_item("이번 bug 고쳤어", "이번bug 고쳤어", 3, 6, B).R_strict == 1


def test_derivational_suffix_is_a_compound_not_a_particle():
    """A syllable that forms another word is not a replacement of the reference's particle."""
    for ref, hyp, start, end, members in (
        ("우리 팀이 이겼어", "우리 팀장 이겼어", 3, 4, {"팀", "team"}),
        ("그 카페에 갔어", "그 카페인 갔어", 2, 4, {"카페", "cafe"}),
        ("그 효과가 컸어", "그 효과적 컸어", 2, 4, {"효과"}),
        ("코드에서는 봤어", "코드리뷰 봤어", 0, 2, {"코드", "code"}),
        ("그 bug를 봤어", "그 버그팀 봤어", 2, 5, {"bug", "버그"}),
        ("그 카페 갔어", "그 카페인 갔어", 2, 4, {"카페", "cafe"}),
        ("그 학생이 왔어", "그 학생인 왔어", 2, 4, {"학생"}),
    ):
        r = score_item(ref, hyp, start, end, members)
        assert r.R == 0 and r.R_strict == 0, (ref, hyp, r)
    for ref, hyp, start, end, members in (
        ("우리 팀이 이겼어", "우리 팀은 이겼어", 3, 4, {"팀"}),
        ("그 카페에 갔어", "그 카페의 갔어", 2, 4, {"카페"}),
        ("코드에서는 봤어", "코드에서 봤어", 0, 2, {"코드"}),
        ("코드에서는 봤어", "코드에서만 봤어", 0, 2, {"코드"}),
        ("그 학생인데 왔어", "그 학생인 왔어", 2, 4, {"학생"}),
        ("코드에서는 봤어", "코드 봤어", 0, 2, {"코드"}),
    ):
        r = score_item(ref, hyp, start, end, members)
        assert r.R == 1 and r.R_strict == 1, (ref, hyp, r)


def test_window_default_is_the_pre_registered_four():
    import inspect

    assert inspect.signature(score_utterance).parameters["window"].default == 4
    assert inspect.signature(score_item).parameters["window"].default == 4


def test_known_latent_paths_are_documented_not_fixed():
    """Three known edge cases of the scorer, pinned so that a scorer change that alters them is noticed."""
    # (a) window match built from context characters aligned EQUAL to reference context (anchor reuse)
    ref = "이 cafe 좋아 카페인 많아"
    s = ref.index("cafe")
    assert score_item(ref, "이 좋아 카페인 많아", s, s + 4, {"cafe", "카페"}).R == 1
    # (b) affix rule: shared attachment character + one more character; particle concatenation
    ref2 = "우리 team은 이겼어"
    s2 = ref2.index("team")
    assert score_item(ref2, "우리 팀은행 이겼어", s2, s2 + 4, {"team", "팀"}).R == 1
    ref3 = "이번 bug 수정했어"
    s3 = ref3.index("bug")
    assert score_item(ref3, "이번 버그나가 수정했어", s3, s3 + 3, {"bug", "버그"}).R == 1
    # (c) cross-script boundary asymmetry
    assert score_item(ref3, "이번 un버그 수정했어", s3, s3 + 3, {"bug", "버그"}).R == 1
    assert score_item(ref3, "이번 바bug 수정했어", s3, s3 + 3, {"bug", "버그"}).R == 0


def test_worked_example_of_chapter3_scoring_table():
    """Worked example: one HiKE Korean-matrix utterance (Apache-2.0) with four items. The item `deadline` (accepted in Latin or
    Hangul) is rendered in five constructed ways; the other items are scored jointly as in the pipeline."""
    from loanword_eval.score import score_utterance

    ref = "이번 sprint에서 API gateway migration 작업이 생각보다 complex해서 deadline을 못 맞출 것 같아."
    spans = [
        (3, 9, {"sprint", "스프린트"}),
        (16, 23, {"gateway", "게이트웨이"}),
        (24, 33, {"migration", "마이그레이션"}),
        (53, 61, {"deadline", "데드라인"}),
    ]
    expect = {
        "deadline": (1, 1, "latin"),
        "데드라인": (1, 0, "hangul"),
        "데드 라인": (1, 0, "hangul"),
        "데드래인": (0, 0, None),
        "데이터": (0, 0, None),
    }
    for rendering, (R, strict, F) in expect.items():
        res = score_utterance(ref, ref[:53] + rendering + ref[61:], spans, window=4)[3]
        assert (res.R, res.R_strict, res.F) == (R, strict, F), (rendering, res)
