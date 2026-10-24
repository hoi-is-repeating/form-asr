from loanword_eval import flags


def test_flags():
    pats, sha = flags.load_off_task_patterns()
    assert len(sha) == 64 and pats
    assert flags.translated("어제 버스 타고 학교에 갔어", "I took the bus to school yesterday") == 1
    assert flags.translated("어제 버스 타고 학교에 갔어", "어제 bus 타고 학교에 갔어") == 0
    assert flags.empty("") == 1 and flags.empty("...") == 1 and flags.empty("안녕") == 0
    assert flags.prompt_copy("아침에 택시를 타고 회사에 갔어요 오늘", "아침에 택시를 타고 회사에 갔어요") == 1
    assert flags.prompt_copy("택시 타고 갔어요", "아침에 택시를 타고 회사에 갔어요") == 0
    assert flags.prompt_copy("그래서 그리고 그런데 학교에 갔어", "네 그래서 그리고 그런데 음 그렇죠") == 1
    assert flags.hallucinated("네 네 네 네 맞아요", "네 네 네 네 맞아요") == 0
    assert flags.hallucinated("네 맞아요", "네 네 네 네 네 네 맞아요") == 1
    assert flags.hallucinated("짧아", "아주 아주 아주 긴 출력이 나왔어요 진짜로") == 1
    assert flags.off_task("죄송합니다 늦었어요", pats, ref="죄송합니다 늦었어요") == 0
    assert flags.off_task("죄송합니다 들리지 않습니다", pats, ref="어제 버스 타고 갔어") == 1
    assert (
        flags.off_task("죄송합니다, 들리지 않습니다", pats) == 1 and flags.off_task("The audio is unclear", pats) == 1
    )
    assert flags.off_task("어제 버스 타고 학교에 갔어", pats) == 0
    assert flags.off_task("네 transcribe the audio into text", pats, frame_text="Transcribe the audio into text.") == 1
    f = flags.utterance_flags("어제 버스 타고 학교에 갔어", "어제 버스 타고 학교에 갔어", None, pats)
    assert flags.any_flag(f) == 0 and set(f) == {
        "translated",
        "hallucinated",
        "empty",
        "off_task",
        "prompt_copy",
        "prompt_copy_ref",
    }
