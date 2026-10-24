from math_eval.corpora.mathspeech import video_id


def test_video_id_from_source_url_forms():
    assert video_id("https://www.youtube.com/watch?v=--lPz7VFnKI&t=10s") == "--lPz7VFnKI"
    assert video_id("https://youtu.be/YBajUR3EFSM") == "YBajUR3EFSM"
    assert video_id("not a url") is None
