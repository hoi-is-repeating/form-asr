import json

import numpy as np
import pytest

from loanword_eval import fixture
from loanword_eval.corpora.hike import load_rows, tokens


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    return fixture.build(tmp_path_factory.mktemp("synthetic"))


def test_fixture_builds_ten_items_with_stable_hashes(synth_dir, tmp_path):
    items = [json.loads(l) for l in (synth_dir / "items.jsonl").read_text().splitlines()]
    assert len(items) == 10
    again = fixture.build(tmp_path / "again")
    items2 = [json.loads(l) for l in (again / "items.jsonl").read_text().splitlines()]
    assert [i["audio_sha256"] for i in items] == [i["audio_sha256"] for i in items2]
    wav = np.load(synth_dir / "audio" / "u01.npy")
    assert wav.dtype == np.float32 and 0.4 * 16000 < len(wav) < 1.2 * 16000


def test_fixture_rows_cover_designed_cases():
    rows = fixture.rows()
    items = list(load_rows(rows))
    by_id = {it.utt_id: it for it in items}
    toks = {it.utt_id: list(tokens(it)) for it in items}
    assert [t.realization for t in toks["u01"]] == ["hangul"]
    assert [t.realization for t in toks["u02"]] == ["latin"]
    assert len(toks["u05"]) == 2 and {t.occurrence for t in toks["u05"]} == {1, 2}
    assert len(toks["u04"]) == 1 and toks["u04"][0].start == by_id["u04"].text.index("bug")
    assert len(toks["u06"]) == 1
