"""End-to-end run of the pipeline on the synthetic fixture with the test decoder and the test converter."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from math_eval import fixture

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ["none", "N0", "W1", "S1", "W2", "S2"]
DISCOURSE = ["D0", "D1", "D2"]
TINY = ["--tokenizer", "openai/whisper-tiny"]


def run(script, *args):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True, cwd=ROOT
    )
    assert r.returncode == 0, f"{script}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_pipeline_on_fixture(tmp_path):
    fx = fixture.build(tmp_path / "fx")
    shutil.copy(ROOT / "labels" / "precedents.v3.json", fx / "labels" / "precedents.v3.json")
    run("prepare_data.py", "--fixture", fx, *TINY)
    run("build_labels.py", "--fixture", fx, *TINY)
    run("decode.py", "--fixture", fx, "--fake", "steer", "--condition", *BENCHMARK)
    run("decode.py", "--fixture", fx, "--fake", "partial", "--condition", *DISCOURSE)
    run("score.py", "--fixture", fx, "--recognizer", "fake", "--condition", *BENCHMARK, *DISCOURSE)
    run("dominant_forms.py", "--fixture", fx, "--recognizer", "fake")
    shutil.copy(fx / "direction_fake.jsonl", fx / "labels" / "direction_fake.v1.jsonl")
    run("convert.py", "--fixture", fx, "--converter", "fake", "--reference")
    for c in ("D0", "D2"):
        run("convert.py", "--fixture", fx, "--converter", "fake", "--recognizer", "fake", "--condition", c)
    run("analyze.py", "--fixture", fx, "--recognizer", "fake", "--B", 200, "--n-perm", 200)
    run("attribute.py", "--fixture", fx, "--recognizer", "fake", "--converter", "fake", "--B", 200, "--n-perm", 200)
    report = run("report.py", "--fixture", fx)

    disc = rows(fx / "items_discourse.jsonl")
    assert len(disc) == 3 and all(
        it["audio"]["d1"]["duration_s"] <= 30.0 and abs(it["audio"]["d1"]["end"] - it["math_end"] - 0.5) < 1e-6
        for it in disc
    )
    assert [r["audio_sha256"] for r in rows(fx / "decodes" / "D0_fake.jsonl")] == [
        r["audio_sha256"] for r in rows(fx / "decodes" / "D2_fake.jsonl")
    ]
    assert len(rows(fx / "accept_sets.jsonl")) == 10 and len(rows(fx / "controls_discourse.jsonl")) > 10
    res = json.loads((fx / "outputs" / "fake.json").read_text())
    h1a = res["H1a"]["benchmark_none"]
    assert (
        h1a["n"] == 10
        and 0 <= h1a["E_A"]["estimate"] <= h1a["E_S"]["estimate"] <= 1
        and res["H1a"]["discourse_D0"]["n"] == 3
    )
    assert res["H1b_full_set_descriptive"]["waterfall"]["controls"] > 10
    assert set(res["H2a"]) == {"W1", "W2", "pooled"} and res["H2a"]["pooled"]["difference_S_minus_W"]["estimate"] > 0
    assert res["H2b"]["pooled"]["conservation_ok"]
    att = json.loads((fx / "outputs" / "attribution_fake.json").read_text())
    assert (
        att["H3a"]["decomposition"]["identity_ok"]
        and att["H3b"]["pairing"]["paired"] == 3
        and att["H3b"]["recovered_0_to_1"] >= 1
    )
    assert "Base rate" in report and "Prompt steering" in report and "Recognition and conversion" in report
