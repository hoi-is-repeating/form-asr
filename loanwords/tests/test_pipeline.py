"""End-to-end run of the pipeline on the synthetic fixture with the test decoder."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from loanword_eval import fixture

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["none", "H1", "L1", "H2", "L2"]


def run(script, *args):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True, cwd=ROOT
    )
    assert r.returncode == 0, f"{script}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_pipeline_on_fixture(tmp_path):
    fx = fixture.build(tmp_path / "fx")
    (fx / "labels").mkdir(exist_ok=True)
    shutil.copy(ROOT / "labels" / "precedents.v1.json", fx / "labels" / "precedents.v1.json")
    run("prepare_data.py", "--fixture", fx)
    run("build_labels.py", "--fixture", fx, "--tokenizer", "openai/whisper-tiny")
    run("count_tokens.py", "--fixture", fx, "--recognizer", "whisper", "--tokenizer", "openai/whisper-tiny")
    run("decode.py", "--fixture", fx, "--fake", "--condition", *CONDITIONS)
    run("score.py", "--fixture", fx, "--condition", *CONDITIONS)
    run("dominant_forms.py", "--fixture", fx)
    run("analyze.py", "--fixture", fx, "--B", 200, "--n-perm", 200)
    report = run("report.py", "--fixture", fx)

    res = json.loads((fx / "outputs" / "whisper.json").read_text())
    assert res["H1a"]["n"] > 0 and 0 <= res["H1a"]["E_A"]["estimate"] <= res["H1a"]["E_S"]["estimate"] <= 1
    assert set(res["H2a"]) == {"H1/L1", "H2/L2", "pooled"}
    assert (
        res["H2a"]["pooled"]["difference_L_minus_H"]["estimate"] != 0
    )  # the test decoder switches form under a Latin prompt
    assert "Base rate" in report and "Prompt steering" in report
