"""Repository paths and small I/O helpers.

Corpus data and intermediate files live under DATA and WORK (git-ignored); `MATH_DATA` / `MATH_WORK` override them.
"""

from __future__ import annotations

import os
from pathlib import Path

from loanword_eval.paths import read_jsonl, sha256_of, write_jsonl  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("MATH_WORK", ROOT / "work"))
DATA = Path(os.environ.get("MATH_DATA", ROOT / "data"))
OUTPUTS = ROOT / "outputs"
LABELS = ROOT / "labels"
CONFIGS = ROOT / "configs"
FIXTURE_TEXT = ROOT / "tests" / "fixtures" / "ch4_synth"


def load_yaml(name: str) -> dict:
    import yaml

    p = Path(name)
    if not p.is_absolute():
        p = CONFIGS / name
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def relpath(path: str | Path) -> str:
    """Path relative to ROOT when possible, else absolute."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def abspath(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
