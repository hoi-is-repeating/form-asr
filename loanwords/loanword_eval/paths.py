"""Repository paths and small I/O helpers.

Corpus data and intermediate files live under DATA and WORK (git-ignored); `LOANWORDS_DATA` / `LOANWORDS_WORK` override them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("LOANWORDS_WORK", ROOT / "work"))
DATA = Path(os.environ.get("LOANWORDS_DATA", ROOT / "data"))
OUTPUTS = ROOT / "outputs"
LABELS = ROOT / "labels"
CONFIGS = ROOT / "configs"


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if dataclasses.is_dataclass(r):
                r = dataclasses.asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_yaml(name: str) -> dict:
    import yaml

    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def relpath(path: str | Path) -> str:
    """Path relative to ROOT when possible (portable item tables), else absolute."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


@dataclasses.dataclass
class Item:
    """One utterance. `audio_path` is relative to ROOT when inside it; `audio_sha256` hashes the 16 kHz mono float32 bytes."""

    utt_id: str
    corpus: str
    split: str
    speaker: str | None
    text: str
    pron: str | None
    audio_path: str
    audio_sha256: str | None = None
    duration_s: float | None = None
    extra: dict = dataclasses.field(default_factory=dict)
