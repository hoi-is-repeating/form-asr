#!/usr/bin/env python
"""Restore the fields of accept_sets.v1.jsonl that are derived from MathSpeech's transcripts.

The file ships without `text`, `canonical` and `canonical_with_grouping`, because the MathSpeech release states no license for its
transcripts. This script reads the transcripts from the MathSpeech release (Hugging Face `AAAI2025/MathSpeech`, or a local JSON Lines
file with `idx` and `transcription`), recomputes the canonical keys with the grammar of the `math_eval` package, puts the fields back
at their original positions and checks the result against `accept_sets.v1.jsonl.original.sha256`. The full file is written as
`accept_sets.v1.full.jsonl`; pass --in-place to replace the stripped file.

Usage: python rebuild_labels.py [--labels DIR] [--transcripts FILE.jsonl] [--in-place]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "accept_sets.v1.jsonl"
sys.path.insert(0, str(HERE.parent))


def default_labels() -> Path:
    return HERE if (HERE / NAME).exists() else HERE.parent / "labels"


def load_transcripts(path: Path | None) -> dict[int, str]:
    if path is not None:
        return {
            int(r["idx"]): r["transcription"]
            for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r
        }
    from datasets import load_dataset

    ds = load_dataset("AAAI2025/MathSpeech", split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c == "audio"])
    return {int(r["idx"]) if "idx" in r else i: r["transcription"] for i, r in enumerate(ds)}


def rebuild(path: Path, transcripts: dict[int, str]) -> bytes:
    try:
        from math_eval.mathforms import grammar
    except ImportError:
        raise SystemExit(
            "needs the math_eval package of https://github.com/hoi-is-repeating/form-asr (pip install -e math)"
        )
    g = grammar()
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = transcripts[r["idx"]]
        c = g.canonical(text)
        items = list(r.items())
        items[4:4] = [("text", text), ("canonical", c.text), ("canonical_with_grouping", c.text_with_grouping)]
        out.append(json.dumps(dict(items), ensure_ascii=False) + "\n")
    return "".join(out).encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=None, help="directory holding accept_sets.v1.jsonl")
    ap.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="JSON Lines with idx and transcription (default: the Hugging Face release)",
    )
    ap.add_argument(
        "--in-place", action="store_true", help="replace the stripped file instead of writing accept_sets.v1.full.jsonl"
    )
    a = ap.parse_args(argv)
    labels = a.labels or default_labels()
    data = rebuild(labels / NAME, load_transcripts(a.transcripts))
    want = (labels / (NAME + ".original.sha256")).read_text().strip()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        print(f"{NAME}: sha256 differs ({got[:12]} vs {want[:12]})")
        return 1
    target = labels / NAME if a.in_place else labels / NAME.replace(".jsonl", ".full.jsonl")
    target.write_bytes(data)
    print(f"{NAME}: sha256 matches ({got[:12]}); wrote {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
