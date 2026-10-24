"""Derive the dominant written form of each form category under one recognizer from its no-prompt transcripts.

The dominant form is the side (words or notation) the recognizer writes more often for the category; labels/direction_<recognizer>.v1.jsonl
holds the files the reported results used.
"""

from __future__ import annotations

from math_eval.cli import common_parser, resolve_dirs
from math_eval.direction import direction_rows, write_direction
from math_eval.paths import read_jsonl


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    spans = list(read_jsonl(work / f"spans_none_{a.recognizer}.jsonl"))
    rows, _ = direction_rows(spans, a.recognizer)
    write_direction(rows, work / f"direction_{a.recognizer}.jsonl")
    print(
        f"{a.recognizer}: "
        + ", ".join(
            f"{r['category']} {r['dominant'] or '-'}{' (fallback)' if r['fallback'] else ''}"
            for r in rows
            if r["deciding"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
