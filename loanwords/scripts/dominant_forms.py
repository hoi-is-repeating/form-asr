"""Derive each item type's dominant written form under one recognizer from its no-prompt transcripts.

The dominant form is the script the recognizer writes more often for the type; ties and types with no recognized token fall back
to the Latin cell. labels/direction_<recognizer>.v1.jsonl holds the files the reported results used.
"""

from __future__ import annotations

from loanword_eval.cli import common_parser, resolve_dirs
from loanword_eval.direction import direction_rows, write_direction
from loanword_eval.paths import read_jsonl


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    spans = list(read_jsonl(work / f"spans_none_{a.recognizer}.jsonl"))
    rows, counts = direction_rows(spans, a.recognizer)
    write_direction(rows, work / f"direction_{a.recognizer}.jsonl")
    print(
        f"{a.recognizer}: {len(rows)} types; dominant Latin {counts.get('dominant_latin', 0)}, Hangul {counts.get('dominant_hangul', 0)}, "
        f"fallback {counts.get('fallback', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
