"""Command-line helpers shared by the pipeline scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from loanword_eval.paths import OUTPUTS, ROOT, WORK

FIXTURE_TEXT = ROOT / "tests" / "fixtures" / "synthetic"


def common_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--work", type=Path, default=None, help="intermediate files (default: work/)")
    ap.add_argument("--out", type=Path, default=None, help="analysis outputs (default: outputs/)")
    ap.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="run on a synthetic fixture built with `python -m loanword_eval.fixture build DIR`",
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most N units")
    return ap


def resolve_dirs(args) -> tuple[Path, Path]:
    if args.fixture is not None:
        return Path(args.work or args.fixture), Path(args.out or args.fixture / "outputs")
    return Path(args.work or WORK), Path(args.out or OUTPUTS)


def labels_dir(args) -> Path:
    from loanword_eval.paths import LABELS

    return args.fixture / "labels" if args.fixture is not None else LABELS


def check(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(f"error: {message}")


def take(it, limit):
    out = []
    for x in it:
        if limit is not None and len(out) >= limit:
            break
        out.append(x)
    return out
