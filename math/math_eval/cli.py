"""Command-line helpers shared by the pipeline scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from loanword_eval.cli import take  # noqa: F401

from math_eval.paths import LABELS, OUTPUTS, WORK


def common_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--work", type=Path, default=None, help="intermediate files (default: work/)")
    ap.add_argument("--out", type=Path, default=None, help="analysis outputs (default: outputs/)")
    ap.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="run on a synthetic fixture built with `python -m math_eval.fixture build DIR`",
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most N units")
    return ap


def resolve_dirs(args) -> tuple[Path, Path]:
    if args.fixture is not None:
        return Path(args.work or args.fixture), Path(args.out or args.fixture / "outputs")
    return Path(args.work or WORK), Path(args.out or OUTPUTS)


def labels_dir(args) -> Path:
    return args.fixture / "labels" if args.fixture is not None else LABELS
