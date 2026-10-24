"""Load HiKE, locate the loanword items, and cache 16 kHz audio as .npy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from loanword_eval.cli import FIXTURE_TEXT, common_parser, resolve_dirs, take
from loanword_eval.corpora import audio as audiolib
from loanword_eval.corpora import hike
from loanword_eval.items import build_items
from loanword_eval.paths import DATA, relpath, write_jsonl


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument(
        "--audio-dir", type=Path, default=None, help="where the .npy files are written (default: data/audio/hike)"
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    if a.fixture is not None:
        rows = [
            json.loads(l)
            for l in (FIXTURE_TEXT / "hike_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        source = hike.load_rows(take(rows, a.limit))
    else:
        source = take(hike.load(split="test", with_audio=False), a.limit)
    items, tokens, counts = build_items(source)

    audio_dir = a.audio_dir or (a.fixture / "audio" if a.fixture is not None else DATA / "audio" / "hike")
    audio_dir.mkdir(parents=True, exist_ok=True)
    if a.fixture is None:
        wanted = {it.utt_id for it in items}
        for h in hike.load(split="test", with_audio=True):
            if h.utt_id in wanted:
                np.save(audio_dir / f"{h.utt_id}.npy", np.asarray(h.audio["array"], dtype=np.float32))
    for it in items:
        x = np.load(audio_dir / f"{it.utt_id}.npy").astype(np.float32)
        it.audio_path = relpath(audio_dir / f"{it.utt_id}.npy")
        it.audio_sha256 = hashlib.sha256(x.tobytes()).hexdigest()
        it.duration_s = round(len(x) / 16000, 4)
        it.extra["audio_problems"] = audiolib.check(x)

    write_jsonl(work / "items_hike.jsonl", items)
    write_jsonl(work / "tokens_hike.jsonl", tokens)
    print(
        f"{counts['utterances_loaded']} utterances, {counts['korean_matrix']} Korean-matrix, "
        f"{counts['tokens_korean_matrix']} loanword items; wrote {work}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
