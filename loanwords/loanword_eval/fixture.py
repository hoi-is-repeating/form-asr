"""Build the synthetic fixture's audio and item table (tests/fixtures/synthetic holds the text side).

Usage: python -m loanword_eval.fixture build <out_dir> [--seed 20260917]
Writes <out_dir>/audio/<sample_id>.npy (16 kHz mono float32 sine waves, distinct frequency and seeded phase per row) and
<out_dir>/items.jsonl (loanword_eval.paths.Item rows, audio_path relative to ROOT when inside it, audio_sha256 of the float32 bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from loanword_eval.paths import ROOT, Item, relpath, write_jsonl

FIXTURE_TEXT = ROOT / "tests" / "fixtures" / "synthetic"
SR = 16000


def rows() -> list[dict]:
    with open(FIXTURE_TEXT / "hike_rows.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sine(duration_s: float, freq: float, phase: float) -> np.ndarray:
    t = np.arange(int(duration_s * SR), dtype=np.float32) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def build(out_dir: str | Path, seed: int = 20260917) -> Path:
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = []
    for i, r in enumerate(rows()):
        wav = sine(r["audioduration"], 220.0 + 55.0 * i, float(rng.uniform(0, 2 * np.pi)))
        p = out / "audio" / f"{r['sample_id']}.npy"
        np.save(p, wav)
        items.append(
            Item(
                utt_id=r["sample_id"],
                corpus="fixture",
                split="test",
                speaker=None,
                text=r["text"],
                pron=None,
                audio_path=relpath(p),
                audio_sha256=hashlib.sha256(wav.tobytes()).hexdigest(),
                duration_s=len(wav) / SR,
                extra={
                    "cs_level": r["cs_level"],
                    "category": r["category"],
                    "pairs": [[d["Korean"], d["English"]] for d in json.loads(r["loanwords"])],
                    "audio_problems": [],
                },
            )
        )
    write_jsonl(out / "items.jsonl", items)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args(argv)
    out = build(a.out_dir, a.seed)
    print(f"fixture built at {out}: {len(rows())} utterances")


if __name__ == "__main__":
    main()
