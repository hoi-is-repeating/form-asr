"""Build the synthetic fixture's audio and in-discourse labels (tests/fixtures/ch4_synth holds the text side).

Usage: python -m math_eval.fixture build <out_dir> [--seed 20261007]
Writes <out_dir>/audio/<utt_id>.npy (16 kHz mono float32 sine waves, distinct frequency and seeded phase per clip),
<out_dir>/lectures/<video_id>.wav (one 60-s wave per video, from which prepare_data.py cuts the in-discourse spans) and
<out_dir>/labels/discourse_clips.v1.jsonl (the in-discourse clips, in the format of labels/discourse_clips.v1.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from math_eval.paths import FIXTURE_TEXT, write_jsonl

SR = 16000
LECTURE_S = 60.0


def rows(name: str = "clips.jsonl") -> list[dict]:
    with open(FIXTURE_TEXT / name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sine(duration_s: float, freq: float, phase: float) -> np.ndarray:
    t = np.arange(int(duration_s * SR), dtype=np.float32) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def build(out_dir: str | Path, seed: int = 20261007) -> Path:
    import soundfile as sf

    out = Path(out_dir)
    for d in ("audio", "lectures", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, r in enumerate(rows()):
        np.save(
            out / "audio" / f"{r['utt_id']}.npy",
            sine(r["duration_s"], 220.0 + 55.0 * i, float(rng.uniform(0, 2 * np.pi))),
        )
    for v in sorted({r["video_id"] for r in rows()}):
        wav = sine(LECTURE_S, 110.0 + 30.0 * (ord(v[-1]) - ord("0")), float(rng.uniform(0, 2 * np.pi)))
        sf.write(out / "lectures" / f"{v}.wav", wav, SR, subtype="FLOAT")
    idx = {r["utt_id"]: r["idx"] for r in rows()}
    clips = [
        {
            "sample_id": d["utt_id"],
            "original_index": idx[d["utt_id"]],
            "video_id": d["video_id"],
            "match_score": d["match_score"],
            "math_start": d["math_start"],
            "math_end": d["math_end"],
            "realigned": False,
            "context_before_start": d["context_before_start"],
            "context_before": d["context_before"],
            "lead_in": "",
            "lead_out": "",
            "context_after": d["context_after"],
        }
        for d in rows("discourse.jsonl")
    ]
    write_jsonl(out / "labels" / "discourse_clips.v1.jsonl", clips)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args(argv)
    out = build(a.out_dir, a.seed)
    print(f"fixture built at {out}: {len(rows())} clips, {len(rows('discourse.jsonl'))} in-discourse")


if __name__ == "__main__":
    main()
