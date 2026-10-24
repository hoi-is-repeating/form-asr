"""Load the MathSpeech benchmark and cache its 16 kHz audio; cut the in-discourse clips from the lecture recordings.

Benchmark: every clip of Hugging Face `AAAI2025/MathSpeech` (spoken-form transcription, LaTeX, video id).
In-discourse: the clips of labels/discourse_clips.v1.jsonl, cut from `--lectures-dir/<video_id>.wav` (16 kHz mono WAV files of the
MIT OpenCourseWare lectures, obtained by the user): the expression span (D0), a window of at most 30 s ending 0.5 s after it (D1), and
the preceding caption text truncated from the left to Whisper's prompt limit (the D2 prompt). Without `--lectures-dir` only the
benchmark is prepared.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from loanword_eval.corpora import audio as audiolib

from math_eval import discourse, fixture
from math_eval.cli import common_parser, labels_dir, resolve_dirs, take
from math_eval.corpora import indiscourse, mathspeech
from math_eval.mathforms import canonical
from math_eval.paths import DATA, load_yaml, read_jsonl, relpath, write_jsonl

SR = 16000


def load_lecture(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"{path}: expected 16 kHz audio, got {sr} Hz")
    return np.ascontiguousarray(x, dtype=np.float32)


def benchmark_items(rows, audio_dir: Path, in_discourse: set[int]) -> list[dict]:
    items = []
    for r in rows:
        p = audio_dir / f"{r['utt_id']}.npy"
        x = np.load(p).astype(np.float32)
        items.append(
            {
                "utt_id": r["utt_id"],
                "idx": r["idx"],
                "corpus": "benchmark",
                "text": r["transcription"],
                "latex": r["latex"],
                "video_id": r["video_id"],
                "source": r["source"],
                "canonical": canonical(r["transcription"]),
                "in_discourse": r["idx"] in in_discourse,
                "audio_path": relpath(p),
                "audio_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "duration_s": round(len(x) / SR, 4),
                "audio_problems": audiolib.check(x),
            }
        )
    return items


def discourse_items(clips, lectures_dir: Path, audio_dir: Path, tokenizer) -> list[dict]:
    seen, items, cache = {}, [], {}
    for c in clips:
        if c.video_id not in cache:
            path = lectures_dir / f"{c.video_id}.wav"
            if not path.exists():
                raise SystemExit(f"missing lecture audio {path}")
            cache.clear()
            cache[c.video_id] = load_lecture(path)
        wav = cache[c.video_id]
        audio = {}
        for key, (s, e) in discourse.spans(c).items():
            arr = discourse.cut(wav, s, e)
            p = audio_dir / f"{c.sample_id}_{key}.npy"
            np.save(p, arr)
            audio[key] = {
                "path": relpath(p),
                "sha256": discourse.sha256_array(arr),
                "duration_s": round(len(arr) / SR, 4),
                "start": s,
                "end": e,
                "problems": audiolib.check(arr, max_s=discourse.WINDOW + 1e-6),
            }
        span = (c.video_id, round(c.math_start, 2), round(c.math_end, 2))
        d2, n_tok = discourse.truncate_left_by_tokens(" ".join(c.context_before.split()), tokenizer)
        items.append(
            {
                "utt_id": c.sample_id,
                "idx": c.original_index,
                "corpus": "discourse",
                "text": c.transcription,
                "latex": c.latex,
                "video_id": c.video_id,
                "canonical": canonical(c.transcription),
                "match_score": c.match_score,
                "math_start": c.math_start,
                "math_end": c.math_end,
                "realigned": c.realigned,
                "lead_in": c.lead_in,
                "duplicate_of": seen.get(span),
                "context_before_start": c.context_before_start,
                "context_before": c.context_before,
                "audio": audio,
                "d2_prompt": d2,
                "d2_prompt_tokens": n_tok,
            }
        )
        seen.setdefault(span, c.sample_id)
    return items


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument(
        "--lectures-dir", type=Path, default=None, help="directory of <video_id>.wav lecture recordings (16 kHz mono)"
    )
    ap.add_argument(
        "--audio-dir", type=Path, default=None, help="where the .npy files are written (default: data/audio)"
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Whisper tokenizer for the D2 truncation (default: the decode_whisper.yaml checkpoint)",
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    if a.fixture is not None:
        rows = take(fixture.rows(), a.limit)
        bench_dir, disc_dir, lectures_dir = (
            a.fixture / "audio",
            a.fixture / "audio_discourse",
            a.lectures_dir or a.fixture / "lectures",
        )
    else:
        root = a.audio_dir or DATA / "audio"
        bench_dir, disc_dir, lectures_dir = root / "benchmark", root / "discourse", a.lectures_dir
        bench_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in mathspeech.load(with_audio=True):
            if a.limit is not None and len(rows) >= a.limit:
                break
            r["utt_id"] = f"ms_{r['idx']:04d}"
            np.save(bench_dir / f"{r['utt_id']}.npy", r.pop("audio"))
            rows.append(r)
    texts = {r["idx"]: (r["transcription"], r["latex"]) for r in rows}
    clip_file = labels_dir(a) / "discourse_clips.v1.jsonl"
    clips = indiscourse.load(texts, clip_file) if clip_file.exists() else []

    shipped = labels_dir(a) / "accept_sets.v1.jsonl"
    located = (
        {r["idx"] for r in read_jsonl(shipped) if r["in_discourse"]}
        if shipped.exists()
        else {c.original_index for c in clips}
    )
    items = benchmark_items(rows, bench_dir, located)
    write_jsonl(work / "items_benchmark.jsonl", items)
    print(f"benchmark: {len(items)} clips, {sum(it['duration_s'] for it in items) / 60:.1f} min")
    if lectures_dir is None:
        print("no --lectures-dir: in-discourse clips not prepared")
        return 0
    from transformers import WhisperTokenizer

    cfg = load_yaml("decode_whisper.yaml")
    tok = WhisperTokenizer.from_pretrained(
        a.tokenizer or cfg["checkpoint"], revision=None if a.tokenizer else cfg.get("revision")
    )
    disc_dir.mkdir(parents=True, exist_ok=True)
    disc = discourse_items(clips, lectures_dir, disc_dir, tok)
    write_jsonl(work / "items_discourse.jsonl", disc)
    print(f"in-discourse: {len(disc)} clips from {len({it['video_id'] for it in disc})} lectures; wrote {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
