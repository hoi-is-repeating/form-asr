"""Transcribe the Korean-matrix utterances with one recognizer under one or more conditions.

Each condition is decoded on the same audio; prompted conditions place the prompt in the recognizer's text slot. Every output row
carries the utterance flags (hallucination, translation, empty output, off-task text, prompt copy) used by intention-to-treat scoring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from loanword_eval import flags
from loanword_eval.cli import common_parser, labels_dir, resolve_dirs, take
from loanword_eval.conditions import ALL, cond_slug, load_precedents, resolve
from loanword_eval.decoders import load_decoder
from loanword_eval.paths import ROOT, load_yaml, read_jsonl, write_jsonl


def load_audio(item: dict) -> np.ndarray:
    p = Path(item["audio_path"])
    return np.load(p if p.is_absolute() else ROOT / p).astype(np.float32)


def decode(dec, items, cond, prompt, language, raw_prompt, patterns, batch_size, frame_text):
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        audio = [load_audio(it) for it in chunk]
        results = dec.transcribe_batch(audio, language=language, prompt=prompt)
        for it, x, r in zip(chunk, audio, results):
            rows.append(
                {
                    "utt_id": it["utt_id"],
                    "audio_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                    "condition": cond,
                    "text": r.text,
                    "raw": r.raw,
                    "language_forced": r.language_forced,
                    "meta": r.meta,
                    "flags": flags.utterance_flags(it["text"], r.text, raw_prompt, patterns, frame_text),
                }
            )
    return rows


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--condition", nargs="+", default=["none"], choices=ALL)
    ap.add_argument("--batch-size", type=int, default=None, help="override the config's batch size")
    ap.add_argument("--fake", action="store_true", help="use the test decoder (synthetic fixture only)")
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    rid = a.recognizer
    items = [it for it in read_jsonl(work / "items_hike.jsonl") if it["extra"].get("korean_matrix")]
    items = take(items, a.limit)
    tokens = list(read_jsonl(work / "tokens_hike.jsonl"))
    prompts, _ = load_precedents(labels_dir(a))
    patterns, _ = flags.load_off_task_patterns()
    if a.fake:
        from loanword_eval.decoders.fake import FlipDecoder

        dec, cfg = FlipDecoder(items, tokens, prompt_slot="prefix"), {"batch_size": 4}
    else:
        cfg = load_yaml(f"decode_{rid}.yaml")
        dec = load_decoder(rid, cfg)
    batch = a.batch_size or int(cfg.get("batch_size", 1))
    frame_text = (
        prompts.get("slot_frames", {}).get(rid, {}).get("frame", "{prompt}").replace("{prompt}", "").strip() or None
    )
    for cond in a.condition:
        prompt, language, raw = resolve(cond, prompts, rid)
        rows = decode(dec, items, cond, prompt, language, raw, patterns, batch, frame_text)
        n = write_jsonl(work / "decodes" / f"{cond_slug(cond)}_{rid}.jsonl", rows)
        print(f"{rid} {cond}: {n} utterances, {sum(flags.any_flag(r['flags']) for r in rows)} flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
