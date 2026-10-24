"""Transcribe the benchmark or the in-discourse clips with one recognizer under one or more conditions.

Each condition decodes the same audio; prompted conditions place the prompt in the recognizer's text slot. Every output row carries
the utterance flags (hallucination, empty output, off-task text, prompt copy, a solved instead of transcribed expression) used by
intention-to-treat scoring. In-discourse conditions decode at batch 1, so D0 and D2 (one prompt per clip) share one batch regime.
"""

from __future__ import annotations

import hashlib

import numpy as np

from math_eval import flags
from math_eval.cli import common_parser, labels_dir, resolve_dirs, take
from math_eval.conditions import ALL, audio_key, corpus_of, load_precedents, resolve
from math_eval.decoders import PROMPT_CAPABLE, load_decoder
from math_eval.paths import abspath, load_yaml, read_jsonl, write_jsonl


def item_audio(item: dict, cond: str) -> np.ndarray:
    path = item["audio"][audio_key(cond)]["path"] if item["corpus"] == "discourse" else item["audio_path"]
    return np.load(abspath(path)).astype(np.float32)


def decode(dec, items, cond, prompts, rid, patterns, solved_patterns, batch_size, frame_text):
    """Items with the same prompt are batched together (D2 has one prompt per clip)."""
    groups: dict[str | None, list] = {}
    for k, it in enumerate(items):
        prompt, language, raw = resolve(cond, prompts, rid, it)
        groups.setdefault(prompt, []).append((k, it, raw, language))
    rows = []
    for prompt, members in groups.items():
        for i in range(0, len(members), batch_size):
            chunk = members[i : i + batch_size]
            audio = [item_audio(it, cond) for _, it, _, _ in chunk]
            results = dec.transcribe_batch(audio, language=chunk[0][3], prompt=prompt)
            for (k, it, raw, _), x, r in zip(chunk, audio, results):
                rows.append(
                    {
                        "utt_id": it["utt_id"],
                        "idx": it.get("idx"),
                        "video_id": it.get("video_id"),
                        "audio_variant": audio_key(cond) if it["corpus"] == "discourse" else "clip",
                        "audio_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                        "condition": cond,
                        "order": k,
                        "prompt_chars": len(raw) if raw else 0,
                        "text": r.text,
                        "raw": r.raw,
                        "language_forced": r.language_forced,
                        "meta": r.meta,
                        "flags": flags.utterance_flags(it["text"], r.text, raw, patterns, solved_patterns, frame_text),
                    }
                )
    rows.sort(key=lambda r: r["order"])
    return rows


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--condition", nargs="+", default=["none"], choices=ALL)
    ap.add_argument("--batch-size", type=int, default=None, help="override the config's batch size")
    ap.add_argument(
        "--fake",
        choices=["words", "steer", "partial"],
        default=None,
        help="use the test decoder (synthetic fixture only)",
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    rid = "fake" if a.fake else a.recognizer
    corpora = {corpus_of(c) for c in a.condition}
    if len(corpora) > 1:
        raise SystemExit("decode benchmark conditions and in-discourse conditions in separate runs")
    corpus = corpora.pop()
    items_all = list(read_jsonl(work / f"items_{corpus}.jsonl"))
    items = take(items_all, a.limit)
    prompted = [c for c in a.condition if c not in ("none", "D0", "D1")]
    if prompted and not a.fake and rid not in PROMPT_CAPABLE:
        raise SystemExit(f"{rid} has no text-conditioning slot; prompted conditions are not defined for it")
    prompts = load_precedents(labels_dir(a))
    patterns, _ = flags.load_off_task_patterns()
    solved_patterns, _ = flags.load_solved_patterns()
    if a.fake:
        from math_eval import fixture
        from math_eval.decoders.fake import MathFakeDecoder

        surfaces = {r["utt_id"]: {"notation": r["notation"], "wrong": r["wrong"]} for r in fixture.rows()}
        dec, cfg = MathFakeDecoder(items_all, mode=a.fake, surfaces=surfaces), {"batch_size": 4}
    else:
        cfg = load_yaml(f"decode_{rid}.yaml")
        dec = load_decoder(rid, cfg)
    batch = a.batch_size or (1 if corpus == "discourse" and not a.fake else int(cfg.get("batch_size", 1)))
    frame_text = (
        prompts.get("slot_frames", {}).get(rid, {}).get("frame", "{prompt}").replace("{prompt}", "").strip() or None
    )
    for cond in a.condition:
        rows = decode(dec, items, cond, prompts, rid, patterns, solved_patterns, batch, frame_text)
        n = write_jsonl(work / "decodes" / f"{cond}_{rid}.jsonl", rows)
        print(f"{rid} {cond}: {n} clips, {sum(flags.any_flag(r['flags']) for r in rows)} flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
