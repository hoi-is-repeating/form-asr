"""Count each item's and comparison word's tokens under one recognizer's tokenizer (a length covariate)."""

from __future__ import annotations

from loanword_eval.cli import common_parser, labels_dir, resolve_dirs
from loanword_eval.paths import CONFIGS, load_yaml, read_jsonl, write_jsonl


def tokenizer_for(rid: str, override: str | None, spm_path: str | None):
    if spm_path:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=spm_path)
        return lambda s: len(sp.encode(s, out_type=int))
    from transformers import AutoTokenizer

    if override:
        tok = AutoTokenizer.from_pretrained(override)
    else:
        cfg = load_yaml(f"decode_{rid}.yaml")
        trust = "trust_remote_code" in str(cfg.get("framework", ""))
        tok = AutoTokenizer.from_pretrained(cfg["checkpoint"], revision=cfg.get("revision"), trust_remote_code=trust)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", required=True)
    ap.add_argument(
        "--tokenizer", default=None, help="Hugging Face tokenizer id overriding the recognizer's checkpoint"
    )
    ap.add_argument(
        "--tokenizer-path", default=None, help="SentencePiece model for recognizers without a Hugging Face tokenizer"
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    rid = a.recognizer
    if not a.tokenizer and not a.tokenizer_path and not (CONFIGS / f"decode_{rid}.yaml").exists():
        raise SystemExit(f"no config for {rid}; pass --tokenizer or --tokenizer-path")
    count = tokenizer_for(rid, a.tokenizer, a.tokenizer_path)
    tokens = [t for t in read_jsonl(work / "tokens_hike.jsonl") if t.get("korean_matrix")]
    ctrl = labels_dir(a) / "controls.v1.jsonl"
    controls = list(read_jsonl(ctrl if ctrl.exists() else work / "controls.jsonl"))
    rows = [
        {"item_id": t["item_id"], f"n_bpe_{rid}": count(t["surface"]), f"n_bpe_{rid}_hangul": count(t["hangul"])}
        for t in tokens
    ]
    rows += [
        {
            "utt_id": c["utt_id"],
            "start": c["start"],
            "end": c["end"],
            f"n_bpe_{rid}": count(c["surface"]),
            f"n_bpe_{rid}_hangul": count(c["surface"]),
        }
        for c in controls
    ]
    write_jsonl(work / f"bpe_{rid}.jsonl", rows)
    print(f"{rid}: token counts for {len(tokens)} items and {len(controls)} comparison words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
