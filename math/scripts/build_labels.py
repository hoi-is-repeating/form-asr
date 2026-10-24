"""Build acceptance sets, comparison words and item covariates.

The acceptance set of an item is its canonical key: the reference spoken form after normalization and canonicalization, which every
written form of the same expression (words, notation or a mix) shares. Comparison words are the ordinary content words of the caption
text inside each item's D1 window (spaCy tags; mathematical words and the item's own words excluded; word times interpolated over the
caption line, words within 2 s of the window start dropped). Covariates: Zipf frequency (an expression takes the minimum over its
words), syllable and phone counts from CMUdict through g2p_en, and Whisper token counts.
"""

from __future__ import annotations

import json

from math_eval import controls as ctl
from math_eval.cli import common_parser, labels_dir, resolve_dirs, take
from math_eval.direction import item_category
from math_eval.mathforms import grammar
from math_eval.paths import load_yaml, read_jsonl, write_jsonl
from math_eval.taxonomy import expression_category, latex_tokens


def acceptance_sets(items) -> list[dict]:
    g = grammar()
    rows = []
    for it in items:
        c = g.canonical(it["text"])
        rows.append(
            {
                "item_id": it["utt_id"],
                "utt_id": it["utt_id"],
                "idx": it["idx"],
                "video_id": it["video_id"],
                "text": it["text"],
                "canonical": c.text,
                "canonical_with_grouping": c.text_with_grouping,
                "n_tokens": len(c.compared_tokens),
                "in_discourse": it.get("in_discourse", False),
                "form_category": item_category(it["text"]),
                "expression_category": expression_category(it["latex"]),
                "latex_tokens": latex_tokens(it["latex"]),
                "grammar_version": g.version,
            }
        )
    return rows


def same_rows(built: list[dict], path, drop=("text", "canonical", "canonical_with_grouping")) -> bool:
    """Rows equal the shipped file's, ignoring the fields derived from MathSpeech's transcripts, which it omits."""
    dump = lambda r: json.dumps({k: v for k, v in r.items() if k not in drop}, ensure_ascii=False)  # noqa: E731
    return [dump(r) for r in built] == [dump(r) for r in read_jsonl(path)]


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer for the length covariate (default: the Whisper checkpoint)"
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    labels = labels_dir(a)
    bench = take(list(read_jsonl(work / "items_benchmark.jsonl")), a.limit)
    acc = acceptance_sets(bench)
    write_jsonl(work / "accept_sets.jsonl", acc)
    shipped = labels / "accept_sets.v1.jsonl"
    note = (
        f" ({'identical to' if same_rows(acc, shipped) else 'differ from'} {shipped.name})" if shipped.exists() else ""
    )
    print(f"acceptance sets: {len(acc)} items{note}")

    disc_path = work / "items_discourse.jsonl"
    if not disc_path.exists():
        print("no items_discourse.jsonl: comparison words and covariates not built")
        return 0
    import spacy
    from g2p_en import G2p
    from transformers import WhisperTokenizer

    cfg = load_yaml("decode_whisper.yaml")
    tok_id = a.tokenizer or cfg["checkpoint"]
    tokenizer = WhisperTokenizer.from_pretrained(
        tok_id, revision=cfg.get("revision") if tok_id == cfg["checkpoint"] else None
    )
    nlp, g2p = spacy.load("en_core_web_sm"), G2p()
    controls = []
    for it in take(list(read_jsonl(disc_path)), a.limit):
        found, _ = ctl.extract_controls(
            it["utt_id"],
            it["context_before"],
            it["context_before_start"],
            it["math_start"],
            it["audio"]["d1"]["start"],
            nlp,
            it["canonical"],
        )
        for c in found:
            controls.append(
                {
                    "utt_id": c.utt_id,
                    "word": c.word,
                    "index": c.index,
                    "est_start": c.est_start,
                    "pos": c.pos,
                    "lemma": c.lemma,
                    **ctl.covariates(text=c.word, tokenizer=tokenizer, g2p=g2p, recognizer_id="whisper"),
                }
            )
    item_cov = [
        {
            "item_id": it["utt_id"],
            **ctl.covariates(
                text=it["text"], tokenizer=tokenizer, g2p=g2p, recognizer_id="whisper", min_over_words=True
            ),
        }
        for it in bench
    ]
    write_jsonl(work / "controls_discourse.jsonl", controls)
    write_jsonl(work / "item_covariates.jsonl", item_cov)
    shipped = labels / "controls.v1.jsonl"
    note = (
        f" ({'identical to' if same_rows(controls, shipped, drop=()) else 'differ from'} {shipped.name})"
        if shipped.exists()
        else ""
    )
    print(f"comparison words: {len(controls)}{note}; item covariates: {len(item_cov)}; wrote {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
