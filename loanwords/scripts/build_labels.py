"""Build acceptance sets, comparison words and item covariates.

The acceptance set of an item holds its attested written forms (HiKE's Hangul and Latin pair and their normalized variants).
Comparison words are the ordinary Hangul content words of the same utterances. Covariates: Zipf frequency, syllable, phone and
token counts. The shipped labels/accept_sets.v1.jsonl and labels/controls.v1.jsonl are the files the reported results used;
kiwipiepy's segmentation differs across CPU architectures, so rebuilt comparison words can differ slightly from the shipped ones.
"""

from __future__ import annotations

from loanword_eval import accept
from loanword_eval.cli import common_parser, labels_dir, resolve_dirs, take
from loanword_eval.controls import covariates, extract_controls
from loanword_eval.g2p import load_g2p
from loanword_eval.paths import load_yaml, read_jsonl, sha256_of, write_jsonl


def acceptance_sets(tokens, g2p):
    rows = accept.rows_for_hike(tokens, g2p=g2p)
    accept.guard_g2p(rows)
    return [
        {
            **r,
            "members": r["members_nog2p"],
            "member_sources": {m: s for m, s in r["member_sources"].items() if m in r["members_nog2p"]},
        }
        for r in rows
    ]


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer for the length covariate (default: the Whisper checkpoint)"
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    from kiwipiepy import Kiwi
    from transformers import WhisperTokenizer

    cfg = load_yaml("decode_whisper.yaml")
    tok_id = a.tokenizer or cfg["checkpoint"]
    tokenizer = WhisperTokenizer.from_pretrained(
        tok_id, revision=cfg.get("revision") if tok_id == cfg["checkpoint"] else None
    )
    g2p = load_g2p()
    items = take(list(read_jsonl(work / "items_hike.jsonl")), a.limit)
    tokens = [t for t in read_jsonl(work / "tokens_hike.jsonl")]

    if g2p is not None and hasattr(g2p, "batch"):
        g2p.batch(sorted({t["latin"] for t in tokens if t.get("korean_matrix")}))
    acc = acceptance_sets(tokens, g2p)
    write_jsonl(work / "accept_sets.jsonl", acc)
    frozen = labels_dir(a) / "accept_sets.v1.jsonl"
    if frozen.exists():
        same = sha256_of(work / "accept_sets.jsonl") == sha256_of(frozen)
        print(f"acceptance sets: {len(acc)} items ({'identical to' if same else 'differ from'} {frozen.name})")

    members = {m for r in read_jsonl(frozen if frozen.exists() else work / "accept_sets.jsonl") for m in r["members"]}
    spans = {}
    for t in tokens:
        spans.setdefault(t["utt_id"], []).append((t["start"], t["end"]))
    if g2p is not None and hasattr(g2p, "batch"):
        g2p.batch(sorted({t["hangul"] for t in tokens if t.get("korean_matrix")}))
    kiwi = Kiwi()
    controls = []
    for it in items:
        if not it["extra"].get("korean_matrix"):
            continue
        found, _ = extract_controls(it["utt_id"], it["text"], spans.get(it["utt_id"], []), kiwi, members)
        if g2p is not None and hasattr(g2p, "batch") and found:
            g2p.batch(sorted({c.surface for c in found}))
        for c in found:
            cov = covariates(
                ref_surface=c.surface, hangul_form=c.surface, tokenizer=tokenizer, g2p=g2p, recognizer_id=cfg["id"]
            )
            controls.append(
                {
                    "utt_id": c.utt_id,
                    "start": c.start,
                    "end": c.end,
                    "surface": c.surface,
                    "tag": c.tag,
                    "lemma": c.lemma,
                    **cov,
                }
            )
    item_cov = [
        {
            "item_id": t["item_id"],
            **covariates(
                ref_surface=t["surface"],
                hangul_form=t["hangul"],
                tokenizer=tokenizer,
                g2p=g2p,
                freq_form=t["hangul"] if t["ref_form"] != "hangul" else None,
                recognizer_id=cfg["id"],
            ),
        }
        for t in tokens
        if t.get("korean_matrix")
    ]
    write_jsonl(work / "controls.jsonl", controls)
    write_jsonl(work / "item_covariates_hike.jsonl", item_cov)
    print(f"comparison words: {len(controls)}; item covariates: {len(item_cov)}; wrote {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
