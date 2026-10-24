"""Convert transcripts to LaTeX with the fixed converter and score them against the reference LaTeX.

The converter (configs/converter.yaml) runs on the item's span of each D0 or D2 transcript (the shortest word window carrying the
item, so boundary words of a loosely cut span do not reach it), or with --reference on the reference spoken forms. Predictions and
references go through the Speech-to-LaTeX normalizer (needs Node.js); C = 1 on an exact match, with character error rate beside it.
"""

from __future__ import annotations

import shutil

from math_eval import converter as conv
from math_eval import latex_norm
from math_eval.cli import common_parser, resolve_dirs, take
from math_eval.paths import load_yaml, read_jsonl, write_jsonl


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--condition", default="D0", choices=["D0", "D2"])
    ap.add_argument("--reference", action="store_true", help="convert the reference spoken forms instead of a decode")
    ap.add_argument("--converter", default=None, help="name in configs/converter.yaml (default: the file's default)")
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    if shutil.which("node") is None:
        raise SystemExit("the LaTeX normalizer (process_formula) needs Node.js")
    cfg = load_yaml("converter.yaml")
    cname = a.converter or cfg["default"]
    items = {r["utt_id"]: r for r in read_jsonl(work / "items_discourse.jsonl")}
    rid = a.recognizer
    if a.reference:
        label, spans = "reference_discourse", {}
        texts = {u: it["text"] for u, it in items.items()}
    else:
        label = f"{a.condition}_{rid}"
        spans = {r["item_id"]: r for r in read_jsonl(work / f"spans_{a.condition}_{rid}.jsonl")}
        hyps = {r["utt_id"]: r["text"] for r in read_jsonl(work / "decodes" / f"{a.condition}_{rid}.jsonl")}
        texts = {
            u: spans[u]["hyp_span_text"] if spans.get(u, {}).get("hyp_span_text") is not None else t
            for u, t in hyps.items()
        }
    utts = take([u for u in texts if u in items], a.limit)
    inputs = [texts[u] for u in utts]
    cv = conv.load_converter(cname, cfg)
    preds = cv.convert(inputs)
    raws = getattr(cv, "last_raw", None) or [None] * len(preds)
    pn, rn = latex_norm.normalize_many(preds), latex_norm.normalize_many([items[u]["latex"] for u in utts])
    rows = []
    for u, t, p, p_n, r_n, raw in zip(utts, inputs, preds, pn, rn, raws):
        s = spans.get(u, {})
        rows.append(
            {
                "item_id": u,
                "utt_id": u,
                "video_id": items[u].get("video_id"),
                "corpus": "discourse",
                "condition": "reference" if a.reference else a.condition,
                "recognizer_id": None if a.reference else rid,
                "converter": cname,
                "input_text": t,
                "latex_pred": p,
                "latex_pred_norm": p_n,
                "latex_ref_norm": r_n,
                "C": latex_norm.exact_match(p_n, r_n),
                "cer": latex_norm.cer(p_n, r_n),
                "empty_pred": int(not p_n),
                "R": s.get("itt_R"),
                "R_raw": s.get("R"),
                "F": s.get("itt_F"),
                "utt_flagged": s.get("utt_flagged"),
                "expression_category": s.get("expression_category"),
                "latex_tokens": s.get("latex_tokens"),
                **({"latex_raw": raw, "bare_latex": int(conv.is_bare_latex(raw))} if raw is not None else {}),
            }
        )
    path = work / f"conv_{cname}_{label}.jsonl"
    write_jsonl(path, rows)
    n = max(1, len(rows))
    print(
        f"{cname} {label}: {len(rows)} items, exact match {sum(r['C'] for r in rows) / n:.3f}, CER {sum(r['cer'] for r in rows) / n:.3f}; wrote {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
