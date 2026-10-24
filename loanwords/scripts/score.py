"""Score every loanword item and comparison word of one decode.

Items: R = 1 when the aligned span matches any member of the item's acceptance set after normalization; R_strict = 1 when it matches
the reference spelling; F is the written form of the matched member (hangul, latin or mixed). Items of a flagged utterance keep
R = 0 under intention-to-treat (`itt_R`). Comparison words are scored strictly against their own surface form.
"""

from __future__ import annotations

from loanword_eval import flags
from loanword_eval.cli import common_parser, labels_dir, resolve_dirs
from loanword_eval.conditions import ALL, cond_slug
from loanword_eval.paths import read_jsonl, write_jsonl
from loanword_eval.score import POLICY, score_item, score_utterance


def score_items(tokens, items, hyps, accept, rid, cond, window):
    by_utt = {}
    for t in tokens:
        if t["utt_id"] in hyps and t["item_id"] in accept:
            by_utt.setdefault(t["utt_id"], []).append(t)
    rows = []
    for utt, ts in by_utt.items():
        results = score_utterance(
            items[utt]["text"],
            hyps[utt]["text"],
            [(t["start"], t["end"], accept[t["item_id"]]) for t in ts],
            window=window,
        )
        for t, s in zip(ts, results):
            if s.outcome == "NO_REF":
                continue
            fl = dict(hyps[utt]["flags"])
            flagged = flags.any_flag(fl)
            rows.append(
                {
                    "item_id": t["item_id"],
                    "utt_id": utt,
                    "type_key": t["type_key"],
                    "corpus": t["corpus"],
                    "cs_level": t.get("cs_level"),
                    "ref_form": t["ref_form"],
                    "condition": cond,
                    "recognizer_id": rid,
                    **s.row(),
                    "utt_flagged": flagged,
                    "itt_R": 0 if flagged else s.R,
                    "itt_F": None if flagged else s.F,
                    **{f"flag_{k}": v for k, v in fl.items()},
                }
            )
    return rows


def score_controls(controls, items, hyps, rid, cond, window):
    rows = []
    for c in controls:
        if c["utt_id"] not in hyps:
            continue
        h = hyps[c["utt_id"]]
        s = score_item(
            items[c["utt_id"]]["text"],
            h["text"],
            c["start"],
            c["end"],
            {POLICY.apply(c["surface"]).text},
            window=window,
        )
        if s.outcome == "NO_REF":
            continue
        flagged = flags.any_flag(dict(h["flags"]))
        rows.append(
            {
                "utt_id": c["utt_id"],
                "start": c["start"],
                "end": c["end"],
                "condition": cond,
                "recognizer_id": rid,
                "R_strict": s.R_strict,
                "err_strict": 1 - s.R_strict,
                "utt_flagged": flagged,
                "itt_err_strict": 1 if flagged else 1 - s.R_strict,
                "R_tolerant": s.R,
                "itt_R_tolerant": 0 if flagged else s.R,
                "outcome": s.outcome,
            }
        )
    return rows


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--condition", nargs="+", default=["none"], choices=ALL)
    ap.add_argument(
        "--window", type=int, default=4, help="characters around the aligned span searched for an accepted form"
    )
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    labels = labels_dir(a)
    rid = a.recognizer
    items = {r["utt_id"]: r for r in read_jsonl(work / "items_hike.jsonl")}
    tokens = [t for t in read_jsonl(work / "tokens_hike.jsonl") if t.get("korean_matrix")]
    acc_path = (
        labels / "accept_sets.v1.jsonl" if (labels / "accept_sets.v1.jsonl").exists() else work / "accept_sets.jsonl"
    )
    accept = {r["item_id"]: set(r["members"]) for r in read_jsonl(acc_path)}
    ctrl_path = labels / "controls.v1.jsonl" if (labels / "controls.v1.jsonl").exists() else work / "controls.jsonl"
    controls = list(read_jsonl(ctrl_path)) if ctrl_path.exists() else []
    for cond in a.condition:
        slug = cond_slug(cond)
        hyps = {r["utt_id"]: r for r in read_jsonl(work / "decodes" / f"{slug}_{rid}.jsonl")}
        rows = score_items(tokens, items, hyps, accept, rid, cond, a.window)
        crow = score_controls(controls, items, hyps, rid, cond, a.window)
        write_jsonl(work / f"spans_{slug}_{rid}.jsonl", rows)
        write_jsonl(work / f"spans_controls_{slug}_{rid}.jsonl", crow)
        n = max(1, len(rows))
        print(
            f"{rid} {cond}: {len(rows)} items, recall {sum(r['R'] for r in rows) / n:.3f}, strict recall {sum(r['R_strict'] for r in rows) / n:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
