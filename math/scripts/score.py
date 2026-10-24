"""Score every item of one decode, and the comparison words of the D1 window.

Benchmark clips: R = 1 when the canonical key of the transcript equals the item's; R_strict = 1 when the normalized transcript equals
the reference spoken form; F is the written form (words, notation, mixed). A transcript that is LaTeX and equals the reference LaTeX
after the Speech-to-LaTeX normalizer counts as recognized with F = latex (needs Node.js). In-discourse clips are scored in context: the
item is recognized when its canonical tokens occur contiguously in the transcript. Flags are recomputed against the caption text the
audio can contain; items of a flagged utterance keep R = 0 under intention-to-treat (`itt_R`). Comparison words are scored strictly.
"""

from __future__ import annotations

import shutil

from math_eval import flags, latex_norm
from math_eval.cli import common_parser, labels_dir, resolve_dirs
from math_eval.conditions import ALL, DISCOURSE, PROMPTED, corpus_of, load_precedents, resolve
from math_eval.corpora import indiscourse
from math_eval.direction import fired_by_category, item_category
from math_eval.paths import read_jsonl, write_jsonl
from math_eval.score import (
    control_strict,
    item_surface_in_context,
    item_surface_window,
    looks_like_latex,
    score_clip,
    score_in_context,
)
from math_eval.taxonomy import expression_category, latex_tokens


def latex_outcome(rows, candidates) -> int:
    """Rows whose transcript is LaTeX equal to the reference LaTeX become R = 1, F = latex."""
    if not candidates:
        return 0
    if shutil.which("node") is None:
        print(f"{len(candidates)} LaTeX candidates not checked (Node.js missing)")
        return 0
    normed = latex_norm.normalize_many([t for _, t, _ in candidates] + [g for _, _, g in candidates])
    k, n = len(candidates), 0
    for j, (ri, _, _) in enumerate(candidates):
        if latex_norm.exact_match(normed[j], normed[k + j]):
            rows[ri].update({"R": 1, "F": "latex", "outcome": "LATEX"})
            n += 1
    return n


def score_items(cond, rid, items, hyps, accept, ctx, prompt, d0_hyps, patterns, solved_patterns, latex_check):
    corpus = corpus_of(cond)
    in_context, in_window = cond in DISCOURSE, cond == "D1"
    rows, candidates = [], []
    for utt, it in items.items():
        if utt not in hyps or utt not in accept:
            continue
        h, acc = hyps[utt], accept[utt]
        s = score_in_context(it["text"], h["text"]) if in_context else score_clip(it["text"], h["text"])
        assert s.canonical_ref == acc["canonical"], utt
        if in_context:
            c = ctx.get(utt, {})
            fref = flags.discourse_flag_reference(
                it["text"],
                it.get("context_before") or c.get("context_before", ""),
                c.get("lead_out", ""),
                c.get("context_after", ""),
                cond,
            )
        else:
            fref = it["text"]
        fl = flags.utterance_flags(
            fref,
            h["text"],
            it.get("d2_prompt") if cond == "D2" else prompt,
            patterns,
            solved_patterns,
            unprompted_hyp=d0_hyps.get(utt) if cond == "D2" else None,
        )
        flagged = flags.any_flag(fl)
        row = {
            "flags": fl,
            "flags_decode": h["flags"],
            "item_id": utt,
            "utt_id": utt,
            "idx": it.get("idx"),
            "video_id": it.get("video_id"),
            "corpus": corpus,
            "condition": cond,
            "recognizer_id": rid,
            "in_discourse": it.get("in_discourse", corpus == "discourse"),
            "form_category": acc.get("form_category") or item_category(it["text"]),
            "expression_category": acc.get("expression_category") or expression_category(it["latex"]),
            "latex_tokens": acc.get("latex_tokens") or latex_tokens(it["latex"]),
            "duration_s": it.get("duration_s") or (it.get("audio") or {}).get("d0", {}).get("duration_s"),
            **s.row(),
            "latex_candidate": int(looks_like_latex(h["text"])),
            "fired_by_category": {k: dict(v) for k, v in fired_by_category(h["text"]).items()},
            "utt_flagged": flagged,
            **{f"flag_{k}": v for k, v in fl.items()},
        }
        if in_window:
            row["hyp_span_text"] = item_surface_in_context(it["text"], h["text"])
        elif in_context:
            sc = score_clip(it["text"], h["text"])
            row["hyp_span_text"] = item_surface_window(it["text"], h["text"])
            row.update(
                {
                    "R_clip": sc.R,
                    "R_strict_clip": sc.R_strict,
                    "F_clip": sc.F,
                    "outcome_clip": sc.outcome,
                    "canonical_hyp_clip": sc.canonical_hyp,
                }
            )
        if row["latex_candidate"] and not s.R:
            candidates.append((len(rows), h["text"], it["latex"]))
        rows.append(row)
    if latex_check:
        latex_outcome(rows, candidates)
    for r in rows:
        r["itt_R"] = 0 if r["utt_flagged"] else r["R"]
        r["itt_F"] = None if r["utt_flagged"] else r["F"]
        r["itt_R_strict"] = 0 if r["utt_flagged"] else r["R_strict"]
    return rows


def score_controls(controls, items, hyps, flagged, cond, rid):
    rows = []
    for c in controls:
        if c["utt_id"] not in hyps:
            continue
        h = hyps[c["utt_id"]]
        fl = flagged.get(c["utt_id"], flags.any_flag(h["flags"]))
        rs = control_strict(c["word"], h["text"])
        rows.append(
            {
                "utt_id": c["utt_id"],
                "index": c["index"],
                "word": c["word"],
                "condition": cond,
                "recognizer_id": rid,
                "video_id": items[c["utt_id"]].get("video_id"),
                "R_strict": rs,
                "err_strict": 1 - rs,
                "utt_flagged": fl,
                "itt_err_strict": 1 if fl else 1 - rs,
            }
        )
    return rows


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--condition", nargs="+", default=["none"], choices=ALL)
    ap.add_argument("--no-latex-check", action="store_true", help="skip the LaTeX outcome (needs Node.js)")
    a = ap.parse_args(argv)
    work, _ = resolve_dirs(a)
    labels = labels_dir(a)
    rid = a.recognizer
    acc_path = work / "accept_sets.jsonl"
    accept = {r["item_id"]: r for r in read_jsonl(acc_path)}
    ctrl_path = (
        labels / "controls.v1.jsonl" if (labels / "controls.v1.jsonl").exists() else work / "controls_discourse.jsonl"
    )
    clip_path = labels / "discourse_clips.v1.jsonl"
    ctx = indiscourse.context(clip_path) if clip_path.exists() else {}
    prompts = load_precedents(labels)
    patterns, _ = flags.load_off_task_patterns()
    solved_patterns, _ = flags.load_solved_patterns()
    for cond in a.condition:
        items = {r["utt_id"]: r for r in read_jsonl(work / f"items_{corpus_of(cond)}.jsonl")}
        hyps = {r["utt_id"]: r for r in read_jsonl(work / "decodes" / f"{cond}_{rid}.jsonl")}
        d0_path = work / "decodes" / f"D0_{rid}.jsonl"
        d0_hyps = {r["utt_id"]: r["text"] for r in read_jsonl(d0_path)} if cond == "D2" and d0_path.exists() else {}
        prompt = resolve(cond, prompts, rid)[2] if cond in PROMPTED else None
        rows = score_items(
            cond, rid, items, hyps, accept, ctx, prompt, d0_hyps, patterns, solved_patterns, not a.no_latex_check
        )
        write_jsonl(work / f"spans_{cond}_{rid}.jsonl", rows)
        msg = ""
        if cond == "D1" and ctrl_path.exists():
            crow = score_controls(
                read_jsonl(ctrl_path), items, hyps, {r["utt_id"]: r["utt_flagged"] for r in rows}, cond, rid
            )
            write_jsonl(work / f"spans_controls_{cond}_{rid}.jsonl", crow)
            msg = f"; {len(crow)} comparison words, strict recall {sum(r['R_strict'] for r in crow) / max(1, len(crow)):.3f}"
        n = max(1, len(rows))
        print(
            f"{rid} {cond}: {len(rows)} items, recall {sum(r['R'] for r in rows) / n:.3f}, strict recall {sum(r['R_strict'] for r in rows) / n:.3f}{msg}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
