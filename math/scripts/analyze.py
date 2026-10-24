"""Estimate the base rate and the prompt-steering effects for one recognizer.

H1a: strict and form-tolerant error without a prompt on the benchmark, and on the in-discourse clips (expression span D0; D1 window
in context), with the share of strict error that is written form.
H1b: odds ratio of error for an item against a comparison word of the same D1 window, GEE with frequency and length covariates,
clustered by video; fitted where items and comparison words share syllable counts, and on the full set.
H2a: change in the notation share of items recognized under both prompts of a pair (notation minus words prompt).
H2b: change in recognition under the prompt matching the dominant form of the item's category, per cell and pooled.
Intervals are cluster-bootstrap over videos; p values are sign-flip permutation tests by video.
"""

from __future__ import annotations

import json

from loanword_eval.models import BOOTSTRAP_SEED

from math_eval import models
from math_eval.cli import common_parser, labels_dir, resolve_dirs
from math_eval.conditions import ALL, PAIRS
from math_eval.direction import load_direction
from math_eval.paths import read_jsonl


def load(path):
    return list(read_jsonl(path)) if path.exists() else None


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--B", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--n-perm", type=int, default=10000, help="permutations")
    a = ap.parse_args(argv)
    work, out = resolve_dirs(a)
    labels = labels_dir(a)
    rid, B, P, seed = a.recognizer, a.B, a.n_perm, BOOTSTRAP_SEED

    spans = {c: load(work / f"spans_{c}_{rid}.jsonl") for c in ALL}
    spans = {c: v for c, v in spans.items() if v is not None}
    if "none" not in spans:
        raise SystemExit(f"missing {work / f'spans_none_{rid}.jsonl'}; run decode.py and score.py first")
    item_cov = {r["item_id"]: r for r in load(work / "item_covariates.jsonl") or []}
    res = {"recognizer_id": rid, "conditions": sorted(spans), "B": B, "n_perm": P}
    res["H1a"] = {"benchmark_none": models.h1a(spans["none"], B=B, seed=seed)}
    if spans.get("D0"):
        res["H1a"]["discourse_D0"] = models.h1a(spans["D0"], B=B, seed=seed)
    if spans.get("D1"):
        res["H1a"]["discourse_D1_in_context"] = models.h1a(spans["D1"], B=B, seed=seed)

    ctrl_spans = load(work / f"spans_controls_D1_{rid}.jsonl")
    if spans.get("D1") and ctrl_spans and item_cov:
        ctrl_file = (
            labels / "controls.v1.jsonl"
            if (labels / "controls.v1.jsonl").exists()
            else work / "controls_discourse.jsonl"
        )
        ctrl_cov = {(c["utt_id"], c["index"], c["index"]): c for c in read_jsonl(ctrl_file)}
        ctrl_rows = [{**c, "start": c["index"], "end": c["index"]} for c in ctrl_spans]
        bpe_rid = rid if any(f"n_bpe_{rid}" in v for v in item_cov.values()) else "whisper"
        full = models.h1b(spans["D1"], ctrl_rows, item_cov, ctrl_cov, bpe_rid, B=B, seed=seed)
        res["H1b_full_set_descriptive"] = full
        rng = models.overlap_range(full["length_support"])
        res["H1b"] = (
            models.h1b(spans["D1"], ctrl_rows, item_cov, ctrl_cov, bpe_rid, B=B, seed=seed, n_syll_range=rng)
            if rng
            else {
                "p": None,
                "fit_note": "no overlapping length support between items and comparison words",
                "length_support": full["length_support"],
            }
        )

    pairs = {w: (spans[w], spans[s]) for w, s in PAIRS if w in spans and s in spans}
    if pairs:
        dpath = labels / f"direction_{rid}.v1.jsonl"
        dpath = dpath if dpath.exists() else work / f"direction_{rid}.jsonl"
        if not dpath.exists():
            raise SystemExit(f"no dominant-form file for {rid}; run dominant_forms.py")
        direction = load_direction(dpath)
        res["H2a"] = {w: models.h2a(rw, rs, B=B, n_perm=P, seed=seed) for w, (rw, rs) in pairs.items()}
        res["H2b"] = {
            w: models.h2b(rw, rs, direction, item_cov, rid, B=B, n_perm=P, seed=seed) for w, (rw, rs) in pairs.items()
        }
        pooled_W = [dict(r, item_id=f"{w}:{r['item_id']}") for w, (rw, _) in pairs.items() for r in rw]
        pooled_S = [dict(r, item_id=f"{w}:{r['item_id']}") for w, (_, rs) in pairs.items() for r in rs]
        res["H2a"]["pooled"] = models.h2a(pooled_W, pooled_S, B=B, n_perm=P, seed=seed + 7)
        res["H2b"]["pooled"] = models.h2b(
            pooled_W,
            pooled_S,
            direction,
            {f"{w}:{k}": v for w in pairs for k, v in item_cov.items()},
            rid,
            B=B,
            n_perm=P,
            seed=seed + 7,
        )

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rid}.json"
    path.write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
