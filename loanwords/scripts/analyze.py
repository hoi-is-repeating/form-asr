"""Estimate the base rate and the prompt-steering effects for one recognizer.

H1a: strict and form-tolerant error of the items without a prompt, and the share of strict error that is written form.
H1b: odds ratio of error for an item against a comparison word, GEE with frequency and length covariates, clustered by utterance.
H2a: change in the Latin share of items recognized under both prompts of a pair (Latin minus Hangul prompt).
H2b: change in recognition within the Latin- and Hangul-dominant cells, and their difference, per pair and pooled.
Intervals are cluster-bootstrap over utterances; p values are sign-flip permutation tests by utterance.
"""

from __future__ import annotations

import json

from loanword_eval import models
from loanword_eval.cli import common_parser, labels_dir, resolve_dirs
from loanword_eval.conditions import ALL, PAIRS, cond_slug
from loanword_eval.paths import read_jsonl


def load(path):
    return list(read_jsonl(path)) if path.exists() else None


def tag_pair(rows, pair):
    return [{**r, "item_id": f"{r['item_id']}#{pair}"} for r in rows]


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--B", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--n-perm", type=int, default=10000, help="permutations")
    a = ap.parse_args(argv)
    work, out = resolve_dirs(a)
    labels = labels_dir(a)
    rid, B, n_perm, seed = a.recognizer, a.B, a.n_perm, models.BOOTSTRAP_SEED

    spans = {c: load(work / f"spans_{cond_slug(c)}_{rid}.jsonl") for c in ALL}
    spans = {c: v for c, v in spans.items() if v is not None}
    ctrls = {c: load(work / f"spans_controls_{cond_slug(c)}_{rid}.jsonl") for c in spans}
    if "none" not in spans:
        raise SystemExit(f"missing {work / f'spans_none_{rid}.jsonl'}; run decode.py and score.py first")
    item_cov = {r["item_id"]: r for r in load(work / "item_covariates_hike.jsonl") or []}
    ctrl_file = labels / "controls.v1.jsonl" if (labels / "controls.v1.jsonl").exists() else work / "controls.jsonl"
    control_cov = {(r["utt_id"], r["start"], r["end"]): r for r in load(ctrl_file) or []}
    for r in load(work / f"bpe_{rid}.jsonl") or []:
        target = (
            item_cov.setdefault(r["item_id"], {})
            if "item_id" in r
            else control_cov.setdefault((r["utt_id"], r["start"], r["end"]), {})
        )
        target.update({k: v for k, v in r.items() if k.startswith("n_bpe_")})
    direction = load(labels / f"direction_{rid}.v1.jsonl") or load(work / f"direction_{rid}.jsonl")

    res = {"recognizer_id": rid, "conditions": sorted(spans), "B": B, "n_perm": n_perm}
    none = spans["none"]
    res["H1a"] = models.h1a(none, B=B, seed=seed)
    res["H1b"] = models.h1b(none, ctrls.get("none") or [], item_cov, control_cov, rid, B=B, seed=seed)

    pairs = [(h, l) for h, l in PAIRS if h in spans and l in spans]
    if pairs and direction is None:
        raise SystemExit(f"no dominant-form file for {rid}; run dominant_forms.py")
    res["H2a"], res["H2b"] = {}, {}
    for h, l in pairs:
        key = f"{h}/{l}"
        res["H2a"][key] = models.h2a(spans[h], spans[l], B=B, n_perm=n_perm, seed=seed)
        res["H2b"][key] = models.h2b(
            spans[h], spans[l], direction, item_cov, rid, ctrls.get(h), ctrls.get(l), B=B, n_perm=n_perm, seed=seed
        )
    if pairs:
        Hs = sum((tag_pair(spans[h], f"{h}/{l}") for h, l in pairs), [])
        Ls = sum((tag_pair(spans[l], f"{h}/{l}") for h, l in pairs), [])
        cov = {**item_cov, **{f"{k}#{h}/{l}": v for h, l in pairs for k, v in item_cov.items()}}
        res["H2a"]["pooled"] = models.h2a(Hs, Ls, B=B, n_perm=n_perm, seed=seed)
        pooled = models.h2b(Hs, Ls, direction, cov, rid, None, None, B=B, n_perm=n_perm, seed=seed)
        guards = [res["H2b"][f"{h}/{l}"]["guard"] for h, l in pairs]
        pooled["guard"] = {
            "holds": all(g.get("holds") is True for g in guards)
            if all(g.get("holds") is not None for g in guards)
            else None
        }
        res["H2b"]["pooled"] = pooled

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rid}.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
