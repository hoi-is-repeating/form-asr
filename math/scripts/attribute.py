"""Attribute final LaTeX errors to recognition or conversion for one recognizer (in-discourse clips).

Decomposition: P(C=0) = P(R=0) P(C=0 | R=0) + P(R=1) P(C=0 | R=1). H3a: the recognition-attributable excess
delta_rec = P(C=0 | transcript) - P(C=0 | reference spoken form) against the converter's own error P(C=0 | reference). H3b: P(R=0) with the
preceding caption text as the prompt (D2) against the expression alone (D0), paired on clips. Clips whose span duplicates another's
are dropped unless --keep-duplicates. Intervals are cluster-bootstrap over videos.
"""

from __future__ import annotations

import json
from pathlib import Path

from loanword_eval.models import BOOTSTRAP_SEED

from math_eval import attribution as attr
from math_eval.cli import common_parser, resolve_dirs
from math_eval.paths import load_yaml, read_jsonl


def join(conv_rows: list[dict], ref_rows: list[dict], spans: dict | None = None) -> list[dict]:
    """Conversion rows with R (intention-to-treat) and the reference row's C; R comes from the spans file when the conversion ran first."""
    ref = {r["item_id"]: r for r in ref_rows}
    out = []
    for r in conv_rows:
        if r.get("R") is None and spans and r["item_id"] in spans:
            sp = spans[r["item_id"]]
            r = {**r, "R": sp.get("itt_R"), "F": sp.get("itt_F"), "expression_category": sp.get("expression_category")}
        if r.get("R") is None or r["item_id"] not in ref:
            continue
        out.append(
            {
                "item_id": r["item_id"],
                "video_id": r["video_id"],
                "R": int(r["R"]),
                "C": int(r["C"]),
                "C_ref": int(ref[r["item_id"]]["C"]),
                "F": r.get("F"),
                "category": r.get("expression_category"),
                "cer": r["cer"],
                "cer_ref": ref[r["item_id"]]["cer"],
            }
        )
    return out


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    ap.add_argument("--recognizer", default="whisper")
    ap.add_argument("--converter", default=None, help="name in configs/converter.yaml (default: the file's default)")
    ap.add_argument("--B", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--n-perm", type=int, default=10000, help="permutations")
    ap.add_argument("--keep-duplicates", action="store_true", help="keep clips whose span duplicates another clip's")
    a = ap.parse_args(argv)
    work, out = resolve_dirs(a)
    cname = a.converter or load_yaml("converter.yaml")["default"]
    rid, B, P, seed = a.recognizer, a.B, a.n_perm, BOOTSTRAP_SEED
    dups = (
        set()
        if a.keep_duplicates
        else {r["utt_id"] for r in read_jsonl(work / "items_discourse.jsonl") if r.get("duplicate_of")}
    )

    def rows(label: str) -> list[dict] | None:
        p = work / f"conv_{cname}_{label}.jsonl"
        return [r for r in read_jsonl(p) if r["item_id"] not in dups] if p.exists() else None

    def spans(cond: str) -> dict | None:
        p = work / f"spans_{cond}_{rid}.jsonl"
        return {r["item_id"]: r for r in read_jsonl(p)} if p.exists() else None

    ref, d0 = rows("reference_discourse"), rows(f"D0_{rid}")
    if ref is None or d0 is None:
        raise SystemExit(f"need conv_{cname}_reference_discourse.jsonl and conv_{cname}_D0_{rid}.jsonl; run convert.py")
    j0 = join(d0, ref, spans("D0"))
    res = {
        "recognizer_id": rid,
        "converter": cname,
        "n_items": len(j0),
        "deduplicated": not a.keep_duplicates,
        "n_duplicates_dropped": len(dups),
        "H3a": attr.h3a(j0, B=B, seed=seed, n_perm=P),
    }
    d2 = rows(f"D2_{rid}")
    if d2 is not None:
        res["H3b"] = attr.h3b(j0, join(d2, ref, spans("D2")), B=B, n_perm=P, seed=seed + 1)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"attribution_{rid}{'_all_clips' if a.keep_duplicates else ''}.json"
    path.write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
