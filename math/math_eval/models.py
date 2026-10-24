"""Estimators for the spoken-mathematics experiments. Span rows in, plain dicts of counts and estimates out.

The bootstrap, permutation, GEE and equivalence helpers are loanword_eval.models'; clusters are lecture videos.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from loanword_eval import stats
from loanword_eval.models import (
    ALPHA,
    TOST_PP,
    _f,
    _paired_difference,
    cluster_bootstrap,
    fit_logit_clustered,
    length_support,
    mde_paired,
    paired,
)
from loanword_eval.models import h1b as _h1b


def h1a(rows: list[dict], *, B: int, seed: int) -> dict:
    """Strict and form-tolerant error (intention-to-treat), the share of strict error that is written form, by category."""
    if not rows:
        return {"n": 0}
    R = np.array([r["itt_R"] for r in rows], float)
    S = np.array([r["itt_R_strict"] for r in rows], float)
    cl = [r["video_id"] for r in rows]
    es = cluster_bootstrap(lambda i: 1 - S[i].mean(), cl, B=B, seed=seed)
    ea = cluster_bootstrap(lambda i: 1 - R[i].mean(), cl, B=B, seed=seed)
    sh = cluster_bootstrap(
        lambda i: ((1 - S[i].mean()) - (1 - R[i].mean())) / (1 - S[i].mean()) if S[i].mean() < 1 else None,
        cl,
        B=B,
        seed=seed,
    )
    by_form = Counter(str(r["itt_F"]) for r in rows if r["itt_R"] and not r["itt_R_strict"])
    forms = Counter(str(r["itt_F"]) for r in rows if r["itt_R"])
    cats = {}
    for c in sorted({r.get("expression_category") or "unknown" for r in rows}):
        sub = [r for r in rows if (r.get("expression_category") or "unknown") == c]
        cats[c] = {
            "n": len(sub),
            "E_S": _f(1 - np.mean([r["itt_R_strict"] for r in sub])),
            "E_A": _f(1 - np.mean([r["itt_R"] for r in sub])),
        }
    unfl = [r for r in rows if not r["utt_flagged"]]
    return {
        "n": len(rows),
        "n_videos": len(set(cl)),
        "E_S": es,
        "E_A": ea,
        "share": sh,
        "difference_by_form": dict(by_form),
        "F_among_recognized": dict(forms),
        "latex_rate": _f(sum(1 for r in rows if r["itt_F"] == "latex") / len(rows)),
        "n_flagged_items": int(sum(r["utt_flagged"] for r in rows)),
        "per_condition_exclusion": {
            "n": len(unfl),
            "E_S": _f(1 - np.mean([r["R_strict"] for r in unfl])) if unfl else None,
            "E_A": _f(1 - np.mean([r["R"] for r in unfl])) if unfl else None,
        },
        "token_cer_mean": _f(np.mean([r["token_cer"] for r in rows])),
        "per_category": cats,
        **(
            {"R_tail_sensitivity": _f(1 - np.mean([0 if r["utt_flagged"] else (r.get("R_tail") or 0) for r in rows]))}
            if any(r.get("R_tail") is not None for r in rows)
            else {}
        ),
    }


def h1b(item_rows, control_rows, item_cov, control_cov, bpe_rid, *, B: int, seed: int, **kw) -> dict:
    """Odds ratio of error for an item against a comparison word of the same D1 window (GEE with frequency and length covariates,
    clustered by video)."""
    ls = length_support({k: v for k, v in item_cov.items() if k in {r["item_id"] for r in item_rows}}, control_cov)
    r = _h1b(item_rows, control_rows, item_cov, control_cov, bpe_rid, B=B, seed=seed, cluster_key="video_id", **kw)
    r["cluster"] = "video"
    r["length_support"] = ls
    r["n_bpe_tokenizer"] = bpe_rid
    if r["waterfall"]["items"] < 2 or r["waterfall"]["controls"] < 2:
        r.update(
            {
                "OR_is_item": None,
                "p": None,
                "rejected": False,
                "fit_note": f"not estimable: {r['waterfall']['items']} items / {r['waterfall']['controls']} comparison words",
            }
        )
    return r


def overlap_range(ls: dict) -> tuple[int, int] | None:
    """Syllable counts at which both items and comparison words occur."""
    both = [int(k) for k, v in ls.items() if v["items"] > 0 and v["controls"] > 0]
    return (min(both), max(both)) if both else None


def h2a(rows_W: list[dict], rows_S: list[dict], *, B: int, n_perm: int, seed: int) -> dict:
    """Notation share among items recognized under both prompts of a pair (notation minus words prompt)."""
    W, S, wf = paired(rows_W, rows_S)
    both = [(w, s) for w, s in zip(W, S) if w["itt_R"] and s["itt_R"]]
    nw = [int(w["itt_F"] == "notation") for w, _ in both]
    ns = [int(s["itt_F"] == "notation") for _, s in both]
    cl = [w["video_id"] for w, _ in both]
    d = (
        _paired_difference(ns, nw, cl, B=B, n_perm=n_perm, seed=seed, cluster_label="video")
        if both
        else {"estimate": None, "n": 0}
    )
    share = lambda rows: (
        _f(np.mean([r["itt_F"] == "notation" for r in rows if r["itt_R"]])) if any(r["itt_R"] for r in rows) else None
    )  # noqa: E731
    rejected = (
        d.get("estimate") is not None
        and d.get("p_permutation") is not None
        and d["p_permutation"] < ALPHA
        and d["estimate"] > 0
    )
    return {
        "pairing": wf,
        "n_recognized_both": len(both),
        "P_notation_given_R": {"W": share(W), "S": share(S)},
        "difference_S_minus_W": d,
        "rejected": bool(rejected),
        "mixed_share": {
            "W": _f(np.mean([r["itt_F"] == "mixed" for r in W if r["itt_R"]])) if any(r["itt_R"] for r in W) else None,
            "S": _f(np.mean([r["itt_F"] == "mixed" for r in S if r["itt_R"]])) if any(r["itt_R"] for r in S) else None,
        },
    }


def h2b(
    rows_W: list[dict],
    rows_S: list[dict],
    direction: dict[str, dict],
    item_cov: dict[str, dict],
    rid: str,
    *,
    B: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Recognition under the prompt matching the dominant form of the item's category minus the other prompt, per cell (notation- and
    words-dominant) and pooled; the difference of the cells; GEE recognized ~ prompt x dominant + covariates, clustered by video; TOST
    and the minimum detectable effect of the larger cell. Fallback categories are excluded."""
    W, S, wf = paired(rows_W, rows_S)
    d_s_minus_w, cell, cl, kept, excl = [], [], [], [], Counter()
    for w, s in zip(W, S):
        cat = w.get("form_category")
        drow = direction.get(cat) if cat else None
        if drow is None or not drow.get("deciding"):
            excl["no_direction_row"] += 1
            continue
        if drow.get("fallback"):
            excl["fallback_excluded"] += 1
            continue
        d_s_minus_w.append(int(s["itt_R"]) - int(w["itt_R"]))
        cell.append(drow["dominant"])
        cl.append(w["video_id"])
        kept.append((w, s))
    n = len(kept)
    out = {
        "pairing": wf,
        "n_analysed": n,
        "exclusions": dict(excl),
        "conservation_ok": n + sum(excl.values()) == wf["paired"],
    }
    if n == 0:
        return {
            **out,
            "cells": {"notation": 0, "words": 0},
            "cell_differences": {},
            "main_cell": None,
            "delta": {"estimate": None},
            "tost": None,
            "mde_pp": {},
        }
    d = np.array(d_s_minus_w, float)
    c = np.array(cell)
    cells = {k: int((c == k).sum()) for k in ("notation", "words")}
    cd = {}
    for k, sign in (("notation", 1), ("words", -1)):
        idx = [i for i in range(n) if c[i] == k]
        if idx:
            xs = [int(kept[i][1]["itt_R"]) for i in idx]
            xw = [int(kept[i][0]["itt_R"]) for i in idx]
            cli = [cl[i] for i in idx]
            cd[k] = (
                _paired_difference(xs, xw, cli, B=B, n_perm=n_perm, seed=seed, cluster_label="video")
                if sign == 1
                else _paired_difference(xw, xs, cli, B=B, n_perm=n_perm, seed=seed, cluster_label="video")
            )
            cd[k]["direction"] = "R_S − R_W" if sign == 1 else "R_W − R_S"
            cd[k]["discordance"] = _f(np.mean([a != b for a, b in zip(xs, xw)]))
        else:
            cd[k] = {"estimate": None, "n": 0}
    delta = cluster_bootstrap(
        lambda i: (
            (d[i][c[i] == "notation"].mean() if (c[i] == "notation").any() else np.nan)
            - (d[i][c[i] == "words"].mean() if (c[i] == "words").any() else np.nan)
        ),
        cl,
        B=B,
        seed=seed,
    )
    y, design, groups = (
        [],
        {
            "precedent_S": [],
            "dominant_notation": [],
            "interaction": [],
            "log_freq": [],
            "n_syll": [],
            "n_phones": [],
            "n_bpe": [],
        },
        [],
    )
    for (w, s), k, v in zip(kept, cell, cl):
        cov = item_cov.get(w["item_id"], {})
        bpe = cov.get(f"n_bpe_{rid}", cov.get("n_bpe_whisper"))
        if any(cov.get(x) is None for x in ("log_freq", "n_syll", "n_phones")) or bpe is None:
            continue
        for r, pS in ((w, 0.0), (s, 1.0)):
            y.append(int(r["itt_R"]))
            groups.append(v)
            dn = 1.0 if k == "notation" else 0.0
            design["precedent_S"].append(pS)
            design["dominant_notation"].append(dn)
            design["interaction"].append(pS * dn)
            design["log_freq"].append(float(cov["log_freq"]))
            design["n_syll"].append(float(cov["n_syll"]))
            design["n_phones"].append(float(cov["n_phones"]))
            design["n_bpe"].append(float(bpe))
    gee = (
        fit_logit_clustered(y, design, groups, focal=["precedent_S", "interaction"])
        if y
        else {"odds_ratios": {}, "fit_note": "no covariates"}
    )
    main_cell = max(cells, key=lambda k: cells[k]) if any(cells.values()) else None
    md = cd.get(main_cell) or {}
    tost = None
    if md.get("estimate") is not None and md.get("lo") is not None and md.get("hi") is not None:
        se = (md["hi"] - md["lo"]) / (2 * 1.96)
        tost = {
            "p": _f(stats.tost_difference(md["estimate"], se, TOST_PP, md.get("n_clusters") or 1)) if se > 0 else None,
            "bound_pp": TOST_PP,
            "se_source": "bootstrap-t interval half-width / 1.96",
        }
        tost["equivalence_holds"] = tost["p"] is not None and tost["p"] < ALPHA
    mde = {k: mde_paired(cells.get(k, 0), (cd.get(k) or {}).get("discordance")) for k in ("notation", "words")}
    return {
        **out,
        "cells": cells,
        "cell_differences": cd,
        "main_cell": main_cell,
        "delta": delta,
        "gee": gee,
        "tost": tost,
        "mde_pp": mde,
        "pooled_matching_minus_nonmatching": _paired_difference(
            [int(s["itt_R"]) if k == "notation" else int(w["itt_R"]) for (w, s), k in zip(kept, cell)],
            [int(w["itt_R"]) if k == "notation" else int(s["itt_R"]) for (w, s), k in zip(kept, cell)],
            cl,
            B=B,
            n_perm=n_perm,
            seed=seed + 5,
            cluster_label="video",
        ),
    }
