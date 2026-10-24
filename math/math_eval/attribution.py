"""Attribution of final LaTeX errors to recognition or conversion. Per-item rows in, counts and estimates out.

Row fields: `R` (recognized, intention-to-treat), `C` (converter output correct on the transcript), `C_ref` (converter output correct on
the reference spoken form), `video_id` (cluster).
  decomposition  P(C=0) = P(R=0) P(C=0 | R=0) + P(R=1) P(C=0 | R=1)
  delta_rec      P(C=0 | transcript) - P(C=0 | reference), the recognition-attributable excess
  h3a            delta_rec against the converter's own error P(C=0 | reference), with a cluster interval of their difference
  h3b            P(R=0) under D2 against D0, paired on clips, clustered by video
"""

from __future__ import annotations

import numpy as np
from loanword_eval.models import _f, _paired_difference, cluster_bootstrap, paired


def _arr(rows, key):
    return np.array([int(r[key]) for r in rows], dtype=float)


def decomposition(rows: list[dict], *, B: int, seed: int) -> dict:
    if not rows:
        return {"n": 0}
    R, C = _arr(rows, "R"), _arr(rows, "C")
    cl = [r["video_id"] for r in rows]
    n = len(rows)
    p_r0 = float((1 - R).mean())
    c0_given_r0 = float((1 - C)[R == 0].mean()) if (R == 0).any() else None
    c0_given_r1 = float((1 - C)[R == 1].mean()) if (R == 1).any() else None
    term_rec = p_r0 * (c0_given_r0 or 0.0)
    term_conv = (1 - p_r0) * (c0_given_r1 or 0.0)
    total = float((1 - C).mean())
    share = cluster_bootstrap(
        lambda i: ((1 - R[i]) * (1 - C[i])).sum() / max(1, (1 - C[i]).sum()) if (1 - C[i]).sum() > 0 else None,
        cl,
        B=B,
        seed=seed,
    )
    out = {
        "n": n,
        "n_videos": len(set(cl)),
        "P_R0": p_r0,
        "P_C0_given_R0": c0_given_r0,
        "P_C0_given_R1": c0_given_r1,
        "term_recognition": term_rec,
        "term_conversion": term_conv,
        "P_C0": total,
        "identity_ok": abs(term_rec + term_conv - total) < 1e-9,
        "recognition_share_of_final_error_upper_bound": share,  # credits recognition with every joint failure: an upper bound
        "cells": {
            "R0_C0": int(((R == 0) & (C == 0)).sum()),
            "R0_C1": int(((R == 0) & (C == 1)).sum()),
            "R1_C0": int(((R == 1) & (C == 0)).sum()),
            "R1_C1": int(((R == 1) & (C == 1)).sum()),
        },
    }
    if all("C_ref" in r for r in rows):
        # per-item attribution: the eight (R, C_ref, C) cells and the item-level classes
        Cref = _arr(rows, "C_ref")
        cells8 = {}
        for r_ in (0, 1):
            for cr in (0, 1):
                for c_ in (0, 1):
                    cells8[f"R{r_}_Cref{cr}_C{c_}"] = int(((R == r_) & (Cref == cr) & (C == c_)).sum())
        rec_item = ((C == 0) & (Cref == 1)).astype(
            float
        )  # the converter would have been right on the reference: recognition-attributable
        conv_item = ((C == 0) & (Cref == 0)).astype(
            float
        )  # the converter fails on the reference too: converter-attributable
        lucky = ((C == 1) & (Cref == 0)).astype(float)  # right on the ASR output although wrong on the reference
        out.update(
            {
                "cells_R_Cref_C": cells8,
                "item_recognition_attributable_rate": cluster_bootstrap(
                    lambda i: rec_item[i].mean(), cl, B=B, seed=seed
                ),
                "item_converter_attributable_rate": cluster_bootstrap(
                    lambda i: conv_item[i].mean(), cl, B=B, seed=seed
                ),
                "item_lucky_rate": cluster_bootstrap(lambda i: lucky[i].mean(), cl, B=B, seed=seed),
                "recognition_share_of_final_error_item_level": cluster_bootstrap(
                    lambda i: rec_item[i].sum() / max(1, (1 - C[i]).sum()) if (1 - C[i]).sum() > 0 else None,
                    cl,
                    B=B,
                    seed=seed,
                ),
                "note": "upper bound credits recognition with every joint failure; item level counts only errors the converter does not make on the reference (lower bound)",
            }
        )
    return out


def delta_rec(rows: list[dict], *, B: int, seed: int) -> dict:
    """delta_rec = P(C=0 | transcript) - P(C=0 | reference) on the same items, cluster bootstrap by video."""
    if not rows:
        return {"n": 0}
    C, Cref = _arr(rows, "C"), _arr(rows, "C_ref")
    cl = [r["video_id"] for r in rows]
    asr = cluster_bootstrap(lambda i: 1 - C[i].mean(), cl, B=B, seed=seed)
    ref = cluster_bootstrap(lambda i: 1 - Cref[i].mean(), cl, B=B, seed=seed)
    d = cluster_bootstrap(lambda i: (1 - C[i]).mean() - (1 - Cref[i]).mean(), cl, B=B, seed=seed)
    return {"n": len(rows), "P_C0_asr": asr, "P_C0_reference": ref, "delta_rec": d}


def h3a(rows: list[dict], *, B: int, seed: int, n_perm: int = 10000) -> dict:
    """Which stage contributes more: delta_rec against P(C=0 | reference), and the cluster interval of their difference."""
    if not rows:
        return {"n": 0}
    C, Cref = _arr(rows, "C"), _arr(rows, "C_ref")
    cl = [r["video_id"] for r in rows]
    dr = delta_rec(rows, B=B, seed=seed)
    # delta_rec - P(C=0|ref) = P(C=0|ASR) - 2 * P(C=0|ref)
    diff = cluster_bootstrap(
        lambda i: ((1 - C[i]).mean() - (1 - Cref[i]).mean()) - (1 - Cref[i]).mean(), cl, B=B, seed=seed
    )
    larger = None
    if dr["delta_rec"]["estimate"] is not None and dr["P_C0_reference"]["estimate"] is not None:
        larger = "recognition" if dr["delta_rec"]["estimate"] > dr["P_C0_reference"]["estimate"] else "conversion"
    excl = diff["lo"] is not None and (diff["lo"] > 0 or diff["hi"] < 0)
    return {
        **dr,
        "difference_rec_minus_conv": diff,
        "larger_term": larger,
        "interval_excludes_zero": bool(excl),
        "decomposition": decomposition(rows, B=B, seed=seed),
    }


def h3b(rows_d0: list[dict], rows_d2: list[dict], *, B: int, n_perm: int, seed: int) -> dict:
    """P(R=0) lower under D2 than D0, paired on items, clustered by video (audio identical by construction)."""
    a, b, wf = paired(rows_d0, rows_d2)
    if not a:
        return {"pairing": wf, "n": 0}
    e0 = [1 - int(r["R"]) for r in a]
    e2 = [1 - int(r["R"]) for r in b]
    cl = [r["video_id"] for r in a]
    d = _paired_difference(e2, e0, cl, B=B, n_perm=n_perm, seed=seed, cluster_label="video")
    f0 = {k: sum(1 for r in a if r.get("F") == k) for k in ("words", "notation", "mixed", "latex")}
    f2 = {k: sum(1 for r in b if r.get("F") == k) for k in ("words", "notation", "mixed", "latex")}
    return {
        "pairing": wf,
        "n": len(a),
        "P_R0_D0": _f(np.mean(e0)),
        "P_R0_D2": _f(np.mean(e2)),
        "difference_D2_minus_D0": d,
        "recovered_0_to_1": int(sum(1 for x, y in zip(e0, e2) if x == 1 and y == 0)),
        "lost_1_to_0": int(sum(1 for x, y in zip(e0, e2) if x == 0 and y == 1)),
        "form_counts_D0": f0,
        "form_counts_D2": f2,
    }
