"""Estimators for the base-rate contrasts, the prompt-steering tests and minimum detectable effects.

Inputs are span rows written by scripts/score.py; outputs are plain dicts of counts and estimates, identical for every
recognizer.
  - clustered logistic model: statsmodels GEE (binomial logit, exchangeable working correlation, clustered by utterance,
    robust covariance);
  - block bootstrap of Liu and Peng (2020): `stats.paired_rate_difference` for paired differences, `cluster_bootstrap` for
    rates and the interaction delta;
  - permutation p-values: sign flips per utterance; label permutation across item types for delta;
  - TOST, Holm, sample size (Hsieh; two proportions; Kish design effect).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

from loanword_eval import stats

ALPHA = 0.05
GUARD_PP = 0.02  # max L-vs-H difference in control-word strict error
TOST_PP = 0.02  # equivalence bound on delta
MDE_LIMIT = 0.10  # MDE above which the directional test counts as underpowered
CONFIRMATORY = ("whisper", "owsm")
COVARIATES = ("log_freq", "freq_oov", "n_syll", "n_phones", "n_bpe")
PAIRS = (("H1", "L1"), ("H2", "L2"))
BOOTSTRAP_SEED = 20260918


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)


def itt_strict(row: dict) -> int:
    """Strict recognition under intention-to-treat: a flagged utterance keeps its items at 0."""
    return 0 if row.get("utt_flagged") else int(row["R_strict"])


def by_item(rows: Iterable[dict]) -> dict[str, dict]:
    return {r["item_id"]: r for r in rows}


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float], clusters: Sequence[str], *, B: int, seed: int, alpha: float = ALPHA
) -> dict:
    """Percentile cluster bootstrap of `stat(row_indices)` (block bootstrap of Liu and Peng, 2020)."""
    ids, idx = stats._cluster_index(list(clusters))
    k = len(ids)
    n = len(clusters)
    est = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        pick = rng.integers(0, k, size=k)
        sel = np.concatenate([idx[p] for p in pick]) if k else np.arange(0)
        v = stat(sel)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            boot.append(v)
    boot_a = np.asarray(boot, dtype=float)
    if boot_a.size == 0:
        return {
            "estimate": _f(est),
            "lo": None,
            "hi": None,
            "se": None,
            "B": B,
            "B_used": 0,
            "n_clusters": k,
            "method": "cluster-percentile",
        }
    lo, hi = np.quantile(boot_a, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": _f(est),
        "lo": _f(lo),
        "hi": _f(hi),
        "se": _f(boot_a.std(ddof=1)) if boot_a.size > 1 else 0.0,
        "B": B,
        "B_used": int(boot_a.size),
        "n_clusters": k,
        "method": "cluster-percentile",
    }


def paired(rows_a: Iterable[dict], rows_b: Iterable[dict]) -> tuple[list[dict], list[dict], dict]:
    """Rows of two conditions paired on item_id; unpaired items are counted, never silently dropped."""
    a, b = by_item(rows_a), by_item(rows_b)
    shared = sorted(set(a) & set(b))
    wf = {"paired": len(shared), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}
    return [a[i] for i in shared], [b[i] for i in shared], wf


def _paired_difference(
    x_a: Sequence[int],
    x_b: Sequence[int],
    clusters: Sequence[str],
    *,
    B: int,
    n_perm: int,
    seed: int,
    cluster_label: str = "utterance",
) -> dict:
    """mean(x_a) - mean(x_b) on paired items: bootstrap-t and percentile cluster intervals and a sign-flip permutation p,
    all blockwise by `clusters` (Liu and Peng, 2020). The bootstrap-t interval is an addition to their method."""
    if len(x_a) == 0:
        return {"estimate": None, "n": 0}
    ci = stats.paired_rate_difference(x_a, x_b, clusters, B=B, seed=seed)
    p = stats.paired_permutation(x_a, x_b, clusters, n_perm=n_perm, seed=seed + 1) if len(set(clusters)) > 1 else None
    t, pc = ci["bootstrap_t"], ci["percentile"]
    # With k discordant clusters all of one sign, the sign-flip p cannot fall below 2 / 2^k and the bootstrap interval
    # cannot cross zero, so a p at this floor means "not testable at this n".
    disc_idx = [i for i, (u, v) in enumerate(zip(x_a, x_b)) if u != v]
    n_pos = int(sum(1 for i in disc_idx if x_a[i] > x_b[i]))
    n_neg = len(disc_idx) - n_pos
    k = len({clusters[i] for i in disc_idx})
    p_floor = _f(2.0 / 2**k) if k > 0 else None
    same_sign = bool(disc_idx) and (n_pos == 0 or n_neg == 0)
    return {
        "estimate": _f(t.estimate),
        "lo": _f(t.lo),
        "hi": _f(t.hi),
        "lo_percentile": _f(pc.lo),
        "hi_percentile": _f(pc.hi),
        "method": f"blockwise cluster bootstrap (Liu & Peng 2020), {t.method} interval",
        "n": len(x_a),
        "n_clusters": t.n_clusters,
        "B": B,
        "B_used_t": t.B,
        "n_degenerate_resamples": ci.get("n_degenerate_resamples", 0),
        "p_permutation": _f(p),
        "permutation_unit": f"{cluster_label} (sign flips per cluster)",
        "n_perm": n_perm,
        "n_discordant": len(disc_idx),
        "n_discordant_positive": n_pos,
        "n_discordant_negative": n_neg,
        "n_discordant_clusters": k,
        "p_floor_same_sign": p_floor,
        "discordance_same_sign": same_sign,
        "p_at_floor": bool(p is not None and p_floor is not None and same_sign and p <= p_floor * 1.1),
    }


def h1a(rows: list[dict], *, B: int, seed: int) -> dict:
    """E_S, E_A and the share (E_S - E_A) / E_S with cluster intervals, and the difference broken down by form produced."""
    if not rows:
        return {"n": 0}
    R = np.array([r["itt_R"] for r in rows], dtype=float)
    S = np.array([itt_strict(r) for r in rows], dtype=float)
    cl = [r["utt_id"] for r in rows]
    es = cluster_bootstrap(lambda i: 1 - S[i].mean(), cl, B=B, seed=seed)
    ea = cluster_bootstrap(lambda i: 1 - R[i].mean(), cl, B=B, seed=seed)

    def share(i):
        e_s = 1 - S[i].mean()
        return (e_s - (1 - R[i].mean())) / e_s if e_s > 0 else None

    sh = cluster_bootstrap(share, cl, B=B, seed=seed)
    breakdown: dict[str, int] = {}
    for r in rows:
        if r["itt_R"] and not itt_strict(r):
            breakdown[str(r["itt_F"])] = breakdown.get(str(r["itt_F"]), 0) + 1
    unflagged = [r for r in rows if not r.get("utt_flagged")]
    excl = {
        "n": len(unflagged),
        "E_S": _f(1 - np.mean([r["R_strict"] for r in unflagged])) if unflagged else None,
        "E_A": _f(1 - np.mean([r["R"] for r in unflagged])) if unflagged else None,
    }
    per_corpus = {}
    for c in sorted({r["corpus"] for r in rows}):
        sub = [r for r in rows if r["corpus"] == c]
        per_corpus[c] = {
            "n": len(sub),
            "E_S": _f(1 - np.mean([itt_strict(r) for r in sub])),
            "E_A": _f(1 - np.mean([r["itt_R"] for r in sub])),
        }
    # Sensitivity rows: exact-aligned recognition (no window; not a lower bound, since cross-script output is mostly
    # matched by the window), type-clustered E_A, per code-switching level.
    NW = np.array([0 if r.get("utt_flagged") else int(r.get("R_nowindow", r["R"])) for r in rows], dtype=float)
    ea_nowindow = cluster_bootstrap(lambda i: 1 - NW[i].mean(), cl, B=B, seed=seed)
    types = [r.get("type_key", r["item_id"]) for r in rows]
    ea_type = cluster_bootstrap(lambda i: 1 - R[i].mean(), types, B=B, seed=seed)
    per_level = {}
    for lv in sorted({str(r.get("cs_level")) for r in rows}):
        sub = [k for k, r in enumerate(rows) if str(r.get("cs_level")) == lv]
        per_level[lv] = {"n": len(sub), "E_S": _f(1 - S[sub].mean()), "E_A": _f(1 - R[sub].mean())}
    return {
        "n": len(rows),
        "n_utterances": len(set(cl)),
        "E_S": es,
        "E_A": ea,
        "share": sh,
        "difference_by_form": breakdown,
        "per_condition_exclusion": excl,
        "per_corpus": per_corpus,
        "n_flagged_items": int(sum(1 for r in rows if r.get("utt_flagged"))),
        "E_A_exact_aligned": {
            **ea_nowindow,
            "note": "R_nowindow: exact-aligned match only; an alignment artefact for cross-script output, not a lower bound",
        },
        "E_A_type_clustered": ea_type,
        "per_cs_level": per_level,
    }


def _gee_logit(y: np.ndarray, X: np.ndarray, names: list[str], groups: np.ndarray) -> dict:
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable

    model = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(), cov_struct=Exchangeable())
    res = model.fit()
    ci = res.conf_int()
    coef = {
        n: {
            "beta": _f(res.params[i]),
            "se": _f(res.bse[i]),
            "p": _f(res.pvalues[i]),
            "lo": _f(ci[i][0]),
            "hi": _f(ci[i][1]),
        }
        for i, n in enumerate(names)
    }
    return {
        "coefficients": coef,
        "n_obs": int(len(y)),
        "n_clusters": int(len(set(groups.tolist()))),
        "working_correlation": _f(getattr(res.cov_struct, "dep_params", None)),
        "converged": bool(getattr(res, "converged", True)),
    }


def fit_logit_clustered(
    y: Sequence[int], design: dict[str, Sequence[float]], groups: Sequence[str], focal: Sequence[str]
) -> dict:
    """GEE logit with a fallback ladder (full -> constant columns dropped -> focal terms only) recorded in `fit_note`.
    Returns odds ratios for the `focal` terms, or None when the model is not estimable."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if y.size == 0 or y.min() == y.max():
        return {
            "odds_ratios": {f: None for f in focal},
            "fit_note": "outcome constant or empty; model not estimable",
            "n_obs": int(y.size),
            "covariates_used": [],
        }
    cols = list(design.keys())
    ladders = [
        cols,
        [c for c in cols if np.std(np.asarray(design[c], dtype=float)) > 0],
        [c for c in cols if c in focal],
    ]
    last_err = None
    for names in ladders:
        if not all(f in names for f in focal):
            continue
        X = np.column_stack([np.ones(y.size)] + [np.asarray(design[c], dtype=float) for c in names])
        try:
            fit = _gee_logit(y, X, ["intercept"] + names, groups)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:200]
            continue
        ors = {}
        for f in focal:
            c = fit["coefficients"][f]
            # a degenerate fit can return beta without an interval
            ors[f] = (
                None
                if c["beta"] is None
                else {
                    "OR": _f(math.exp(c["beta"])),
                    "lo": None if c["lo"] is None else _f(math.exp(c["lo"])),
                    "hi": None if c["hi"] is None else _f(math.exp(c["hi"])),
                    "p": c["p"],
                    "beta": c["beta"],
                    "se": c["se"],
                }
            )
        note = (
            "full model"
            if names == cols
            else ("constant covariates dropped" if names == ladders[1] else "focal terms only (covariate model failed)")
        )
        if last_err:
            note += f"; earlier attempt failed: {last_err}"
        return {"odds_ratios": ors, "fit_note": note, "covariates_used": [n for n in names if n not in focal], **fit}
    return {
        "odds_ratios": {f: None for f in focal},
        "fit_note": f"no model estimable ({last_err})",
        "n_obs": int(y.size),
        "covariates_used": [],
    }


def _cov(row: dict, cov: dict, rid: str, bpe_form: str = "reference") -> dict | None:
    """Covariate vector of an item or control, or None when any covariate is missing.
    bpe_form="hangul" uses the tokenizer length of the Hangul form (`n_bpe_<rid>_hangul`)."""
    bpe = cov.get(f"n_bpe_{rid}" if bpe_form == "reference" else f"n_bpe_{rid}_hangul")
    vals = {
        "log_freq": cov.get("log_freq"),
        "freq_oov": cov.get("freq_oov"),
        "n_syll": cov.get("n_syll"),
        "n_phones": cov.get("n_phones"),
        "n_bpe": bpe,
    }
    return None if any(v is None for v in vals.values()) else {k: float(v) for k, v in vals.items()}


def length_support(item_cov: dict[str, dict], control_cov: dict[tuple, dict]) -> dict:
    """Items and controls per syllable count; the length adjustment is supported only where both are non-zero."""
    tab: dict[int, dict] = {}
    for cov in item_cov.values():
        n = cov.get("n_syll")
        if n is not None:
            tab.setdefault(int(n), {"items": 0, "controls": 0})["items"] += 1
    for cov in control_cov.values():
        n = cov.get("n_syll")
        if n is not None:
            tab.setdefault(int(n), {"items": 0, "controls": 0})["controls"] += 1
    return {str(k): tab[k] for k in sorted(tab)}


def word_cluster_id(row: dict, cov: dict, is_item: bool) -> str:
    """Word-identity cluster: the item type for items, the lemma (else surface, else span) for controls.
    Analogue of the word random effect of Goldwater, Jurafsky and Manning (2010), used as an alternative clustering."""
    if is_item:
        return f"item:{row.get('type_key') or row.get('item_id')}"
    return f"ctrl:{cov.get('lemma') or cov.get('surface') or (row['utt_id'], row['start'], row['end'])}"


def h1b(
    item_rows: list[dict],
    control_rows: list[dict],
    item_cov: dict[str, dict],
    control_cov: dict[tuple, dict],
    rid: str,
    *,
    B: int,
    seed: int,
    n_syll_range: tuple[int, int] | None = None,
    cluster_by: str = "utterance",
    bpe_form: str = "reference",
    utt_levels: set[str] | None = None,
    controls_scoring: str = "strict",
    items_scoring: str = "tolerant",
    exclude_flagged: bool = False,
    cluster_key: str = "utt_id",
) -> dict:
    """Items vs ordinary words: error ~ is_item + covariates, GEE clustered by utterance; OR(is_item) > 1 means more item errors.

    Items use acceptance error 1 - itt_R; controls use strict error; flagged utterances count as errors (intention-to-treat).
    Sensitivity options:
      cluster_key: row field naming the utterance cluster (the math project passes `video_id`);
      n_syll_range=(lo, hi): only rows with lo <= n_syll <= hi;
      cluster_by="type": cluster by word identity (`word_cluster_id`);
      bpe_form="hangul": tokenizer length of the Hangul form for every row;
      utt_levels: only items of these HiKE code-switching levels and the controls of the same utterances;
      controls_scoring="tolerant" / items_scoring="strict": one error definition for every word, as in Goldwater, Jurafsky
        and Manning (2010);
      exclude_flagged: drop utterances with a whole-utterance flag (Thabane et al., 2013)."""
    if cluster_by not in ("utterance", "type"):
        raise ValueError(f"cluster_by must be 'utterance' or 'type', got {cluster_by!r}")
    y, groups, design, wf = (
        [],
        [],
        {k: [] for k in ("is_item",) + COVARIATES},
        {
            "items": 0,
            "controls": 0,
            "items_missing_cov": 0,
            "controls_missing_cov": 0,
            "items_outside_length_range": 0,
            "controls_outside_length_range": 0,
            "items_outside_level": 0,
            "controls_outside_level": 0,
        },
    )

    def _in_range(cv: dict) -> bool:
        return n_syll_range is None or (n_syll_range[0] <= cv["n_syll"] <= n_syll_range[1])

    level_utts = {r["utt_id"] for r in item_rows if utt_levels is not None and r.get("cs_level") in utt_levels}
    if controls_scoring not in ("strict", "tolerant") or items_scoring not in ("strict", "tolerant"):
        raise ValueError("controls_scoring / items_scoring must be 'strict' or 'tolerant'")
    wf["items_flagged_excluded"] = 0
    wf["controls_flagged_excluded"] = 0
    for r in item_rows:
        if utt_levels is not None and r.get("cs_level") not in utt_levels:
            wf["items_outside_level"] += 1
            continue
        if exclude_flagged and r.get("utt_flagged"):
            wf["items_flagged_excluded"] += 1
            continue
        raw = item_cov.get(r["item_id"], {})
        cv = _cov(r, raw, rid, bpe_form)
        if cv is None:
            wf["items_missing_cov"] += 1
            continue
        if not _in_range(cv):
            wf["items_outside_length_range"] += 1
            continue
        y_i = (
            (1 - r["itt_R"]) if items_scoring == "tolerant" else (1 if r.get("utt_flagged") else 1 - int(r["R_strict"]))
        )
        y.append(y_i)
        groups.append(r[cluster_key] if cluster_by == "utterance" else word_cluster_id(r, raw, True))
        design["is_item"].append(1.0)
        for k in COVARIATES:
            design[k].append(cv[k])
        wf["items"] += 1
    for r in control_rows:
        if utt_levels is not None and r["utt_id"] not in level_utts:
            wf["controls_outside_level"] += 1
            continue
        if exclude_flagged and r.get("utt_flagged"):
            wf["controls_flagged_excluded"] += 1
            continue
        raw = control_cov.get((r["utt_id"], r["start"], r["end"]), {})
        cv = _cov(r, raw, rid, bpe_form)
        if cv is None:
            wf["controls_missing_cov"] += 1
            continue
        if not _in_range(cv):
            wf["controls_outside_length_range"] += 1
            continue
        if controls_scoring == "tolerant":
            if "itt_R_tolerant" not in r:
                raise KeyError("controls_scoring='tolerant' needs control spans with itt_R_tolerant; re-run score.py")
            y_c = 1 - r["itt_R_tolerant"]
        else:
            y_c = r["itt_err_strict"]
        y.append(y_c)
        groups.append(r[cluster_key] if cluster_by == "utterance" else word_cluster_id(r, raw, False))
        design["is_item"].append(0.0)
        for k in COVARIATES:
            design[k].append(cv[k])
        wf["controls"] += 1
    fit = fit_logit_clustered(y, design, groups, focal=["is_item"])
    ya = np.asarray(y, dtype=float)
    isit = np.asarray(design["is_item"], dtype=float)
    unadj = (
        cluster_bootstrap(
            lambda i: (
                (ya[i][isit[i] == 1].mean() - ya[i][isit[i] == 0].mean())
                if (isit[i] == 1).any() and (isit[i] == 0).any()
                else None
            ),
            groups,
            B=B,
            seed=seed,
        )
        if wf["items"] and wf["controls"]
        else {"estimate": None}
    )
    err_items = _f(ya[isit == 1].mean()) if wf["items"] else None
    err_controls = _f(ya[isit == 0].mean()) if wf["controls"] else None
    orr = fit["odds_ratios"]["is_item"]
    # model-based detectable OR (Bloom, 1995, Table 1: two-sided alpha 0.05, power 0.80 -> exp(2.80 * SE))
    mde_model = _f(math.exp(2.80 * orr["se"])) if orr and orr.get("se") is not None else None
    return {
        "waterfall": wf,
        "error_items": err_items,
        "error_controls": err_controls,
        "unadjusted_difference": unadj,
        "detectable_OR_model_se": mde_model,
        "controls_scoring": controls_scoring,
        "items_scoring": items_scoring,
        "exclude_flagged": exclude_flagged,
        "OR_is_item": orr,
        "p": (orr or {}).get("p"),
        "rejected": (orr is not None and orr["p"] is not None and orr["p"] < ALPHA and orr["OR"] > 1),
        "fit_note": fit["fit_note"],
        "covariates_used": fit.get("covariates_used", []),
        "n_obs": fit.get("n_obs"),
        "n_clusters": fit.get("n_clusters"),
        "n_syll_range": list(n_syll_range) if n_syll_range else None,
        "cluster_by": cluster_by,
        "bpe_form": bpe_form,
        "utt_levels": sorted(utt_levels) if utt_levels is not None else None,
        "post_hoc": cluster_by != "utterance"
        or bpe_form != "reference"
        or utt_levels is not None
        or controls_scoring != "strict"
        or items_scoring != "tolerant"
        or exclude_flagged,
        "coefficients": fit.get("coefficients"),
        "model": (
            "GEE binomial logit, exchangeable, clustered by utterance"
            if cluster_by == "utterance"
            else "GEE binomial logit, exchangeable, clustered by word identity (item type / control lemma; sensitivity)"
        )
        + (f"; restricted to {n_syll_range[0]}–{n_syll_range[1]} syllables (sensitivity)" if n_syll_range else "")
        + ("; n_bpe of the Hangul form for every row (sensitivity)" if bpe_form == "hangul" else "")
        + (
            f"; items in {'/'.join(sorted(utt_levels))}-level utterances and their comparison words only (sensitivity)"
            if utt_levels is not None
            else ""
        )
        + (
            "; comparison words scored with the item path (tolerant; sensitivity)"
            if controls_scoring == "tolerant"
            else ""
        )
        + ("; items scored strictly (sensitivity)" if items_scoring == "strict" else "")
        + ("; utterances with a whole-utterance flag excluded (sensitivity)" if exclude_flagged else ""),
    }


def h2a(rows_H: list[dict], rows_L: list[dict], *, B: int, n_perm: int, seed: int) -> dict:
    """Manipulation check: P(F = latin | R = 1) under L minus under H, paired on items recognized under both."""
    H, L, wf = paired(rows_H, rows_L)
    # intention-to-treat: items of a flagged utterance count as not recognized
    both = [(h, l) for h, l in zip(H, L) if h["itt_R"] and l["itt_R"]]
    lat_h = [int(h["itt_F"] == "latin") for h, _ in both]
    lat_l = [int(l["itt_F"] == "latin") for _, l in both]
    cl = [h["utt_id"] for h, _ in both]
    diff = _paired_difference(lat_l, lat_h, cl, B=B, n_perm=n_perm, seed=seed)
    diff_type = _paired_difference(
        lat_l,
        lat_h,
        [h.get("type_key", h["item_id"]) for h, _ in both],
        B=B,
        n_perm=n_perm,
        seed=seed + 17,
        cluster_label="item type",
    )
    share_h = _f(np.mean([r["itt_F"] == "latin" for r in H if r["itt_R"]])) if any(r["itt_R"] for r in H) else None
    share_l = _f(np.mean([r["itt_F"] == "latin" for r in L if r["itt_R"]])) if any(r["itt_R"] for r in L) else None
    rejected = (
        diff.get("estimate") is not None
        and diff.get("p_permutation") is not None
        and diff["p_permutation"] < ALPHA
        and diff["estimate"] > 0
    )
    return {
        "pairing": wf,
        "n_recognized_both": len(both),
        "P_latin_given_R": {"H": share_h, "L": share_l},
        "difference_L_minus_H": diff,
        "difference_L_minus_H_type_clustered": diff_type,
        "rejected": bool(rejected),
        "note": "paired on items recognized under BOTH conditions (intention-to-treat); a selection on the outcome",
    }


def _delta(d: np.ndarray, cell: np.ndarray, i: np.ndarray) -> float | None:
    a, b = d[i][cell[i] == 1], d[i][cell[i] == 0]
    return float(a.mean() - b.mean()) if a.size and b.size else None


def h2b(
    rows_H: list[dict],
    rows_L: list[dict],
    direction_rows: list[dict],
    item_cov: dict[str, dict],
    rid: str,
    controls_H: list[dict] | None,
    controls_L: list[dict] | None,
    *,
    B: int,
    n_perm: int,
    seed: int,
    include_fallback: bool = False,
    min_recognized: int | None = None,
) -> dict:
    """Steering test: delta = mean(R_L - R_H | dominant = Latin) - mean(R_L - R_H | dominant = Hangul), fallback types
    excluded, intention-to-treat. Reports a cluster-bootstrap interval, a type-level label-permutation p, the GEE interaction,
    the paired difference in each dominance cell, the pooled matching-minus-nonmatching difference, the control-word guard and
    TOST on delta."""
    H, L, wf = paired(rows_H, rows_L)
    dirmap = {r["type_key"]: r for r in direction_rows}
    d, cell, cl, types, kept_H, kept_L = [], [], [], [], [], []
    excl = {"fallback_excluded": 0, "no_direction_row": 0, "singleton_type_excluded": 0}
    match_R, nonmatch_R, match_cl = [], [], []
    for h, l in zip(H, L):
        dr = dirmap.get(h["type_key"])
        if dr is None:
            excl["no_direction_row"] += 1
            continue
        if dr.get("fallback") and not include_fallback:
            excl["fallback_excluded"] += 1
            continue
        if min_recognized and (dr.get("n_recognized") or 0) < min_recognized:
            excl["singleton_type_excluded"] += 1
            continue
        dom_latin = 1 if dr["dominant"] == "latin" else 0
        d.append(l["itt_R"] - h["itt_R"])
        cell.append(dom_latin)
        cl.append(h["utt_id"])
        types.append(h["type_key"])
        kept_H.append(h)
        kept_L.append(l)
        match_R.append(l["itt_R"] if dom_latin else h["itt_R"])
        nonmatch_R.append(h["itt_R"] if dom_latin else l["itt_R"])
        match_cl.append(h["utt_id"])
    d_a, cell_a = np.asarray(d, dtype=float), np.asarray(cell)
    n_lat, n_han = int((cell_a == 1).sum()), int((cell_a == 0).sum())
    out: dict = {
        "pairing": wf,
        "exclusions": excl,
        "n_analysed": len(d),
        "cells": {"latin": n_lat, "hangul": n_han},
        "conservation_ok": wf["paired"]
        == len(d) + excl["fallback_excluded"] + excl["no_direction_row"] + excl["singleton_type_excluded"],
        "min_recognized": min_recognized,
    }
    # paired difference per dominance cell, computed whenever the cell exists
    out["cell_differences"] = {}
    for name, flag in (("latin", 1), ("hangul", 0)):
        sel = [k for k in range(len(d)) if cell[k] == flag]
        out["cell_differences"][name] = (
            _paired_difference(
                [kept_L[k]["itt_R"] for k in sel],
                [kept_H[k]["itt_R"] for k in sel],
                [cl[k] for k in sel],
                B=B,
                n_perm=n_perm,
                seed=seed + 3,
            )
            if sel
            else {"estimate": None, "n": 0}
        )
    lat = out["cell_differences"]["latin"]
    # the final rejection uses Holm-adjusted p-values across recognizers and both clusterings (set by the family step)
    out["latin_cell_rejected_pre_holm"] = bool(
        lat.get("estimate") is not None
        and lat.get("p_permutation") is not None
        and lat["p_permutation"] < ALPHA
        and lat["estimate"] > 0
    )
    sel_l = [k for k in range(len(d)) if cell[k] == 1]
    out["cell_differences_type_clustered"] = {
        "latin": _paired_difference(
            [kept_L[k]["itt_R"] for k in sel_l],
            [kept_H[k]["itt_R"] for k in sel_l],
            [types[k] for k in sel_l],
            B=B,
            n_perm=n_perm,
            seed=seed + 19,
            cluster_label="item type",
        )
        if sel_l
        else {"estimate": None, "n": 0}
    }
    # items of a type move together, so a Latin-cell rejection also requires the type-clustered permutation p
    lat_t = out["cell_differences_type_clustered"]["latin"]
    out["latin_cell_rejected_pre_holm"] = bool(
        out["latin_cell_rejected_pre_holm"]
        and lat_t.get("p_permutation") is not None
        and lat_t["p_permutation"] < ALPHA
    )
    out["latin_cell_rejected"] = None  # set by the family step
    out["fallback_included"] = include_fallback
    # Latin-cell items recognized under L only whose H hypothesis is a Hangul substitution (candidate spelling variants)
    cand = [kept_H[k] for k in sel_l if kept_L[k]["itt_R"] == 1 and kept_H[k]["itt_R"] == 0]
    hang = sum(
        1
        for h in cand
        if h.get("outcome") == "SUB" and any("가" <= ch <= "힣" for ch in str(h.get("hyp_material", "")))
    )
    out["latin_cell_L_only_variant_candidates"] = {
        "n_L_only": len(cand),
        "n_H_hangul_sub": hang,
        "share": _f(hang / len(cand)) if cand else None,
    }

    # the same count in both cells
    def _l_only(flag: int) -> dict:
        sel_ = [k for k in range(len(d)) if cell[k] == flag]
        c_ = [kept_H[k] for k in sel_ if kept_L[k]["itt_R"] == 1 and kept_H[k]["itt_R"] == 0]
        h_ = sum(
            1
            for h in c_
            if h.get("outcome") == "SUB" and any("가" <= ch <= "힣" for ch in str(h.get("hyp_material", "")))
        )
        return {"n_L_only": len(c_), "n_H_hangul_sub": h_, "share": _f(h_ / len(c_)) if c_ else None}

    out["L_only_variant_candidates_by_cell"] = {"latin": _l_only(1), "hangul": _l_only(0), "post_hoc": True}
    if n_lat == 0 or n_han == 0:
        out.update({"delta": {"estimate": None}, "status": "cell_empty", "rejected": False})
    else:
        delta = cluster_bootstrap(lambda i: _delta(d_a, cell_a, i), cl, B=B, seed=seed)
        # dominance is a property of the type, so labels are permuted over types
        tkeys = sorted(set(types))
        tlab = {t: cell[types.index(t)] for t in tkeys}
        rng = np.random.default_rng(seed + 7)
        obs = abs(delta["estimate"])
        hits = 0
        used = 0
        labs = np.array([tlab[t] for t in tkeys])
        tidx = np.array([tkeys.index(t) for t in types])
        for _ in range(n_perm):
            perm = labs[rng.permutation(len(labs))][tidx]
            v = _delta(d_a, perm, np.arange(len(d)))
            if v is None:
                continue
            used += 1
            hits += abs(v) >= obs - 1e-12
        p_perm = (hits + 1) / (used + 1) if used else None
        delta["p_permutation"] = _f(p_perm)
        delta["n_perm_used"] = used
        delta["permutation_unit"] = "item type"
        out["delta"] = delta
        out["status"] = "estimated"
        out["rejected"] = bool(p_perm is not None and p_perm < ALPHA and delta["estimate"] > 0)
        # TOST on delta with bound TOST_PP
        se = delta.get("se")
        if se is None or se == 0:
            out["tost"] = {
                "bound": TOST_PP,
                "p": None,
                "equivalence_holds": None,
                "se_source": "cluster bootstrap",
                "note": "SE zero or missing: not estimable",
            }
        else:
            p_tost = stats.tost_difference(delta["estimate"], se, TOST_PP, delta["n_clusters"])
            out["tost"] = {
                "bound": TOST_PP,
                "p": _f(p_tost),
                "equivalence_holds": bool(p_tost < ALPHA),
                "se_source": "cluster bootstrap",
            }
        # GEE with a precedent x dominant-form interaction
        y, groups, design = [], [], {k: [] for k in ("precedent_L", "dominant_latin", "interaction") + COVARIATES}
        miss = 0
        for k, (h, l) in enumerate(zip(kept_H, kept_L)):
            cv = _cov(h, item_cov.get(h["item_id"], {}), rid)
            if cv is None:
                miss += 1
                continue
            for prec, r in ((0, h), (1, l)):
                y.append(r["itt_R"])
                groups.append(h["utt_id"])
                design["precedent_L"].append(prec)
                design["dominant_latin"].append(cell[k])
                design["interaction"].append(prec * cell[k])
                for c in COVARIATES:
                    design[c].append(cv[c])
        fit = fit_logit_clustered(y, design, groups, focal=["precedent_L", "dominant_latin", "interaction"])
        out["gee"] = {
            "interaction": fit["odds_ratios"].get("interaction"),
            "precedent_L": fit["odds_ratios"].get("precedent_L"),
            "fit_note": fit["fit_note"],
            "n_obs": fit.get("n_obs"),
            "items_missing_covariates": miss,
            "model": "recognized ~ precedent × dominant_form + "
            + " + ".join(COVARIATES)
            + ", GEE clustered by utterance (type random effect not available in GEE)",
        }
    # pooled matching minus non-matching
    out["pooled_matching_minus_nonmatching"] = (
        _paired_difference(match_R, nonmatch_R, match_cl, B=B, n_perm=n_perm, seed=seed + 5)
        if match_R
        else {"estimate": None, "n": 0}
    )
    out["pooled_L_minus_H_all_kept"] = (
        _paired_difference(
            [l["itt_R"] for l in kept_L], [h["itt_R"] for h in kept_H], cl, B=B, n_perm=n_perm, seed=seed + 9
        )
        if kept_H
        else {"estimate": None, "n": 0}
    )
    # guard: strict error on the control words of the same utterances
    out["guard"] = control_guard(
        controls_H, controls_L, {r["utt_id"] for r in kept_H}, B=B, n_perm=n_perm, seed=seed + 11
    )
    # discordance per cell (input to the MDE simulation)
    out["discordance"] = {
        "latin": _f(np.mean(np.abs(d_a[cell_a == 1]))) if n_lat else None,
        "hangul": _f(np.mean(np.abs(d_a[cell_a == 0]))) if n_han else None,
    }
    return out


def control_guard(
    controls_H: list[dict] | None, controls_L: list[dict] | None, utts: set[str], *, B: int, n_perm: int, seed: int
) -> dict:
    """Control-word guard: holds when the cluster-bootstrap interval of the L minus H strict control error lies within
    +-GUARD_PP."""
    if not controls_H or not controls_L:
        return {"status": "no_controls", "holds": None}
    key = lambda r: (r["utt_id"], r["start"], r["end"])
    ch = {key(r): r for r in controls_H if r["utt_id"] in utts}
    cl_ = {key(r): r for r in controls_L if r["utt_id"] in utts}
    shared = sorted(set(ch) & set(cl_))
    if not shared:
        return {"status": "no_shared_controls", "holds": None}
    diff = _paired_difference(
        [cl_[k]["itt_err_strict"] for k in shared],
        [ch[k]["itt_err_strict"] for k in shared],
        [k[0] for k in shared],
        B=B,
        n_perm=n_perm,
        seed=seed,
    )
    lo, hi = diff.get("lo"), diff.get("hi")
    holds = lo is not None and hi is not None and max(abs(lo), abs(hi)) <= GUARD_PP
    return {
        "status": "evaluated",
        "n_controls": len(shared),
        "bound": GUARD_PP,
        "difference_L_minus_H": diff,
        "holds": bool(holds),
    }


def mde_paired(n: int, q: float, *, alpha: float = ALPHA, power: float = 0.8) -> float | None:
    """Minimum detectable paired difference for n items with discordance q = P(R_L != R_H):
    (z_{1-alpha/2} + z_power) * sqrt(q / n), the normal approximation of the sign test."""
    from scipy import stats as st

    if not n or not q:
        return None
    return _f((st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) * math.sqrt(q / n))


# an R row whose flagged share exceeds its L row's by more than this is not a usable instrument
S3_FLAG_MARGIN = 0.10
