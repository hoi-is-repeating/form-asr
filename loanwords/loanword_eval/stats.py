"""Paired cluster bootstrap (Bisani and Ney, 2004), cluster sign-flip permutation test, Holm correction, TOST and sample-size
helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CI:
    estimate: float
    lo: float
    hi: float
    method: str
    n_clusters: int
    B: int


def _cluster_index(clusters: Sequence[str]) -> tuple[list[str], list[np.ndarray]]:
    ids = sorted(set(clusters))
    arr = np.asarray(clusters)
    return ids, [np.flatnonzero(arr == c) for c in ids]


def paired_rate_difference(
    err_a: Sequence[int],
    err_b: Sequence[int],
    clusters: Sequence[str],
    *,
    B: int = 2000,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, CI]:
    """Paired cluster bootstrap of mean(err_a) - mean(err_b); returns bootstrap-t and percentile CIs."""
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    assert a.shape == b.shape
    ids, idx = _cluster_index(clusters)
    k = len(ids)
    rng = np.random.default_rng(seed)
    d_hat = a.mean() - b.mean()

    def se_of(a_, b_, idx_list):
        diffs = a_ - b_
        tot = np.array([diffs[i].sum() for i in idx_list])
        sz = np.array([len(i) for i in idx_list])
        m = diffs.mean()
        resid = tot - m * sz
        return math.sqrt(k / max(k - 1, 1) * (resid**2).sum()) / diffs.size

    se_hat = se_of(a, b, idx)
    boot_d = np.empty(B)
    boot_t = []
    n_degenerate = 0
    for r in range(B):
        pick = rng.integers(0, k, size=k)
        sel = np.concatenate([idx[p] for p in pick])
        ab, bb = a[sel], b[sel]
        db = ab.mean() - bb.mean()
        boot_d[r] = db
        # duplicated clusters count as distinct clusters in the resample
        sel_idx = []
        off = 0
        for p in pick:
            L = len(idx[p])
            sel_idx.append(np.arange(off, off + L))
            off += L
        seb = se_of(ab, bb, sel_idx)
        if seb == 0:  # no discordance: t is undefined; skipped rather than floored
            n_degenerate += 1
            continue
        boot_t.append((db - d_hat) / seb)
    lo_p, hi_p = np.quantile(boot_d, [alpha / 2, 1 - alpha / 2])
    ci_p = CI(d_hat, lo_p, hi_p, "percentile", k, B)
    if se_hat > 0 and len(boot_t) >= max(50, B // 2):
        t_lo, t_hi = np.quantile(np.asarray(boot_t), [alpha / 2, 1 - alpha / 2])
        ci_t = CI(d_hat, d_hat - t_hi * se_hat, d_hat - t_lo * se_hat, "bootstrap-t", k, len(boot_t))
    else:
        ci_t = CI(d_hat, lo_p, hi_p, "percentile (bootstrap-t not estimable: sparse discordance)", k, B)
    return {"bootstrap_t": ci_t, "percentile": ci_p, "n_degenerate_resamples": n_degenerate}


def paired_permutation(
    err_a: Sequence[int], err_b: Sequence[int], clusters: Sequence[str] | None = None, *, n_perm: int = 10000, seed: int
) -> float:
    """Two-sided sign-flip permutation p-value for mean(err_a - err_b) = 0, flipping signs per cluster (utterance).
    Without `clusters` every item is its own cluster."""
    d = np.asarray(err_a, dtype=float) - np.asarray(err_b, dtype=float)
    n = d.size
    if clusters is None:
        sums = d
    else:
        _, idx = _cluster_index(list(clusters))
        sums = np.array([d[i].sum() for i in idx])
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, sums.size))
    perm = np.abs(signs @ sums) / n
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def tost_difference(diff: float, se: float, bound: float, n_clusters: int) -> float:
    """Two one-sided tests for |diff| < bound; returns the larger of the two one-sided p-values."""
    from scipy import stats as st

    df = max(n_clusters - 1, 1)
    p1 = st.t.sf((diff + bound) / se, df)  # H0: diff <= -bound
    p2 = st.t.cdf((diff - bound) / se, df)  # H0: diff >= bound
    return float(max(p1, p2))


def two_proportion_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    from scipy import stats as st

    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
    return int(math.ceil(n))


def hsieh_logistic_n(p: float, odds_ratio_per_sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Sample size for a continuous predictor (Hsieh, Bloch and Larsen, 1998): n = (z_a + z_b)^2 / (p(1-p) beta^2)."""
    from scipy import stats as st

    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    beta = math.log(odds_ratio_per_sd)
    return int(math.ceil((za + zb) ** 2 / (p * (1 - p) * beta**2)))
