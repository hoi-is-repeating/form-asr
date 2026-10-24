"""Estimator tests: each estimator recovers a known quantity on simulated data or agrees with a library implementation."""

import math

import numpy as np
import pytest

from loanword_eval import models, stats


def _sig(x):
    return 1 / (1 + math.exp(-x))


def _cov(rng):
    ns = int(rng.integers(1, 5))
    return {
        "freq_oov": 0,
        "log_freq": float(rng.normal(5, 1)),
        "n_syll": ns,
        "n_phones": int(ns * 2 + rng.integers(0, 3)),
        "n_bpe_whisper": int(rng.integers(1, 6)),
    }


def simulate_h1b(seed=1, n_utt=400, per_utt=6, or_true=2.0, p_control=0.15):
    """Items and controls in the same utterances with an utterance random intercept; only is_item has a real effect."""
    rng = np.random.default_rng(seed)
    b0 = math.log(p_control / (1 - p_control))
    items, controls, icov, ccov = [], [], {}, {}
    for u in range(n_utt):
        re_u = rng.normal(0, 0.5)
        for j in range(per_utt):
            is_item = j < 2
            err = rng.random() < _sig(b0 + re_u + math.log(or_true) * is_item)
            cv = {**_cov(rng), "freq_oov": 0}
            if is_item:
                iid = f"u{u}:{j}"
                items.append(
                    {
                        "item_id": iid,
                        "utt_id": f"u{u}",
                        "type_key": f"t{j}",
                        "corpus": "hike",
                        "R": int(not err),
                        "itt_R": int(not err),
                        "R_strict": int(not err),
                        "F": None if err else "latin",
                        "itt_F": None if err else "latin",
                        "utt_flagged": 0,
                    }
                )
                icov[iid] = cv
            else:
                controls.append(
                    {
                        "utt_id": f"u{u}",
                        "start": j * 3,
                        "end": j * 3 + 2,
                        "itt_err_strict": int(err),
                        "err_strict": int(err),
                        "utt_flagged": 0,
                        "R_tolerant": int(not err),
                        "itt_R_tolerant": int(not err),
                    }
                )
                ccov[(f"u{u}", j * 3, j * 3 + 2)] = cv
    return items, controls, icov, ccov


def test_s1_gee_recovers_known_odds_ratio():
    items, controls, icov, ccov = simulate_h1b()
    out = models.h1b(items, controls, icov, ccov, "whisper", B=200, seed=3)
    orr = out["OR_is_item"]
    assert out["fit_note"] == "full model" and set(out["covariates_used"]) == set(models.COVARIATES)
    assert 1.5 < orr["OR"] < 2.7 and orr["lo"] < 2.0 < orr["hi"], orr
    assert orr["p"] < 0.01 and out["rejected"] is True
    assert out["waterfall"] == {
        "items": 800,
        "controls": 1600,
        "items_missing_cov": 0,
        "controls_missing_cov": 0,
        "items_outside_level": 0,
        "controls_outside_level": 0,
        "items_outside_length_range": 0,
        "controls_outside_length_range": 0,
        "items_flagged_excluded": 0,
        "controls_flagged_excluded": 0,
    }
    assert (
        out["controls_scoring"] == "strict"
        and out["items_scoring"] == "tolerant"
        and out["exclude_flagged"] is False
        and out["post_hoc"] is False
    )
    assert abs(out["detectable_OR_model_se"] - math.exp(2.80 * orr["se"])) < 1e-9
    assert out["unadjusted_difference"]["estimate"] > 0 and out["unadjusted_difference"]["lo"] > 0


def test_s1_gee_reports_inestimable_models_instead_of_inventing():
    items, controls, icov, ccov = simulate_h1b(n_utt=30)
    for r in items:
        r["itt_R"] = 1
    for r in controls:
        r["itt_err_strict"] = 0
    out = models.h1b(items, controls, icov, ccov, "whisper", B=50, seed=1)
    assert (
        out["OR_is_item"] is None
        and out["p"] is None
        and out["rejected"] is False
        and "not estimable" in out["fit_note"]
    )


def simulate_h2b(seed=2, n_types=60, per_type=25, q=0.3, delta=0.10, fallback_types=2, n_utt=300):
    """Paired items; a type's dominance is fixed; Latin-cell items favour L by delta, Hangul-cell items are balanced."""
    rng = np.random.default_rng(seed)
    H, L, direction, icov = [], [], [], {}
    k = 0
    for t in range(n_types):
        dom = "latin" if t % 2 == 0 else "hangul"
        fb = 1 if t >= n_types - fallback_types else 0
        direction.append(
            {"type_key": f"type{t}", "dominant": dom, "fallback": fb, "fallback_reason": "tie" if fb else None}
        )
        for _ in range(per_type):
            d_shift = delta if dom == "latin" else 0.0
            u = rng.random()
            if u < (q + d_shift) / 2:
                rh, rl = 0, 1
            elif u < q:
                rh, rl = 1, 0
            else:
                rh = rl = int(rng.random() < 0.6)
            iid = f"i{k}"
            utt = f"u{k % n_utt}"
            k += 1
            base = {"item_id": iid, "utt_id": utt, "type_key": f"type{t}", "corpus": "hike", "utt_flagged": 0}
            H.append(
                {
                    **base,
                    "R": rh,
                    "itt_R": rh,
                    "R_strict": rh,
                    "F": "hangul" if rh else None,
                    "itt_F": "hangul" if rh else None,
                }
            )
            L.append(
                {
                    **base,
                    "R": rl,
                    "itt_R": rl,
                    "R_strict": 0,
                    "F": "latin" if rl else None,
                    "itt_F": "latin" if rl else None,
                }
            )
            icov[iid] = _cov(rng)
    return H, L, direction, icov


def test_s2_s4_interaction_delta_recovers_truth_and_conserves_items():
    H, L, direction, icov = simulate_h2b()
    out = models.h2b(H, L, direction, icov, "whisper", None, None, B=300, n_perm=500, seed=5)
    d = out["delta"]
    assert (
        out["conservation_ok"]
        and out["exclusions"]["fallback_excluded"] == 2 * 25
        and out["exclusions"]["no_direction_row"] == 0
    )
    assert out["cells"] == {"latin": 29 * 25, "hangul": 29 * 25}
    assert abs(d["estimate"] - 0.10) < 0.04 and d["lo"] < 0.10 < d["hi"], d
    assert d["p_permutation"] < 0.05 and out["rejected"] is True and d["permutation_unit"] == "item type"
    assert out["cell_differences"]["latin"]["estimate"] > out["cell_differences"]["hangul"]["estimate"]
    assert out["gee"]["interaction"]["OR"] > 1 and out["gee"]["fit_note"] == "full model"
    assert out["guard"] == {"status": "no_controls", "holds": None}
    assert 0.2 < out["discordance"]["latin"] < 0.4 and 0.2 < out["discordance"]["hangul"] < 0.4


def test_h2b_null_delta_is_not_rejected_and_tost_holds_at_large_n():
    H, L, direction, icov = simulate_h2b(seed=9, delta=0.0, per_type=60)
    out = models.h2b(H, L, direction, icov, "whisper", None, None, B=300, n_perm=500, seed=6)
    assert abs(out["delta"]["estimate"]) < 0.03 and out["rejected"] is False
    assert out["tost"]["bound"] == 0.02 and out["tost"]["p"] is not None


def test_control_guard_holds_and_fails():
    rng = np.random.default_rng(4)
    ctrl = [
        {"utt_id": f"u{i % 80}", "start": i, "end": i + 2, "itt_err_strict": int(rng.random() < 0.1)}
        for i in range(800)
    ]
    same = [dict(c) for c in ctrl]
    g = models.control_guard(ctrl, same, {c["utt_id"] for c in ctrl}, B=200, n_perm=200, seed=1)
    assert g["status"] == "evaluated" and g["holds"] is True and g["n_controls"] == 800
    worse = [{**c, "itt_err_strict": 1 if rng.random() < 0.12 else c["itt_err_strict"]} for c in ctrl]
    g2 = models.control_guard(ctrl, worse, {c["utt_id"] for c in ctrl}, B=200, n_perm=200, seed=1)
    assert g2["holds"] is False and g2["difference_L_minus_H"]["estimate"] > 0.05


def test_s2_cluster_bootstrap_coverage_on_clustered_rates():
    """Block bootstrap (Liu and Peng, 2020) interval for a rate: coverage near 0.95 when clusters share a random shift."""
    rng = np.random.default_rng(11)
    p, cover, reps = 0.3, 0, 150
    for r in range(reps):
        shifts = rng.uniform(-0.15, 0.15, size=60)
        y = np.concatenate([(rng.random(6) < p + s).astype(float) for s in shifts])
        cl = [f"c{i}" for i in range(60) for _ in range(6)]
        ci = models.cluster_bootstrap(lambda idx: y[idx].mean(), cl, B=200, seed=r)
        cover += ci["lo"] <= p <= ci["hi"]
    assert 0.87 <= cover / reps <= 0.99, cover / reps


def test_s5_tost_matches_statsmodels_ttost_paired():
    from statsmodels.stats.weightstats import ttost_paired

    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 0.2, 200)
    b = a + rng.normal(0.005, 0.1, 200)
    d = a - b
    se = d.std(ddof=1) / math.sqrt(d.size)
    p_ours = stats.tost_difference(d.mean(), se, 0.02, d.size)
    p_sm = ttost_paired(a, b, -0.02, 0.02)[0]
    assert abs(p_ours - p_sm) < 1e-9


def test_s6_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    ps = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20, "e": 0.001}
    ours = stats.holm(ps)
    sm = multipletests(list(ps.values()), method="holm")[1]
    for k, v in zip(ps, sm):
        assert abs(ours[k] - v) < 1e-12


def test_h1a_share_breakdown_and_itt():
    rows = [
        {
            "item_id": f"i{k}",
            "utt_id": f"u{k // 2}",
            "corpus": "hike",
            "R": 1,
            "itt_R": 1,
            "R_strict": int(k % 3 == 0),
            "F": "latin" if k % 2 else "hangul",
            "itt_F": "latin" if k % 2 else "hangul",
            "utt_flagged": 0,
        }
        for k in range(30)
    ]
    rows[0]["utt_flagged"] = 1
    rows[0]["itt_R"] = 0
    rows[0]["itt_F"] = None
    out = models.h1a(rows, B=100, seed=1)
    assert out["n"] == 30 and out["n_flagged_items"] == 1
    assert out["E_S"]["estimate"] >= out["E_A"]["estimate"] and 0 < out["share"]["estimate"] <= 1
    assert sum(out["difference_by_form"].values()) == sum(1 for r in rows if r["itt_R"] and not models.itt_strict(r))
    assert out["per_condition_exclusion"]["n"] == 29


def test_h2a_manipulation_check_direction():
    rng = np.random.default_rng(8)
    H = [
        {
            "item_id": f"i{k}",
            "utt_id": f"u{k // 3}",
            "type_key": f"t{k // 5}",
            "R": 1,
            "itt_R": 1,
            "F": "latin" if rng.random() < 0.3 else "hangul",
            "utt_flagged": 0,
        }
        for k in range(300)
    ]
    for h in H:
        h["itt_F"] = h["F"]
    L = [{**h, "F": "latin" if rng.random() < 0.7 else "hangul"} for h in H]
    for l in L:
        l["itt_F"] = l["F"]
    out = models.h2a(H, L, B=200, n_perm=300, seed=2)
    assert (
        out["n_recognized_both"] == 300 and out["rejected"] is True and out["difference_L_minus_H"]["estimate"] > 0.25
    )
    assert out["P_latin_given_R"]["L"] > out["P_latin_given_R"]["H"]


def test_cov_requires_the_recognizers_own_bpe_length():
    """No silent fallback to Whisper's tokenizer length."""
    cov = {"log_freq": 4.0, "freq_oov": 0, "n_syll": 2, "n_phones": 5, "n_bpe_whisper": 3}
    assert models._cov({}, cov, "whisper") is not None
    assert models._cov({}, cov, "owsm") is None
    assert models._cov({}, {**cov, "n_bpe_owsm": 4}, "owsm")["n_bpe"] == 4.0


def test_h2a_uses_intention_to_treat_fields():
    """A translated utterance under L (F=latin raw, itt_F=None) must not inflate P(latin | R)."""
    H = [
        {
            "item_id": f"i{k}",
            "utt_id": f"u{k}",
            "type_key": f"t{k}",
            "R": 1,
            "itt_R": 1,
            "F": "hangul",
            "itt_F": "hangul",
            "utt_flagged": 0,
        }
        for k in range(30)
    ]
    L = [
        {
            "item_id": f"i{k}",
            "utt_id": f"u{k}",
            "type_key": f"t{k}",
            "R": 1,
            "itt_R": 0,
            "F": "latin",
            "itt_F": None,
            "utt_flagged": 1,
        }
        for k in range(30)
    ]
    out = models.h2a(H, L, B=50, n_perm=100, seed=1)
    assert out["n_recognized_both"] == 0 and out["P_latin_given_R"]["L"] is None and out["rejected"] is False


def test_delta_permutation_is_type_level_and_bootstrap_resamples_clusters():
    """With type-level dependence, an item-level shuffle or item-level resampling would be anticonservative."""
    import numpy as np

    rng = np.random.default_rng(3)
    n_types, per_type = 60, 4
    rows_H, rows_L, direction, cov = [], [], [], {}
    for t in range(n_types):
        dom = "latin" if t % 2 == 0 else "hangul"
        direction.append({"type_key": f"t{t}", "dominant": dom, "fallback": 0})
        shift = rng.choice([-1, 0, 1])
        for j in range(per_type):
            iid = f"i{t}_{j}"
            utt = f"u{t}_{j}"
            rH = int(rng.random() < 0.5)
            rL = int(np.clip(rH + shift, 0, 1))
            rows_H.append(
                {
                    "item_id": iid,
                    "utt_id": utt,
                    "type_key": f"t{t}",
                    "R": rH,
                    "itt_R": rH,
                    "F": "hangul",
                    "itt_F": "hangul",
                    "utt_flagged": 0,
                }
            )
            rows_L.append(
                {
                    "item_id": iid,
                    "utt_id": utt,
                    "type_key": f"t{t}",
                    "R": rL,
                    "itt_R": rL,
                    "F": "latin",
                    "itt_F": "latin",
                    "utt_flagged": 0,
                }
            )
            cov[iid] = {"log_freq": 4.0, "freq_oov": 0, "n_syll": 2, "n_phones": 5, "n_bpe_whisper": 2}
    out = models.h2b(rows_H, rows_L, direction, cov, "whisper", None, None, B=100, n_perm=400, seed=2)
    assert out["delta"]["permutation_unit"] == "item type" and out["delta"]["n_perm_used"] > 0
    lat_u, lat_t = out["cell_differences"]["latin"], out["cell_differences_type_clustered"]["latin"]
    assert lat_t["n_clusters"] < lat_u["n_clusters"]
    assert (lat_t["hi"] - lat_t["lo"]) >= 0.8 * (lat_u["hi"] - lat_u["lo"])


def _res_all_good():
    return {
        "H1b": {"fit_note": "full model", "OR_is_item": {"OR": 1.5}},
        "consistency": {"H2a": {"agree": True}, "H2b": {"agree": True}},
        "H2a": {"pooled": {"rejected": True}},
        "H2b": {
            "pooled": {
                "guard": {"holds": True},
                "cell_differences": {"latin": {"estimate": 0.05, "p_permutation": 0.01}},
            },
            "mde_latin_cell": {"underpowered": False},
        },
        "sensitivity_g2p_acceptance": {
            "present": True,
            "sign_agreement": {"H1b": True, "H2a": True, "latin_cell": True},
        },
    }


def test_h1b_type_clustered_uses_word_identity_clusters():
    """Word-identity clustering: clusters are item types and control lemmas, the OR stays near the truth, and the row is
    labelled post hoc."""
    items, controls, icov, ccov = simulate_h1b(n_utt=300)
    for k in ccov:
        ccov[k] = {**ccov[k], "lemma": f"w{k[1] % 7}"}
    out = models.h1b(items, controls, icov, ccov, "whisper", B=50, seed=3, cluster_by="type")
    assert out["cluster_by"] == "type" and out["post_hoc"] is True and "word identity" in out["model"]
    n_lemmas = len({v["lemma"] for v in ccov.values()})
    assert out["n_clusters"] == 2 + n_lemmas
    assert 1.4 < out["OR_is_item"]["OR"] < 2.8, out["OR_is_item"]
    full = models.h1b(items, controls, icov, ccov, "whisper", B=50, seed=3)
    assert full["cluster_by"] == "utterance" and full["post_hoc"] is False and full["n_clusters"] == 300
    with pytest.raises(ValueError):
        models.h1b(items, controls, icov, ccov, "whisper", B=5, seed=3, cluster_by="speaker")


def test_h1b_word_level_filter_and_hangul_bpe_and_coefficients():
    """The level filter keeps items of the named level and only the comparison words of their utterances; bpe_form="hangul"
    reads the Hangul-form tokenizer length; the coefficient table is returned."""
    items, controls, icov, ccov = simulate_h1b(n_utt=200)
    for i, it in enumerate(items):
        it["cs_level"] = "word" if int(it["utt_id"][1:]) % 2 == 0 else "phrase"
    full = models.h1b(items, controls, icov, ccov, "whisper", B=30, seed=4)
    assert set(full["coefficients"]) == {"intercept", "is_item", *models.COVARIATES} and full["post_hoc"] is False
    word = models.h1b(items, controls, icov, ccov, "whisper", B=30, seed=4, utt_levels={"word"})
    w = word["waterfall"]
    assert (
        w["items"] + w["items_outside_level"] + w["items_missing_cov"] == len(items)
        and w["items_outside_level"] == len(items) // 2
    )
    assert (
        w["controls"] + w["controls_outside_level"] + w["controls_missing_cov"] == len(controls)
        and w["controls_outside_level"] == len(controls) // 2
    )
    assert (
        word["post_hoc"] is True
        and word["utt_levels"] == ["word"]
        and "word-level" in word["model"]
        and 1.2 < word["OR_is_item"]["OR"] < 3.5
    )
    hb = models.h1b(items, controls, icov, ccov, "whisper", B=30, seed=4, bpe_form="hangul")
    assert hb["waterfall"]["items_missing_cov"] == len(items) and hb["OR_is_item"] is None
    for k in icov:
        icov[k] = {**icov[k], "n_bpe_whisper_hangul": icov[k]["n_bpe_whisper"] + 1}
    for k in ccov:
        ccov[k] = {**ccov[k], "n_bpe_whisper_hangul": ccov[k]["n_bpe_whisper"]}
    hb = models.h1b(items, controls, icov, ccov, "whisper", B=30, seed=4, bpe_form="hangul")
    assert (
        hb["waterfall"]["items"] == len(items)
        and hb["bpe_form"] == "hangul"
        and hb["post_hoc"] is True
        and 1.2 < hb["OR_is_item"]["OR"] < 3.5
    )


def test_paired_difference_reports_the_same_sign_permutation_floor():
    """With k same-signed discordant clusters the sign-flip p cannot fall below 2/2^k; the output must flag this floor."""
    clusters = [f"u{i}" for i in range(37)] + ["u0", "u1", "u2", "u3", "u4"]
    x_l = [0] * 42
    x_h = [0] * 42
    for i in (5, 6, 7, 8):
        x_l[i] = 1
    d = models._paired_difference(x_l, x_h, clusters, B=200, n_perm=4000, seed=3)
    assert (
        d["n_discordant"] == 4
        and d["n_discordant_positive"] == 4
        and d["n_discordant_negative"] == 0
        and d["n_discordant_clusters"] == 4
    )
    assert d["p_floor_same_sign"] == 0.125 and d["discordance_same_sign"] is True and d["p_at_floor"] is True
    assert d["p_permutation"] >= 0.11 and d["lo_percentile"] >= 0.0
    x_h2 = list(x_h)
    x_h2[9] = 1
    d2 = models._paired_difference(x_l, x_h2, clusters, B=200, n_perm=4000, seed=3)
    assert d2["n_discordant_negative"] == 1 and d2["discordance_same_sign"] is False and d2["p_at_floor"] is False
    d3 = models._paired_difference(x_h, x_h, clusters, B=50, n_perm=200, seed=1)
    assert d3["n_discordant"] == 0 and d3["p_floor_same_sign"] is None and d3["p_at_floor"] is False


def test_h1b_like_with_like_and_flag_exclusion_rows():
    """Sensitivity rows that score items and comparison words alike, or drop flagged utterances, move the OR as expected."""
    items, controls, icov, ccov = simulate_h1b()
    base = models.h1b(items, controls, icov, ccov, "whisper", B=20, seed=1)
    tol = models.h1b(items, controls, icov, ccov, "whisper", B=20, seed=1, controls_scoring="tolerant")
    assert (
        tol["post_hoc"] is True
        and tol["controls_scoring"] == "tolerant"
        and abs(tol["OR_is_item"]["OR"] - base["OR_is_item"]["OR"]) < 1e-9
    )
    rescued = [dict(c, R_tolerant=1, itt_R_tolerant=1) for c in controls]
    tol2 = models.h1b(items, rescued, icov, ccov, "whisper", B=20, seed=1, controls_scoring="tolerant")
    assert tol2["error_controls"] == 0.0 and (
        tol2["OR_is_item"] is None or tol2["OR_is_item"]["OR"] > base["OR_is_item"]["OR"]
    )
    strict = models.h1b(items, controls, icov, ccov, "whisper", B=20, seed=1, items_scoring="strict")
    assert strict["items_scoring"] == "strict" and abs(strict["OR_is_item"]["OR"] - base["OR_is_item"]["OR"]) < 1e-9
    flagged_utts = {f"u{u}" for u in range(0, 400, 10)}
    items_f = [
        dict(r, utt_flagged=int(r["utt_id"] in flagged_utts), itt_R=0 if r["utt_id"] in flagged_utts else r["itt_R"])
        for r in items
    ]
    controls_f = [
        dict(
            c,
            utt_flagged=int(c["utt_id"] in flagged_utts),
            itt_err_strict=1 if c["utt_id"] in flagged_utts else c["itt_err_strict"],
        )
        for c in controls
    ]
    itt = models.h1b(items_f, controls_f, icov, ccov, "whisper", B=20, seed=1)
    unfl = models.h1b(items_f, controls_f, icov, ccov, "whisper", B=20, seed=1, exclude_flagged=True)
    assert (
        unfl["waterfall"]["items_flagged_excluded"] == 80
        and unfl["waterfall"]["controls_flagged_excluded"] == 160
        and unfl["waterfall"]["items"] == 720
    )
    assert (
        unfl["exclude_flagged"] is True
        and unfl["post_hoc"] is True
        and itt["OR_is_item"]["OR"] < base["OR_is_item"]["OR"] + 0.3
    )
    import pytest

    with pytest.raises(KeyError):
        models.h1b(
            items,
            [dict(c) for c in controls if not c.pop("itt_R_tolerant", None) or True][:0]
            or [{k: v for k, v in controls[0].items() if k != "itt_R_tolerant"}],
            icov,
            ccov,
            "whisper",
            B=5,
            seed=1,
            controls_scoring="tolerant",
        )


def test_l_only_variant_candidates_are_counted_in_both_cells():
    """The Hangul-substitution count for L-only discordant items exists for both cells; the Latin entry equals the Latin-cell
    diagnostic."""
    H, L, direction, icov = simulate_h2b()
    for h in H:
        if h["itt_R"] == 0:
            h.update({"outcome": "SUB", "hyp_material": "카페" if int(h["item_id"][1:]) % 2 else "cafe"})
    out = models.h2b(H, L, direction, icov, "whisper", None, None, B=50, n_perm=50, seed=5)
    by = out["L_only_variant_candidates_by_cell"]
    assert by["latin"] == out["latin_cell_L_only_variant_candidates"] and by["post_hoc"] is True
    assert by["hangul"]["n_L_only"] > 0 and 0 < by["hangul"]["n_H_hangul_sub"] < by["hangul"]["n_L_only"]


def test_h1b_cluster_key_keeps_covariate_lookup(monkeypatch):
    """The cluster field is a parameter (the math project clusters by video); the covariate lookup key is not."""
    import numpy as np

    from loanword_eval.models import h1b

    rng = np.random.default_rng(0)
    items, controls, icov, ccov = [], [], {}, {}
    for i in range(40):
        u, v = f"u{i}", f"v{i % 5}"
        items.append(
            {
                "item_id": f"it{i}",
                "utt_id": u,
                "video_id": v,
                "itt_R": int(rng.random() > 0.6),
                "R_strict": 0,
                "utt_flagged": 0,
            }
        )
        icov[f"it{i}"] = {"log_freq": 3.0, "freq_oov": 0, "n_syll": 2, "n_phones": 5, "n_bpe_whisper": 2}
        for k in range(3):
            controls.append(
                {
                    "utt_id": u,
                    "video_id": v,
                    "start": k,
                    "end": k,
                    "itt_err_strict": int(rng.random() > 0.8),
                    "utt_flagged": 0,
                }
            )
            ccov[(u, k, k)] = {"log_freq": 4.0, "freq_oov": 0, "n_syll": 2, "n_phones": 5, "n_bpe_whisper": 1}
    r = h1b(items, controls, icov, ccov, "whisper", B=50, seed=1, cluster_key="video_id")
    assert r["waterfall"]["controls"] == 120 and r["waterfall"]["controls_missing_cov"] == 0 and r["n_clusters"] == 5
    r0 = h1b(items, controls, icov, ccov, "whisper", B=50, seed=1)
    assert r0["n_clusters"] == 40
