import numpy as np

from loanword_eval.stats import holm, paired_permutation, paired_rate_difference


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(0)
    clusters = [f"s{i // 10}" for i in range(400)]
    a = (rng.random(400) < 0.25).astype(int)
    b = (rng.random(400) < 0.15).astype(int)
    ci = paired_rate_difference(a, b, clusters, B=500, seed=1)
    t = ci["bootstrap_t"]
    assert t.lo < 0.10 < t.hi and t.n_clusters == 40
    assert abs(t.estimate - (a.mean() - b.mean())) < 1e-12


def test_bootstrap_seed_reproducible():
    a = [1, 0, 1, 0, 1, 1, 0, 0]
    b = [0, 0, 1, 0, 0, 1, 0, 0]
    c = ["x", "x", "y", "y", "z", "z", "w", "w"]
    r1 = paired_rate_difference(a, b, c, B=200, seed=5)
    r2 = paired_rate_difference(a, b, c, B=200, seed=5)
    assert r1["percentile"] == r2["percentile"]


def test_permutation_identical_arrays_p_near_one_and_signal_detected():
    a = [1, 0, 1, 1, 0, 1, 0, 1]
    assert paired_permutation(a, a, n_perm=500, seed=0) > 0.99
    rng = np.random.default_rng(1)
    x = (rng.random(300) < 0.4).astype(int)
    y = (rng.random(300) < 0.1).astype(int)
    assert paired_permutation(x, y, n_perm=2000, seed=0) < 0.01


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] <= adj["c"] <= adj["b"] <= 1.0 and abs(adj["a"] - 0.03) < 1e-12


def test_paired_permutation_flips_signs_per_cluster_not_per_item():
    """Under an utterance-level null (a whole-utterance shift with zero mean across utterances), per-item
    flips reject far too often; per-cluster flips keep the size near alpha."""
    import numpy as np

    rng = np.random.default_rng(0)
    n_utt, per = 100, 4
    rej_item = rej_cluster = 0
    for rep in range(60):
        shift = rng.choice([-1, 1], size=n_utt)
        d = np.repeat(shift, per)
        a = np.where(d > 0, 1, 0)
        b = np.where(d < 0, 1, 0)
        cl = [f"u{i}" for i in range(n_utt) for _ in range(per)]
        rej_item += paired_permutation(a, b, n_perm=400, seed=rep) < 0.05
        rej_cluster += paired_permutation(a, b, cl, n_perm=400, seed=rep) < 0.05
    assert rej_item / 60 > 0.2, rej_item
    assert rej_cluster / 60 < 0.12, rej_cluster


def test_bootstrap_t_not_estimable_under_sparse_discordance_falls_back_to_percentile():
    """Regression: 2 discordant controls among 800 gave a bootstrap-t interval of +-10^6 through a 1e-12 floor."""
    a = [0] * 800
    b = [0] * 800
    b[3] = 1
    b[400] = 1
    cl = [f"u{i // 10}" for i in range(800)]
    out = paired_rate_difference(a, b, cl, B=200, seed=1)
    t = out["bootstrap_t"]
    assert out["n_degenerate_resamples"] > 0
    assert -0.05 < t.lo <= t.hi < 0.05, t
    a2 = [0] * 40
    b2 = [0] * 40
    out2 = paired_rate_difference(a2, b2, [f"u{i // 10}" for i in range(40)], B=100, seed=2)
    assert "percentile" in out2["bootstrap_t"].method and out2["bootstrap_t"].lo == out2["bootstrap_t"].hi == 0.0
