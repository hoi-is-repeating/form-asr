import numpy as np

from loanword_eval.corpora.audio import check, stats


def test_stats_duration_and_clean_signal():
    y = (np.sin(np.linspace(0, 200, 16000)) * 0.3).astype(np.float32)
    assert y.dtype == np.float32 and y.size == 16000 and abs(stats(y).duration_s - 1.0) < 1e-9
    assert check(y) == []


def test_check_flags_long_silent_clipped():
    assert any(p.startswith("long_form") for p in check(np.zeros(16000 * 31, dtype=np.float32) + 0.1))
    assert any(p.startswith("silent") for p in check(np.zeros(16000, dtype=np.float32)))
    assert any(p.startswith("clipping") for p in check(np.ones(16000, dtype=np.float32)))


def test_hash_is_content_based():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    assert stats(a).sha256 == stats(b).sha256
    b[0] = 0.5
    assert stats(a).sha256 != stats(b).sha256
