"""Tests for the statistical-validation module (A)."""
from __future__ import annotations

import numpy as np
import pytest

from nikkei_leverage_sim.backtest import prepare_market_data, simulate
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim import validation as V


# --- purged K-fold --------------------------------------------------------- #
def test_purged_kfold_covers_all_test_indices_once():
    folds = V.purged_kfold_indices(100, n_splits=5, embargo=0)
    assert len(folds) == 5
    all_test = np.concatenate([te for _tr, te in folds])
    assert np.array_equal(np.sort(all_test), np.arange(100))  # exact partition


def test_purged_kfold_embargo_excluded_from_train():
    folds = V.purged_kfold_indices(100, n_splits=5, embargo=4)
    for train, test in folds:
        gap = set(range(test.min() - 4, test.max() + 5))
        assert gap.isdisjoint(set(train.tolist()))
        assert set(test.tolist()).isdisjoint(set(train.tolist()))


def test_purged_kfold_validates_n_splits():
    with pytest.raises(ValueError):
        V.purged_kfold_indices(10, n_splits=1)
    with pytest.raises(ValueError):
        V.purged_kfold_indices(3, n_splits=5)


# --- Benjamini-Hochberg ---------------------------------------------------- #
def test_benjamini_hochberg_known_case():
    reject, thr = V.benjamini_hochberg([0.001, 0.2, 0.03, 0.5, 0.04], alpha=0.1)
    assert reject.tolist() == [True, False, True, False, True]
    assert abs(thr - 0.04) < 1e-12


def test_benjamini_hochberg_none_and_empty():
    reject, thr = V.benjamini_hochberg([0.9, 0.8, 0.95], alpha=0.05)
    assert not reject.any() and thr == 0.0
    reject2, thr2 = V.benjamini_hochberg([], alpha=0.05)
    assert reject2.size == 0 and thr2 == 0.0


# --- permutation test ------------------------------------------------------ #
def _small_joined(n=250, seed=7):
    t, b = make_synthetic_data(n_days=n, seed=seed)
    return join_target_benchmark(t, b)


def test_permutation_test_pvalue_range_and_determinism():
    joined = _small_joined()
    cfg = Config()
    cfg.optimization.enabled = False
    params = cfg.strategy.default_params
    a = V.permutation_test(joined, cfg, params, n_perm=25, seed=3)
    b = V.permutation_test(joined, cfg, params, n_perm=25, seed=3)
    assert 0.0 < a.p_value <= 1.0
    assert a.p_value == b.p_value  # deterministic given seed
    assert len(a.null) == 25
    # p-value floor: (1 + k) / (n_perm + 1), so >= 1/26.
    assert a.p_value >= 1.0 / 26.0


# --- bootstrap CI ---------------------------------------------------------- #
def test_bootstrap_ci_ordering_and_determinism():
    joined = _small_joined()
    cfg = Config()
    cfg.optimization.enabled = False
    md = prepare_market_data(joined, cfg)
    res = simulate(md, 0, md.n, cfg.strategy.default_params, cfg)
    a = V.bootstrap_metric_ci(res.equity_curve, n_boot=80, seed=5)
    b = V.bootstrap_metric_ci(res.equity_curve, n_boot=80, seed=5)
    assert a.ci_low <= a.ci_high
    assert a.ci_low == b.ci_low and a.ci_high == b.ci_high
    assert a.confidence == 0.95


def test_bootstrap_ci_degenerate_short_curve():
    ci = V.bootstrap_metric_ci([100.0], n_boot=10, seed=1)
    assert ci.ci_low == ci.ci_high == ci.point_estimate


# --- driver helpers + output ---------------------------------------------- #
def test_segment_returns_length():
    joined = _small_joined()
    cfg = Config()
    cfg.optimization.enabled = False
    md = prepare_market_data(joined, cfg)
    res = simulate(md, 0, md.n, cfg.strategy.default_params, cfg)
    segs = V.segment_returns(res.equity_curve, n_splits=5)
    assert len(segs) == 5


def test_permutation_pvalue_tail_direction():
    # A metric where the OBSERVED (first call) is the unique maximum must give the
    # p-value floor; where it is the unique minimum must give p == 1.0. This pins
    # the one-sided upper-tail direction.
    joined = _small_joined(n=200)
    cfg = Config()
    cfg.optimization.enabled = False
    params = cfg.strategy.default_params

    calls = {"n": 0}

    def winner(_res):  # observed first, then nulls
        calls["n"] += 1
        return 100.0 if calls["n"] == 1 else 0.0

    pr = V.permutation_test(joined, cfg, params, n_perm=20, seed=1, metric_fn=winner)
    assert pr.p_value == pytest.approx(1.0 / 21.0)  # (1 + 0) / (20 + 1)

    calls["n"] = 0

    def loser(_res):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 100.0

    pr2 = V.permutation_test(joined, cfg, params, n_perm=20, seed=1, metric_fn=loser)
    assert pr2.p_value == pytest.approx(1.0)  # (1 + 20) / 21


def test_write_validation_outputs(tmp_path):
    joined = _small_joined()
    cfg = Config()
    cfg.optimization.enabled = False
    perm = V.permutation_test(joined, cfg, cfg.strategy.default_params, n_perm=10, seed=1)
    summary = {"permutation": perm.to_dict()}
    V.write_validation_outputs(tmp_path, summary, perm)
    assert (tmp_path / "validation.json").exists()
    assert (tmp_path / "permutation_null.png").exists()
