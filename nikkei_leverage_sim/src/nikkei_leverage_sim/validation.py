"""Statistical validation (roadmap item A — "is the edge real or data-snooping?").

Walk-forward + random search over many parameters invites multiple-comparison
and overfitting bias, and a no-stop-loss take-profit strategy is especially good
at *looking* profitable by chance.  This module provides four guards, all
dependency-light (numpy / pandas / the existing engine) and reproducible:

1. :func:`purged_kfold_indices` — time-series cross-validation splits with an
   embargo (gap) around each test fold so train/test do not leak across the
   boundary.
2. :func:`permutation_test` — shuffle the (paired) daily returns to destroy all
   timing structure, re-run the strategy, and ask how often chance beats the
   real result.  A small p-value means the strategy exploits real structure.
3. :func:`bootstrap_metric_ci` — moving-block bootstrap confidence interval for a
   performance metric (so a point estimate is never reported naked).
4. :func:`benjamini_hochberg` — Benjamini-Hochberg false-discovery-rate control
   across many tested configurations.

The permutation / bootstrap re-runs use a FIXED parameter set (no walk-forward),
so they test a *given strategy's* edge, not the optimizer's search; this is the
tractable and honest scope and is documented in the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .backtest import prepare_market_data, simulate
from .config import Config, StrategyParams
from .data import join_target_benchmark
from .metrics import annualized_return, daily_returns
from .stress import _bar_ratios, _synthetic_frame


# --------------------------------------------------------------------------- #
# 1. Purged / embargoed K-fold time-series splits
# --------------------------------------------------------------------------- #
def purged_kfold_indices(
    n: int, n_splits: int = 5, embargo: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous time-series CV folds with a symmetric embargo gap.

    The timeline ``[0, n)`` is cut into ``n_splits`` contiguous test blocks; for
    each, the training set is everything outside the test block **and** outside an
    ``embargo``-row gap on either side of it (so a decision whose multi-day
    outcome straddles the boundary cannot leak between train and test).
    """
    if n_splits < 2 or n_splits > n:
        raise ValueError("n_splits must be in [2, n]")
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        t0, t1 = bounds[i], bounds[i + 1]
        test = np.arange(t0, t1)
        lo = max(0, t0 - embargo)
        hi = min(n, t1 + embargo)
        train = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        folds.append((train, test))
    return folds


# --------------------------------------------------------------------------- #
# 2. Permutation test (timing-skill significance)
# --------------------------------------------------------------------------- #
MetricFn = Callable[[object], float]


def net_realized_metric(sim_result) -> float:
    return float(sim_result.portfolio.realized_after_tax)


def annual_return_metric(sim_result) -> float:
    eq = sim_result.equity_curve
    return annualized_return(eq, len(eq))


@dataclass
class PermutationResult:
    metric_name: str
    observed: float
    p_value: float
    n_perm: int
    null_mean: float
    null_p95: float
    seed: int
    null: List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("null", None)
        return d


def _split_target_benchmark(joined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = joined[[c for c in joined.columns if c.startswith("target_")]].rename(
        columns=lambda c: c[len("target_"):].title().replace("Adj_Close", "Adj Close")
    )
    benchmark = joined[[c for c in joined.columns if c.startswith("benchmark_")]].rename(
        columns=lambda c: c[len("benchmark_"):].title().replace("Adj_Close", "Adj Close")
    )
    return target, benchmark


def permutation_test(
    joined: pd.DataFrame,
    cfg: Config,
    params: StrategyParams,
    *,
    n_perm: int = 200,
    seed: int = 42,
    metric_fn: MetricFn = net_realized_metric,
    metric_name: str = "net_realized_after_tax",
) -> PermutationResult:
    """One-sided permutation test that the strategy beats timing-shuffled chance.

    The observed metric is computed on the real series.  Each permutation applies
    the SAME random reordering to the paired (target, benchmark) daily bars
    (preserving their cross-sectional relationship while destroying the temporal
    ordering / autocorrelation the strategy might exploit), rebuilds prices, and
    re-runs the strategy with the fixed ``params``.  The p-value is the
    right-tail rank of the observed metric among the null,
    ``(1 + #{null >= observed}) / (n_perm + 1)`` (never zero).
    """
    md = prepare_market_data(joined, cfg)
    observed = metric_fn(simulate(md, 0, md.n, params, cfg, record=False))

    target, benchmark = _split_target_benchmark(joined)
    t_ratios = _bar_ratios(target)
    b_ratios = _bar_ratios(benchmark)
    t_start = float(target["Close"].iloc[0])
    b_start = float(benchmark["Close"].iloc[0])
    n = len(joined)
    base_index = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=n))
    rng = np.random.default_rng(seed)

    null: List[float] = []
    for _ in range(n_perm):
        # Reorder rows [1, n) (row 0 has no close-to-close return); prepend the
        # zero-return anchor row 0 so the null path has exactly ``n`` bars,
        # matching the observed run's horizon.
        perm = np.concatenate(([0], 1 + rng.permutation(n - 1)))
        t_df = _synthetic_frame(base_index, t_ratios, perm, t_start)
        b_df = _synthetic_frame(base_index, b_ratios, perm, b_start)
        md_p = prepare_market_data(join_target_benchmark(t_df, b_df), cfg)
        null.append(metric_fn(simulate(md_p, 0, md_p.n, params, cfg, record=False)))

    null_arr = np.asarray(null, dtype=float)
    p_value = float((1 + int(np.sum(null_arr >= observed))) / (n_perm + 1))
    return PermutationResult(
        metric_name=metric_name,
        observed=float(observed),
        p_value=p_value,
        n_perm=n_perm,
        null_mean=float(null_arr.mean()) if null_arr.size else 0.0,
        null_p95=float(np.percentile(null_arr, 95)) if null_arr.size else 0.0,
        seed=seed,
        null=null,
    )


# --------------------------------------------------------------------------- #
# 3. Block-bootstrap confidence interval for a metric
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapCI:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_boot: int
    block_size: int
    seed: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def bootstrap_metric_ci(
    equity_curve: Sequence[float],
    *,
    n_boot: int = 500,
    block_size: int = 20,
    confidence: float = 0.95,
    seed: int = 42,
    periods: int = 252,
    metric_name: str = "annualized_return",
) -> BootstrapCI:
    """Moving-block bootstrap CI for the annualized return of an equity curve."""
    rets = daily_returns(equity_curve)
    n = rets.size
    point = annualized_return(equity_curve, len(equity_curve))
    if n < 2:
        return BootstrapCI(metric_name, point, point, point, confidence, n_boot, block_size, seed)

    rng = np.random.default_rng(seed)
    max_start = max(1, n - block_size)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx: List[int] = []
        while len(idx) < n:
            s = int(rng.integers(0, max_start + 1))
            idx.extend(range(s, min(s + block_size, n)))
        r = rets[np.asarray(idx[:n], dtype=int)]
        growth = float(np.prod(1.0 + r))
        samples[b] = growth ** (periods / n) - 1.0 if growth > 0 else -1.0

    tail = (1.0 - confidence) / 2.0
    lo = float(np.percentile(samples, tail * 100.0))
    hi = float(np.percentile(samples, (1.0 - tail) * 100.0))
    return BootstrapCI(metric_name, point, lo, hi, confidence, n_boot, block_size, seed)


# --------------------------------------------------------------------------- #
# 4. Benjamini-Hochberg false-discovery-rate control
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR procedure.

    Returns ``(reject, threshold)`` where ``reject[i]`` is True if hypothesis ``i``
    is rejected at FDR ``alpha`` and ``threshold`` is the largest p-value that
    passes (0.0 if none).
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    crit = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= crit
    if not passed.any():
        return np.zeros(m, dtype=bool), 0.0
    kmax = int(np.max(np.where(passed)[0]))
    threshold = float(ranked[kmax])
    return p <= threshold, threshold


# --------------------------------------------------------------------------- #
# Driver helpers + output writing
# --------------------------------------------------------------------------- #
def segment_returns(equity_curve: Sequence[float], n_splits: int = 5) -> List[float]:
    """Per-segment annualized return of one fixed-strategy equity curve.

    This is a **cross-era consistency check, NOT train/test cross-validation**: it
    simply cuts the (single) equity curve into ``n_splits`` equal contiguous
    segments and annualizes each, to see whether the result is driven by one
    lucky era or is broadly consistent.  There is no training step (the strategy
    is fixed), so an embargo is irrelevant to these numbers.  For true optimizer
    cross-validation use :func:`purged_kfold_indices` (the embargoed splitter
    provided here) to separate train/test windows.
    """
    eq = np.asarray(equity_curve, dtype=float)
    n = eq.size
    if n < n_splits + 1:
        return []
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    out: List[float] = []
    for i in range(n_splits):
        seg = eq[bounds[i]:bounds[i + 1]]
        out.append(annualized_return(seg, len(seg)))
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_validation_outputs(out_dir, summary: Dict[str, object], permutation: PermutationResult, charts: bool = True):
    """Write validation.json (+ null-distribution chart) and return the summary."""
    from pathlib import Path
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if charts and permutation.null:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(np.asarray(permutation.null) / 1e6, bins=40, color="#1f77b4", alpha=0.8)
        ax.axvline(
            permutation.observed / 1e6, color="red", linestyle="--",
            label=f"observed (p={permutation.p_value:.3f})",
        )
        ax.set_title("Permutation null vs observed (net realized after tax)")
        ax.set_xlabel("net realized after tax (M JPY)")
        ax.set_ylabel("permutations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "permutation_null.png", dpi=110)
        plt.close(fig)
    return summary
