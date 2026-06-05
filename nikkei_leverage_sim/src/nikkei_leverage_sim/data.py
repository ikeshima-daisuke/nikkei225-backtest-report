"""Data loading utilities.

The backtest engine only ever consumes a *joined* DataFrame produced by
:func:`load_market_data`.  Data can come from CSV files (the canonical input,
works fully offline) or, optionally, from yfinance via :func:`fetch_to_csv`.

A synthetic data generator (:func:`make_synthetic_data`) is provided so tests
and quick demos can run with no network access.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Price columns rescaled when an isolated glitch bar is repaired (Volume is left
# untouched — it is not part of the price-level anomaly).
_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]


def repair_price_glitches(
    df: pd.DataFrame,
    *,
    move_threshold: float = 0.40,
    revert_tol: float = 0.10,
    max_run: int = 3,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Repair *isolated* price glitches (a level that teleports and reverts).

    A vendor data feed occasionally emits a short run of bars at a wildly wrong
    level (e.g. 1570.T printing ~¥8,040 for two sessions in 2021-04 between
    ¥16,250 and ¥15,900 closes — a fake ‑50%/+100% round-trip).  Such a glitch
    is invisible to a naive P&L total but it (a) corrupts every tail-risk metric
    and (b) fires spurious draw-down-based buy signals for weeks afterwards.

    This is an **offline data-hygiene step**, not part of the live signal path:
    like any historical data cleaning it inspects neighbouring bars to *identify*
    a vendor misprint, but the value it *writes* is causal — the last known-good
    close (forward-fill), never a future price — so no future price level leaks
    into the indicators.  The trading loop's no-look-ahead contract
    (``close_t`` decides → ``open_{t+1}`` executes) is untouched.

    The detector is deliberately conservative so it never touches a *real* crash:
    a run of up to ``max_run`` bars is repaired only when **every** bar in the run
    deviates from the last good close by more than ``move_threshold`` *in the same
    direction* **and** the price snaps back — the bar right after the run is within
    ``revert_tol`` (default 10%) of the bar right before it.  A genuine crash does
    not move >40% and return to within 10% of its pre-crash level inside three
    sessions, so it is left alone.

    Repaired bars are set to the last known-good close (forward-fill); the whole
    OHLC bar is rescaled by that factor to keep its intraday shape.

    Returns ``(repaired_df, repairs)`` where ``repairs`` is a list of
    ``{"date", "original_close", "repaired_close"}`` dicts (empty when clean).
    """
    if df.empty:
        return df, []
    close = df["Close"].to_numpy(dtype=float).copy()
    n = len(close)
    lo = 1.0 - move_threshold
    hi = 1.0 + move_threshold
    repairs: List[Dict[str, object]] = []
    targets: Dict[int, float] = {}

    i = 1
    while i < n - 1:
        matched = False
        for run in range(1, max_run + 1):
            j = i + run  # index of the bar just after the run
            if j >= n:
                break
            pre = close[i - 1]
            post = close[j]
            if pre <= 0 or post <= 0:
                continue
            seg = close[i:j]
            if not np.all(np.isfinite(seg)) or np.any(seg <= 0):
                continue
            down = bool(np.all(seg < lo * pre))
            up = bool(np.all(seg > hi * pre))
            reverts = (1.0 - revert_tol) <= (post / pre) <= (1.0 + revert_tol)
            if (down or up) and reverts:
                # Forward-fill with the last known-good close (causal value).
                for k in range(run):
                    targets[i + k] = pre
                    close[i + k] = pre  # so later brackets read repaired values
                i = j
                matched = True
                break
        if not matched:
            i += 1

    if not targets:
        return df, []

    orig_close = df["Close"].to_numpy(dtype=float)
    out = df.copy()
    for idx, new_close in targets.items():
        oc = orig_close[idx]
        factor = (new_close / oc) if oc else 1.0
        date = df.index[idx]
        for col in _PRICE_COLUMNS:
            out.iloc[idx, out.columns.get_loc(col)] = df.iloc[idx][col] * factor
        repairs.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "original_close": float(oc),
                "repaired_close": float(new_close),
            }
        )
    return out, repairs


def read_ohlc_csv(path: str | Path, repair_glitches: bool = True) -> pd.DataFrame:
    """Read a single OHLCV CSV and return a frame indexed by ``Date``.

    Validates that all required columns are present.  Unless ``repair_glitches``
    is ``False``, isolated price glitches are repaired (see
    :func:`repair_price_glitches`); a warning is emitted and the repair record is
    stashed on ``df.attrs["data_repairs"]`` for audit.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} is missing required 'Date' column")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    df = df[REQUIRED_COLUMNS].astype(float)

    repairs: List[Dict[str, object]] = []
    if repair_glitches:
        df, repairs = repair_price_glitches(df)
        if repairs:
            warnings.warn(
                f"{path.name}: repaired {len(repairs)} isolated price glitch "
                f"bar(s): {[r['date'] for r in repairs]}",
                stacklevel=2,
            )
    df.attrs["data_repairs"] = repairs
    return df


def join_target_benchmark(
    target: pd.DataFrame, benchmark: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join target and benchmark on the date index.

    Only trading days present in *both* series are kept (per the spec).
    Columns are prefixed ``target_`` / ``benchmark_``.
    """
    t = target.rename(columns=lambda c: "target_" + c.lower().replace(" ", "_"))
    b = benchmark.rename(columns=lambda c: "benchmark_" + c.lower().replace(" ", "_"))
    joined = t.join(b, how="inner").sort_index()
    if joined.empty:
        raise ValueError("No overlapping trading days between target and benchmark")
    # Carry any data-repair audit records forward, labelled by series, since the
    # join does not preserve the inputs' ``.attrs``.
    repairs: List[Dict[str, object]] = []
    for series_name, frame in (("target", target), ("benchmark", benchmark)):
        for rec in getattr(frame, "attrs", {}).get("data_repairs", []) or []:
            repairs.append({"series": series_name, **rec})
    joined.attrs["data_repairs"] = repairs
    return joined


def load_market_data(
    target_csv: str | Path,
    benchmark_csv: str | Path,
    repair_glitches: bool = True,
) -> pd.DataFrame:
    """Load both CSVs and return the joined frame (no indicators yet)."""
    target = read_ohlc_csv(target_csv, repair_glitches=repair_glitches)
    benchmark = read_ohlc_csv(benchmark_csv, repair_glitches=repair_glitches)
    return join_target_benchmark(target, benchmark)


def make_synthetic_data(
    n_days: int = 600,
    seed: int = 0,
    start: str = "2019-01-01",
    leverage: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic benchmark + leveraged-target OHLCV data.

    The benchmark follows a geometric random walk with mild drift and
    occasional draw-down regimes.  The target tracks the *daily* benchmark
    return times ``leverage`` (the classic daily-rebalanced leverage ETF
    behaviour, including volatility decay), so it is a realistic stand-in for
    1570.T.

    Returns ``(target_df, benchmark_df)``, each indexed by ``Date`` with the
    standard OHLCV columns, ready for :func:`join_target_benchmark`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Benchmark daily log-returns with a couple of stress regimes.
    mu = 0.0003
    vol = np.full(n_days, 0.011)
    # Inject two higher-volatility / negative-drift draw-down windows.
    for lo, hi in [(n_days // 5, n_days // 5 + 30), (n_days // 2, n_days // 2 + 25)]:
        vol[lo:hi] = 0.025
    shocks = rng.normal(0.0, 1.0, n_days)
    drift = np.full(n_days, mu)
    drift[n_days // 5 : n_days // 5 + 30] = -0.004
    drift[n_days // 2 : n_days // 2 + 25] = -0.005
    bench_ret = drift + vol * shocks

    bench_close = 20000.0 * np.exp(np.cumsum(bench_ret))
    target_ret = leverage * bench_ret  # daily-rebalanced leverage
    target_close = 15000.0 * np.exp(np.cumsum(target_ret))

    def _ohlc(close: np.ndarray, day_vol: np.ndarray) -> pd.DataFrame:
        # Build plausible OHLC around the close path.
        open_ = np.empty_like(close)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.002, n_days - 1))
        intraday = np.abs(rng.normal(0.0, day_vol, n_days)) * close
        high = np.maximum(open_, close) + intraday * 0.5
        low = np.minimum(open_, close) - intraday * 0.5
        volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        return pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            },
            index=pd.Index(dates, name="Date"),
        )

    target_df = _ohlc(target_close, np.abs(target_ret) + 0.005)
    benchmark_df = _ohlc(bench_close, vol)
    return target_df, benchmark_df


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write an OHLCV frame (indexed by Date) to CSV with a Date column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().rename(columns={"index": "Date"}).to_csv(path, index=False)


def fetch_to_csv(
    tickers: dict[str, str], start: str, end: str, out_dir: str | Path
) -> None:
    """Fetch OHLCV data with yfinance and write CSVs.

    ``tickers`` maps a friendly filename stem to a yfinance symbol, e.g.
    ``{"target_1570": "1570.T", "benchmark_n225": "^N225"}``.

    This is a *convenience* helper only; the backtest itself never requires
    network access.  yfinance is imported lazily so the package works without
    it installed.
    """
    try:
        import yfinance as yf  # noqa: WPS433  (lazy import on purpose)
    except Exception as exc:  # pragma: no cover - network/optional dep
        raise RuntimeError(
            "yfinance is required for `fetch`; install it or supply CSVs"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, symbol in tickers.items():
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:  # pragma: no cover - network dependent
            raise RuntimeError(f"No data returned for {symbol}")
        # yfinance may return a MultiIndex on columns for single tickers.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[REQUIRED_COLUMNS]
        write_csv(df, out_dir / f"{stem}.csv")
