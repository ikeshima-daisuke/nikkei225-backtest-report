"""Data loading utilities.

The backtest engine only ever consumes a *joined* DataFrame produced by
:func:`load_market_data`.  Data can come from CSV files (the canonical input,
works fully offline) or, optionally, from yfinance via :func:`fetch_to_csv`.

A synthetic data generator (:func:`make_synthetic_data`) is provided so tests
and quick demos can run with no network access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def read_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """Read a single OHLCV CSV and return a frame indexed by ``Date``.

    Validates that all required columns are present.
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
    return df[REQUIRED_COLUMNS].astype(float)


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
    return joined


def load_market_data(
    target_csv: str | Path, benchmark_csv: str | Path
) -> pd.DataFrame:
    """Load both CSVs and return the joined frame (no indicators yet)."""
    target = read_ohlc_csv(target_csv)
    benchmark = read_ohlc_csv(benchmark_csv)
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
