"""Command line interface.

Examples
--------
Fetch data with yfinance (convenience only, needs network)::

    python -m nikkei_leverage_sim.cli fetch --start 2019-01-01 --end 2026-12-31 --out data/

Generate offline synthetic CSVs::

    python -m nikkei_leverage_sim.cli synth --out data/ --days 900 --seed 7

Run a backtest from CSVs::

    python -m nikkei_leverage_sim.cli run --config examples/sample_config.yaml \
        --target data/target_1570.csv --benchmark data/benchmark_n225.csv --out outputs/

Run a backtest on freshly generated synthetic data (no network/CSV needed)::

    python -m nikkei_leverage_sim.cli run --config examples/sample_config.yaml --synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import data as data_mod
from .backtest import prepare_market_data, run_backtest
from .config import load_config
from .data import join_target_benchmark, load_market_data, make_synthetic_data
from .reporting import write_outputs


def _cmd_fetch(args: argparse.Namespace) -> int:
    tickers = {
        f"target_{args.target_symbol.replace('.', '_').replace('^', '')}": args.target_symbol,
        f"benchmark_{args.benchmark_symbol.replace('.', '_').replace('^', '')}": args.benchmark_symbol,
    }
    data_mod.fetch_to_csv(tickers, args.start, args.end, args.out)
    print(f"Fetched {list(tickers.values())} -> {args.out}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    target_df, benchmark_df = make_synthetic_data(
        n_days=args.days, seed=args.seed, start=args.start
    )
    out = Path(args.out)
    data_mod.write_csv(target_df, out / "target_synthetic.csv")
    data_mod.write_csv(benchmark_df, out / "benchmark_synthetic.csv")
    print(f"Wrote synthetic CSVs ({args.days} rows) to {out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    # CLI overrides (handy for running the same config under several scenarios
    # without duplicating YAML files).  ``None`` means "leave the config value".
    if args.seed is not None:
        cfg.optimization.random_seed = int(args.seed)
    if args.force_liquidation is not None:
        cfg.force_liquidation = bool(args.force_liquidation)
    if args.initial_equity is not None:
        cfg.initial_equity = float(args.initial_equity)

    if args.synthetic:
        target_df, benchmark_df = make_synthetic_data(
            n_days=args.synthetic_days, seed=args.synthetic_seed
        )
        joined = join_target_benchmark(target_df, benchmark_df)
    else:
        if not args.target or not args.benchmark:
            print(
                "error: --target and --benchmark CSVs are required "
                "(or pass --synthetic)",
                file=sys.stderr,
            )
            return 2
        joined = load_market_data(args.target, args.benchmark)

    md = prepare_market_data(joined, cfg)
    print(f"Loaded {md.n} rows ({int(md.valid.sum())} tradable). Running backtest...")
    result = run_backtest(md, cfg)
    summary = write_outputs(result, args.out)

    print(f"\nDone. Outputs written to {Path(args.out).resolve()}")
    print("--- Summary (selected) ---")
    for key in (
        "final_equity",
        "net_realized_profit_after_tax",
        "ending_unrealized_pnl",
        "total_interest_paid",
        "total_tax_paid",
        "buy_trade_count",
        "sell_trade_count",
        "win_rate_of_closed_lots",
        "average_profit_per_take_profit_day",
        "average_profit_per_trading_day",
        "max_unrealized_loss",
        "max_drawdown_equity",
        "max_consecutive_days_without_take_profit",
        "margin_call_count",
        "forced_liquidation_count",
        "min_maintenance_ratio",
        "exposure_limit_hit_count",
        "force_liquidation",
        "random_seed",
    ):
        print(f"  {key}: {summary.get(key)}")
    return 0


def _cmd_stress(args: argparse.Namespace) -> int:
    """Stress testing (D1): historical regimes, cost sensitivity, bootstrap ruin."""
    from .backtest import simulate
    from .optimizer import WalkForwardOptimizer
    from . import stress as stress_mod
    import pandas as pd

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.optimization.random_seed = int(args.seed)

    if args.synthetic:
        target_df, benchmark_df = make_synthetic_data(
            n_days=args.synthetic_days, seed=args.synthetic_seed
        )
        joined = join_target_benchmark(target_df, benchmark_df)
    else:
        if not args.target or not args.benchmark:
            print("error: --target and --benchmark CSVs are required (or --synthetic)", file=sys.stderr)
            return 2
        joined = load_market_data(args.target, args.benchmark)

    md = prepare_market_data(joined, cfg)
    print(f"Loaded {md.n} rows. Running base backtest for regime slicing ...", flush=True)

    # 1. Historical regimes from a full walk-forward run at the base config.
    result = run_backtest(md, cfg)
    daily = pd.DataFrame(result.daily_rows)
    regimes = stress_mod.all_regime_metrics(daily)

    # 2. Cost sensitivity: capture the walk-forward parameter sequence once, then
    #    replay under the cost grid (decisions fixed).
    print("Capturing walk-forward parameter sequence for sensitivity ...", flush=True)
    opt = WalkForwardOptimizer(md, cfg)
    param_seq = [opt.params_at_close(i) for i in range(md.n)]
    provider = lambda i: param_seq[i]  # noqa: E731
    sensitivity = stress_mod.sensitivity_grid(
        md, cfg, provider,
        slippage_bps_list=[2.0, 5.0, 10.0, 20.0],
        interest_rate_list=[0.015, 0.028, 0.04, 0.06],
    )

    # Per-regime maintenance ratios / forced closes at a THIN own-funds level
    # (replay the same captured tape at the ruin-equity with forced liquidation
    # on), so the regime maintenance ratios are meaningful (not trivial as at
    # ¥100M).
    from dataclasses import replace as _replace
    thin_cfg = _replace(cfg, initial_equity=float(args.ruin_equity), force_liquidation=True)
    thin_res = simulate(md, 0, md.n, provider, thin_cfg, record=True)
    daily_thin = pd.DataFrame(thin_res.daily_rows)
    regimes_thin = stress_mod.all_regime_metrics(daily_thin)

    # 3. Bootstrap ruin at a realistic own-funds level with forced liquidation on.
    ruin_cfg = load_config(args.config)
    ruin_cfg.initial_equity = float(args.ruin_equity)
    ruin_cfg.force_liquidation = True
    print(
        f"Running {args.bootstrap_paths} bootstrap paths at "
        f"¥{ruin_cfg.initial_equity:,.0f} (force_liquidation on) ...",
        flush=True,
    )
    bootstrap = stress_mod.block_bootstrap_ruin(
        joined, ruin_cfg,
        n_paths=args.bootstrap_paths, block_size=args.block_size,
        seed=cfg.optimization.random_seed,
    )

    meta = {
        "initial_equity": cfg.initial_equity,
        "ruin_initial_equity": ruin_cfg.initial_equity,
        "random_seed": cfg.optimization.random_seed,
        "sessions": md.n,
        "data_repairs": list(md.data_repairs),
    }
    stress_mod.write_stress_outputs(
        args.out, regimes, sensitivity, bootstrap, regimes_thin=regimes_thin, meta=meta
    )

    print(f"\nDone. Stress outputs written to {Path(args.out).resolve()}")
    print(f"  regimes: {len(regimes)} | sensitivity cells: {len(sensitivity)}")
    print(
        f"  bootstrap: ruin {bootstrap.ruin_probability:.1%}, "
        f"forced {bootstrap.forced_liquidation_probability:.1%}, "
        f"median final ¥{bootstrap.median_final_equity:,.0f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nikkei_leverage_sim",
        description="Nikkei leverage ETF margin-long accumulation backtester",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch OHLCV data via yfinance")
    p_fetch.add_argument("--start", required=True)
    p_fetch.add_argument("--end", required=True)
    p_fetch.add_argument("--out", default="data/")
    p_fetch.add_argument("--target-symbol", default="1570.T")
    p_fetch.add_argument("--benchmark-symbol", default="^N225")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_synth = sub.add_parser("synth", help="Generate offline synthetic CSVs")
    p_synth.add_argument("--out", default="data/")
    p_synth.add_argument("--days", type=int, default=900)
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.add_argument("--start", default="2019-01-01")
    p_synth.set_defaults(func=_cmd_synth)

    p_run = sub.add_parser("run", help="Run a backtest from CSVs or synthetic data")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--target", help="Target ETF OHLCV CSV path")
    p_run.add_argument("--benchmark", help="Benchmark OHLCV CSV path")
    p_run.add_argument("--out", default="outputs/")
    p_run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override optimization.random_seed (reproducibility)",
    )
    p_run.add_argument(
        "--force-liquidation",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="force_liquidation",
        help="Override force_liquidation: enable/disable the margin-call model",
    )
    p_run.add_argument(
        "--initial-equity",
        type=float,
        default=None,
        help="Override initial_equity (own funds / committed margin, JPY)",
    )
    p_run.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    p_run.add_argument("--synthetic-days", type=int, default=900)
    p_run.add_argument("--synthetic-seed", type=int, default=7)
    p_run.set_defaults(func=_cmd_run)

    p_stress = sub.add_parser(
        "stress", help="Stress test: historical regimes, cost sensitivity, bootstrap ruin"
    )
    p_stress.add_argument("--config", required=True)
    p_stress.add_argument("--target", help="Target ETF OHLCV CSV path")
    p_stress.add_argument("--benchmark", help="Benchmark OHLCV CSV path")
    p_stress.add_argument("--out", default="outputs_stress/")
    p_stress.add_argument("--seed", type=int, default=None, help="Override optimization.random_seed")
    p_stress.add_argument(
        "--ruin-equity", type=float, default=5_000_000.0,
        help="Own funds for the bootstrap ruin study (forced liquidation on)",
    )
    p_stress.add_argument("--bootstrap-paths", type=int, default=300)
    p_stress.add_argument("--block-size", type=int, default=20)
    p_stress.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    p_stress.add_argument("--synthetic-days", type=int, default=900)
    p_stress.add_argument("--synthetic-seed", type=int, default=7)
    p_stress.set_defaults(func=_cmd_stress)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
