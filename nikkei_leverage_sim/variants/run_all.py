"""End-to-end driver for the strategy-variant grid search.

Pipeline
--------
1. Load the real 1570.T / N225 data and the *fast* config.
2. Capture the fast walk-forward's per-day buy params **once** (cached to disk).
3. Run the full 432-combo grid under two buy engines:
     * ``walkforward`` -- replay the captured fast params (primary, realistic).
     * ``fixed_default`` -- the config's default params (robustness check that
       the exit-rule ranking does not depend on the particular buy params).
4. Write per-engine ``summary_all_variants.csv`` + ``rows.json``, plus
   ``top_per_category.md`` and ``variants_comparison.png`` for the primary set.

Usage::

    python -m variants.run_all --out outputs_variants [--processes 8] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import List

# Windows consoles default to cp932 (Shift-JIS) here, which cannot encode the
# characters used in progress prints; force UTF-8 so logging never crashes the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - older interpreters / non-reconfigurable streams
    pass

from nikkei_leverage_sim.backtest import prepare_market_data
from nikkei_leverage_sim.config import load_config
from nikkei_leverage_sim.data import load_market_data

from . import aggregate
from .grid import build_grid, run_grid
from .variant_engine import prepare_market_data_v
from .wf_capture import capture_walkforward_params

HERE = Path(__file__).resolve().parent
PKG = HERE.parent  # nikkei_leverage_sim/


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the strategy-variant grid search")
    ap.add_argument("--config", default=str(PKG / "examples" / "config_fast.yaml"))
    ap.add_argument("--target", default=str(PKG / "data" / "target_1570_T.csv"))
    ap.add_argument("--benchmark", default=str(PKG / "data" / "benchmark_N225.csv"))
    ap.add_argument("--out", default=str(PKG / "outputs_variants"))
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute the cached walk-forward param sequence")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    joined = load_market_data(args.target, args.benchmark)
    md = prepare_market_data(joined, cfg)
    mdv = prepare_market_data_v(joined, cfg)
    print(f"Loaded {md.n} rows ({int(md.valid.sum())} tradable).")

    combos = build_grid()
    print(f"Grid: {len(combos)} combos "
          f"({sum(c.variant=='per_lot' for c in combos)} per_lot, "
          f"{sum(c.variant=='bulk_yen' for c in combos)} bulk_yen, "
          f"{sum(c.variant=='bulk_pct' for c in combos)} bulk_pct, "
          f"{sum(c.variant=='combo' for c in combos)} combo).")

    # --- 1. capture (cached) walk-forward param sequence ---
    # NOTE: wf_params.pkl is a *self-produced local cache* of StrategyParams
    # written by this same run (not external/untrusted input); pickle is used
    # only to round-trip our own dataclass objects. Safe to load here.
    cache = out / "wf_params.pkl"
    if cache.exists() and not args.refresh:
        seq = pickle.loads(cache.read_bytes())
        print(f"Loaded cached walk-forward params ({len(seq)} days) from {cache.name}.")
    else:
        print("Capturing fast walk-forward params (one-off, ~2.5 min)...")
        t0 = time.time()
        seq = capture_walkforward_params(md, cfg)
        cache.write_bytes(pickle.dumps(seq))
        print(f"  captured in {time.time()-t0:.0f}s -> {cache.name}")

    engines = {
        "walkforward": seq,
        "fixed_default": cfg.strategy.default_params,
    }
    rows_by_engine = {}
    for name, provider in engines.items():
        print(f"\n=== Running grid under buy engine: {name} ===")
        t0 = time.time()
        rows = aggregate.annotate(run_grid(mdv, cfg, provider, combos, processes=args.processes))
        rows_by_engine[name] = rows
        dt = time.time() - t0
        print(f"  {len(rows)} combos in {dt:.1f}s ({dt/len(rows)*1000:.0f} ms/combo)")

        eng_dir = out / name
        eng_dir.mkdir(parents=True, exist_ok=True)
        aggregate.write_csv(rows, eng_dir / "summary_all_variants.csv")
        (eng_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str),
                                           encoding="utf-8")
        aggregate.write_top_markdown(rows, eng_dir / "top_per_category.md",
                                     baseline_label="per_lot__init0")
        base = next((r for r in rows if r["label"] == "per_lot__init0"), None)
        if base:
            print(f"  anchor per_lot/init0: net Y{base['net_profit_after_tax']/1e6:.2f}M, "
                  f"maxUL Y{base['max_unrealized_loss']/1e6:.2f}M, "
                  f"harvest {base['harvest_days']}, sharpe {base['sharpe_like']:.3f}")

    # Primary plots + top tables at the top level use the walkforward engine.
    print("\n=== Rendering primary outputs (walkforward engine) ===")
    rows = rows_by_engine["walkforward"]
    aggregate.write_csv(rows, out / "summary_all_variants.csv")
    aggregate.write_top_markdown(rows, out / "top_per_category.md",
                                 baseline_label="per_lot__init0")
    aggregate.make_plots(rows, out / "variants_comparison.png")
    print(f"Wrote primary outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
