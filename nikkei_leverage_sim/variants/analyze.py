"""Post-grid analysis: print the tables/figures that feed REPORT_VARIANTS.md.

Reads the two engine CSVs and emits:
* the in-grid control anchor (`per_lot__init0`),
* best-N overall (by net profit and by comfort score),
* best per category under both buy engines (ranking-consistency check),
* a few descriptive cuts (initial-lump effect; bulk vs per-lot).

Pure stdlib (csv) so it has no extra deps.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

NUM = {
    "initial_amount", "n_trading_days", "net_profit_after_tax", "realized_after_tax",
    "ending_unrealized_pnl", "final_equity", "annualized_return", "sharpe_like",
    "max_drawdown_equity", "max_unrealized_loss", "total_interest_paid",
    "total_tax_paid", "buy_trade_count", "sell_trade_count", "harvest_days",
    "bulk_exit_count", "avg_profit_per_harvest_day", "median_profit_per_harvest_day",
    "win_rate_closed_lots", "profit_factor", "max_consecutive_days_without_take_profit",
    "max_lot_holding_days", "days_underwater", "pct_days_underwater",
    "max_consecutive_underwater_days", "margin_call_count", "exposure_limit_hit_count",
    "max_gross_exposure", "average_gross_exposure", "comfort_score",
}


def load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in list(r):
                if k in NUM and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def m(v: float) -> str:
    return f"{v/1_000_000:.2f}"


def by(rows, key, n=10, rev=True):
    fin = [r for r in rows if isinstance(r.get(key), float) and math.isfinite(r[key])]
    return sorted(fin, key=lambda r: r[key], reverse=rev)[:n]


def table(rows, title):
    print(f"\n### {title}")
    print("| label | net ¥M | maxUL ¥M | maxDD ¥M | harvest | bulkExits | "
          "avg/harvest ¥ | underwater% | UWmax(d) | sharpe | comfort |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        print(f"| `{r['label']}` | {m(r['net_profit_after_tax'])} | "
              f"{m(r['max_unrealized_loss'])} | {m(r['max_drawdown_equity'])} | "
              f"{int(r['harvest_days'])} | {int(r['bulk_exit_count'])} | "
              f"{r['avg_profit_per_harvest_day']:,.0f} | "
              f"{r['pct_days_underwater']*100:.0f}% | "
              f"{int(r['max_consecutive_underwater_days'])} | "
              f"{r['sharpe_like']:.3f} | {r['comfort_score']:.2f} |")


def main():
    base = Path("outputs_variants")
    wf = load(base / "walkforward" / "summary_all_variants.csv")
    fx = load(base / "fixed_default" / "summary_all_variants.csv")

    anchor = next(r for r in wf if r["label"] == "per_lot__init0")
    print("## ANCHOR — per_lot__init0 (published fast strategy)")
    table([anchor], "control")

    table(by(wf, "net_profit_after_tax", 12), "WALKFORWARD — top 12 by NET PROFIT")
    table(by(wf, "comfort_score", 12), "WALKFORWARD — top 12 by COMFORT SCORE")

    print("\n## Per-category best (NET PROFIT) — walkforward vs fixed_default")
    for cat in ("per_lot", "bulk_yen", "bulk_pct", "combo"):
        w = by([r for r in wf if r["category"] == cat], "net_profit_after_tax", 1)[0]
        x = by([r for r in fx if r["category"] == cat], "net_profit_after_tax", 1)[0]
        print(f"- **{cat}**: WF `{w['label']}` net ¥{m(w['net_profit_after_tax'])}M "
              f"(maxUL ¥{m(w['max_unrealized_loss'])}M, harvest {int(w['harvest_days'])}) "
              f"|| FIXED `{x['label']}` net ¥{m(x['net_profit_after_tax'])}M")

    print("\n## Per-category best (COMFORT) — walkforward")
    for cat in ("per_lot", "bulk_yen", "bulk_pct", "combo"):
        w = by([r for r in wf if r["category"] == cat], "comfort_score", 1)[0]
        print(f"- **{cat}**: `{w['label']}` comfort {w['comfort_score']:.2f} "
              f"net ¥{m(w['net_profit_after_tax'])}M maxUL ¥{m(w['max_unrealized_loss'])}M "
              f"harvest {int(w['harvest_days'])} UWmax {int(w['max_consecutive_underwater_days'])}d")

    # Initial-lump effect on the baseline per_lot exit.
    print("\n## Initial-lump effect (per_lot, walkforward)")
    pl = sorted([r for r in wf if r["category"] == "per_lot"], key=lambda r: r["initial_amount"])
    for r in pl:
        print(f"- init ¥{m(r['initial_amount'])}M: net ¥{m(r['net_profit_after_tax'])}M, "
              f"maxUL ¥{m(r['max_unrealized_loss'])}M, harvest {int(r['harvest_days'])}, "
              f"annual {r['annualized_return']*100:.2f}%, sharpe {r['sharpe_like']:.3f}")

    # Dump a compact json for the report writer.
    out = {
        "anchor": anchor,
        "top_net": by(wf, "net_profit_after_tax", 15),
        "top_comfort": by(wf, "comfort_score", 15),
    }
    Path("outputs_variants/analysis.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nWrote outputs_variants/analysis.json")


if __name__ == "__main__":
    main()
