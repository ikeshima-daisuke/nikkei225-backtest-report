"""Aggregation, ranking and plotting for the variant grid results."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Column order for the summary CSV (kept stable for diffing).
CSV_COLUMNS: List[str] = [
    "label", "category", "initial_amount", "bulk_exit_yen", "bulk_exit_pct",
    "n_trading_days", "net_profit_after_tax", "realized_after_tax",
    "ending_unrealized_pnl", "final_equity", "annualized_return", "sharpe_like",
    "max_drawdown_equity", "max_unrealized_loss", "total_interest_paid",
    "total_tax_paid", "buy_trade_count", "sell_trade_count", "harvest_days",
    "bulk_exit_count", "avg_profit_per_harvest_day", "median_profit_per_harvest_day",
    "win_rate_closed_lots", "profit_factor",
    "max_consecutive_days_without_take_profit", "max_lot_holding_days",
    "days_underwater", "pct_days_underwater", "max_consecutive_underwater_days",
    "margin_call_count", "exposure_limit_hit_count", "max_gross_exposure",
    "average_gross_exposure", "comfort_score",
]


def comfort_score(row: Dict[str, Any]) -> float:
    """A 'satisfying take-profit' (kimochiyoku rikaku) score for the *harvesting
    experience* — deliberately **excludes total profit**, which is reported
    separately, so a few giant rare windfalls cannot masquerade as "comfortable".

    Higher = a nicer strategy to actually live with.  It rewards:

    * harvesting often enough to feel active (``freq``, saturating ~monthly),
    * harvests being *meaningful* in size, not yen-dust (``size``, saturating at
      ~¥300k so a single ¥8M windfall does not dominate a steady ¥70k cadence),
    * decent risk-adjusted quality (``sharpe``),

    and penalises the two stressors REPORT_REAL flags as the real cost of this
    loss-cut-free strategy:

    * the *fraction of days* sat in unrealized loss, and
    * the *longest unbroken stretch* stuck underwater (capital lock-up).

    Scales make each term roughly O(1); this is a ranking aid, not money.
    """
    avg = row["avg_profit_per_harvest_day"]
    harvest = row["harvest_days"]
    sharpe = row["sharpe_like"]
    uw = row["pct_days_underwater"]
    uw_streak_years = row["max_consecutive_underwater_days"] / 252.0

    freq = min(harvest / 120.0, 1.0)          # ~monthly over 12y saturates the reward
    size = min(avg / 300_000.0, 1.0)          # harvests feel "real" by ~¥300k
    return (
        1.5 * freq
        + 1.5 * size
        + 1.0 * sharpe
        - 3.0 * uw
        - 2.0 * uw_streak_years
    )


def annotate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach the comfort score to each row (in place) and return the list."""
    out = []
    for r in rows:
        r = dict(r)
        r["comfort_score"] = comfort_score(r)
        out.append(r)
    return out


def write_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def top_n(
    rows: Sequence[Dict[str, Any]], key: str, n: int = 3, reverse: bool = True
) -> List[Dict[str, Any]]:
    finite = [r for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return sorted(finite, key=lambda r: r[key], reverse=reverse)[:n]


def top_per_category(
    rows: Sequence[Dict[str, Any]], key: str, n: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    cats: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {c: top_n(rs, key, n) for c, rs in cats.items()}


def _fmt_yen(v: float) -> str:
    return f"¥{v/1_000_000:,.2f}M"


def write_top_markdown(
    rows: Sequence[Dict[str, Any]], path: str | Path, baseline_label: str
) -> None:
    """Write ``top_per_category.md`` ranking each category by several lenses."""
    path = Path(path)
    base = next((r for r in rows if r["label"] == baseline_label), None)
    lines: List[str] = []
    lines.append("# Variant grid — category leaders\n")
    if base is not None:
        lines.append(
            f"**In-grid control** (`{baseline_label}` = published fast strategy, "
            f"per-lot take-profit, no lump): net {_fmt_yen(base['net_profit_after_tax'])}, "
            f"max unrealized loss {_fmt_yen(base['max_unrealized_loss'])}, "
            f"{base['harvest_days']} harvest days, Sharpe {base['sharpe_like']:.3f}, "
            f"longest underwater {base['max_consecutive_underwater_days']} days.\n"
        )

    lenses = [
        ("net_profit_after_tax", "Most total profit (net, after tax)"),
        ("comfort_score", "Most 'satisfying take-profit' (comfort score)"),
        ("sharpe_like", "Best risk-adjusted (Sharpe-like)"),
    ]
    cat_order = ["per_lot", "bulk_yen", "bulk_pct", "combo"]
    for key, title in lenses:
        lines.append(f"\n## {title}\n")
        per_cat = top_per_category(rows, key, 3)
        for cat in cat_order:
            if cat not in per_cat:
                continue
            lines.append(f"\n### `{cat}`\n")
            lines.append(
                "| rank | label | net | max unreal. loss | harvest days | "
                "longest underwater | Sharpe | comfort |\n"
                "|---:|---|---:|---:|---:|---:|---:|---:|"
            )
            for i, r in enumerate(per_cat[cat], 1):
                lines.append(
                    f"| {i} | `{r['label']}` | {_fmt_yen(r['net_profit_after_tax'])} | "
                    f"{_fmt_yen(r['max_unrealized_loss'])} | {r['harvest_days']} | "
                    f"{r['max_consecutive_underwater_days']} | {r['sharpe_like']:.3f} | "
                    f"{r['comfort_score']:.2f} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    """Render the multi-panel ``variants_comparison.png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_label = {r["label"]: r for r in rows}

    def get(cat, **filt):
        out = []
        for r in rows:
            if r["category"] != cat:
                continue
            if all(r.get(k) == v for k, v in filt.items()):
                out.append(r)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    M = 1_000_000.0

    # Panel 1: risk/return frontier — max unrealized loss vs net profit.
    ax = axes[0][0]
    colors = {"per_lot": "#1f77b4", "bulk_yen": "#ff7f0e",
              "bulk_pct": "#2ca02c", "combo": "#d62728"}
    for cat, col in colors.items():
        pts = get(cat)
        if not pts:
            continue
        ax.scatter([p["max_unrealized_loss"] / M for p in pts],
                   [p["net_profit_after_tax"] / M for p in pts],
                   s=22, alpha=0.55, color=col, label=cat)
    base = by_label.get("per_lot__init0")
    if base:
        ax.scatter([base["max_unrealized_loss"] / M], [base["net_profit_after_tax"] / M],
                   s=160, marker="*", color="black", zorder=5,
                   label="fast baseline (per_lot/init0)")
    ax.set_xlabel("Max unrealized loss (¥M)  — lower is safer")
    ax.set_ylabel("Net profit after tax (¥M)")
    ax.set_title("Risk / return frontier (all 432 combos)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: bulk_yen threshold sweep at initial=0.
    ax = axes[0][1]
    ys = sorted(get("bulk_yen", initial_amount=0.0), key=lambda r: r["bulk_exit_yen"])
    if ys:
        x = [r["bulk_exit_yen"] / M for r in ys]
        ax.plot(x, [r["net_profit_after_tax"] / M for r in ys], "o-", label="net profit")
        ax.plot(x, [r["max_unrealized_loss"] / M for r in ys], "s--", color="#d62728",
                label="max unrealized loss")
        ax.set_xlabel("bulk_exit_yen threshold (¥M)")
        ax.set_ylabel("¥M")
        ax.set_title("bulk_yen sweep (initial=0)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Panel 3: bulk_pct threshold sweep at initial=0 (profit + harvest cadence).
    ax = axes[1][0]
    ps = sorted(get("bulk_pct", initial_amount=0.0), key=lambda r: r["bulk_exit_pct"])
    if ps:
        x = [r["bulk_exit_pct"] * 100 for r in ps]
        ax.plot(x, [r["net_profit_after_tax"] / M for r in ps], "o-", label="net profit (¥M)")
        ax.set_xlabel("bulk_exit_pct (%)")
        ax.set_ylabel("net profit (¥M)")
        ax.set_title("bulk_pct sweep (initial=0)")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(x, [r["harvest_days"] for r in ps], "^--", color="#2ca02c",
                 label="harvest days")
        ax2.set_ylabel("harvest days")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")

    # Panel 4: effect of the initial lump on net profit, best threshold per category.
    ax = axes[1][1]
    inits = sorted({r["initial_amount"] for r in rows})
    for cat, col in colors.items():
        ys = []
        for init in inits:
            pts = get(cat, initial_amount=init)
            if not pts:
                ys.append(float("nan"))
                continue
            ys.append(max(p["net_profit_after_tax"] for p in pts) / M)
        ax.plot([i / M for i in inits], ys, "o-", color=col, label=cat)
    ax.set_xlabel("Initial lump sum (¥M)")
    ax.set_ylabel("Best net profit in category (¥M)")
    ax.set_title("Initial lump-sum effect (best exit per category)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Nikkei 1570.T leverage accumulation — strategy-variant grid (fast buy engine)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
