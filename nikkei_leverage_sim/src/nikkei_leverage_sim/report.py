"""Render a structured, audit-friendly Markdown report for one backtest run.

Design priority (Week 1, item F): the **headline table is survival/tail-risk**,
not average return.  "Worst-day equity", "max unrealized loss", "CVaR95",
"margin-call rate" and "minimum maintenance ratio" sit at the top — average
return and Sharpe are deliberately demoted, because for a no-stop-loss leveraged
strategy the average hides the thing that can ruin it.

The report is generated purely from the summary dict (see
:func:`nikkei_leverage_sim.metrics.build_summary`) plus the passive benchmark
results, so it is reproducible and contains no hidden state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .benchmark import BenchmarkResult


def _yen(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"¥{v:,.0f}"


def _yen_m(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"¥{v / 1_000_000:,.2f}M"


def _pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "—"
    if v == 0:  # collapse negative zero
        v = 0.0
    return f"{v * 100:.{digits}f}%"


def _ratio(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    if v == 0:  # collapse negative zero
        v = 0.0
    return f"{v:.{digits}f}"


def render_report_md(
    summary: Dict[str, Any],
    benchmarks: Sequence[BenchmarkResult],
    *,
    period_start: str = "",
    period_end: str = "",
    n_sessions: int = 0,
) -> str:
    """Return the full Markdown report as a string."""
    risk: Dict[str, Any] = summary.get("risk", {}) or {}
    lines: List[str] = []

    # --- Title + run header -------------------------------------------------
    lines.append("# バックテスト結果レポート（自動生成）")
    lines.append("")
    period = (
        f"{period_start} 〜 {period_end}" if period_start and period_end else "（期間情報なし）"
    )
    lines.append(
        f"> **対象期間:** {period}（{n_sessions:,} 営業日） / "
        f"**自己資金:** {_yen(summary.get('initial_equity'))} / "
        f"**最大建玉（実績）:** {_yen(summary.get('max_gross_exposure'))}"
    )
    lines.append(
        f"> **強制ロスカット:** {'ON' if summary.get('force_liquidation') else 'OFF'} / "
        f"**維持率閾値:** {_pct(summary.get('maintenance_margin_ratio'), 0)} / "
        f"**乱数seed:** {summary.get('random_seed')}"
    )
    lines.append(
        "> ⚠️ コスト（手数料・スリッページ・信用金利・税）は理論前提。利益はサンプル内であり将来を保証しない。"
    )
    repairs = (summary.get("data_quality", {}) or {}).get("price_glitch_repairs", []) or []
    if repairs:
        dates = ", ".join(str(r.get("date")) for r in repairs)
        lines.append(
            f"> 🛠 **データ補修:** ベンダー価格の孤立グリッチ {len(repairs)} 件を補正済み"
            f"（{dates}）。詳細は `summary.json` の `data_quality`。"
        )
    lines.append("")

    # --- §1 Headline: survival / tail risk ---------------------------------
    lines.append("## 1. 最重要：生存・テールリスク指標")
    lines.append("")
    lines.append("| 指標 | 値 | 意味 |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| 最悪日資産（最低エクイティ） | {_yen(risk.get('worst_day_equity'))} | "
        "口座が到達した最低水準 |"
    )
    lines.append(
        f"| 最大含み損 | {_yen(summary.get('max_unrealized_loss'))} | "
        "未実現での最大の痛み |"
    )
    lines.append(
        f"| 最大ドローダウン（額／％） | {_yen(summary.get('max_drawdown_equity'))}"
        f" ／ {_pct(risk.get('max_drawdown_pct'))} | 資産のピーク→谷 |"
    )
    lines.append(
        f"| CVaR95（日次・期待ショートフォール） | {_pct(risk.get('cvar_95_daily'))} | "
        "最悪5%の日の平均損失率 |"
    )
    lines.append(
        f"| VaR95（日次） | {_pct(risk.get('var_95_daily'))} | "
        "95%信頼で1日に被りうる損失率 |"
    )
    lines.append(
        f"| 最低維持率 | {_ratio(summary.get('min_maintenance_ratio'))} | "
        "追証ラインまでの距離（閾値 "
        f"{_pct(summary.get('maintenance_margin_ratio'), 0)}）|"
    )
    lines.append(
        f"| 追証発生率 | {_pct(risk.get('margin_call_rate'))} | "
        f"全{n_sessions:,}営業日中の維持率割れ日割合 |"
    )
    lines.append(
        f"| 強制決済回数 | {summary.get('forced_liquidation_count')} | "
        "実際に強制ロスカットされた回数 |"
    )
    lines.append("")

    # --- §2 Return summary --------------------------------------------------
    lines.append("## 2. リターン要約（参考：上の生存指標とセットで読む）")
    lines.append("")
    lines.append("| 指標 | 値 |")
    lines.append("|---|---:|")
    lines.append(f"| 期末資産 | {_yen(summary.get('final_equity'))} |")
    lines.append(
        f"| 実現益（税引後） | {_yen(summary.get('net_realized_profit_after_tax'))} |"
    )
    lines.append(f"| 期末含み損益 | {_yen(summary.get('ending_unrealized_pnl'))} |")
    lines.append(f"| 年率リターン (CAGR) | {_pct(summary.get('annualized_return'))} |")
    lines.append(f"| 支払金利 合計 | {_yen(summary.get('total_interest_paid'))} |")
    lines.append(f"| 支払税 合計 | {_yen(summary.get('total_tax_paid'))} |")
    lines.append(f"| 決済勝率（ロット単位） | {_pct(summary.get('win_rate_of_closed_lots'))} |")
    lines.append("")

    # --- §3 Full risk-adjusted metrics -------------------------------------
    lines.append("## 3. リスク調整後・分布指標")
    lines.append("")
    lines.append("| 指標 | 値 |")
    lines.append("|---|---:|")
    lines.append(f"| Sharpe-like | {_ratio(summary.get('sharpe_like_ratio'))} |")
    lines.append(f"| Sortino | {_ratio(risk.get('sortino_ratio'))} |")
    lines.append(f"| Calmar (CAGR / maxDD) | {_ratio(risk.get('calmar_ratio'))} |")
    lines.append(f"| Ulcer Index | {_ratio(risk.get('ulcer_index'), 2)} |")
    lines.append(f"| 日次リターン 歪度 (skew) | {_ratio(risk.get('return_skew'))} |")
    lines.append(
        f"| 日次リターン 超過尖度 (excess kurtosis) | {_ratio(risk.get('return_kurtosis_excess'))} |"
    )
    lines.append(f"| 最悪日リターン | {_pct(risk.get('worst_daily_return'))} |")
    lines.append("")
    lines.append("ドローダウン深さの分布（正の値＝下落率）:")
    lines.append("")
    lines.append("| p50 | p90 | p95 | p99 |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(
        f"| {_pct(risk.get('drawdown_pct_p50'))} | {_pct(risk.get('drawdown_pct_p90'))} | "
        f"{_pct(risk.get('drawdown_pct_p95'))} | {_pct(risk.get('drawdown_pct_p99'))} |"
    )
    lines.append("")

    # --- §4 Benchmark comparison -------------------------------------------
    lines.append("## 4. パッシブ・ベンチマーク比較")
    lines.append("")
    lines.append(
        "同一の自己資金を「ただ買って持つ」だけの基準と比較（**終値ベース**。"
        "本戦略の翌寄付約定とは執行前提が異なる）。各資産で2通りを併記する: "
        "**一括 Buy&Hold**（初日に全額投入）と **定額積立(DCA)**（同じ総額を全営業日に均等投下）。"
        "本戦略は建玉を自己資金より大幅に絞りつつ積み増す（accumulate）ため、"
        "**一括 Buy&Hold は資金投入がより積極的な参照点**、"
        "**定額積立は「淡々と積み立てて持つだけ」の等質な参照点**（レバ・利確なし）。"
        "いずれもリスクの完全な等価物ではない（本戦略はエクスポージャを絞り利確する）ので、"
        "上の生存・DD指標と必ずセットで読む。"
    )
    lines.append("")
    lines.append("| 戦略 | 期末資産 | 総リターン | CAGR | 最大DD% | Sharpe | Sortino | Ulcer | CVaR95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| **本戦略** | {_yen_m(summary.get('final_equity'))} | "
        f"{_pct(_safe_total_return(summary))} | {_pct(summary.get('annualized_return'))} | "
        f"{_pct(risk.get('max_drawdown_pct'))} | {_ratio(summary.get('sharpe_like_ratio'))} | "
        f"{_ratio(risk.get('sortino_ratio'))} | {_ratio(risk.get('ulcer_index'), 2)} | "
        f"{_pct(risk.get('cvar_95_daily'))} |"
    )
    for b in benchmarks:
        lines.append(
            f"| {b.name} | {_yen_m(b.final_equity)} | {_pct(b.total_return)} | "
            f"{_pct(b.annualized_return)} | {_pct(b.max_drawdown_pct)} | "
            f"{_ratio(b.sharpe_like)} | {_ratio(b.sortino_ratio)} | "
            f"{_ratio(b.ulcer_index, 2)} | {_pct(b.cvar_95_daily)} |"
        )
    lines.append("")

    # --- Footer -------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append(
        "*このレポートは `summary.json` から自動生成。再現には同一 seed・同一データ・同一 config を使用。*"
    )
    lines.append("")
    return "\n".join(lines)


def _safe_total_return(summary: Dict[str, Any]) -> Optional[float]:
    init = summary.get("initial_equity")
    final = summary.get("final_equity")
    if not init:
        return None
    return final / init - 1.0
