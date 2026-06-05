"""Tests for the structured Markdown report and its end-to-end wiring."""
from __future__ import annotations

import json

from nikkei_leverage_sim.backtest import prepare_market_data, run_backtest
from nikkei_leverage_sim.benchmark import build_benchmarks
from nikkei_leverage_sim.config import Config
from nikkei_leverage_sim.data import join_target_benchmark, make_synthetic_data
from nikkei_leverage_sim.metrics import build_summary
from nikkei_leverage_sim.report import render_report_md
from nikkei_leverage_sim.reporting import write_outputs


def _fast_result():
    """A small, optimization-free backtest on synthetic data (fast + offline)."""
    target, benchmark = make_synthetic_data(n_days=300, seed=123)
    joined = join_target_benchmark(target, benchmark)
    cfg = Config()
    cfg.optimization.enabled = False  # default params throughout -> quick
    md = prepare_market_data(joined, cfg)
    return run_backtest(md, cfg), cfg


def test_render_report_has_survival_first_and_benchmarks():
    result, cfg = _fast_result()
    summary = build_summary(result, cfg)
    daily = result.daily_rows
    benchmarks = build_benchmarks(
        [r["target_close"] for r in daily],
        [r["benchmark_close"] for r in daily],
        cfg.initial_equity,
        len(daily),
    )
    md = render_report_md(
        summary, benchmarks, period_start="2019-01-01", period_end="2020-01-01", n_sessions=len(daily)
    )

    # Headline is survival / tail risk, and it appears before the return section.
    assert "生存・テールリスク" in md
    assert "パッシブ・ベンチマーク比較" in md
    assert md.index("生存・テールリスク") < md.index("リターン要約")
    # The strategy row and all three baselines are present.
    assert "本戦略" in md
    for name in ("1570.T Buy & Hold", "N225 Buy & Hold", "Cash (no position)"):
        assert name in md
    # Key tail-risk fields are rendered.
    for label in ("CVaR95", "最低維持率", "最悪日資産", "Sortino", "Ulcer"):
        assert label in md


def test_render_report_handles_missing_period():
    result, cfg = _fast_result()
    summary = build_summary(result, cfg)
    md = render_report_md(summary, [], n_sessions=0)
    assert "期間情報なし" in md


def test_write_outputs_emits_report_and_benchmarks(tmp_path):
    result, _cfg = _fast_result()
    summary = write_outputs(result, tmp_path)

    # New Week 1 artifacts exist alongside the legacy ones.
    for fname in (
        "report.md",
        "benchmarks.json",
        "summary.json",
        "underwater_curve.png",
        "return_distribution.png",
    ):
        assert (tmp_path / fname).exists(), f"missing {fname}"

    # summary.json carries the nested risk block.
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "risk" in on_disk
    assert "cvar_95_daily" in on_disk["risk"]
    assert "margin_call_rate" in on_disk["risk"]
    # Data-quality audit block is always present (empty list for clean synthetic).
    assert on_disk["data_quality"]["price_glitch_repairs"] == []

    # benchmarks.json is a list of three baselines.
    benches = json.loads((tmp_path / "benchmarks.json").read_text(encoding="utf-8"))
    assert [b["name"] for b in benches] == [
        "1570.T Buy & Hold",
        "N225 Buy & Hold",
        "Cash (no position)",
    ]

    # The returned summary matches what was written (risk block included).
    assert summary["risk"]["max_drawdown_pct"] == on_disk["risk"]["max_drawdown_pct"]
