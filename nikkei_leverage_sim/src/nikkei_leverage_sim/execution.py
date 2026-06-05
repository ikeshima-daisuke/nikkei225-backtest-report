"""Execution-model realism comparison (roadmap item D2).

The engine's default fill is the next open at +/- slippage.  Real fills for a
leveraged ETF are worse: the open is one print in a thin tape, VWAP/TWAP differ
from it, the fill can be on the unfavourable side of the bar, latency delays
execution, and large orders only partially fill against the day's volume.

This module replays a *fixed* captured strategy (a walk-forward parameter
sequence, or fixed params) under a set of execution scenarios — changing only the
``execution`` config — so each scenario isolates the execution effect (no
re-optimization).  Everything runs through :func:`simulate`, so the no-look-ahead
and no-loss-cut invariants are inherited unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .backtest import MarketData, simulate
from .config import Config, StrategyParams

ParamsProvider = Union[StrategyParams, Callable[[int], StrategyParams]]


# (name, execution-field overrides) — the first is the engine default baseline.
EXECUTION_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "baseline (next_open)", "overrides": {}},
    {"name": "VWAP proxy", "overrides": {"fill_model": "vwap"}},
    {"name": "adverse (High/Low)", "overrides": {"fill_model": "adverse"}},
    {"name": "delay +1 day", "overrides": {"execution_delay_days": 1}},
    {"name": "delay +3 days", "overrides": {"execution_delay_days": 3}},
    {"name": "partial 0.5% volume", "overrides": {"volume_participation": 0.005}},
    {"name": "partial 0.001% volume", "overrides": {"volume_participation": 0.00001}},
]


@dataclass
class ExecutionCell:
    name: str
    overrides: Dict[str, Any]
    net_realized_after_tax: float
    final_equity: float
    max_drawdown_equity: float
    max_unrealized_loss: float
    average_gross_exposure: float
    buy_trade_count: int
    sell_trade_count: int
    forced_liquidations: int
    min_maintenance_ratio: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _cfg_with_execution(base_cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Build a Config whose execution D2 fields are reset to defaults then
    overridden, so each scenario is independent of the caller's execution state
    (the "baseline" cell is a true next_open / full-fill / zero-delay run even if
    the input config already set non-default D2 fields)."""
    fields = {"fill_model": "next_open", "volume_participation": 1.0, "execution_delay_days": 0}
    fields.update(overrides)
    exec_cfg = replace(base_cfg.execution, **fields)
    return replace(base_cfg, execution=exec_cfg)


def compare_execution(
    md: MarketData,
    base_cfg: Config,
    params_provider: ParamsProvider,
    scenarios: Sequence[Dict[str, Any]] = EXECUTION_SCENARIOS,
) -> List[ExecutionCell]:
    """Replay ``params_provider`` under each execution scenario."""
    cells: List[ExecutionCell] = []
    for sc in scenarios:
        cfg = _cfg_with_execution(base_cfg, sc["overrides"])
        res = simulate(md, 0, md.n, params_provider, cfg, record=False)
        pf = res.portfolio
        avg_exposure = (pf.exposure_sum / pf.exposure_obs) if pf.exposure_obs else 0.0
        min_maint = (
            pf.min_maintenance_ratio_seen
            if np.isfinite(pf.min_maintenance_ratio_seen)
            else None
        )
        cells.append(
            ExecutionCell(
                name=sc["name"],
                overrides=dict(sc["overrides"]),
                net_realized_after_tax=pf.realized_after_tax,
                final_equity=res.final_equity,
                max_drawdown_equity=res.max_drawdown_equity,
                max_unrealized_loss=pf.max_unrealized_loss,
                average_gross_exposure=avg_exposure,
                buy_trade_count=pf.buy_trade_count,
                sell_trade_count=pf.sell_trade_count,
                forced_liquidations=pf.forced_liquidation_count,
                min_maintenance_ratio=min_maint,
            )
        )
    return cells


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def write_execution_outputs(
    out_dir: str | Path,
    cells: Sequence[ExecutionCell],
    *,
    meta: Optional[Dict[str, Any]] = None,
    charts: bool = True,
) -> Dict[str, Any]:
    """Write execution.json + execution.csv (+ chart) and return the summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"meta": meta or {}, "scenarios": [c.to_dict() for c in cells]}
    (out / "execution.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([c.to_dict() for c in cells]).to_csv(out / "execution.csv", index=False)
    if charts and cells:
        _write_execution_chart(out, cells)
    return summary


def _write_execution_chart(out: Path, cells: Sequence[ExecutionCell]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    names = [c.name for c in cells]
    net = [c.net_realized_after_tax / 1e6 for c in cells]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in net]
    ax.bar(np.arange(len(names)), net, color=colors)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("net realized after tax (M JPY)")
    ax.set_title("Execution-model impact on net realized P&L")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "execution_impact.png", dpi=110)
    plt.close(fig)
