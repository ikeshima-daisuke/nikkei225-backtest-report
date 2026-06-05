"""Configuration models for the Nikkei leverage ETF margin-long simulator.

Everything is expressed with plain :mod:`dataclasses` so the package has no hard
dependency on pydantic.  Configuration can be loaded from YAML or JSON via
:func:`load_config`.

The searchable strategy parameters live in :class:`StrategyParams` (a slotted
dataclass for speed, because the optimizer evaluates it inside tight loops).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict

# YAML is optional at import time; JSON always works.
try:  # pragma: no cover - exercised indirectly
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(slots=True)
class StrategyParams:
    """All strategy parameters that the optimizer is allowed to search.

    The first block controls *how much* to buy each day, the second block
    controls *when* to take profit on an individual lot.
    """

    # --- Buy sizing parameters ---
    base_buy_amount: float = 200_000.0
    max_daily_buy_amount: float = 1_000_000.0
    score_threshold: float = 0.0
    score_scale: float = 2.0
    w_drawdown: float = 2.5
    w_rsi: float = 1.5
    w_ma_gap_25: float = 1.5
    w_ret_5: float = 1.0
    w_trend: float = 0.5
    w_vol: float = 2.0
    w_exposure: float = 5.0
    w_unrealized_loss: float = 5.0

    # --- Take-profit parameters ---
    fixed_profit_yen: float = 5_000.0
    base_take_profit_pct: float = 0.006
    min_take_profit_pct: float = 0.002
    exposure_tp_sensitivity: float = 0.5
    vol_tp_multiplier: float = 1.0

    def replace(self, **kwargs: Any) -> "StrategyParams":
        """Return a copy with the given fields overridden."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class OptimizationConfig:
    enabled: bool = True
    method: str = "random"  # "random" | "grid"
    random_seed: int = 42
    n_trials: int = 100
    lookback_days: int = 252
    min_train_days: int = 200
    apply_days: int = 1
    rebalance_frequency: str = "daily"  # "daily" | "weekly"


@dataclass
class ExecutionConfig:
    signal_timing: str = "close_t"
    execution_timing: str = "next_open"
    valuation_price: str = "close"  # "close" | "adj_close"
    # --- Execution-realism model (D2) ---
    # fill_model: how the execution price is formed on the execution day.
    #   "next_open" (default) — open * (1 ± slippage); fills use only the open, so
    #                           the no-look-ahead guarantee is unchanged.
    #   "vwap"      — ((O+H+L+C)/4) * (1 ± slippage); a VWAP proxy over the day.
    #   "adverse"   — buy at High*(1+slip), sell at Low*(1-slip); worst-case fill.
    fill_model: str = "next_open"
    # volume_participation < 1.0 caps BUY shares at participation * day volume
    # (a partial-fill model for the accumulation side; sells stay whole-lot).
    volume_participation: float = 1.0
    # execution_delay_days adds latency beyond the standard next-open execution
    # (0 = decide at close_t, fill at open_{t+1}; 1 = fill at open_{t+2}, ...).
    execution_delay_days: int = 0


@dataclass
class StrategyConfig:
    default_params: StrategyParams = field(default_factory=StrategyParams)


@dataclass
class ObjectiveConfig:
    weight_max_drawdown_equity: float = 0.5
    weight_max_unrealized_loss: float = 0.5
    margin_call_penalty: float = 1_000_000.0
    exposure_limit_hit_penalty: float = 100_000.0
    no_take_profit_streak_penalty: float = 50_000.0
    no_take_profit_grace_days: int = 20


@dataclass
class Config:
    """Top-level configuration object."""

    initial_equity: float = 100_000_000.0
    max_gross_exposure: float = 10_000_000.0
    maintenance_margin_ratio: float = 0.30
    warning_margin_ratio: float = 0.50
    force_liquidation: bool = False

    commission_bps: float = 0.0
    slippage_bps: float = 2.0
    annual_margin_interest_rate: float = 0.028
    tax_rate: float = 0.20315

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)


def _filter_known(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that correspond to dataclass fields of ``cls``."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


def config_from_dict(data: Dict[str, Any]) -> Config:
    """Build a :class:`Config` from a nested dictionary.

    Unknown keys raise ``ValueError`` so typos in YAML do not silently get
    ignored.
    """
    data = dict(data)  # shallow copy so we can pop sub-sections

    opt = data.pop("optimization", {}) or {}
    exe = data.pop("execution", {}) or {}
    strat = data.pop("strategy", {}) or {}
    obj = data.pop("objective", {}) or {}

    default_params = (strat or {}).get("default_params", {}) or {}
    strategy = StrategyConfig(
        default_params=StrategyParams(**_filter_known(StrategyParams, default_params))
    )

    return Config(
        **_filter_known(Config, data),
        optimization=OptimizationConfig(**_filter_known(OptimizationConfig, opt)),
        execution=ExecutionConfig(**_filter_known(ExecutionConfig, exe)),
        strategy=strategy,
        objective=ObjectiveConfig(**_filter_known(ObjectiveConfig, obj)),
    )


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML or JSON file.

    The file extension decides the parser (``.json`` -> JSON, otherwise YAML).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML config files")
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError("Top level of config must be a mapping")
    return config_from_dict(data)
