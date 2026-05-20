#!/usr/bin/env python3
"""Backtest single + 2-way + 3-way AND combos of Nikkei 225 entry signals.

Strategy mechanics (from spec):
- Universe: ^N225, daily, past 5 years (yfinance period="5y").
- Entry: at next bar's open after the signal fires (no look-ahead).
- One position at a time (signals while holding are ignored).
- Take-profit levels evaluated independently: +5 / +10 / +15 / +20%.
- Exit: when intraday High of any day on/after the entry bar reaches
  entry * (1 + tp/100). Fill price = that day's *close* (conservative —
  assumes the touch and possible giveback).
- Max hold = 250 trading days; otherwise close at the 250th-day close
  and mark as failure.
- No commissions, no taxes, no slippage, no dividends.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf

from indicators import compute_indicators, define_signals

TP_LEVELS = (5, 10, 15, 20)
MAX_HOLD = 250
MIN_SIGNALS = 20
TICKER = "^N225"


# ----------------------------- Data ---------------------------------------- #


def fetch_data() -> pd.DataFrame:
    """Fetch 5y daily OHLC for ^N225. yfinance returns the index timezone-aware
    for indices; strip it so date arithmetic is straightforward."""
    df = yf.Ticker(TICKER).history(period="5y", auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty data for ^N225")
    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df.dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ----------------------------- Simulation ---------------------------------- #


@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    days: int
    entry_price: float
    exit_price: float
    return_pct: float
    hit_tp: bool
    mfe: float  # max favourable excursion (%)
    mae: float  # max adverse excursion (%)


@dataclass
class StratStats:
    combo: tuple[str, ...]
    tp_pct: int
    n_trades: int
    hit_rate: float            # %
    avg_days_to_tp: float | None
    avg_win_return: float
    failure_mean: float        # %
    failure_median: float      # %
    avg_mfe: float
    avg_mae: float
    expected_value: float      # %
    n_signals_raw: int
    trades: list[Trade] = field(default_factory=list)


def simulate(signal: np.ndarray,
             opens: np.ndarray,
             highs: np.ndarray,
             lows: np.ndarray,
             closes: np.ndarray,
             dates: np.ndarray,
             tp_pct: int,
             max_hold: int = MAX_HOLD) -> list[Trade]:
    """Single-position simulation."""
    trades: list[Trade] = []
    n = len(signal)
    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    mfe = 0.0
    mae = 0.0
    tp_mult = 1.0 + tp_pct / 100.0

    for i in range(n):
        if in_pos:
            # Update MFE/MAE on this bar (intraday extremes).
            up = (highs[i] - entry_price) / entry_price * 100.0
            dn = (lows[i] - entry_price) / entry_price * 100.0
            if up > mfe:
                mfe = up
            if dn < mae:
                mae = dn

            days_held = i - entry_idx
            hit_tp = highs[i] >= entry_price * tp_mult

            if hit_tp:
                exit_price = closes[i]
                ret = (exit_price - entry_price) / entry_price * 100.0
                trades.append(Trade(
                    entry_date=dates[entry_idx],
                    exit_date=dates[i],
                    days=days_held,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=ret,
                    hit_tp=True,
                    mfe=mfe,
                    mae=mae,
                ))
                in_pos = False
            elif days_held >= max_hold:
                exit_price = closes[i]
                ret = (exit_price - entry_price) / entry_price * 100.0
                trades.append(Trade(
                    entry_date=dates[entry_idx],
                    exit_date=dates[i],
                    days=days_held,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=ret,
                    hit_tp=False,
                    mfe=mfe,
                    mae=mae,
                ))
                in_pos = False
        else:
            # Try to open a position the day after a signal.
            if signal[i] and i + 1 < n:
                entry_idx = i + 1
                entry_price = opens[entry_idx]
                mfe = 0.0
                mae = 0.0
                in_pos = True

    # Force-close any open position at the very last bar (treated as failure).
    if in_pos:
        i = n - 1
        if i > entry_idx:  # we did get to next bar at least
            exit_price = closes[i]
            ret = (exit_price - entry_price) / entry_price * 100.0
            trades.append(Trade(
                entry_date=dates[entry_idx],
                exit_date=dates[i],
                days=i - entry_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=ret,
                hit_tp=False,
                mfe=mfe,
                mae=mae,
            ))

    return trades


def aggregate(trades: list[Trade], tp_pct: int,
              combo: tuple[str, ...], n_signals_raw: int) -> StratStats | None:
    if not trades:
        return None
    wins = [t for t in trades if t.hit_tp]
    losses = [t for t in trades if not t.hit_tp]
    n_total = len(trades)
    hit_rate = len(wins) / n_total * 100.0
    avg_days = float(np.mean([t.days for t in wins])) if wins else None
    avg_win_ret = float(np.mean([t.return_pct for t in wins])) if wins else float(tp_pct)
    fail_mean = float(np.mean([t.return_pct for t in losses])) if losses else 0.0
    fail_median = float(np.median([t.return_pct for t in losses])) if losses else 0.0
    mfe = float(np.mean([t.mfe for t in trades]))
    mae = float(np.mean([t.mae for t in trades]))
    ev = hit_rate / 100.0 * avg_win_ret + (1 - hit_rate / 100.0) * fail_mean
    return StratStats(
        combo=combo,
        tp_pct=tp_pct,
        n_trades=n_total,
        hit_rate=hit_rate,
        avg_days_to_tp=avg_days,
        avg_win_return=avg_win_ret,
        failure_mean=fail_mean,
        failure_median=fail_median,
        avg_mfe=mfe,
        avg_mae=mae,
        expected_value=ev,
        n_signals_raw=n_signals_raw,
        trades=trades,
    )


# ----------------------------- Combo search -------------------------------- #


def run_all(df: pd.DataFrame, label: str = "全期間") -> list[StratStats]:
    """Run all single/2-way/3-way combos on the supplied dataframe."""
    sigs = define_signals(df)
    names = list(sigs.keys())

    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    dates = df.index.to_numpy()

    combos: list[tuple[str, ...]] = []
    for r in (1, 2, 3):
        combos.extend(combinations(names, r))

    out: list[StratStats] = []
    t0 = time.time()
    for combo in combos:
        sig = sigs[combo[0]].to_numpy()
        for nm in combo[1:]:
            sig = sig & sigs[nm].to_numpy()
        n_raw = int(sig.sum())
        if n_raw < MIN_SIGNALS:
            continue
        for tp in TP_LEVELS:
            trades = simulate(sig, opens, highs, lows, closes, dates, tp)
            s = aggregate(trades, tp, combo, n_raw)
            if s:
                out.append(s)
    elapsed = time.time() - t0
    print(f"[{label}] {len(combos)} combos screened, "
          f"{len(out) // len(TP_LEVELS) if out else 0} passed ≥{MIN_SIGNALS}-signals "
          f"filter, {len(out)} stat rows. ({elapsed:.1f}s)")
    return out


# ----------------------------- Benchmarks ---------------------------------- #


def buy_and_hold(df: pd.DataFrame) -> dict:
    first = df["Close"].iloc[0]
    last = df["Close"].iloc[-1]
    ret = (last - first) / first * 100.0
    days = (df.index[-1] - df.index[0]).days
    return {"return_pct": ret, "days": days,
            "annualized_pct": ((last / first) ** (365.0 / max(days, 1)) - 1) * 100.0}


def always_buy(df: pd.DataFrame, tp_pct: int) -> dict:
    """If you bought at *every* close (sequential, single-pos) and exited at
    +tp / 250d, what % of entries would hit TP?"""
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    dates = df.index.to_numpy()
    sig = np.ones(len(df), dtype=bool)
    sig[-1] = False  # need a next-day open
    trades = simulate(sig, opens, highs, lows, closes, dates, tp_pct)
    if not trades:
        return {"hit_rate": 0.0, "n_trades": 0, "avg_days_to_tp": None,
                "expected_value": 0.0}
    s = aggregate(trades, tp_pct, ("always_buy",), int(sig.sum()))
    return {
        "hit_rate": s.hit_rate,
        "n_trades": s.n_trades,
        "avg_days_to_tp": s.avg_days_to_tp,
        "expected_value": s.expected_value,
    }


def day_forward_baseline(df: pd.DataFrame, tp_pct: int,
                         max_hold: int = MAX_HOLD) -> dict:
    """For *every* trading day i in the data, ask: if you entered at open of i+1
    and held until either high touched entry*(1+tp/100) or max_hold days, would
    it have hit? This is the proper "random-day baseline" — non-sequential and
    independent across days, so all 1,200+ entry points are evaluated."""
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    n = len(df)
    tp_mult = 1.0 + tp_pct / 100.0
    hits = 0
    fails = 0
    days_to_hit = []
    returns = []
    for i in range(n - 1):  # need next-day open
        entry = opens[i + 1]
        target = entry * tp_mult
        hit = False
        for j in range(i + 1, min(i + 1 + max_hold + 1, n)):
            if highs[j] >= target:
                hits += 1
                days_to_hit.append(j - (i + 1))
                returns.append((closes[j] - entry) / entry * 100.0)
                hit = True
                break
        if not hit:
            # Either reached max_hold or end of data
            last_j = min(i + 1 + max_hold, n - 1)
            if last_j > i + 1:
                fails += 1
                returns.append((closes[last_j] - entry) / entry * 100.0)
    total = hits + fails
    return {
        "hit_rate": hits / total * 100.0 if total else 0.0,
        "n_entries": total,
        "avg_days_to_tp": float(np.mean(days_to_hit)) if days_to_hit else None,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
    }


# ----------------------------- Equity curve -------------------------------- #


def equity_curve(trades: list[Trade], dates: np.ndarray,
                 start_equity: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compound equity curve, one point per trading day in `dates`.

    Equity is constant when flat, multiplies by (1 + return/100) on exit day."""
    eq = np.full(len(dates), start_equity, dtype=np.float64)
    # Build exit-date -> return map
    by_date = {pd.Timestamp(t.exit_date).to_datetime64(): t.return_pct for t in trades}
    current = start_equity
    for i, d in enumerate(dates):
        ret = by_date.get(d)
        if ret is not None:
            current *= 1.0 + ret / 100.0
        eq[i] = current
    return dates, eq


# ----------------------------- CSV ----------------------------------------- #


def to_dataframe(results: list[StratStats]) -> pd.DataFrame:
    rows = []
    for s in results:
        rows.append({
            "combo": " + ".join(s.combo),
            "n_components": len(s.combo),
            "tp_pct": s.tp_pct,
            "n_trades": s.n_trades,
            "n_signals_raw": s.n_signals_raw,
            "hit_rate": s.hit_rate,
            "avg_days_to_tp": s.avg_days_to_tp,
            "avg_win_return": s.avg_win_return,
            "failure_mean": s.failure_mean,
            "failure_median": s.failure_median,
            "avg_mfe": s.avg_mfe,
            "avg_mae": s.avg_mae,
            "expected_value": s.expected_value,
        })
    return pd.DataFrame(rows)


# ----------------------------- HTML report --------------------------------- #


CAVEAT_HTML = """
<div class="caveat">
  <strong>⚠️ 読む前の注意</strong>
  <ul>
    <li>5年バックテストは特定の相場レジームに偏る可能性が高い（直近の上昇トレンドが結果を持ち上げている可能性あり）。</li>
    <li>多重比較（1,350通り超のシグナルを試している）で偽陽性が出やすい。下の<em>Out-of-Sample</em>欄を必ず一緒に見ること。</li>
    <li>手数料・税・スリッページ・配当の<strong>すべて未考慮</strong>の理論値です。</li>
    <li>過去のパフォーマンスは将来を保証しません。</li>
  </ul>
</div>
"""


def format_pct(v, signed=False, prec=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v:.{prec}f}%"


def format_days(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f}日"


def make_ranking_table(df: pd.DataFrame, oos_lookup: dict, top_n: int = 25) -> str:
    """Build an HTML table of the top N strategies by expected value."""
    cols = [
        ("combo", "シグナル"),
        ("tp_pct", "TP"),
        ("n_trades", "回数"),
        ("hit_rate", "到達率"),
        ("avg_days_to_tp", "平均日数"),
        ("avg_win_return", "勝平均"),
        ("failure_mean", "失敗平均"),
        ("avg_mfe", "MFE"),
        ("avg_mae", "MAE"),
        ("expected_value", "期待値"),
        ("oos_hit", "OOS到達率"),
        ("oos_n", "OOS回数"),
    ]
    top = df.head(top_n)
    lines = ["<table class='ranking'><thead><tr>"]
    for _, label in cols:
        lines.append(f"<th>{label}</th>")
    lines.append("</tr></thead><tbody>")
    for _, row in top.iterrows():
        oos = oos_lookup.get((row["combo"], int(row["tp_pct"])), None)
        oos_hit = oos["hit_rate"] if oos else None
        oos_n = oos["n_trades"] if oos else 0
        cells = {
            "combo": row["combo"],
            "tp_pct": f"+{int(row['tp_pct'])}%",
            "n_trades": int(row["n_trades"]),
            "hit_rate": format_pct(row["hit_rate"]),
            "avg_days_to_tp": format_days(row["avg_days_to_tp"]),
            "avg_win_return": format_pct(row["avg_win_return"], signed=True),
            "failure_mean": format_pct(row["failure_mean"], signed=True),
            "avg_mfe": format_pct(row["avg_mfe"], signed=True),
            "avg_mae": format_pct(row["avg_mae"], signed=True),
            "expected_value": format_pct(row["expected_value"], signed=True),
            "oos_hit": format_pct(oos_hit) if oos_hit is not None else "—",
            "oos_n": oos_n if oos else "—",
        }
        ev = row["expected_value"]
        klass = "good" if ev > 0 else "bad"
        lines.append("<tr>")
        for key, _ in cols:
            cls = ""
            if key == "expected_value":
                cls = f" class='{klass}'"
            lines.append(f"<td{cls}>{cells[key]}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "".join(lines)


def make_summary_cards(df: pd.DataFrame, oos_lookup: dict) -> str:
    """One card per TP level with the best in-sample combo and its OOS perf."""
    parts = []
    for tp in TP_LEVELS:
        sub = df[df["tp_pct"] == tp]
        if sub.empty:
            continue
        top = sub.iloc[0]
        oos = oos_lookup.get((top["combo"], tp))
        oos_str = (
            f"OOS 到達率 {oos['hit_rate']:.1f}% ({oos['n_trades']}回)"
            if oos and oos["n_trades"] > 0
            else "OOS 該当なし"
        )
        parts.append(f"""
        <div class="card">
          <div class="card-tp">利確 +{tp}%</div>
          <div class="card-combo">{top['combo']}</div>
          <div class="card-stat">
            <span>到達率 <b>{top['hit_rate']:.1f}%</b></span>
            <span>期待値 <b class="{ 'good' if top['expected_value']>0 else 'bad' }">{format_pct(top['expected_value'], signed=True)}</b></span>
            <span>回数 <b>{int(top['n_trades'])}</b></span>
          </div>
          <div class="card-oos">{oos_str}</div>
        </div>""")
    return "<div class='cards'>" + "\n".join(parts) + "</div>"


def make_equity_chart(top_results: list[StratStats], dates_all: np.ndarray,
                      bnh_curve: tuple[np.ndarray, np.ndarray]) -> str:
    fig = go.Figure()
    # Buy & Hold reference
    bnh_dates, bnh_eq = bnh_curve
    fig.add_trace(go.Scatter(
        x=bnh_dates, y=bnh_eq, mode="lines",
        name="Buy&Hold (参照)", line=dict(width=2, dash="dot", color="#888"),
    ))
    for s in top_results:
        d, eq = equity_curve(s.trades, dates_all)
        label = f"{' + '.join(s.combo)} (TP +{s.tp_pct}%, hit {s.hit_rate:.0f}%)"
        fig.add_trace(go.Scatter(x=d, y=eq, mode="lines", name=label))
    fig.update_layout(
        title="エクイティカーブ (1倍スタート、複利)",
        xaxis_title="日付", yaxis_title="エクイティ倍率",
        legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        margin=dict(l=40, r=20, t=60, b=80),
        height=420,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"responsive": True})


def make_return_hist(results: list[StratStats]) -> str:
    """Histogram of all trade returns from the top strategies, separated by hit/miss."""
    wins, losses = [], []
    for s in results:
        for t in s.trades:
            (wins if t.hit_tp else losses).append(t.return_pct)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wins, nbinsx=20, name="利確トレード",
                               marker_color="#2ed573", opacity=0.8))
    fig.add_trace(go.Histogram(x=losses, nbinsx=20, name="失敗トレード",
                               marker_color="#ff4757", opacity=0.8))
    fig.update_layout(
        barmode="overlay",
        title="上位戦略の全トレード・リターン分布",
        xaxis_title="リターン (%)", yaxis_title="回数",
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"responsive": True})


def make_signal_count_chart(df: pd.DataFrame) -> str:
    """Show distribution of signal counts vs expected value, sized by hit rate."""
    fig = go.Figure()
    for tp in TP_LEVELS:
        sub = df[df["tp_pct"] == tp]
        fig.add_trace(go.Scatter(
            x=sub["n_trades"], y=sub["expected_value"],
            mode="markers",
            name=f"TP +{tp}%",
            marker=dict(size=6, opacity=0.6),
            text=sub["combo"],
            hovertemplate="%{text}<br>回数: %{x}<br>期待値: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        title="シグナル回数 × 期待値 (TP水準別)",
        xaxis_title="トレード回数", yaxis_title="期待値 (%)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"responsive": True})


def build_html(df_is: pd.DataFrame,
               results_all: list[StratStats],
               oos_lookup: dict,
               bnh: dict,
               always_buy_stats: dict[int, dict],
               day_forward_stats: dict[int, dict],
               df_data: pd.DataFrame,
               top_n_charts: int = 5,
               generated_at: str = "") -> str:
    # Plot HTML fragments
    dates_all = df_data.index.to_numpy()
    bnh_first = df_data["Close"].iloc[0]
    bnh_eq = (df_data["Close"] / bnh_first).to_numpy()
    top_results_for_chart = [s for s in results_all
                             if (s.combo, s.tp_pct) in
                             {(tuple(r["combo"].split(" + ")), int(r["tp_pct"]))
                              for _, r in df_is.head(top_n_charts).iterrows()}]
    # Sort top_results_for_chart by EV from df_is
    rank = {(tuple(r["combo"].split(" + ")), int(r["tp_pct"])): i
            for i, (_, r) in enumerate(df_is.head(top_n_charts).iterrows())}
    top_results_for_chart.sort(key=lambda s: rank.get((s.combo, s.tp_pct), 999))

    equity_html = make_equity_chart(top_results_for_chart, dates_all,
                                    (dates_all, bnh_eq))
    hist_html = make_return_hist(top_results_for_chart)
    scatter_html = make_signal_count_chart(df_is)
    ranking_html = make_ranking_table(df_is, oos_lookup, top_n=30)
    summary_html = make_summary_cards(df_is, oos_lookup)

    # Benchmark block
    bm_lines = [
        f"<li>Buy &amp; Hold (5年): "
        f"<b class='{'good' if bnh['return_pct']>0 else 'bad'}'>{bnh['return_pct']:+.2f}%</b> "
        f"(年率 {bnh['annualized_pct']:+.2f}%)</li>"
    ]
    for tp in TP_LEVELS:
        df_st = day_forward_stats.get(tp)
        ab = always_buy_stats.get(tp)
        if df_st:
            bm_lines.append(
                f"<li>「ランダム1日エントリー (全{df_st['n_entries']}日)」TP +{tp}%: "
                f"到達率 <b>{df_st['hit_rate']:.1f}%</b> "
                f"(平均到達 {format_days(df_st['avg_days_to_tp'])}, "
                f"平均リターン {format_pct(df_st['avg_return'], signed=True)})</li>"
            )
        if ab:
            bm_lines.append(
                f"<li class='sub'>参考: いつでも買い (1ポジ・シーケンシャル) TP +{tp}%: "
                f"到達率 {ab['hit_rate']:.1f}% ({ab['n_trades']}回)</li>"
            )

    head = """<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5">
<title>日経225 シグナル・バックテスト</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
  --bg: #fff; --fg: #222; --muted: #666; --line: #ddd;
  --good: #1e8a3c; --bad: #c52a2a; --card: #f7f7fa;
  --caveat-bg: #fff8e1; --caveat-bd: #f1c40f;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #1a1a1a; --fg: #eee; --muted: #aaa; --line: #444;
          --good: #2ed573; --bad: #ff4757; --card: #252535;
          --caveat-bg: #3a300f; --caveat-bd: #f1c40f; }
}
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--fg); margin: 0; padding: 0;
       line-height: 1.5; }
.container { max-width: 1100px; margin: 0 auto; padding: 1.2rem; }
h1 { margin-top: 0.3em; font-size: 1.5em; }
h2 { margin-top: 1.8em; border-bottom: 2px solid var(--line); padding-bottom: 4px; }
h3 { color: var(--muted); font-size: 1em; margin-top: 1.5em; }
.caveat { background: var(--caveat-bg); border-left: 4px solid var(--caveat-bd);
          padding: 0.8em 1em; margin: 1em 0; border-radius: 4px; }
.caveat ul { margin: 0.4em 0 0; padding-left: 1.3em; }
.meta { color: var(--muted); font-size: 0.85em; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
         gap: 0.8em; margin: 1em 0 1.6em; }
.card { background: var(--card); padding: 0.9em 1em; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.card-tp { font-size: 0.85em; color: var(--muted); text-transform: uppercase;
           letter-spacing: 1px; }
.card-combo { font-weight: 700; font-size: 1.05em; margin: 0.3em 0; word-break: break-all; }
.card-stat { display: flex; flex-direction: column; gap: 0.15em; font-size: 0.95em; }
.card-stat b { font-variant-numeric: tabular-nums; }
.card-oos { margin-top: 0.5em; font-size: 0.85em; color: var(--muted);
            border-top: 1px dashed var(--line); padding-top: 0.4em; }
table.ranking { width: 100%; border-collapse: collapse; font-size: 0.85em;
                font-variant-numeric: tabular-nums; }
table.ranking th, table.ranking td { padding: 4px 6px; border-bottom: 1px solid var(--line);
                                     text-align: right; white-space: nowrap; }
table.ranking th:first-child, table.ranking td:first-child {
                                     text-align: left; white-space: normal; min-width: 12em; }
table.ranking thead th { position: sticky; top: 0; background: var(--bg); }
.good { color: var(--good); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }
.scroll-x { overflow-x: auto; -webkit-overflow-scrolling: touch; }
ul.bm { list-style: none; padding-left: 0; }
ul.bm li { padding: 0.3em 0; border-bottom: 1px dashed var(--line); }
@media (max-width: 600px) {
  .container { padding: 0.7rem; }
  h1 { font-size: 1.3em; }
  table.ranking { font-size: 0.75em; }
  table.ranking th, table.ranking td { padding: 3px 4px; }
}
</style>
</head><body><div class="container">
"""
    body = f"""
<h1>📊 日経225 シグナル・バックテスト</h1>
<p class="meta">過去5年 (日足) ・ 利確水準 +5/+10/+15/+20% ・ 最大保有 250営業日 ・
   1ポジション運用 ・ 翌日寄付エントリー ・ 手数料/税/配当/スリッページ未考慮</p>
<p class="meta">生成日時: {generated_at} (UTC) ・
   データソース: yfinance <code>^N225</code></p>

{CAVEAT_HTML}

<h2>① 各利確水準のトップ戦略</h2>
<p>In-Sample (過去4年) で期待値が最も高かったシグナル組み合わせ。OOSは直近1年の検証成績。</p>
{summary_html}

<h2>② ベンチマーク</h2>
<ul class="bm">
{''.join(bm_lines)}
</ul>
<p class="meta">「いつでも買い」= 毎日エントリーした場合の1ポジション運用結果（次のシグナルを待たずに即エントリー）。
これを上回らないシグナルは "ただランダムに買ってるのと変わらない"。</p>

<h2>③ ランキング表 (期待値順 上位30件)</h2>
<p class="meta">緑=期待値プラス, 赤=期待値マイナス。OOS列は直近1年での再現結果。OOSのhitが低い、または回数0/1なら過剰適合の疑い。</p>
<div class="scroll-x">{ranking_html}</div>

<h2>④ エクイティカーブ (上位5戦略 + Buy&amp;Hold)</h2>
{equity_html}

<h2>⑤ トレード・リターン分布</h2>
<p class="meta">上位5戦略の全トレードを「利確トレード(緑)」と「失敗トレード(赤)」に分けた頻度分布。</p>
{hist_html}

<h2>⑥ シグナル回数 vs 期待値 (全候補)</h2>
<p class="meta">右下に外れているものは「回数は多いが期待値マイナス」=単に長期保有しただけのケース。
左上の点ほど効率の良いシグナル候補。点にカーソルを当てるとシグナル名が出ます。</p>
{scatter_html}

<h2>⑦ メソッドメモ</h2>
<ul>
  <li><strong>エントリー</strong>: シグナル発生日の翌日<em>始値</em>で約定。1ポジション運用、保有中の新シグナルは無視。</li>
  <li><strong>エグジット</strong>: 当日<em>高値</em>が利確水準にタッチした日の<em>終値</em>で利確（保守的）。
      最大保有250営業日で強制決済 (=失敗扱い)。</li>
  <li><strong>シグナル定義</strong>: 単一/2-way AND/3-way ANDを全網羅 (合計1,350通り)。
      5年間で発生回数20回未満のものは除外。</li>
  <li><strong>In-Sample / Out-of-Sample</strong>: 直近約252営業日を OOS（検証）として切り出し、
      残りで IS のランキング作成。OOS のヒット率と回数を併記して過剰適合の度合いを示しています。</li>
  <li><strong>MFE/MAE</strong>: 保有中に到達した最大含み益 / 最大含み損 (% で、各トレード平均)。
      MAE が大きく負なら、利確まで耐える勇気が要る戦略。</li>
</ul>

<p class="meta">Generated by <code>backtest/backtest.py</code>.</p>
</div></body></html>"""
    return head + body


# ----------------------------- Entrypoint ---------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent),
                        help="Output directory (default: this script's folder)")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    print("Fetching ^N225 5y daily data...")
    df = fetch_data()
    print(f"  rows={len(df)}, first={df.index[0].date()}, last={df.index[-1].date()}")

    print("Computing indicators...")
    df = compute_indicators(df)

    # Out-of-sample split: last ~252 bars are OOS, the rest is IS.
    if len(df) > 252 + 60:
        cutoff = len(df) - 252
        df_is = df.iloc[:cutoff].copy()
        df_oos = df.iloc[cutoff - 200:].copy()  # carry warm-up
        oos_start = df.index[cutoff].date()
    else:
        df_is = df
        df_oos = None
        oos_start = None
    print(f"In-sample bars: {len(df_is)} ; OOS starts at {oos_start}")

    print("Running In-Sample combo screen...")
    results_is = run_all(df_is, "IS")
    df_is_table = to_dataframe(results_is)
    df_is_table = df_is_table.sort_values("expected_value", ascending=False).reset_index(drop=True)

    # Run the same combos on full data so charts show whole 5y equity curve.
    # We re-simulate just for the top set to keep things tractable.
    top_keys = {(tuple(r["combo"].split(" + ")), int(r["tp_pct"]))
                for _, r in df_is_table.head(50).iterrows()}

    print("Re-simulating top combos on full 5y for charts...")
    full_results: list[StratStats] = []
    sigs_full = define_signals(df)
    opens = df["Open"].to_numpy(); highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy(); closes = df["Close"].to_numpy()
    dates = df.index.to_numpy()
    for combo, tp in top_keys:
        sig = sigs_full[combo[0]].to_numpy()
        for nm in combo[1:]:
            sig = sig & sigs_full[nm].to_numpy()
        trades = simulate(sig, opens, highs, lows, closes, dates, tp)
        s = aggregate(trades, tp, combo, int(sig.sum()))
        if s:
            full_results.append(s)

    # OOS lookup table for the strategies we care about.
    oos_lookup: dict[tuple[str, int], dict] = {}
    if df_oos is not None:
        print("Evaluating same strategies on OOS slice...")
        sigs_oos = define_signals(df_oos)
        # Only the OOS portion's dates for entries; but our simulate above already
        # iterates on rows. We pass df_oos but include the warm-up rows so signals
        # have valid history; trades opened *before* OOS start are filtered out.
        opens_o = df_oos["Open"].to_numpy(); highs_o = df_oos["High"].to_numpy()
        lows_o = df_oos["Low"].to_numpy(); closes_o = df_oos["Close"].to_numpy()
        dates_o = df_oos.index.to_numpy()
        oos_start_ts = pd.Timestamp(oos_start).to_datetime64() if oos_start else None
        for combo, tp in top_keys:
            sig = sigs_oos[combo[0]].to_numpy()
            for nm in combo[1:]:
                sig = sig & sigs_oos[nm].to_numpy()
            trades = simulate(sig, opens_o, highs_o, lows_o, closes_o, dates_o, tp)
            # Keep only trades that opened on/after OOS start
            if oos_start_ts is not None:
                trades = [t for t in trades
                          if pd.Timestamp(t.entry_date).to_datetime64() >= oos_start_ts]
            s = aggregate(trades, tp, combo, len(trades))
            if s:
                oos_lookup[(" + ".join(combo), tp)] = {
                    "hit_rate": s.hit_rate,
                    "n_trades": s.n_trades,
                    "expected_value": s.expected_value,
                }
            else:
                oos_lookup[(" + ".join(combo), tp)] = {
                    "hit_rate": 0.0, "n_trades": 0, "expected_value": 0.0,
                }

    print("Computing benchmarks...")
    bnh = buy_and_hold(df)
    always_buy_stats = {tp: always_buy(df, tp) for tp in TP_LEVELS}
    day_forward_stats = {tp: day_forward_baseline(df, tp) for tp in TP_LEVELS}

    # --- Write CSV ---
    df_is_table.to_csv(out_dir / "data" / "results.csv", index=False, float_format="%.4f")
    print(f"Wrote {out_dir / 'data' / 'results.csv'} ({len(df_is_table)} rows)")

    # --- Write JSON of summary for downstream tooling ---
    with open(out_dir / "data" / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "ticker": TICKER,
            "rows_total": len(df),
            "rows_in_sample": len(df_is),
            "oos_start": str(oos_start) if oos_start else None,
            "buy_and_hold": bnh,
            "always_buy": always_buy_stats,
            "day_forward": day_forward_stats,
            "top_5": df_is_table.head(5).to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2, default=str)

    # --- Build HTML ---
    html = build_html(
        df_is=df_is_table,
        results_all=full_results,
        oos_lookup=oos_lookup,
        bnh=bnh,
        always_buy_stats=always_buy_stats,
        day_forward_stats=day_forward_stats,
        df_data=df,
        top_n_charts=5,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"Wrote {out_dir / 'report.html'} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
