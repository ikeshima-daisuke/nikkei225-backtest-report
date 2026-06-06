"""Statistical validation of the accumulation×exit study.

The grid in :mod:`run_study` is a *descriptive* in-sample horse race; the best
of 214 noisy metrics is upward-biased (winner's curse).  This module asks the
harder questions, reusing the project's stance from ``nikkei_leverage_sim.
validation`` (permutation null, block bootstrap, Benjamini-Hochberg FDR) but
**re-running each plan on each resampled price path** (the statistically correct
unit for a strategy comparison, not the realized equity returns):

1. **Block-bootstrap CIs** for each plan's Calmar (moving blocks preserve vol
   clustering), plus bootstrap CIs and p-values for the *pairwise differences*
   that the headline rests on (VA vs lump, VA vs DCA, best-hold vs best-exit).
2. **Permutation test** — shuffle daily returns to destroy timing structure,
   re-run each plan, p = P(null Calmar >= observed).
3. **Benjamini-Hochberg FDR** over those permutation p-values.
4. **Out-of-sample split** — pick the in-sample (first-half) Calmar winner, then
   read its rank on the untouched second half.
5. **Cost sensitivity** — re-rank all plans at a round-trip fee.

Everything is seeded for reproducibility; no network, no core changes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nikkei_leverage_sim.validation import benjamini_hochberg

from .engine import evaluate, month_start_indices, simulate
from .run_study import CAPITAL, build_grid, load_prices

PKG = Path(__file__).resolve().parent


def _calmar_on_path(plan, closes, buy_idx, signals, cost_bps=0.0) -> float:
    res = simulate(closes, CAPITAL, plan.accumulation(), plan.exit(),
                   buy_day_indices=buy_idx, repeated=plan.repeated,
                   signals=signals, cost_bps=cost_bps)
    return evaluate(res, CAPITAL, len(closes)).calmar


def _log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(closes))


def _path_from_logrets(p0: float, logrets: np.ndarray) -> np.ndarray:
    out = np.empty(logrets.size + 1)
    out[0] = p0
    out[1:] = p0 * np.exp(np.cumsum(logrets))
    return out


def _moving_block(logrets: np.ndarray, block: int, rng) -> np.ndarray:
    n = logrets.size
    max_start = max(1, n - block)
    idx: List[int] = []
    while len(idx) < n:
        s = int(rng.integers(0, max_start + 1))
        idx.extend(range(s, min(s + block, n)))
    return logrets[np.asarray(idx[:n], dtype=int)]


def _ci(samples: np.ndarray, conf: float = 0.95) -> Tuple[float, float]:
    tail = (1.0 - conf) / 2.0
    return (float(np.percentile(samples, tail * 100.0)),
            float(np.percentile(samples, (1.0 - tail) * 100.0)))


def _build_signals(closes):
    # local import keeps signals construction in one place
    from .signals import build_signals
    return build_signals(closes)


def run_validation(
    prices_path: str,
    *,
    n_boot: int = 1000,
    block: int = 21,
    n_perm: int = 1000,
    cost_bps: float = 20.0,
    seed: int = 42,
    top_k: int = 10,
) -> Dict:
    dates, closes = load_prices(prices_path)
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    n_months = len(month_start_indices(dates))
    buy_idx = month_start_indices(dates)
    base_sig = _build_signals(closes)

    plans = build_grid(n, n_months)
    by_label = {p.label: p for p in plans}
    # Validate the monthly plans only (daily are sensitivity); keep the panel
    # to the headline contenders + controls so the bootstrap stays affordable.
    monthly = [p for p in plans if p.cadence == "monthly"]
    obs = {p.label: _calmar_on_path(p, closes, buy_idx, base_sig) for p in monthly}
    ranked = sorted(obs, key=lambda k: obs[k], reverse=True)
    panel = ranked[:top_k]
    for ctrl in ("lump|hold|monthly|1shot", "dca48|hold|monthly|1shot",
                 "mom252|momexit|monthly|rot"):
        if ctrl in obs and ctrl not in panel:
            panel.append(ctrl)
    # Best exit-using plan (exit != hold) for the hold-vs-exit test.
    best_exit = max((l for l in obs if l.split("|")[1] != "hold"), key=lambda l: obs[l])
    if best_exit not in panel:
        panel.append(best_exit)

    logr = _log_returns(closes)
    p0 = float(closes[0])

    # --- 1) Block bootstrap: Calmar distribution per panel plan ---------------
    rng = np.random.default_rng(seed)
    boot = {l: np.empty(n_boot) for l in panel}
    for b in range(n_boot):
        path = _path_from_logrets(p0, _moving_block(logr, block, rng))
        sig = _build_signals(path)
        for l in panel:
            boot[l][b] = _calmar_on_path(by_label[l], path, buy_idx, sig)

    boot_ci = {l: {"observed": obs[l], "ci_low": _ci(boot[l])[0],
                   "ci_high": _ci(boot[l])[1], "median": float(np.median(boot[l]))}
               for l in panel}

    def _pair(a: str, b: str) -> Dict:
        d = boot[a] - boot[b]
        lo, hi = _ci(d)
        # one-sided bootstrap p that A is NOT better than B
        p = float(np.mean(d <= 0.0))
        return {"a": a, "b": b, "obs_diff": obs[a] - obs[b],
                "ci_low": lo, "ci_high": hi, "p_a_not_gt_b": p}

    best_hold = max((l for l in panel if l.split("|")[1] == "hold"), key=lambda l: obs[l])
    pairs = []
    for a, b in [
        (best_hold, "lump|hold|monthly|1shot"),
        (best_hold, "dca48|hold|monthly|1shot"),
        (best_hold, best_exit),
        ("lump|hold|monthly|1shot", best_exit),
    ]:
        if a in boot and b in boot and a != b:
            pairs.append(_pair(a, b))

    # --- 2) Permutation null (timing destroyed) + 3) FDR ----------------------
    rng2 = np.random.default_rng(seed + 1)
    perm_p = {}
    for l in panel:
        null = np.empty(n_perm)
        for j in range(n_perm):
            path = _path_from_logrets(p0, rng2.permutation(logr))
            sig = _build_signals(path)
            null[j] = _calmar_on_path(by_label[l], path, buy_idx, sig)
        perm_p[l] = float((1 + int(np.sum(null >= obs[l]))) / (n_perm + 1))
    labels_sorted = sorted(perm_p)
    reject, thresh = benjamini_hochberg([perm_p[l] for l in labels_sorted], alpha=0.05)
    fdr = {l: {"p_value": perm_p[l], "reject_fdr05": bool(r)}
           for l, r in zip(labels_sorted, reject)}

    # --- 4) Out-of-sample split (in-sample winner -> OOS rank) ----------------
    half = n // 2
    in_c, out_c = closes[:half], closes[half:]
    in_d, out_d = dates[:half], dates[half:]
    in_idx, out_idx = month_start_indices(in_d), month_start_indices(out_d)
    in_sig, out_sig = _build_signals(in_c), _build_signals(out_c)
    in_cal = {p.label: _calmar_on_path(p, in_c, in_idx, in_sig) for p in monthly}
    out_cal = {p.label: _calmar_on_path(p, out_c, out_idx, out_sig) for p in monthly}
    out_rank = sorted(out_cal, key=lambda k: out_cal[k], reverse=True)
    is_winner = max(in_cal, key=lambda k: in_cal[k])
    oos = {
        "split_date": out_d[0],
        "in_sample_winner": is_winner,
        "in_sample_calmar": in_cal[is_winner],
        "winner_oos_calmar": out_cal[is_winner],
        "winner_oos_rank": out_rank.index(is_winner) + 1,
        "n_plans": len(monthly),
        "oos_top5": [(l, out_cal[l]) for l in out_rank[:5]],
        "in_top5": [(l, in_cal[l]) for l in sorted(in_cal, key=lambda k: in_cal[k], reverse=True)[:5]],
    }

    # --- 5) Cost sensitivity (round-trip fee) ---------------------------------
    cost_cal = {p.label: _calmar_on_path(p, closes, buy_idx, base_sig, cost_bps=cost_bps)
                for p in monthly}
    cost_rank = sorted(cost_cal, key=lambda k: cost_cal[k], reverse=True)
    cost = {
        "cost_bps": cost_bps,
        "top5_no_cost": [(l, obs[l]) for l in ranked[:5]],
        "top5_with_cost": [(l, cost_cal[l]) for l in cost_rank[:5]],
    }

    return {
        "config": {"n_boot": n_boot, "block": block, "n_perm": n_perm,
                   "seed": seed, "cost_bps": cost_bps, "panel_size": len(panel)},
        "bootstrap_calmar_ci": boot_ci,
        "pairwise": pairs,
        "permutation_fdr": fdr,
        "out_of_sample": oos,
        "cost_sensitivity": cost,
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="outputs_real/daily.csv")
    ap.add_argument("--out", default="accumulation_study/outputs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args(argv)

    res = run_validation(args.prices, n_boot=args.n_boot, n_perm=args.n_perm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out / "validation.json")
    o = res["out_of_sample"]
    print(f"  OOS: in-sample winner {o['in_sample_winner']} -> OOS rank "
          f"{o['winner_oos_rank']}/{o['n_plans']} (Calmar {o['winner_oos_calmar']:.3f})")
    for pr in res["pairwise"]:
        print(f"  {pr['a']} vs {pr['b']}: diff {pr['obs_diff']:+.3f} "
              f"CI[{pr['ci_low']:+.3f},{pr['ci_high']:+.3f}] p={pr['p_a_not_gt_b']:.3f}")


if __name__ == "__main__":
    main()
