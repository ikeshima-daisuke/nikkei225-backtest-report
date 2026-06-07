"""Statistical validation of the regime replay.

The descriptive replay (:mod:`run_study`) shows the bull-era ranking *inverting*
in the lost decades (exits beat hold).  But is that inversion itself signal, or
just a different draw of noise?  This module reuses ``accumulation_study.validate``
verbatim — block-bootstrap Calmar CIs, the hold-vs-best-exit pairwise difference,
permutation null + Benjamini-Hochberg FDR, out-of-sample split, cost sensitivity
— but on the **synthetic lost-decade path** instead of the real bull path.

It writes one ``validation_<regime>.json`` per regime validated.  Runtime scales
with ``n_boot``/``n_perm``; defaults are smaller than the accumulation study to
stay tractable on the long (≈5,900-day) windows (documented in the report).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
for _p in (str(_PKG / "src"), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402
from typing import List  # noqa: E402

from accumulation_study.validate import run_validation  # noqa: E402

from .build_target import calibrate_base_drag, read_ohlc, synth_close_path, write_daily_csv  # noqa: E402
from .regimes import REGIME_BY_KEY  # noqa: E402
from nikkei_leverage_sim.data import read_ohlc_csv  # noqa: E402

# Regimes worth the (expensive) statistical battery: the iconic lost decade, the
# full two-decade window, and the bull baseline for contrast.
DEFAULT_KEYS = ["lost_decade_1", "two_lost_decades", "bull_real"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n225", default="data/benchmark_N225_long.csv")
    ap.add_argument("--real", default="data/target_1570_T.csv")
    ap.add_argument("--out", default="regime_study/outputs")
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-perm", type=int, default=400)
    args = ap.parse_args(argv)

    n225 = read_ohlc(args.n225)
    real = read_ohlc_csv(args.real)[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    calib = calibrate_base_drag(n225, real)
    base_drag, calib_rate = calib["base_drag"], calib["calib_rate"]

    # Build the synthetic close ONCE over the whole series, then slice per regime
    # (consistent first-bar anchoring with run_study/survival).
    full_close = synth_close_path(n225, base_drag=base_drag, calib_rate=calib_rate)

    out = Path(args.out)
    for key in args.keys:
        regime = REGIME_BY_KEY[key]
        close = full_close.loc[regime.start:regime.end]
        rdir = out / key
        rdir.mkdir(parents=True, exist_ok=True)
        daily_csv = rdir / "daily.csv"
        write_daily_csv(close, daily_csv)

        print(f"[{key}] validating {len(close)}d "
              f"(n_boot={args.n_boot}, n_perm={args.n_perm}) ...")
        res = run_validation(str(daily_csv), n_boot=args.n_boot, n_perm=args.n_perm)
        res["regime"] = {"key": key, "label": regime.label,
                         "start": str(close.index[0].date()), "end": str(close.index[-1].date())}
        (rdir / "validation.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        o = res["out_of_sample"]
        any_sig = any(v["reject_fdr05"] for v in res["permutation_fdr"].values())
        print(f"  OOS in-sample winner {o['in_sample_winner']} -> OOS rank "
              f"{o['winner_oos_rank']}/{o['n_plans']};  any FDR-significant: {any_sig}")
        print("  wrote", rdir / "validation.json")


if __name__ == "__main__":
    main()
