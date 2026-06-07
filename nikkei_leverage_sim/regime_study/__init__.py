"""Adverse-regime robustness study for the 2x leveraged-ETF strategy.

This independent sub-package (no core changes) replays the *whole* accumulation×
exit comparison — and the margin/ruin survival test — on Japan's **actual lost
decades** (real ``^N225`` 1989–2013, before 1570.T existed), to ask whether the
tail-wind-period conclusions (``hold`` wins, 2x accumulation survives) hold up in
a real 20-year sideways/bear grind where daily-rebalanced leverage decays.

The synthetic 1570.T is reconstructed from real N225 daily returns at 2x with a
volatility-decay-aware, **time-varying JPY financing** drag (high rates in the
early 1990s make leverage far more expensive than the ZIRP era).  The
construction is calibrated and validated against the *real* 1570.T over
2014–2026 (see :mod:`regime_study.build_target`).

Layout mirrors ``accumulation_study/``: ``financing`` → ``build_target`` →
``regimes`` → ``run_study`` → ``validate`` → ``make_report``.
"""
