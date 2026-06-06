"""Accumulation-and-exit comparison study for 1570.T (cash-only, ¥10M).

A **fully self-contained** sub-package (mirrors ``variants/``): it imports the
validated core only for metric helpers and never modifies it.  The question it
answers is the user's: *with a fixed pot of usable capital, which way of
**buying in** and **selling out** best avoids large draw-downs while maximising
profit?*  Plans pair an **accumulation policy** (how to deploy cash over time)
with an **exit policy** (when to sell), run on the real 1570.T close series, and
are ranked by Calmar (CAGR ÷ max draw-down) on the **deployed capital** basis —
the honest denominator established in ``REPORT_REAL.md §10``.

No leverage is added beyond the ETF's built-in 2x; cash earns 0%; fractional
shares are allowed (an idealisation, flagged in the report); decisions are
strictly lookahead-safe (day ``t`` sees only closes up to ``t``).
"""
