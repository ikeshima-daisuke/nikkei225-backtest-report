# Strategy variants — design notes

A **self-contained** add-on to the core simulator (`src/nikkei_leverage_sim/`).
It changes nothing in the core package: it imports the validated `Portfolio`,
`strategy` scoring and `metrics` and layers new **entry/exit rules** on top of
the same daily DCA buy engine. The core test-suite is untouched and still green.

> Why a separate package? A parallel task is implementing a *forced-liquidation*
> model that edits the core engine. Keeping the variants fully isolated means the
> two efforts cannot collide — this folder only ever **reads** the core API.

## The variants

| `variant`  | Entry                         | Exit rule |
|------------|-------------------------------|-----------|
| `per_lot`  | DCA (+ optional initial lump) | existing per-lot take-profit (the published baseline) |
| `bulk_yen` | DCA (+ lump)                  | when **aggregate** net unrealized P&L ≥ `bulk_exit_yen`, sell **all** lots at the **next open**, reset, resume DCA |
| `bulk_pct` | DCA (+ lump)                  | when the day's **high** ≥ `avg_entry × (1 + bulk_exit_pct)`, sell **all** lots at that day's **close**, reset |
| `combo`    | DCA (+ lump)                  | both bulk rules active (pct fires same-day, yen next-open) |

* **Initial lump sum** (`initial_amount`, variant A): a one-off fixed buy folded
  into the very first buy decision (executed next open, capped by gross
  exposure). `0` = pure DCA; `≥` the exposure cap ≈ buy-and-hold.
* **Bulk exits sell everything**, including individually underwater lots, so they
  realize an *aggregate* result. Tax is applied to the **netted** positive gain
  (gains/losses offset, as in a tokutei-kouza same-day close) — this is the fair
  comparison against the per-lot strategy, which never realizes a loss and so has
  no offset to surrender.

## How the grid stays both realistic and fast

The only expensive part of a real run is the **fast walk-forward** that
re-selects buy params every 5 days (~2.5 min). We:

1. run that walk-forward **once** and **capture the exact per-day params**
   (`wf_capture.py`) — fully causal, identical to the published fast run;
2. **replay** that fixed per-day buy sequence through the cheap variant engine
   (~0.1 s/combo) for **every** grid combo.

So every combo buys *exactly like the published fast strategy*; only the
entry/exit rule differs — a clean controlled comparison. Replaying the captured
sequence with `per_lot / init0` reproduces the published fast result to the yen
(¥10,482,989 / max DD ¥5.26M / max unrealized loss ¥5.28M), which is asserted as
a validation check.

## Robustness: two buy engines

`run_all.py` runs the full grid under **two** buy engines and writes both:

* `outputs_variants/walkforward/` — replayed fast walk-forward (primary).
* `outputs_variants/fixed_default/` — the config's fixed default params.

If the exit-rule ranking agrees across both, it isn't an artifact of the
particular buy params.

## Grid (432 combos)

* `initial_amount ∈ {0, 0.5, 1, 2, 5, 10} M`
* `bulk_exit_yen ∈ {0.1, 0.3, 0.5, 1, 2, 5, 10} M`
* `bulk_exit_pct ∈ {3, 5, 8, 10, 12, 15, 20, 25} %`
* `per_lot` (6) + `bulk_yen` (6×7) + `bulk_pct` (6×8) + `combo` (6×7×8) = **432**

## Run it

```bash
cd nikkei_leverage_sim
PYTHONPATH="src;." python -m variants.run_all --out outputs_variants
# data/target_1570_T.csv and data/benchmark_N225.csv must be present
```

Outputs: `summary_all_variants.csv`, `top_per_category.md`,
`variants_comparison.png` (+ per-engine subfolders). Tests:
`PYTHONPATH="src;." python -m pytest variants/tests -q`.
