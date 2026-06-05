# アーキテクチャ（nikkei_leverage_sim）

日経2倍レバETF（1570.T）の「信用ロング・損切りなし・毎日積立・利確」戦略バックテスター。
src レイアウト・pytest・外部通信なし（人工データ＋ローカル実データ）。本書は **モジュール構成 /
データフロー / 不変条件 / 成熟度（L3→L4）** をまとめる。コードは `src/nikkei_leverage_sim/`
（コア 16 モジュール ≒ 4,100 行）と `variants/`（独立オーバーレイ ≒ 1,240 行）。

---

## 1. レイヤ構成（モジュールマップ）

```
                 ┌─────────────────────────── CLI (cli.py) ───────────────────────────┐
                 │  fetch   synth   run        stress      validate     execmodel       │
                 └────┬───────┬──────┬────────────┬────────────┬─────────────┬──────────┘
                      │       │      │            │            │             │
            ┌─────────▼───────▼──────▼─┐   ┌──────▼─────┐ ┌────▼──────┐ ┌────▼───────┐
   入力層    │  data.py                 │   │ stress.py  │ │validation │ │execution.py│   分析層
            │  - load/fetch/synth OHLCV│   │  (D1)      │ │  .py (A)  │ │   (D2)     │  （エンジン上に
            │  - join target+benchmark │   │ レジーム/  │ │ 置換検定/ │ │ VWAP/不利/ │   構築・ロード
            │  - repair_price_glitches │   │ コスト感応 │ │ ブートCI/ │ │ 遅延/部分  │   マップ項目）
            │  - prepare_market_data   │   │ /破綻確率  │ │ FDR       │ │ 約定       │
            └─────────┬────────────────┘   └──────┬─────┘ └────┬──────┘ └────┬───────┘
                      │ MarketData                │            │             │
        ┌─────────────▼─────────────────────────────────────────────────────────────┐
        │                          コア・エンジン                                      │
        │  indicators.py ── 指標（RSI/MA乖離/DD/ボラ…ベクトル化）                       │
        │  strategy.py   ── 買いサイズ（スコア→金額）＋利確ルール（StrategyParams）      │
        │  portfolio.py  ── ロット式信用ポートフォリオ（cash/equity/維持率/制度イベント）│
        │  backtest.py   ── simulate()（1パス）＋ run_backtest()（ウォークフォワード駆動）│
        │  optimizer.py  ── WalkForwardOptimizer.params_at_close（再選択・候補評価）      │
        │  metrics.py    ── objective_score ＋ サマリ指標（risk ブロック）               │
        │  benchmark.py  ── パッシブ B&H 基準（1570.T / N225 / 現金）                     │
        └─────────────┬─────────────────────────────────────────────────────────────┘
                      │ BacktestResult（daily_rows / trades / equity_curve / summary）
        ┌─────────────▼─────────────────────────────────────────────────────────────┐
   出力層 │  reporting.py ── summary.json / daily・trades・optimization.csv /            │
        │                  PNG（equity/exposure/drawdown/realized/underwater/return） /│
        │                  benchmarks.json                                            │
        │  report.py    ── report.md（生存・テールリスク優先の自動生成レポート）         │
        └─────────────────────────────────────────────────────────────────────────────┘

   variants/（コア無改変のオーバーレイ）:
     wf_capture → variant_engine → grid(432×並列) → aggregate(comfort/top) → analyze(analysis.json) → run_all
```

`config.py` は全層が参照する設定モデル（`Config / StrategyParams / Optimization / Execution /
Objective` ＋ YAML ロード）。

---

## 2. データフロー（1 ラン）

```
CSV / synth
   │  data.load_market_data / join_target_benchmark
   ▼
prepare_market_data(joined, cfg)
   │   ・repair_price_glitches（2021-04 偽半値を補修・監査証跡を保持）
   │   ・indicators 計算  ・tradable マスク（valid）
   ▼
MarketData ──► run_backtest(md, cfg)
                  │  WalkForwardOptimizer.params_at_close(i)   ← 各日のパラメータ供給
                  │     （リバランス周期で候補を再選択／enabled=False なら default 固定）
                  ▼
               simulate(md, lo, hi, provider, cfg)   ←★ ルックアヘッド禁止の3段
                  close_t で意思決定 → open_{t+1} で約定 → close_{t+1} で評価
                  Portfolio が約定・含み損益・維持率・制度イベントを記録
                  ▼
               BacktestResult（daily_rows / trades / equity_curve / optimization_rows）
                  │  metrics.summarize（risk ブロック含む）＋ benchmark.compare
                  ▼
               reporting.write_outputs → summary.json / CSV / PNG / benchmarks.json
                                         report.py → report.md
```

分析層（stress/validation/execution）は同じ `prepare_market_data` → `WalkForwardOptimizer`
で**パラメータ列を1回キャプチャ**し、固定列をシナリオ別に**リプレイ**する設計（意思決定を固定して
コスト/執行/帰無分布だけを変える統制比較）。variants も同じ「捕捉→リプレイ」方式。

---

## 3. 設計上の不変条件（厳守）

| 不変条件 | 実装箇所 | 検証 |
|---|---|---|
| **損切りなし** | `Portfolio.sell_lot()` が含み損（`net_pnl_before_tax < 0`）の売却を拒否 | コアテスト |
| **ルックアヘッド厳禁** | `close_t` 意思決定 → `open_{t+1}` 約定 → `close_{t+1}` 評価 | `tests/test_backtest_no_lookahead.py`（実行日 Close を改変しても同日 Open 約定は不変） |
| **制度イベントは記録のみ** | 追証・建玉上限・買付不能は記録するだけ（`force_liquidation` ON 時のみ強制決済が例外で `force=True` bypass） | metrics の追証ペナルティは ON 時のみ適用 |
| **再現性** | 最適化は seed 固定、テストは人工データのみ・外部通信なし | core 107 + variants 7 緑 |

維持率は `(own_funds + min(unrealized_pnl, 0)) / gross_position`、`own_funds = cash() =
initial_equity + realized_after_tax`（実現損益を反映・評価益は保守的に除外・金利二重控除なし）。

---

## 4. CLI（6 コマンド）

| コマンド | 役割 | 主な出力 |
|---|---|---|
| `fetch` | yfinance で OHLCV 取得（任意・要ネット） | `data/*.csv` |
| `synth` | オフライン人工データ生成 | `data/*_synthetic.csv` |
| `run` | バックテスト本体（ウォークフォワード） | `outputs*/`（summary/CSV/PNG/report.md/benchmarks.json） |
| `stress` | D1: 歴史レジーム・コスト感応・ブートストラップ破綻確率 | `outputs_stress/`（stress.json ほか） |
| `validate` | A: セグメント整合・置換検定・ブートCI・FDR | `outputs_validation/`（validation.json ほか） |
| `execmodel` | D2: 執行現実化（VWAP/不利/遅延/部分約定）の比較 | `outputs_execution/`（execution.json ほか） |

実データは `data/target_1570_T.csv` / `data/benchmark_N225.csv`（gitignore 済・ローカルのみ）。
文字コードは `.claude/settings.json` の `PYTHONUTF8=1` で対応済み（¥ が cp932 で落ちない）。

---

## 5. 成熟度（到達レベルと次段）

**到達済み ≒ L3（現実的な単一銘柄リサーチ・エンジン）**
- ルックアヘッド禁止の約定/評価エンジン、信用・損切り禁止・追証（強制ロスカット任意）。
- リスク指標（Sortino/Calmar/Ulcer/VaR/CVaR/最大DD/アンダーウォーター分布）＋パッシブ比較。
- ストレス（暴落レジーム＋ブートストラップ破綻確率）、統計的検証（置換検定/CV/FDR）、
  執行現実化（VWAP/不利/遅延/部分約定）。
- データ品質補修＋監査証跡（summary.json `data_quality` / report.md）。
- 「捕捉→リプレイ」による統制比較、seed 固定の再現性、外部通信なしの 114 テスト。

**L4 へ向けて（本リポジトリのロードマップ外 B/E、または任意拡張 ④）**
- 複数銘柄ポートフォリオ（相関・配分・共通追証）。
- 分足/ティック約定（板・部分約定の高解像度化）。
- 高度最適化（Optuna/ベイズ最適化、purged walk-forward でのリーク防止）。
- ライブデータ取り込みパイプライン、CI、アウトオブサンプルの体系化。

> 関連レポート: 主力 3 版比較 [REPORT_REAL.md](../REPORT_REAL.md) ／ 利確バリアント
> [REPORT_VARIANTS.md](../REPORT_VARIANTS.md) ／ ストレス [REPORT_STRESS.md](../REPORT_STRESS.md) ／
> 統計検証 [REPORT_VALIDATION.md](../REPORT_VALIDATION.md) ／ 執行 [REPORT_EXECUTION.md](../REPORT_EXECUTION.md) ／
> 追証 [REPORT_MARGIN_CALL.md](../REPORT_MARGIN_CALL.md)。
