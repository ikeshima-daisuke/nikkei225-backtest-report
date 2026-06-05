# nikkei_leverage_sim

日経レバレッジ ETF（**1570.T** / NEXT FUNDS 日経平均レバレッジ・インデックス連動型上場投信）を
**信用ロングのみ**で「相場状況に応じて毎日積立し、ロット単位で利確する」戦略をシミュレーションする
バックテスターです。ベンチマークは日経平均株価（`^N225`）。

> ⚠️ これは研究用のバックテストであり、**将来の成績を一切保証しません**。後述の「注意点」を必ず読んでください。

> 📊 **実データ結果**は [`REPORT_REAL.md`](REPORT_REAL.md)（fast/フル/Codex の3版比較）。
> 🧪 **戦略バリアント**（初期一括・一括売却[固定円/固定%]・combo の432通りグリッドサーチ）は
> [`REPORT_VARIANTS.md`](REPORT_VARIANTS.md) と [`variants/`](variants/README.md)。コア無改変の独立実装です。

---

## 目的

- 最大建玉 **1,000 万円以内**、自己資金 **1 億円**、**損切りなし** の制約下で
- 毎日の買付額を相場状況に応じて最適化し、利確条件も最適化したうえで
- **1 営業日あたり / 利確発生日あたり 5,000 円程度以上**の実現利益が現実的に狙えるかを検証する。

ただし利益最大化だけを目的にせず、**最大含み損・建玉膨張・利確不能期間・過剰最適化**もまとめて評価します。

---

## 前提条件

| 項目 | 値 |
|---|---|
| 売買 | 信用取引ロングのみ（ショート禁止） |
| 損切り | **禁止**（含み損ロットは売らない） |
| 最大建玉総額 | 10,000,000 円 |
| 自己資金 | 100,000,000 円 |
| 維持率（追証 / 警告） | 0.30 / 0.50 |
| 約定 | 翌営業日の **寄付（Open）** |
| 評価 | 当日 **引け（Close, 設定で Adj Close 可）** |
| コスト | スリッページ 2bps / 手数料 0bps / 信用金利 2.8%(年) / 税率 20.315% |

### 「損切りなし」の意味

このシミュレーターは **損失を確定するための売却を一切行いません**。
ロットを売るのは「利確条件を満たした（=含み益が出ている）ロット」だけです。
含み損のロットは、回復して利確条件を満たすまで持ち続けます。
`Portfolio.sell_lot()` は安全装置として `net_pnl_before_tax < 0` の売却を拒否します。

### 追証・強制決済は「戦略」ではなく「制度イベント」

自己資金が大きいため維持率割れは基本的に起きにくい想定ですが、起きた場合でも
**戦略としての売り（損切り）は行いません**。維持率・追証相当・建玉上限到達・買付不能日は
**イベントとして記録するだけ**です。強制決済ロジックは `force_liquidation`（初期値 `false`）で
ON/OFF でき、ON のときのみ制度イベントとして全建玉を翌寄付で処分します（これは唯一、含み損でも
売却しうる経路で、ストップロスではなく制度イベントです）。

---

## ルックアヘッド（先読み）を避ける売買タイミング

未来情報は一切使いません。エンジンのループ構造そのものでこれを保証しています。

```
day t   引け後: day t までの情報で指標を計算 → 買付額・利確対象ロットを決定（"pending")
day t+1 寄付  : pending を Open で約定（口数 = floor(買付額 / Open約定価格)）
day t+1 引け  : Close で評価・含み損益・信用金利を計上
```

実行日 `i` の注文は必ず「前日 `i-1` の引けで決めた `pending`」に基づくため、
`i` 日の Close を見て `i` 日の Open で売買することは構造上できません。
これは `tests/test_backtest_no_lookahead.py` で検証しています（実行日の Close を改変しても、
その日の Open 約定が変化しないことを確認）。

### 売買単位・コストの扱い

- 口数は整数。`buy_shares = floor(buy_amount / 寄付約定価格)`。
- 買い約定価格 = `Open * (1 + slippage_bps/10000)`、売り約定価格 = `Open * (1 - slippage_bps/10000)`。
- 手数料 = 約定代金 × `commission_bps/10000`。
- 信用金利（日次）= 建玉評価額 × `annual_margin_interest_rate / 365`。
- 税金は **その日の実現利益がプラスのときだけ** `tax_rate` を控除（含み益には非課税）。

---

## インストール

```bash
cd nikkei_leverage_sim
python -m pip install -e .          # pandas / numpy / scipy / matplotlib / pyyaml
python -m pip install -e ".[fetch]" # yfinance も入れる場合（任意）
```

`src` レイアウトのため、インストールせずに使う場合は `PYTHONPATH=src` を付けてください。

---

## データ取得方法

### 1. yfinance で取得（任意・要ネットワーク）

```bash
python -m nikkei_leverage_sim.cli fetch \
    --start 2019-01-01 --end 2026-12-31 --out data/ \
    --target-symbol 1570.T --benchmark-symbol ^N225
```

`data/target_1570_T.csv` と `data/benchmark_N225.csv` が出力されます。
**yfinance はあくまで便利機能**で、バックテスト本体は CSV だけで動きます。

### 2. オフラインの人工データを生成（ネット不要）

```bash
python -m nikkei_leverage_sim.cli synth --out data/ --days 900 --seed 7
```

`data/target_synthetic.csv` / `data/benchmark_synthetic.csv` を生成します
（日次リバランス型レバレッジ ETF の挙動＝逓減を含む合成系列）。

### CSV 入力方法

target / benchmark とも以下の列が必須です（`Date` で内部結合し、両方に存在する営業日のみ使用）。

```
Date, Open, High, Low, Close, Adj Close, Volume
```

- 指標計算は原則 **benchmark の Adj Close**。
- 約定は **target ETF の Open**。
- 日次評価は **target ETF の Close**（`execution.valuation_price: adj_close` で Adj Close に変更可）。

---

## 実行方法

```bash
# CSV から
python -m nikkei_leverage_sim.cli run \
    --config examples/sample_config.yaml \
    --target data/target_1570_T.csv \
    --benchmark data/benchmark_N225.csv \
    --out outputs/

# 人工データから（CSV/ネット不要、動作確認向け）
python -m nikkei_leverage_sim.cli run \
    --config examples/sample_config.yaml --synthetic --out outputs/
```

> **パフォーマンス注意:** ウォークフォワード最適化は「毎営業日 × `n_trials` 個の候補 × `lookback_days`
> 営業日の学習シミュレーション」を回すため計算量が大きいです（例: 500 営業日・`n_trials=100` で約 4 分）。
> 素早い動作確認には `n_trials` / `lookback_days` を小さくするか、`optimization.enabled: false` を使ってください。

---

## テスト方法

```bash
pytest -q
```

すべてのテストは **外部通信なし・人工データのみ** で完結します。

| ファイル | 主な検証内容 |
|---|---|
| `test_indicators.py` | RSI / MA gap / drawdown_252 / vol_20 |
| `test_portfolio.py` | ロット追加・対象ロットのみ売却・損失ロット非売却・日次金利・利益にのみ課税・建玉上限 |
| `test_backtest_no_lookahead.py` | 翌寄付約定・当日 Close 不参照・建玉 1,000 万円上限 |
| `test_strategy.py` | 下落で buy_score 上昇・exposure/含み損で買付抑制・max_daily 上限・利確条件 |
| `test_optimizer.py` | seed 固定で再現・学習窓外の未来データ不使用・最良候補の選択 |
| `test_metrics.py` | 最大含み損・最大ドローダウン・利確なし連続日数 |

---

## 出力ファイル説明（`outputs/`）

| ファイル | 内容 |
|---|---|
| `summary.json` | サマリー指標（最終資産・税引後実現益・最大含み損・最大DD・利確発生日あたり利益 など） |
| `daily.csv` | 日次ログ（買付額・口数・売却額・実現/含み損益・金利・建玉・現金・equity・維持率・採用パラメータID・イベント） |
| `trades.csv` | 約定明細（side / 口数 / 価格 / 金額 / lot_id / 実現損益 / reason） |
| `optimization.csv` | 日次の最適化ログ（採用パラメータ・学習スコア・学習期間の純益/最大DD/最大含み損/追証数） |
| `equity_curve.png` | equity 推移 |
| `exposure_curve.png` | 建玉推移（上限ライン付き） |
| `drawdown_curve.png` | equity ドローダウン |
| `realized_profit_by_day.png` | 日次の税引後実現利益 |

---

## 主要な設定項目（`examples/sample_config.yaml`）

- **口座/制度**: `initial_equity`, `max_gross_exposure`, `maintenance_margin_ratio`,
  `warning_margin_ratio`, `force_liquidation`
- **コスト**: `commission_bps`, `slippage_bps`, `annual_margin_interest_rate`, `tax_rate`
- **最適化**: `optimization.{enabled, method(random/grid), random_seed, n_trials, lookback_days,
  min_train_days, apply_days, rebalance_frequency}`
- **約定**: `execution.{signal_timing, execution_timing, valuation_price}`
- **戦略既定値**: `strategy.default_params.*`（買付スコア重み・利確パラメータ）
- **目的関数**: `objective.*`（最大DD/最大含み損の重み・追証/建玉上限/利確不能のペナルティ）

### 買付額ロジック（要約）

```
buy_score = w_drawdown·dd + w_rsi·rsi_oversold + w_ma_gap_25·ma_gap
          + w_ret_5·short_drop + w_trend·trend
          - w_vol·vol - w_exposure·exposure_ratio - w_unrealized_loss·unrealized_loss_ratio

buy_amount = clip(base_buy_amount · sigmoid(score_scale·(buy_score - score_threshold)),
                  0, max_daily_buy_amount)
```
さらに `gross_exposure + buy_amount <= max_gross_exposure` を満たすよう減額します。

### 利確ロジック（要約）

```
required_profit_pct = max(min_take_profit_pct,
                          base_take_profit_pct·(1 - exposure_ratio·exposure_tp_sensitivity)
                          + vol_20·vol_tp_multiplier)
```
ロットの `net_pnl_before_tax >= fixed_profit_yen` **または** `profit_pct >= required_profit_pct`
を満たした **プラスのロットだけ** を利確します。

### 目的関数（最適化スコア）

```
score = 税引後実現益 + 最終含み損益
      - 0.5·最大DD(equity) - 0.5·最大含み損
      - 1,000,000·追証相当数 - 100,000·建玉上限到達数 - 50,000·利確不能超過日数
```
重み・ペナルティは `objective` セクションで変更できます。

---

## 注意点（重要）

- **バックテストは将来の成績を保証しません。** 特定の相場レジームに過剰適合しがちです。
- **レバレッジ ETF には逓減（ボラティリティ・ディケイ）リスク**があります。横ばい・乱高下相場で
  原指数より不利に減価しやすく、長期保有・ナンピン積立と相性が悪い局面があります。
- **信用金利・税金・スリッページで利益は確実に削られます。** 本シミュレーターはこれらを計上しますが、
  実際のスプレッド・約定不成立・逆日歩・銘柄個別事情などはモデル化していません。
- **毎日最適化は過剰最適化（オーバーフィッティング）のリスクが高い**です。ウォークフォワードで
  学習窓は過去データのみに限定していますが、`n_trials` を増やすほど in-sample スコアに過適合しやすく
  なります。`optimization.csv` の学習スコアと実現結果の乖離、`summary.json` の最大含み損・最大DD・
  利確不能日数を必ず併読してください。

---

## プロジェクト構成

```
nikkei_leverage_sim/
  pyproject.toml
  README.md
  src/nikkei_leverage_sim/
    __init__.py  cli.py  config.py  data.py  indicators.py
    portfolio.py strategy.py optimizer.py backtest.py metrics.py reporting.py
  tests/
    test_indicators.py test_portfolio.py test_backtest_no_lookahead.py
    test_strategy.py test_optimizer.py test_metrics.py
  examples/sample_config.yaml
```

ライセンス: MIT
