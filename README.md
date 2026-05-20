# backtest/

日経225（`^N225`）の押し目買いシグナル候補を 5年分のデータで網羅探索する一次調査ツール。

## 📊 レポートを読む

**まずは [`REPORT.md`](./REPORT.md) を開いてください**（GitHub アプリ／モバイルブラウザでそのまま見れます）。
グラフは `figures/*.png` に静的画像として保存されているので、スマホでもサクサク表示できます。

インタラクティブな Plotly 版を見たい場合は `report.html` をローカルでブラウザに開いてください。

> public Web公開はしていません。公開したい場合の選択肢は [`OPTIONAL_HOSTING.md`](./OPTIONAL_HOSTING.md) を参照。

## 何をしているか

- yfinance で `^N225` の日足5年を取得
- 20種類の単一条件（RSI / BB位置 / 52週位置 / ドローダウン / 連続下落日数 / リターン / MA乖離 / ゴールデンクロス）から、
  単一 + 2-way AND + 3-way AND の **1,350通り** を生成
- 5年で発生回数 ≥ 20 のシグナルだけ残す
- 各シグナル × 利確水準 (+5 / +10 / +15 / +20%) で単一ポジション・バックテスト
- 翌日寄付エントリー、当日高値が利確水準にタッチした日の終値で利確（保守的）、250営業日で強制決済
- 期待値で並べ替え、直近約252営業日を Out-of-Sample として切り出して再評価

## 重要な但し書き

- **手数料・税・配当・スリッページは未考慮の理論値です。**
- 5年バックテストは特定相場レジームに偏りやすい。多重比較で偽陽性も出やすいので、OOS列を必ず一緒に見る。
- 過去のパフォーマンスは将来を保証しません。

## ファイル

| ファイル | 役割 |
|---|---|
| **`REPORT.md`** | **主要アウトプット。Markdown レポート (GitHub上でモバイル閲覧)** |
| `figures/*.png` | matplotlib で書き出した静的画像 (Markdown から参照) |
| `report.html` | **Plotly インタラクティブ版** (ローカルブラウザで開く用) |
| `data/results.csv` | 全戦略の集計表（生データ、1,240 行） |
| `data/summary.json` | 上位5戦略 + ベンチマークの JSON |
| `backtest.py` | データ取得 + 全コンビ探索 + 全出力生成 |
| `indicators.py` | 指標とシグナル定義 (vectorised) |
| `OPTIONAL_HOSTING.md` | Web公開したい場合の選択肢一覧（実行はしていない） |

## 実行方法（ローカル）

```bash
cd backtest
pip install -r requirements.txt
python backtest.py
```

完了すると `REPORT.md` / `figures/*.png` / `report.html` / `data/results.csv` / `data/summary.json` が更新される。

## このフォルダと `notify.py` の関係

このフォルダはあくまでオフラインの調査用。スマホへの日次通知（`notify.py`）には影響しない独立した実装です。
バックテスト結果でめぼしいシグナルが見つかったら、`notify.py` 側に組み込んでアラート対象を増やす想定。
