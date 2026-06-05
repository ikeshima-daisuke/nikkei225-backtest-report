# CLAUDE.md

このファイルは、このリポジトリで作業する Claude Code 向けのガイドです。

## リポジトリ概要

日経平均（`^N225`）関連の **2 つの独立したバックテスト・プロジェクト**を含むモノレポです。
両者はコードを共有せず、別々に実行・テストします。

| プロジェクト | 場所 | 内容 |
|---|---|---|
| **① シグナル探索レポート** | リポジトリルート直下 | `^N225` の押し目買いシグナル候補を 5 年分で網羅探索（1,350 通り）し、Markdown/HTML レポートを生成 |
| **② レバレッジ ETF シミュレータ** | `nikkei_leverage_sim/` | 日経レバ ETF（1570.T）の「信用ロング・損切りなし・毎日積立・利確」戦略のバックテスター（src レイアウト + pytest） |

> 📁 **整理予定**: ① をルートから `signal_report/` へ移す対称モノレポ化を計画中。
> 詳細と未決事項は `MIGRATION.md` を参照。**現状はまだ ① がルート直下にある**ので、
> パスはこのファイルの記述に従うこと。

## ① シグナル探索レポート（ルート直下）

### 主要ファイル
- `backtest.py` — データ取得 + 全コンビ探索 + 全出力生成（CSV / JSON / PNG / REPORT.md / report.html）
- `indicators.py` — 指標・シグナル定義（ベクトル化）
- `REPORT.md` — **主要アウトプット**（GitHub / モバイルでそのまま閲覧）。`figures/*.png` を相対参照
- `report.html` / `index.html` — Plotly インタラクティブ版（自己完結 HTML）。`index.html` は GitHub Pages ランディング
- `data/` — `results.csv`（全集計）, `summary.json`（上位戦略）
- `figures/` — matplotlib 製の静的 PNG（REPORT.md から参照）

### 実行
```bash
pip install -r requirements.txt
python backtest.py                 # 既定の出力先は backtest.py と同じディレクトリ
python backtest.py --out-dir DIR   # 出力先を変更
```
`--out-dir` のデフォルトは `Path(__file__).resolve().parent`。**ファイルを移動すれば出力先も追従する。**

### 注意
- 手数料・税・配当・スリッページは**未考慮の理論値**。多重比較による偽陽性に注意し、OOS 列を必ず併読する。
- `index.html` を動かすと GitHub Pages の公開 URL が壊れる可能性がある（Settings → Pages を要確認）。

## ② nikkei_leverage_sim/

### レイアウト
```
nikkei_leverage_sim/
├── src/nikkei_leverage_sim/   # backtest, portfolio, strategy, optimizer, metrics, indicators, data, cli, reporting, config
├── tests/                     # pytest（外部通信なし・人工データのみ）
├── examples/                  # sample_config.yaml / config_fast.yaml
├── outputs/  outputs_real_fast/  # 生成物（CSV/PNG/JSON）
├── pyproject.toml             # setuptools, src レイアウト, console script: nikkei-leverage-sim
├── README.md  REPORT_REAL.md
```

### 実行（必ず `nikkei_leverage_sim/` 内で）
```bash
cd nikkei_leverage_sim
pip install -e ".[dev,fetch]"   # fetch は yfinance（任意）

python -m nikkei_leverage_sim.cli synth --out data/ --days 900 --seed 7        # オフライン人工データ生成
python -m nikkei_leverage_sim.cli run   --config examples/sample_config.yaml --synthetic
pytest -q                        # 33 tests, 外部通信なし
```

### 設計上の不変条件（変更時は厳守）
- **損切りなし**: `Portfolio.sell_lot()` は含み損（`net_pnl_before_tax < 0`）の売却を拒否する。
- **ルックアヘッド厳禁**: `close_t` で意思決定 → `open_{t+1}` で約定 → `close_{t+1}` で評価。
  `tests/test_backtest_no_lookahead.py` がこの不変条件を検証する。実行日 Close を改変しても同日 Open 約定は不変。
- **制度イベントは記録のみ**: 追証・建玉上限・買付不能は記録するだけで戦略行動には使わない
  （`force_liquidation` ON 時のみ例外）。
- **再現性**: 最適化は seed 固定。テストは人工データのみで外部通信しない。

## 全体的な作業方針

- 2 つのプロジェクトは独立。一方を触る変更が他方に波及しないようにする。
- ① と ② で依存関係ファイルが別（ルート `requirements.txt` / `nikkei_leverage_sim/pyproject.toml`）。混同しない。
- 生成物（`data/`, `figures/`, `outputs*/`）の追跡方針は未確定（`MIGRATION.md` の「未決事項」参照）。
  大量の生成物を新規にコミットする前に方針を確認する。
- レポートは「GitHub / モバイルでそのまま読む」ことを重視した設計。Markdown と相対パス画像の表示を壊さない。
