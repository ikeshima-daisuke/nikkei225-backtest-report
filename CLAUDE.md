# CLAUDE.md

このファイルは、このリポジトリで作業する Claude Code 向けのガイドです。

## リポジトリ概要

日経平均（`^N225`）関連の **2 つの独立したバックテスト・プロジェクト**を含むモノレポです。
両者はコードを共有せず、別々に実行・テストします。

| プロジェクト | 場所 | 内容 |
|---|---|---|
| **① シグナル探索レポート** | `signal_report/` | `^N225` の押し目買いシグナル候補を 5 年分で網羅探索（1,350 通り）し、Markdown/HTML レポートを生成 |
| **② レバレッジ ETF シミュレータ** | `nikkei_leverage_sim/` | 日経レバ ETF（1570.T）の「信用ロング・損切りなし・毎日積立・利確」戦略のバックテスター（src レイアウト + pytest） |

リポジトリルートには両プロジェクトへ誘導する `README.md`、GitHub Pages ランディングの
`index.html`、本ガイド、`MIGRATION.md` のみを置く（対称モノレポ構成）。整理の経緯は `MIGRATION.md` を参照。

## ① signal_report/

### 主要ファイル（すべて `signal_report/` 配下）
- `backtest.py` — データ取得 + 全コンビ探索 + 全出力生成（CSV / JSON / PNG / REPORT.md / report.html）
- `indicators.py` — 指標・シグナル定義（ベクトル化）
- `REPORT.md` — **主要アウトプット**（GitHub / モバイルでそのまま閲覧）。`figures/*.png` を相対参照
- `report.html` — Plotly インタラクティブ版（自己完結 HTML、ローカル閲覧用）
- `data/` — `results.csv`（全集計）, `summary.json`（上位戦略）
- `figures/` — matplotlib 製の静的 PNG（REPORT.md から参照）

※ ルートの `index.html` は ① のレポートを GitHub Pages で配信するための自己完結 HTML ランディング。

### 実行（必ず `signal_report/` 内で）
```bash
cd signal_report
pip install -r requirements.txt
python backtest.py                 # 既定の出力先は backtest.py と同じディレクトリ
python backtest.py --out-dir DIR   # 出力先を変更
```
`--out-dir` のデフォルトは `Path(__file__).resolve().parent`。**ファイルを移動すれば出力先も追従する。**

### 注意
- 手数料・税・配当・スリッページは**未考慮の理論値**。多重比較による偽陽性に注意し、OOS 列を必ず併読する。
- `index.html` はルートに固定。動かすと GitHub Pages の公開 URL が壊れる可能性がある（Settings → Pages を要確認）。

## ② nikkei_leverage_sim/

### レイアウト
```
nikkei_leverage_sim/
├── src/nikkei_leverage_sim/   # backtest, portfolio, strategy, optimizer, metrics, indicators, data, cli, reporting, config, benchmark, validation
├── tests/                     # pytest（外部通信なし・人工データのみ）
├── variants/                  # 戦略バリアント・グリッドサーチ（コア無改変の独立実装）
├── accumulation_study/        # 積立×Exit 比較スタディ＋統計的検証（コア無改変の独立実装）
├── examples/                  # sample_config.yaml / config_fast.yaml
├── outputs/  outputs_real_fast/  # 生成物（CSV/PNG/JSON）
├── pyproject.toml             # setuptools, src レイアウト, console script: nikkei-leverage-sim
├── README.md  REPORT_REAL.md  REPORT_VARIANTS.md  REPORT_ACCUMULATION.md ...
```

### サブパッケージ（コア無改変・独立）
- **`variants/`** — 既存 fast 戦略に初期一括・一括売却ルールを載せた 432 通りグリッド。`REPORT_VARIANTS.md`。
- **`accumulation_study/`** — 「使える資本 ¥10M で、どう積立→いつ売る」を総当たり比較（Calmar ランキング）し、
  **統計的に検証**（ブロックブートストラップ CI・置換検定・BH-FDR・アウトオブサンプル・コスト感応度）するスタディ。
  `REPORT_ACCUMULATION.md`。重要な学び: **グリッド最良は統計的有意でなく OOS で再現しない**（過剰最適化の実例）。
  実行: `python -m accumulation_study.run_study` → `make_report`（検証は `validate`）。テストは `pytest accumulation_study/tests`。
- パッシブ比較は `src/.../benchmark.py`（一括 B&H＋**定額積立 DCA**：1570.T/N225/現金）。`REPORT_REAL.md §10` は
  **使える資本 ¥10M を分母**にした公平比較（¥100M 分母は本戦略の DD を過小評価するため）。

### 実行（必ず `nikkei_leverage_sim/` 内で）
```bash
cd nikkei_leverage_sim
pip install -e ".[dev,fetch]"   # fetch は yfinance（任意）

python -m nikkei_leverage_sim.cli synth --out data/ --days 900 --seed 7        # オフライン人工データ生成
python -m nikkei_leverage_sim.cli run   --config examples/sample_config.yaml --synthetic
pytest -q                              # コア（外部通信なし・人工データ）
pytest variants/tests accumulation_study/tests -q   # 独立サブパッケージのテスト（別途実行）
```
※ コアの `pytest -q` は `testpaths=["tests"]` のため**サブパッケージのテストは拾わない**。
`variants/` と `accumulation_study/` は各自のローカル `conftest.py` で `src`/PKG を sys.path に追加して別途実行する。

### 設計上の不変条件（変更時は厳守）
- **損切りなし**: `Portfolio.sell_lot()` は含み損（`net_pnl_before_tax < 0`）の売却を拒否する。
- **ルックアヘッド厳禁**: `close_t` で意思決定 → `open_{t+1}` で約定 → `close_{t+1}` で評価。
  `tests/test_backtest_no_lookahead.py` がこの不変条件を検証する。実行日 Close を改変しても同日 Open 約定は不変。
- **制度イベントは記録のみ**: 追証・建玉上限・買付不能は記録するだけで戦略行動には使わない
  （`force_liquidation` ON 時のみ例外）。
- **再現性**: 最適化は seed 固定。テストは人工データのみで外部通信しない。

### 実行環境のはまりどころ（Claude Code 向け・毎回プロンプトに書かない）
- **venv**: `nikkei_leverage_sim/.venv`。実行は `./.venv/Scripts/python.exe`、テストは `./.venv/Scripts/python.exe -m pytest`。
- **文字コード**: ルートの `.claude/settings.json` で `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` を常設済み。
  これにより CLI が ¥ 記号を出しても cp932 で落ちない。**`PYTHONUTF8=1` の前置きは不要**（settings.json が肩代わり）。
- **git-bash の `cd`**: バックスラッシュ path だと壊れる。**forward slash** を使う
  （例: `cd /c/Users/rief5/nikkei225-backtest-report/nikkei_leverage_sim`）。
- **実データ**: `data/target_1570_T.csv`・`data/benchmark_N225.csv`（**gitignore 済・ローカルのみ**、2014–2026・3,032営業日）。
  2021-04-27/28 のベンダー偽半値はエンジンが**自動補修**（既定 ON、`repair_glitches=False` で無効化可）。補修跡は summary.json の `data_quality` と report.md に残る。
- **生成物の追跡方針**: 嵩む per-row 出力（`outputs_margincall/*/{daily,trades,optimization}.csv`・`outputs_variants/*/rows.json`・`outputs_variants/wf_params.pkl`）は `.gitignore` で除外。
  **summary.json・小さい CSV・PNG・report.md は追跡**。大量 CSV を新規コミットする前にユーザーへ確認。
- **Codex レビュー**: `codex:codex-rescue` サブエージェント経由。プロンプトに **「pytest」「-m pytest」の語を入れない**
  （companion の引数パーサが `--model pytest` と誤認し 400）。「静的レビューのみ、テストはローカルで全件パス」と書く。

## 全体的な作業方針

- 2 つのプロジェクトは独立。一方を触る変更が他方に波及しないようにする。
- ① と ② で依存関係ファイルが別（`signal_report/requirements.txt` / `nikkei_leverage_sim/pyproject.toml`）。混同しない。
- 生成物（`data/`, `figures/`, `outputs*/`）の追跡方針は未確定（`MIGRATION.md` の「未決事項」参照）。
  大量の生成物を新規にコミットする前に方針を確認する。
- レポートは「GitHub / モバイルでそのまま読む」ことを重視した設計。Markdown と相対パス画像の表示を壊さない。
