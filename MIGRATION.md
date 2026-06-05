# フォルダ整理計画（MIGRATION）

> ステータス: **実施済み**。本計画に沿って ① を `signal_report/` へ移動済み。
> `index.html` は GitHub Pages 保護のためルートに残置。生成物の追跡方針は **C（折衷）** で確定
> （下記セクション参照）。残るユーザー確認事項は GitHub Pages の配信元設定のみ。

## 背景

PR #1 で信用ロング・利確戦略のバックテスター `nikkei_leverage_sim/` が
独立サブプロジェクトとしてマージされました。その結果、リポジトリには
**2 つのプロジェクトが非対称な形で同居**しています。

### 現状（マージ後）

```
nikkei225-backtest-report/
├── README.md / REPORT.md / OPTIONAL_HOSTING.md   ┐
├── index.html / report.html                       │ ① 元の「シグナル探索レポート」が
├── backtest.py / indicators.py                    │   ルート直下に散在
├── requirements.txt                               │
├── data/ (results.csv, summary.json)              │
├── figures/ (*.png ×4)                            ┘
└── nikkei_leverage_sim/      ← ② src レイアウトで自己完結（綺麗）
    ├── README.md / REPORT_REAL.md / pyproject.toml
    ├── src/nikkei_leverage_sim/  examples/  tests/
    └── outputs/  outputs_real_fast/
```

### 問題点

1. **非対称** — ② はパッケージ化されているのに、① はルートに散らばっており、
   どれが「リポジトリ共通」でどれが「プロジェクト① 固有」か区別がつかない。
2. **ルートが煩雑** — Python・CSV・PNG・HTML が混在し、リポジトリ全体の入口
   （README）とプロジェクト① の README が同一ファイルに同居している。
3. **生成物の混在** — 再生成可能な `data/`・`figures/`・`outputs*/` が
   ソースと同じ階層に並んでいる。

## 目標構成（対称モノレポ化）

① を `nikkei_leverage_sim/` と対になる独立フォルダ `signal_report/` に格納し、
ルートは「リポジトリの入口」に専念させる。

```
nikkei225-backtest-report/
├── README.md                    ★ 新規: リポジトリ全体の入口（2 プロジェクトへ誘導）
├── index.html                   GitHub Pages ランディング（ルート維持・後述）
│
├── signal_report/               ① を丸ごと移動（旧ルート群）
│   ├── README.md   REPORT.md   OPTIONAL_HOSTING.md
│   ├── backtest.py   indicators.py   requirements.txt
│   ├── report.html
│   ├── data/        (results.csv, summary.json)
│   └── figures/     (*.png ×4)
│
└── nikkei_leverage_sim/         ② 変更なし
```

## 移行手順（合意後に実施）

`git mv` を使い履歴を保ったまま移動する。

```bash
mkdir signal_report
git mv README.md REPORT.md OPTIONAL_HOSTING.md \
       backtest.py indicators.py requirements.txt report.html \
       data figures signal_report/

# リポジトリ全体の新しい入口を作成
$EDITOR README.md   # 2 プロジェクトの概要と各 README へのリンク
```

### 安全性の根拠（調査済み）

| 項目 | 確認結果 | 影響 |
|---|---|---|
| `backtest.py` の出力先 | `--out-dir` のデフォルトが `Path(__file__).resolve().parent`（自身の場所） | **移動するだけで出力先も自動追従。コード修正不要** |
| `REPORT.md` の画像参照 | `figures/*.png` を**相対パス**で参照 | REPORT.md と figures/ を一緒に動かせば**リンク不変** |
| `index.html` | Plotly 埋め込みの**自己完結型**（外部ファイル参照なし） | 単独で移動可能 |
| `report.html` | 同上（自己完結） | 単独で移動可能 |

### 注意点 — GitHub Pages（要確認）

リポジトリに `.github/workflows` が無いため、GitHub Pages は
**Settings でルート `/` の `index.html` を配信**していると推測される。

- ⇒ **`index.html` はルートに残す**のが安全（移動すると公開 URL が 404 になる恐れ）。
- もし Pages を使っていない / 別設定なら、`index.html` も `signal_report/` へ移してよい。
- **実施前に Settings → Pages の配信元設定を確認すること。**

## 生成物（outputs / data / figures）の扱い — 決定: C（折衷）

「レポートを GitHub / モバイルでそのまま読む」という本リポジトリの設計思想を踏まえ、
**レポート表示に必要な成果物は追跡し、埋め込まれていない大きな生データ羅列は追跡外**とする。

### 追跡を維持（GitHub 上でレポートを表示するのに必要）
- `signal_report/figures/*.png` … `REPORT.md` に埋め込み
- `signal_report/data/{results.csv,summary.json}` … ① の主要データ出力（モダレートなサイズ・README が明示）
- `signal_report/report.html` / ルート `index.html` … レポートの HTML 版
- `nikkei_leverage_sim/outputs_real_fast/*.png` … `REPORT_REAL.md` に埋め込み
- `nikkei_leverage_sim/outputs_real_fast/summary.json` … 小さく、レポートが参照

### 追跡外（`.gitignore` 化・再生成可能で埋め込みなし）
- `nikkei_leverage_sim/outputs/`（人工データのデモ一式）… **元々 `.gitignore` 済みだったが追跡されていた**矛盾を解消
- `nikkei_leverage_sim/outputs_real_fast/*.csv`（daily / optimization / trades、計 約 1.1MB の生データ羅列）

> 判断理由: PNG はレポートの図として GitHub 上で必要なので残す。一方、行数の多い CSV は
> `cli run` で再生成でき、どのレポートにも埋め込まれていないため追跡外にしてリポジトリを軽量化した。
> ① の `data/` は ② の CSV ほど大きくなく、README が明示的な成果物として扱っているため維持。

## 実施チェックリスト

- [x] `git mv` で ① を `signal_report/` へ移動（履歴保持）
- [x] `index.html` はルートに残置（GitHub Pages 保護）
- [x] リポジトリ全体の新 `README.md` を作成（2 プロジェクトへ誘導）
- [x] `REPORT.md` の画像参照（相対 `figures/*.png`）が移動後も解決することを確認
- [x] 文書内のパス記述（旧 `backtest/...` → `signal_report/...`）を更新。生成元 `backtest.py` の
      テンプレートも合わせて修正し、再生成時も正しいパスになるようにした
- [x] 追跡されていた stale な `__pycache__/*.pyc` を削除（`.gitignore` 済み）
- [x] 生成物の扱いを **C（折衷）** に決定し反映（上記セクション参照、`.gitignore` 更新・大きな CSV を追跡解除）
- [ ] **要確認（ユーザー）**: GitHub Pages の配信元設定を Settings → Pages で確認（`index.html` がルート配信か）
