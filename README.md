# nikkei225-backtest-report

日経平均（`^N225`）関連の **2 つの独立したバックテスト・プロジェクト**を含むモノレポです。
両者はコードを共有せず、それぞれ別ディレクトリで完結しています。

| プロジェクト | ディレクトリ | 内容 |
|---|---|---|
| **① シグナル探索レポート** | [`signal_report/`](./signal_report/) | `^N225` の押し目買いシグナル候補を 5 年分で網羅探索（1,350 通り）し、Markdown / HTML レポートを生成 |
| **② レバレッジ ETF シミュレータ** | [`nikkei_leverage_sim/`](./nikkei_leverage_sim/) | 日経レバ ETF（1570.T）の「信用ロング・損切りなし・毎日積立・利確」戦略のバックテスター（src レイアウト + pytest） |

## ① signal_report/

`^N225` の 5 年データから単一 / 2-way / 3-way の条件を網羅し、利確水準別にバックテストして
期待値で並べたレポートを生成します。

- **読む**: [`signal_report/REPORT.md`](./signal_report/REPORT.md)（GitHub / モバイルでそのまま閲覧可）
- **インタラクティブ版**: `signal_report/report.html`（ローカルブラウザで開く）
- **実行**: 詳細は [`signal_report/README.md`](./signal_report/README.md)

```bash
cd signal_report
pip install -r requirements.txt
python backtest.py
```

## ② nikkei_leverage_sim/

日経レバ ETF を信用ロング・損切りなしでシミュレーションするバックテスター。
詳細は [`nikkei_leverage_sim/README.md`](./nikkei_leverage_sim/README.md) を参照。

```bash
cd nikkei_leverage_sim
pip install -e ".[dev,fetch]"
python -m nikkei_leverage_sim.cli synth --out data/ --days 900 --seed 7
python -m nikkei_leverage_sim.cli run   --config examples/sample_config.yaml --synthetic
pytest -q
```

## その他

- `index.html` — GitHub Pages 用ランディングページ（① のレポートの自己完結 HTML 版）。ルートに配置。
- [`CLAUDE.md`](./CLAUDE.md) — Claude Code 向けの作業ガイド。
- [`MIGRATION.md`](./MIGRATION.md) — 本構成への整理計画と未決事項（生成物の追跡方針など）。

> ⚠️ いずれのバックテストも研究用であり、**将来の成績を一切保証しません**。
> 手数料・税・スリッページの扱いは各プロジェクトの注意書きを参照してください。
