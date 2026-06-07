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
詳細は [`nikkei_leverage_sim/README.md`](./nikkei_leverage_sim/README.md)、**開発を継続するなら**
[`nikkei_leverage_sim/DEVELOPMENT.md`](./nikkei_leverage_sim/DEVELOPMENT.md)（到達点・結論・ロードマップ）を参照。

```bash
cd nikkei_leverage_sim
pip install -e ".[dev,fetch]"
python -m nikkei_leverage_sim.cli synth --out data/ --days 900 --seed 7
python -m nikkei_leverage_sim.cli run   --config examples/sample_config.yaml --synthetic
pytest -q
```

主な検証レポート（GitHub / モバイルでそのまま閲覧可）:
[実データ3版](./nikkei_leverage_sim/REPORT_REAL.md) ・ [戦略バリアント](./nikkei_leverage_sim/REPORT_VARIANTS.md) ・
[積立×Exit](./nikkei_leverage_sim/REPORT_ACCUMULATION.md) ・ [ストレス](./nikkei_leverage_sim/REPORT_STRESS.md) ・
[追証](./nikkei_leverage_sim/REPORT_MARGIN_CALL.md) ・ [約定](./nikkei_leverage_sim/REPORT_EXECUTION.md) ・
[統計検証](./nikkei_leverage_sim/REPORT_VALIDATION.md) ・ [逆風レジーム](./nikkei_leverage_sim/REPORT_REGIME.md) ・
[公平性監査](./nikkei_leverage_sim/REPORT_FAIRNESS.md)

> 横断的な学び: **タイミングの α は検出されず（実体は増幅された β）、勝率100%は損切りしない裏返し（含み損在庫）。
> 「hold 最良」等の結論は追い風相場限定で、失われた30年では反転・崩壊する。**

## その他

- `index.html` — GitHub Pages 用ランディングページ（① のレポートの自己完結 HTML 版）。ルートに配置。
- [`CLAUDE.md`](./CLAUDE.md) — Claude Code 向けの作業ガイド（不変条件・実行環境・公平性規約）。
- [`MIGRATION.md`](./MIGRATION.md) — 本構成への整理経緯（生成物の追跡方針は **C/折衷で確定済**。残タスクは GitHub Pages 配信元の確認のみ）。

> ⚠️ いずれのバックテストも研究用であり、**将来の成績を一切保証しません**。
> 手数料・税・スリッページの扱いは各プロジェクトの注意書きを参照してください。
