# accumulation_study/ — 積立×Exit 比較スタディ

決まった**使える資本 ¥10M**（現金のみ・ETF内蔵2倍以外のレバなし）で、1570.T を
**どう買い増し（積立）→ いつ売る（Exit）**のが「大きなドローダウンを避けつつ利益最大」かを
総当たり比較する独立サブパッケージ。**コア（`src/`）は無改変**で、`nikkei_leverage_sim` のメトリクスのみ再利用する（`variants/` と同じ方針）。

主要アウトプット: リポジトリの **`nikkei_leverage_sim/REPORT_ACCUMULATION.md`**（Calmar ランキング・局面別頑健性・但し書き）。

## 構成
| ファイル | 役割 |
|---|---|
| `engine.py` | ルックアヘッド無しの「積立→Exit」シミュレータ＋Calmar 等のメトリクス。`simulate()` / `evaluate()` |
| `signals.py` | 事前計算の lookahead-safe 指標（trailing peak / SMA100,200 / RSI14 / 20日ボラ / 12mモメンタム） |
| `policies.py` | 積立手法（一括・定額DCA・押し目買い・バリューアベレージング・トレンド濾過・ボラ/RSI調整・デュアルモメンタム）と Exit（hold/利確/トレーリング/MA割れ/期日glide/ボラ退避） |
| `run_study.py` | 事前登録グリッド（~214プラン）を実行し `outputs/rows.csv`・`summary.json` を出力 |
| `make_report.py` | `outputs/` から `REPORT_ACCUMULATION.md` を生成（表はデータ駆動・考察は記述） |
| `tests/` | エンジン不変条件（資本基準・**ノールックアヘッド**・Exit往復）と各ポリシー・指標の検証 |

## 実行（必ず `nikkei_leverage_sim/` 内で）
```bash
python -m accumulation_study.run_study --prices outputs_real/daily.csv --out accumulation_study/outputs
python -m accumulation_study.make_report
python -m pytest accumulation_study/tests -q     # 13 tests, 外部通信なし
```
価格系列は追跡済みの `outputs_real/daily.csv`（`target_close`=1570.T、3,032営業日）を既定で使用。

## 不変条件 / 注意
- **ルックアヘッド厳禁**: 当日 `t` の判断は `closes[:t+1]` のみ。`tests/test_engine.py` が「未来の値を変えても過去のエクイティは不変」を検証。
- **現金のみ・追加レバなし**、現金金利0%、端株可、**コスト未計上**（手数料・税・スリッページ・トラッキング誤差ゼロ）。回転の多い手法ほど理想化で有利に出る点は report に明記。
- **過去1本の経路**（2倍レバに有利な追い風期間・端点効果）の結果であり将来を保証しない。局面別 Calmar を併読すること。
