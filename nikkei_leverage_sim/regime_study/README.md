# regime_study — 逆風レジーム頑健性スタディ（失われた30年 replay）

`accumulation_study/` ・ 各 `REPORT_*.md` の結論（「hold が最良」「VA+hold」「¥5M(2倍)なら生存」）は
**すべて 2014-2026 の追い風 1 本**に依存している。このサブパッケージは、その最大の積み残し
——「結論は強気相場限定なのか？」——を、**日本が実際に経験した失われた30年**で直接検証する。

1570.T は 2012 年上場なので逆風期は実在しない。そこで **実 `^N225` 1989-2013** から
**日次2倍リバランス（逓減込み）の合成 1570.T** を再構成する。構成は実 1570.T（2014-2026）に対し
**較正・検証**してある（beta≈2.0・日次相関≈0.99・累積一致）。較正後、同じ式を逆風期に適用し、
**時変 JPY 短期金利**（1990-92 は 4〜7.4%）を `(レバ-1)×金利` のファイナンスコストとして上乗せする。

**コア（`src/`）は無改変。** `accumulation_study` のグリッド・エンジン・統計検証を再利用する。

## 構成

| ファイル | 役割 |
|---|---|
| `financing.py` | 時変 JPY 無担保コール翌日物（年平均近似）→ 日次ファイナンスドラッグ |
| `build_target.py` | 実 N225 → 合成 1570.T OHLC（較正 `calibrate_base_drag` ＋構築 `build_synthetic_target`） |
| `regimes.py` | レジーム窓の定義（バブル崩壊/失われた10年/デフレ横ばい/リーマン/二つの失われた10年/強気ベースライン） |
| `run_study.py` | 各レジームで積立×Exit グリッドを replay（`accumulation_study` を再利用） |
| `survival.py` | 各レジーム×資本(¥100M/¥5M/¥3.3M)で追証/破綻をコアエンジンで測定（固定戦略・`force_liquidation` ON） |
| `validate.py` | 主要レジームで統計検証（ブートストラップ CI・置換 FDR・OOS）を再実行 |
| `make_report.py` | `REPORT_REGIME.md` を生成 |

## 実行（必ず `nikkei_leverage_sim/` 内で）

```bash
# データ取得（^N225 1989-2026、要ネットワーク・gitignore 済）
python -m nikkei_leverage_sim.cli fetch --start 1989-01-01 --end 2026-06-07 \
    --out data/ --target-symbol ^N225 --benchmark-symbol ^N225   # 便宜上。実際は下記スクリプトでも可

python -m regime_study.run_study      # 積立×Exit グリッド（全レジーム）
python -m regime_study.survival       # 追証/破綻（資本×レジーム）
python -m regime_study.validate       # 統計検証（時間がかかる）
python -m regime_study.make_report    # REPORT_REGIME.md

pytest regime_study/tests -q          # テスト（人工データ。実データ忠実度テストは data 不在時 skip）
```

> 入力データ `data/benchmark_N225_long.csv`（`Date,Open,High,Low,Close,Adj Close,Volume`）が必要。
> `accumulation_study` と同様、ローカル `tests/conftest.py` が `src` と PKG を `sys.path` に追加する
> （コアの `pytest -q` は `testpaths=["tests"]` のため拾わない。別途実行）。

## 主要な学び

- **追い風の3結論は逆風で反転/崩壊する。** hold は失われた10年で buy&hold 2倍が ≈−98%。勝つのは
  現金退避（trail/maexit）・トレンド+利確。積立法の順位も総入れ替わり。¥3.3M(3倍)は逆風で債務超過、
  ¥5M(2倍)も自己資金 −40〜−76%。
- **唯一頑健**: 2倍レバで高リターンを狙えば最大DD ~50%超は不可避。逆風では逓減＋金利がレバを食い尽くす。
- 詳細は [`REPORT_REGIME.md`](../REPORT_REGIME.md)。
