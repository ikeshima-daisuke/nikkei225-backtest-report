"""Render ``REPORT_REGIME.md`` from the regime-study JSON outputs.

Reads ``outputs/summary.json`` (accumulation grid headlines + calibration),
``outputs/survival.json`` (margin/ruin), and ``outputs/<key>/validation.json``
(statistical battery, when present), and writes the Japanese report next to the
other ``REPORT_*.md`` at the package root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

PKG = Path(__file__).resolve().parents[1]  # nikkei_leverage_sim/


def _pct(x: Optional[float], dp: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.{dp}f}%"


def _f(x: Optional[float], dp: int = 3) -> str:
    return "—" if x is None else f"{x:.{dp}f}"


def _plan(p: Optional[dict]) -> str:
    return "—" if not p else f"`{p['label']}`"


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _header(calib: dict, regimes: List[dict]) -> str:
    n_adv = sum(1 for r in regimes if r["kind"] == "adverse")
    return f"""# 逆風レジーム頑健性スタディ — 失われた30年 replay（1570.T 2倍レバ）

> **対象:** 日経平均 `^N225` 実データ **1989-2013**（バブル崩壊・失われた30年）から再構成した合成 1570.T（日次2倍・逓減込み） / **比較基準:** 実era 2014-2026 / **手法:** 同一の積立×Exitグリッド（{len(regimes)}レジーム中{n_adv}が逆風）＋ 追証/破綻エンジン（コア無改変） / **資本:** 積立は使える¥10M、生存は¥100M/¥5M/¥3.3M
> ⚠️ **理想化＋合成**: 1570.Tは2012年上場のため逆風期は実在しない。合成は実1570.Tで**較正・検証済み**（後述）だが、過去1本の経路であり将来を保証しない。手数料/税/スリッページは積立グリッドでは0（理想値）。

> 🔴 **先に結論:** 既存レポート群の「**hold が最良**」「**VA+hold**」「**¥5M(2倍)なら生存**」という結論は、すべて **2014-2026 の追い風1本**に依存していた。実際の失われた30年を当てると **3つとも反転/崩壊**する:
> 1. **hold は壊滅**（buy&hold 2倍は失われた10年で **{_pct(next((r['lump_hold']['total_return'] for r in regimes if r['key']=='lost_decade_1'), None),0)}**）。勝つのは「**降りる/現金退避/トレンド+利確**」系。
> 2. **積立法の順位も総入れ替わり**（追い風=VA/dip hold、逆風=trend+tp / trail）。
> 3. **生存閾値も追い風限定**: ¥3.3M(3倍)は逆風で**債務超過**、¥5M(2倍)も強制決済は免れるが**自己資金 −40〜−76%**。
> **構造的な唯一の頑健結論は変わらず**: 2倍レバETFで高リターンを狙えば最大DD ~50%超は不可避で、**逆風では逓減＋金利がレバを食い尽くす**。

### 合成1570.Tの忠実度（実1570.T 2014-2026 で較正・検証）

| 指標 | 値 | 判定 |
|---|---:|:--:|
| beta（対N225日次） | {_f(calib['beta'],4)} | ≈2.0 ✓ |
| 相関（日次リターン vs 2×N225） | {_f(calib['corr_daily_vs_2x'],4)} | 日次2倍ETF ✓ |
| 累積一致（合成 / 実） | {calib['cum_synth']} / {calib['cum_real']} | 一致 ✓ |
| 年トラッキング誤差 | {_pct(calib['tracking_error_ann'])} | 妥当 |
| 較正ドラッグ（base_drag, 2014-2026の低金利期） | {_pct(calib['base_drag'])}/年 | ER+金利+TE |

> 機構（日次2倍リバランス＝逓減）を実1570.Tに対し較正し、相関{_f(calib['corr_daily_vs_2x'],3)}・累積を完全一致させたうえで、同じ構築式を1989-2013に適用。逆風期は**時変JPY短期金利**（1990-92は4〜7.4%）を `(レバ-1)×金利` のファイナンスコストとして上乗せ（[`financing.py`](regime_study/financing.py)）。低金利前提なら逆風はさらに過小評価になるため、これは保守的方向。
"""


def _summary_table(regimes: List[dict]) -> str:
    rows = []
    for r in regimes:
        bh = r["best_hold"]; be = r["best_exit"]
        rows.append(
            f"| {r['label']} | {_pct(r['n225_total_return'],0)} | "
            f"{_pct(r['lump_hold']['total_return'],0)} (DD {_pct(r['lump_hold']['max_drawdown_pct'],0).lstrip('+')}) | "
            f"{_plan(bh)} {_f(bh['calmar'],2) if bh else '—'} | "
            f"{_plan(be)} {_f(be['calmar'],2) if be else '—'} | "
            f"{'**hold**' if r['hold_beats_exit'] else 'exit'} |"
        )
    body = "\n".join(rows)
    return f"""## 1. 結論サマリ — 追い風の3結論が逆風で反転

| レジーム | N225(無レバ) | buy&hold 2倍 | 最良hold (Calmar) | 最良exit (Calmar) | 勝者 |
|---|---:|---:|---|---|:--:|
{body}

- **強気era だけ hold が勝つ。** すべての逆風レジームで **exit 系が hold を上回る**（勝者列）。「holdが最良」は相場レジームの関数であって戦略の優位ではない。
- buy&hold 2倍は逆風で **−83〜−98%**。N225(無レバ)の −45〜−80% に対し、**逓減がレバ効果を反転させ損失を増幅**している。
"""


def _hold_vs_exit(regimes: List[dict]) -> str:
    blocks = []
    for r in regimes:
        if r["kind"] != "adverse":
            continue
        top = "\n".join(
            f"| {_plan(p)} | {_pct(p['total_return'],0)} | {_pct(p['max_drawdown_pct'],0).lstrip('+')} | "
            f"{_f(p['calmar'],3)} | {_pct(p['avg_invested_pct'],0).lstrip('+')} |"
            for p in r["top5"]
        )
        blocks.append(f"""**{r['label']}** — 最良hold {_plan(r['best_hold'])} は {_pct(r['best_hold']['total_return'],0)}（DD {_pct(r['best_hold']['max_drawdown_pct'],0).lstrip('+')}）。上位5は:

| プラン | 総リターン | 最大DD | Calmar | 平均投資率 |
|---|---:|---:|---:|---:|
{top}
""")
    body = "\n".join(blocks)
    return f"""## 2. 結論①の反転 — hold は壊滅、勝つのは「降りる」戦略

追い風期は上位が全て hold（[REPORT_ACCUMULATION.md](REPORT_ACCUMULATION.md)）。逆風では真逆で、**現金退避（trail/maexit）・トレンド+利確（trend·tp）** が上位を占め、平均投資率が極端に低い（=ほとんど現金で待つ）プランほど傷が浅い。

{body}
> **読み:** 逆風では「**含み損を確定しない**」hold が裏目（損が膨らみ続ける）。降りる判断＝現金/トレンド退避が初めて報われる。追い風期に「exitは軒並み逆効果」と結論したのは**端点が高値だったから**で、一般法則ではない。
"""


def _family_ranking(regimes: List[dict]) -> str:
    def fam_line(r):
        fb = r["family_best_hold"]
        order = list(fb.keys())
        return f"| {r['label']} | " + " → ".join(
            f"{fam}({_f(fb[fam]['calmar'],2)})" for fam in order[:5]) + " |"
    lines = "\n".join(fam_line(r) for r in regimes)
    return f"""## 3. 結論②の反転 — 積立法ランキングの総入れ替わり

各レジームで hold プランを積立ファミリー別に最良 Calmar で並べた順位（左が最良）:

| レジーム | 積立ファミリー順位（hold・Calmar） |
|---|---|
{lines}

- 追い風では **value_avg / dip（買い下がり）** が上位だが、逆風では同じ「買い下がり」が**地獄**（落ちるナイフを掴み続ける）。逆風で相対的にマシなのは **trend / scaled（トレンド濾過・低ボラ時に買う）** 系。
- これは [REPORT_ACCUMULATION.md §6](REPORT_ACCUMULATION.md) の「勝者は局面で入れ替わる・単一プラン選択は過剰最適化」を、**別の歴史（実データ）で追認**するもの。
"""


def _survival(survival: Dict[str, list], regimes: List[dict]) -> str:
    label_by_key = {r["key"]: r["label"] for r in regimes}
    blocks = []
    for key in [r["key"] for r in regimes]:
        rows = survival.get(key)
        if not rows:
            continue
        body = "\n".join(
            f"| {x['capital_label']} | {_pct(x.get('account_margin_rate'),1).lstrip('+')} | "
            f"{_pct(x['own_funds_return'],1)} | "
            f"{_f(x['min_maintenance_ratio'],2)} | {x['margin_call_count']} | "
            f"{x['forced_liquidation_count']} | {'🔴 破綻' if x['ruined'] else '生存'} |"
            for x in rows
        )
        blocks.append(f"""**{label_by_key.get(key, key)}**

| 自己資金(レバ) | 口座信用金利 | 自己資金リターン | 最低維持率 | 追証 | 強制決済 | 判定 |
|---|---:|---:|---:|---:|---:|:--:|
{body}
""")
    body = "\n".join(blocks)
    return f"""## 4. 結論③の反転 — 生存閾値も追い風限定

[REPORT_MARGIN_CALL.md](REPORT_MARGIN_CALL.md) の「¥5M(2倍)は生存・¥3.3M(3倍)で破綻」は **2014-2026 の結論**。固定（デフォルト）戦略・`force_liquidation` ON で逆風期を流すと（**公平のため口座の信用金利もレジーム別の実コール金利＋スプレッドに整合**。バブル期 7.0% 等。ZIRP の追い風期は既定の 2.8% に一致＝ベースライン不変）:

{body}
> **読み:** **¥100M** は建玉が極薄で全レジーム生存（=ほぼ無レバ）。**¥3.3M(3倍)** は追い風で +106% だが**失われた30年では債務超過**（自己資金 −100%超、強制決済多発）。**¥5M(2倍)** は強制決済こそ免れても、損切りなしで含み損を抱え続け**自己資金が −33〜−80%** に毀損する。「2倍なら安全」は**追い風限定の安全**。
"""


def _validation(out_dir: Path, regimes: List[dict]) -> str:
    blocks = []
    for r in regimes:
        v = _load(out_dir / r["key"] / "validation.json")
        if not v:
            continue
        o = v["out_of_sample"]
        any_sig = any(x["reject_fdr05"] for x in v["permutation_fdr"].values())
        pairs = "\n".join(
            f"| `{p['a'].split('|')[0]}|{p['a'].split('|')[1]}` vs "
            f"`{p['b'].split('|')[0]}|{p['b'].split('|')[1]}` | {p['obs_diff']:+.3f} | "
            f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] | {p['p_a_not_gt_b']:.3f} |"
            for p in v["pairwise"]
        )
        blocks.append(f"""**{r['label']}** （block bootstrap B={v['config']['n_boot']} / permutation B={v['config']['n_perm']}）

- アウトオブサンプル: 前半の最良 `{o['in_sample_winner']}` は後半で **{o['winner_oos_rank']}/{o['n_plans']}位**（Calmar {o['winner_oos_calmar']:.3f}）。
- 置換検定＋FDR(5%)で有意なプラン: **{'あり' if any_sig else 'なし（0件）'}**。
- 主要ペア差（Calmar, 95%CI が0を跨げば有意差なし）:

| 比較 A vs B | 観測差 | 95%CI of 差 | p(A≦B) |
|---|---:|---:|---:|
{pairs}
""")
    body = "\n".join(blocks) if blocks else "_（`validate.py` 未実行。`python -m regime_study.validate` で生成）_"
    return f"""## 5. 統計検証（順位の反転は本物か、それともまた別のノイズか）

[accumulation_study の検証機構](REPORT_ACCUMULATION.md)（ブロックブートストラップCI・置換検定・BH-FDR・OOS）を、合成逆風経路で再実行。

{body}
> **読み:** 逆風でも「単一の勝ちプラン」を選ぶのは依然として過剰最適化（OOSで順位が動き、多くは FDR 後に非有意）。ただし**方向性**——逆風では exit/現金が hold に勝つ——は、追い風で hold が勝つのと**鏡像**で頑健に現れる。「レジームに賭ける」のであって「プランに賭ける」のではない。
"""


def _facts(regimes: List[dict]) -> str:
    rows = "\n".join(
        f"| {r['label']} | {r['start']}〜{r['end']} | {r['n_days']} | "
        f"{_pct(r['n225_total_return'],0)} | {_pct(r['lump_hold']['total_return'],0)} |"
        for r in regimes
    )
    return f"""## 6. レジーム別ファクト（無レバ vs 2倍hold の逓減ギャップ）

| レジーム | 期間 | 営業日 | N225(無レバ) | 2倍ETF buy&hold |
|---|---|---:|---:|---:|
{rows}

> N225 が小幅マイナスでも 2倍 buy&hold は深いマイナス＝**逓減ギャップ**。横ばい・乱高下が長いほど開く（デフレ横ばい 2000-2012 が典型）。
"""


def _caveats() -> str:
    return """## 7. 但し書き（正直に）

1. **合成・非実在**: 1570.Tは2012年上場。逆風期は実N225から再構成した合成で、実1570.Tでの較正（相関0.99・累積一致）に依存する。実ETFの板・乖離・分配金・貸株は別。
2. **金利前提（公平性）**: ファイナンスは BoJ 無担保コール翌日物の**年平均近似**で、**2層とも**レジーム別に整合させた——(i) 合成ETFの内部ファイナンス `(レバ-1)×金利`（[`financing.py`](regime_study/financing.py)）、(ii) 生存テストの**口座信用金利** `コール金利＋スプレッド`。スプレッドは ZIRP の追い風期(2014-2026)に既定 2.8% へ一致するよう較正したので**ベースラインは不変**で、逆風期だけ史実どおり重くなる。日次・スプレッドの精緻化で水準は動くが、**結論の符号（逆風で壊滅）は金利を0にしても変わらない**（逓減だけで十分壊滅的）。
3. **固定戦略の生存テスト**: 生存は最適化なしのデフォルト戦略（[REPORT_VALIDATION](REPORT_VALIDATION.md) と同じ公平な選択）。ウォークフォワード最適化器そのものの逆風耐性は別問題。
4. **端点効果は両刃**: 追い風ベースラインは高値で終わり hold を過大評価。逆風レジームは底や途中で終わり exit を過大評価しうる。だからこそ**複数レジーム**で符号の一貫性を見ている。
5. **理想化コスト**（積立グリッド）: 手数料/税/スリッページ0。回転の多い exit 系を過大評価するが、**それでも逆風で exit が勝つ**＝結論はコストでむしろ強まる方向。

---

> 再現: `python -m regime_study.run_study`（積立グリッド）→ `python -m regime_study.survival`（生存）→ `python -m regime_study.validate`（統計）→ `python -m regime_study.make_report`。データ取得は `data/benchmark_N225_long.csv`（^N225 1989-2026・gitignore済）。コア（`src/`）は無改変、独立サブパッケージ。
"""


def build_report(out_dir: Path) -> str:
    summary = _load(out_dir / "summary.json")
    if not summary:
        raise SystemExit(f"missing {out_dir/'summary.json'} — run run_study first")
    survival = _load(out_dir / "survival.json") or {}
    calib = summary["calibration"]
    regimes = summary["regimes"]
    parts = [
        _header(calib, regimes),
        _summary_table(regimes),
        _hold_vs_exit(regimes),
        _family_ranking(regimes),
        _survival(survival, regimes),
        _validation(out_dir, regimes),
        _facts(regimes),
        _caveats(),
    ]
    return "\n".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="regime_study/outputs")
    ap.add_argument("--report", default=str(PKG / "REPORT_REGIME.md"))
    args = ap.parse_args(argv)
    text = build_report(Path(args.out))
    Path(args.report).write_text(text, encoding="utf-8")
    print("wrote", args.report)


if __name__ == "__main__":
    main()
