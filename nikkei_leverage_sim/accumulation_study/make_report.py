"""Render REPORT_ACCUMULATION.md from outputs/rows.csv + summary.json.

Tables are data-driven (so the report regenerates with the grid); the analysis
prose is curated.  Run after ``run_study`` from ``nikkei_leverage_sim/``::

    python -m accumulation_study.make_report
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

PKG = Path(__file__).resolve().parent
OUT = PKG / "outputs"
REPORT = PKG.parent / "REPORT_ACCUMULATION.md"

_LABELS = {
    "lump": "一括(全額初日)", "dca": "定額DCA", "dip": "押し目買い",
    "value_avg": "バリューアベレージング", "trend": "トレンド濾過DCA",
    "scaled": "ボラ/RSI調整DCA", "momentum": "デュアルモメンタム",
}


def _pct(x: float, d: int = 1) -> str:
    return f"{float(x) * 100:+.{d}f}%"


def _disp(label: str) -> str:
    # Pipes break GitHub table cells (even inside code spans); show with middots.
    return label.replace("|", "·")


def _row(r: Dict[str, str]) -> str:
    return (
        f"| `{_disp(r['label'])}` | {_pct(r['total_return'])} | "
        f"{float(r['max_drawdown_pct']) * 100:.1f}% | **{float(r['calmar']):.3f}** | "
        f"{float(r['sortino']):.2f} | {float(r['avg_invested_pct']) * 100:.0f}% | "
        f"{r['n_sells']} | {'✓' if r['exited_early'] == 'True' else '—'} |"
    )


def main() -> None:
    rows = list(csv.DictReader(open(OUT / "rows.csv")))
    summ = json.load(open(OUT / "summary.json"))
    by_label = {r["label"]: r for r in rows}
    w = summ["window"]

    header = "| プラン (`積立·Exit·頻度·往復`) | 総リターン | 最大DD | Calmar | Sortino | 平均投資率 | 売却 | 早期撤退 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|:--:|"

    L: List[str] = []
    L.append("# 積立×Exit 比較スタディ — 1570.T（使える資本 ¥10M）")
    L.append("")
    L.append(
        f"> **対象:** 1570.T 終値 / **期間:** {w['start']} 〜 {w['end']}"
        f"（{w['n_days']:,}営業日・{w['n_months']}ヶ月）/ **資本:** ¥{w['capital']:,.0f}（現金のみ・追加レバなし）/ "
        f"**検証プラン数:** {summ['n_plans']} / **ランキング:** Calmar = CAGR ÷ 最大DD"
    )
    L.append(
        "> ⚠️ **理想化**: 手数料・スリッページ・税・トラッキング誤差ゼロ、端株可、現金金利0%。"
        "回転の多いExitほどこの理想化で有利に出る（実際はコストで悪化）。**過去1本の経路**の結果で将来を保証しない。"
    )
    L.append("")
    L.append("## 0. 何を比べたか")
    L.append("")
    L.append(
        "「決まった資本¥10Mを、**どう買い増し（積立）**、**いつ売る（Exit）**のが、"
        "**大きなドローダウンを避けつつ利益を最大化**できるか」を総当たりで比較。"
        "各プランは〔積立手法〕×〔Exitルール〕×〔頻度: 毎月/毎営業日〕×〔Exit往復: 単発1shot/回転rot〕。"
        "全て**ルックアヘッド無し**（当日終値までの情報のみ）・**現金のみ**（ETF内蔵の2倍以外のレバなし）。"
        "リターン・DD は §10 と同じ**使える資本¥10Mを分母**にしたエクイティ基準。"
    )
    L.append("")
    L.append("**積立手法:** " + " / ".join(_LABELS.values()))
    L.append("")
    L.append(
        "**Exitルール:** hold(売らない) / tp50・tp100(利確+50/100%) / "
        "trail15・25・35(エクイティ高値からのトレーリングストップ) / "
        "maexit200(終値<200日線で全売) / glide12(期日前12ヶ月で分割売却) / volx45(20日ボラ>45%で退避)"
    )
    L.append("")

    # --- A. headline ---
    L.append("## 1. ★Calmar 上位（大DD回避×利益のバランス最良）")
    L.append("")
    L.append(header)
    L.append(sep)
    for r in rows[:15]:
        L.append(_row(r))
    L.append("")
    L.append(
        "- **上位は全て `hold`（売らない）。** 利確・トレーリング・MA退避などExitを足すと、"
        "この強い上昇相場では**トレンドから降りてしまい Calmar が必ず悪化**した（次節）。"
    )
    L.append(
        "- **最良は バリューアベレージング＋hold**（`va48_0|hold`：Calmar "
        f"{float(by_label['va48_0|hold|monthly|1shot']['calmar']):.3f}）。"
        "下落時に多く・上昇時に少なく買う反循環の買い方が、**最大DDを 55%→48% に削りつつ +1,055% を確保**。"
    )
    L.append(
        "- **DDの下限は ~43〜48%**。現金を多めに残すボラ/トレンド系でもDDは43%超で、"
        "リターンは落ちる。**2倍レバETFでは『高リターン×低DD』は両立しない**のが実データの結論。"
    )
    L.append("")

    # --- controls ---
    L.append("### 参考：基準プラン")
    L.append("")
    L.append(header)
    L.append(sep)
    for lbl in ("lump|hold|monthly|1shot", "dca48|hold|monthly|1shot",
                "dca48|hold|daily|1shot", "mom252|momexit|monthly|rot"):
        if lbl in by_label:
            L.append(_row(by_label[lbl]))
    L.append("")
    L.append(
        "- **一括(全額初日)＋hold** はリターン最大級（+1,191.9%）だが最大DD 55.4%・Calmar 0.43 で、"
        "VA に Calmar で負ける（DDが深い）。**デュアルモメンタム**は局面切替の空振り（whipsaw）で Calmar 0.24 と"
        "buy&hold に劣り、「賢く避ける」系が裏目に出る典型。"
    )
    L.append("")

    # --- B. exits hurt ---
    ex = [r for r in rows if r["label"].split("|")[1] != "hold"]
    ex.sort(key=lambda r: -float(r["calmar"]))
    L.append("## 2. 重要な発見：Exit（売り）は軒並み逆効果")
    L.append("")
    L.append("Exit を使うプランの中で Calmar 最良でも以下どまり（hold 系に遠く及ばない）:")
    L.append("")
    L.append(header)
    L.append(sep)
    for r in ex[:6]:
        L.append(_row(r))
    L.append("")
    L.append(
        "- トレーリングストップやMA退避は**確かにDDを下げる**が、**現金滞留率が跳ね上がり（平均投資率 1〜7%）**、"
        "大トレンドを取り逃して総リターンが消える。`trail15` の極端例では一度売って二度と良い位置で戻れず**マイナス**に。"
    )
    L.append(
        "- 利確(tp)は『+50/100%で降りる』ことで ×12.9 の複利を途中で打ち切り、リターンを大きく毀損。"
        "**この銘柄・この期間では『売る判断』そのものが負けの主因**。"
    )
    L.append("")

    # --- C. sub-window robustness ---
    sw = summ["subwindow_calmar"]
    sp = summ["splits"]
    L.append("## 3. 局面別の頑健性（Calmar をサブ期間で再計算）")
    L.append("")
    L.append(
        f"全期間1本の順位は**終盤の追い風**に依存する。3分割して各局面のCalmarを見ると頑健性が分かる:"
    )
    L.append("")
    L.append(
        f"| プラン | {sp[0]['start']}〜{sp[0]['end']} | "
        f"{sp[1]['start']}〜{sp[1]['end']} | {sp[2]['start']}〜{sp[2]['end']} |"
    )
    L.append("|---|---:|---:|---:|")
    for lbl, vals in sw.items():
        L.append(f"| `{_disp(lbl)}` | {vals[0]:+.2f} | {vals[1]:+.2f} | {vals[2]:+.2f} |")
    L.append("")
    L.append(
        "- **バリューアベレージングが最も頑健**：3局面ともプラスで、コロナを含む中間期"
        f"（{sp[1]['start']}〜{sp[1]['end']}）でも一括(lump)の {sw.get('lump|hold|monthly|1shot', ['','',''])[1]:+.2f} に対し優位。"
    )
    L.append(
        "- **直近期（2022〜）は全プランCalmar≈1.0 と異常に高い**＝終盤が高値圏で終わる『端点効果』。"
        "全期間ランキングはこの追い風で底上げされている点に注意。"
    )
    L.append("")

    # --- D. recommendation ---
    L.append("## 4. 目的（大DD回避×利益最大）への答え")
    L.append("")
    L.append(
        "- **積立の工夫（買い方）は効く。売りの工夫（Exit）は効かない。** "
        "大DDを避けつつ利益を残す最良は **バリューアベレージング（毎月・下落で多めに買う）を、売らずに持ち切る**。"
        "次点で押し目買い＋hold。トレーリングストップ等の『損失限定』は、この2倍レバ×上昇相場では裏目。"
    )
    L.append(
        "- **ただしDDは避けきれない**（最良でも最大DD ~48%）。本当にDDを30%未満に抑えたいなら、"
        "投資率を落とす（=リターンも落ちる）か、2倍レバETF自体を諦めて無レバ指数にするしかない"
        "（§10 で N225 定額積立が DD 24.6% だったのを参照）。"
    )
    L.append(
        "- **回転売買は理想化の恩恵を最も受ける**（コスト無視）。実コストを入れると Exit 系はさらに悪化するため、"
        "結論（hold 優位）はむしろ強まる。"
    )
    L.append("")

    # --- E. caveats ---
    L.append("## 5. 正直な但し書き")
    L.append("")
    L.append(
        "1. **単一経路・追い風バイアス**: 12.4年・1銘柄・2倍レバに歴史的に有利な期間で、"
        "終点も高値圏。順位はこの経路条件付き。局面別表（§3）で頑健性を併読。\n"
        "2. **レバETFの逓減**: 1570.Tは日次2倍リバランスで、長期・レンジ・下落で減価。"
        "hold優位は『この期間が強トレンドだった』ことに強く依存し、逆風局面では一括holdが最悪化しうる。\n"
        "3. **コスト理想化**: 手数料/スリッページ/税/トラッキング誤差ゼロ。回転の多い`rot`系・Exit系を過大評価。\n"
        "4. **多重比較**: 214プランの最良は一部運。単独スパイクでなく『近傍も良い手法ファミリー』を重視（VA/holdは近傍も良好）。\n"
        "5. **現金ドラッグ**: 押し目・トレンド系は無利子現金で待つ機会費用が大きい。平均投資率を併記した理由。"
    )
    L.append("")
    L.append(
        f"> 全 {summ['n_plans']} プランの全指標は `accumulation_study/outputs/rows.csv`、"
        "局面別は `summary.json`。再現は `python -m accumulation_study.run_study` → `make_report`。"
        "コア（`src/`）は無改変、独立サブパッケージ。"
    )
    L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
