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


def _validation_section(L: List[str], v: Dict) -> None:
    cfg = v["config"]
    L.append("## 6. 統計的検証（有意性・多重比較・アウトオブサンプル）")
    L.append("")
    L.append(
        f"§1〜5 は記述的な総当たり。ここで *有意に区別できるのはどこまでか* を検定する: "
        f"ブロックブートストラップ（block={cfg['block']}営業日・B={cfg['n_boot']}）でCalmarの信頼区間と"
        f"プラン間差を出し、置換検定（B={cfg['n_perm']}・時系列をシャッフルした帰無）→ "
        "Benjamini-Hochberg FDR で多重比較を補正、さらに前半で選び後半で確認する。"
        "全て価格経路を再サンプルしてプランを毎回再実行（seed固定）。"
    )
    L.append("")

    # 6-1 bootstrap CIs
    L.append("### 6-1. Calmar の95%ブートストラップ信頼区間")
    L.append("")
    L.append("| プラン | 観測Calmar | 95%CI |")
    L.append("|---|---:|---:|")
    bci = v["bootstrap_calmar_ci"]
    for lbl in sorted(bci, key=lambda k: bci[k]["observed"], reverse=True):
        c = bci[lbl]
        L.append(f"| `{_disp(lbl)}` | {c['observed']:.3f} | "
                 f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}] |")
    L.append("")
    L.append("- CIが広く互いに**重なる**＝順位の細かい差は経路ノイズに埋もれる。")
    L.append("")

    # 6-2 pairwise
    L.append("### 6-2. 主要ペアの差（ブートストラップ）")
    L.append("")
    L.append("| 比較 A vs B | 観測差(Calmar) | 95%CI of 差 | p(A≦B) |")
    L.append("|---|---:|---:|---:|")
    for pr in v["pairwise"]:
        sig = "" if pr["ci_low"] > 0 else " ⚠️"
        L.append(f"| `{_disp(pr['a'])}` vs `{_disp(pr['b'])}` | {pr['obs_diff']:+.3f} | "
                 f"[{pr['ci_low']:+.3f}, {pr['ci_high']:+.3f}]{sig} | {pr['p_a_not_gt_b']:.3f} |")
    L.append("")
    L.append(
        "- 差のCIが0をまたぐ（⚠️）＝**有意差なし**。p(A≦B)が小さいほど「AがBより良い」が確からしい。"
    )
    L.append("")

    # 6-3 permutation + FDR
    L.append("### 6-3. 置換検定 ＋ FDR（多重比較補正）")
    L.append("")
    L.append("| プラン | 置換p値 | FDR5%で有意 |")
    L.append("|---|---:|:--:|")
    fdr = v["permutation_fdr"]
    for lbl in sorted(fdr, key=lambda k: fdr[k]["p_value"]):
        f = fdr[lbl]
        L.append(f"| `{_disp(lbl)}` | {f['p_value']:.3f} | {'✓' if f['reject_fdr05'] else '—'} |")
    L.append("")
    L.append(
        "- 「✓」＝時系列構造をシャッフルしたまぐれでは説明できない（FDR補正後も有意）。"
        "「—」＝偶然と区別できない。"
    )
    L.append("")

    # 6-4 OOS
    o = v["out_of_sample"]
    L.append("### 6-4. アウトオブサンプル（前半で選び後半で確認）")
    L.append("")
    L.append(
        f"- 前半（最初の50%）の最良は `{_disp(o['in_sample_winner'])}`"
        f"（in-sample Calmar {o['in_sample_calmar']:.3f}）。"
        f"これを後半（{o['split_date']}〜）で見ると **Calmar {o['winner_oos_calmar']:.3f}・"
        f"順位 {o['winner_oos_rank']}/{o['n_plans']}位に転落**。"
    )
    L.append(f"  - 前半の実際の上位5: " +
             ", ".join(f"`{_disp(l)}`({c:.2f})" for l, c in o["in_top5"]))
    L.append(f"  - 後半の実際の上位5: " +
             ", ".join(f"`{_disp(l)}`({c:.2f})" for l, c in o["oos_top5"]))
    L.append(
        "- **勝つ手法が前半↔後半で総入れ替わり**（前半=利確/exit系、後半=trend+利確系、"
        "全期間=VA/hold）。前半で選んだものは後半で上位に来ない＝**選択そのものが過剰最適化**。"
    )
    L.append("")

    # 6-5 cost
    cs = v["cost_sensitivity"]
    L.append(f"### 6-5. コスト感応度（往復 {cs['cost_bps']:.0f}bps）")
    L.append("")
    L.append("| | コスト無し上位5 | コスト有り上位5 |")
    L.append("|--:|---|---|")
    for i in range(5):
        a = cs["top5_no_cost"][i]
        b = cs["top5_with_cost"][i]
        L.append(f"| {i+1} | `{_disp(a[0])}` ({a[1]:.3f}) | `{_disp(b[0])}` ({b[1]:.3f}) |")
    L.append("")

    # interpretation
    L.append("### 6-6. 検証の結論（重要）")
    L.append("")
    L.append(
        "- **プラン間に統計的有意差は無い。** 全プランの Calmar 95%CI は概ね ≈[0, 1] と極めて広く"
        "互いに重なる（§6-1）。主要ペアの差も**全て CI が0をまたぐ**（§6-2、VA vs 一括 p=0.68 等）。"
        "置換検定では**FDR補正後に有意なプランは0件**（§6-3、最良VAでも p=0.17）。\n"
        "- **アウトオブサンプルで順位が崩壊（§6-4）。** 前半の最良は実は exit 系で、後半 31/205 位へ転落。"
        "前半=利確/exit、全期間=VA/hold、後半=trend+利確 と**勝者が局面で入れ替わる**。"
        "→ §1〜2 の『Exitは効かない・VA+holdが最良』も**期間依存の観察にすぎず、統計的に支持されない**"
        "（前半だけ見れば逆に exit が勝っていた）。\n"
        "- **唯一頑健に言えること**: (a) hold はコストに強い（回転ゼロ、§6-5 で上位不変）、"
        "(b) **高リターンのプランは構造的に最大DD ~46〜55% を伴う**（2倍レバの性質で、回避不能）。\n"
        "- **投資判断としては、単一の勝ちプランを選ぶのは過剰最適化。** 言えるのはせいぜい"
        "『この銘柄でDDを浅くしたいなら投資率を落とすか無レバ指数にする』という構造的な話まで。"
    )
    L.append("")


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
    L.append(
        "> 🔴 **先に結論（統計的検証 §6）**: プラン間の Calmar 差は**統計的に有意でない**"
        "（信頼区間が大きく重なり、置換検定でFDR後に有意なものは0件）。しかも**最良プランは"
        "アウトオブサンプルで再現しない**（前半の最良 exit 系は後半 31/205 位に転落、勝つ手法は"
        "局面で入れ替わる）。**以下 §1〜5 の順位は『この1本の経路での記述的観察』として読むこと。**"
        "単一の勝ちプランを選ぶのは過剰最適化。"
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
        "- **観測上の最良は バリューアベレージング＋hold**（`va48_0·hold`：Calmar "
        f"{float(by_label['va48_0|hold|monthly|1shot']['calmar']):.3f}）。"
        "下落時に多く・上昇時に少なく買う反循環の買い方が、最大DDを 55%→48% に削りつつ +1,055% を確保。"
        "**ただし §6 の通り一括・DCA との差は統計的有意でない（CIが重なる）**ので『VAが最良』と断定はできない。"
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
        "- **全期間で見れば**、大DDを避けつつ利益を残す観測上の最良は "
        "**バリューアベレージング（毎月・下落で多めに買う）を売らずに持ち切る**、次点で押し目買い＋hold。"
        "トレーリングストップ等の『損失限定』は全期間では裏目に出た。"
        "**ただし §6 の通りこれは統計的有意でなく、前半だけ見れば逆に利確/exit が勝っていた**ので、"
        "『買い方が効き売りが効かない』と一般化はできない（局面依存）。"
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

    # --- F. statistical validation (if validate.py has been run) -------------
    vpath = OUT / "validation.json"
    if vpath.exists():
        _validation_section(L, json.load(open(vpath)))

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
