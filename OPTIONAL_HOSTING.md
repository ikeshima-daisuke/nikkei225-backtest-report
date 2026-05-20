# OPTIONAL: Web公開したい場合の手順

このリポジトリは **private のまま** で、レポートも基本 `REPORT.md` を GitHub アプリ／モバイルブラウザで閲覧する想定です。

もし、後日「Web 上で URL 共有したい」「Plotly のインタラクティブ版を `report.html` で見せたい」という用途が出てきた場合の選択肢を、影響範囲とあわせて並べておきます。**現状ではどれも有効化していません。**

---

## 選択肢 A: このリポジトリを public 化して GitHub Pages を有効化（最短）

**メリット**: 1ステップで公開URLが生える。
**デメリット / 影響**:
- リポジトリ全体が公開される
- 注意点として、`README.md` に **ntfy トピック名** が記載されている。public 化前に **トピックを別の文字列にローテートしてから** 公開しないと、誰でも通知を購読・偽投稿できる
- 同梱の通知用 GitHub Secrets（`NTFY_TOPIC`）自体は public 化しても露出しない（Secrets はリポ可視性と独立）

**手順**:

```bash
# 1) 新しいランダムトピックに切り替え
NEW_TOPIC="nikkei-alert-$(python -c 'import secrets;print(secrets.token_hex(6))')"
echo "$NEW_TOPIC" | gh secret set NTFY_TOPIC --repo ikeshima-daisuke/nikkei225-drawdown-notifier
# README.md の旧トピック名を $NEW_TOPIC に書き換えてコミット → push
# スマホ ntfy アプリで新トピックを subscribe しなおす

# 2) Public 化
gh repo edit ikeshima-daisuke/nikkei225-drawdown-notifier --visibility=public --accept-visibility-change-consequences

# 3) Pages 有効化（main branch の / を公開）
gh api -X POST repos/ikeshima-daisuke/nikkei225-drawdown-notifier/pages \
  -f source[branch]=main -f source[path]=/

# 数十秒〜数分待つと:
# https://ikeshima-daisuke.github.io/nikkei225-drawdown-notifier/backtest/report.html
```

---

## 選択肢 B: バックテスト成果物だけ別の public リポへ複製してホスティング

**メリット**: 本リポは private のまま、レポートだけ公開できる。
**デメリット**: リポジトリが2つに増える。更新時は2リポ同期が必要。

**手順**:

```bash
mkdir -p /tmp/nikkei225-backtest-report
cp -r backtest/ /tmp/nikkei225-backtest-report/
cd /tmp/nikkei225-backtest-report
cp report.html index.html   # Pages のデフォルトURLで表示させたい場合
git init -b main && git add -A && git commit -m "publish report"
gh repo create nikkei225-backtest-report --public --source=. --push
gh api -X POST repos/ikeshima-daisuke/nikkei225-backtest-report/pages \
  -f source[branch]=main -f source[path]=/

# URL: https://ikeshima-daisuke.github.io/nikkei225-backtest-report/
```

> ⚠️ 過去（2026-05-20）にこの方針で一度公開→即 private 化済みの履歴あり。public プランがあるアカウントなので Pages 自体は使えるが、確認なしで公開はしない方針で運用中。

---

## 選択肢 C: GitHub Pro/Enterprise プランで private リポのまま Pages を有効化

**メリット**: 本リポ private + Pages 公開 を両立できる（ただし Pages 自体はインターネットに公開される）。
**デメリット**: GitHub の有料プランが必要（個人 Pro: 月$4 など）。

```bash
# Pro/Enterprise 加入後:
gh api -X POST repos/ikeshima-daisuke/nikkei225-drawdown-notifier/pages \
  -f source[branch]=main -f source[path]=/
```

無料プランの private リポでは `Your current plan does not support GitHub Pages` でリジェクトされます（2026-05-20 確認済み）。

---

## 選択肢 D: 外部ホスト (Cloudflare Pages / Vercel / Netlify)

`report.html` を静的ファイルとしてアップロードできる無料サービスで公開する方式。GitHub と分離できる反面、アカウント連携・デプロイ設定の手間が発生します。深くは触れません。

---

## 選択肢 E: ローカル閲覧 (今の運用)

何もせず、`REPORT.md` を GitHub モバイルアプリで開く。Plotly インタラクティブ版が必要な時だけ `report.html` をローカルブラウザで開く。

```bash
# Windows
start backtest/report.html
# macOS
open backtest/report.html
```

---

## 推奨

今のように **ある程度個人/家族向けの数値出力** であれば、選択肢 E（ローカル + GitHub Markdown）で十分です。
将来 X やブログで議論したい等の理由が発生した場合のみ B を選び、その際は本ファイルにある手順を実行してください。
