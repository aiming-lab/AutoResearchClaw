<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>アイデアを話す。論文を手に入れる。完全自動。</b></h2>



<p align="center">
  <i><a href="#openclaw-integration">OpenClaw</a> にチャットするだけ：「Xを研究して」→ 完了。</i>
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/Tests-1284%20passed-brightgreen?logo=pytest&logoColor=white" alt="1284 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-integration"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
</p>

<p align="center">
  <a href="../README.md">🇺🇸 English</a> ·
  <a href="README_CN.md">🇨🇳 中文</a> ·
  <a href="README_JA.md">🇯🇵 日本語</a> ·
  <a href="README_KO.md">🇰🇷 한국어</a> ·
  <a href="README_FR.md">🇫🇷 Français</a> ·
  <a href="README_DE.md">🇩🇪 Deutsch</a> ·
  <a href="README_ES.md">🇪🇸 Español</a> ·
  <a href="README_PT.md">🇧🇷 Português</a> ·
  <a href="README_RU.md">🇷🇺 Русский</a> ·
  <a href="README_AR.md">🇸🇦 العربية</a>
</p>

<p align="center">
  <a href="integration-guide.md">📖 統合ガイド</a>
</p>

---

## ⚡ ワンライナー

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```

---

## 🤔 これは何？

アイデアがある。論文が欲しい。**それだけです。**

AutoResearchClawは研究トピックを受け取り、完全な学術論文を自律的に生成します — arXivとSemantic Scholar（マルチソース、レート制限回避のためarXiv優先）からの実際の文献検索、ハードウェア対応のサンドボックス実験（GPU/MPS/CPUを自動検出）、統計分析、査読、学会対応のLaTeX（NeurIPS/ICML/ICLR向け5,000〜6,500語）まで含みます。手動作業も、ツール間のコピペも不要です。

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>完全な学術論文（序論、関連研究、手法、実験、結果、結論）</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>学会対応LaTeX（NeurIPS / ICLR / ICMLテンプレート）</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Semantic ScholarとarXivからの実際のBibTeX参考文献 — 本文中の引用に合わせて自動整理</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4層の引用整合性＋関連性検証（arXiv、CrossRef、DataCite、LLM）</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>生成されたコード＋サンドボックス実行結果＋構造化JSONメトリクス</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>誤差棒と信頼区間付きの条件比較チャートを自動生成</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>手法-証拠の一貫性チェック付きマルチエージェント査読</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>各実行から抽出された自己学習の教訓</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>すべての最終成果物を1フォルダに集約 — Overleafですぐにコンパイル可能</td></tr>
</table>

パイプラインは**人手の介入なしにエンドツーエンドで実行**されます（手動レビュー用のゲートステージを設定しない限り）。実験が失敗すれば自己修復し、仮説が成り立たなければ方向転換します。

### 🎯 試してみよう

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 他と何が違うのか

### 🔄 PIVOT / REFINE 判定ループ

パイプラインは単に直線的に実行されるわけではありません。ステージ15（RESEARCH_DECISION）は実験結果を仮説と照合し、自律的に判定を行います：

- **PROCEED** — 結果が仮説を支持しているため、論文執筆に進む
- **REFINE** — 結果は有望だが改善が必要なため、コード/パラメータの改良にループバック
- **PIVOT** — 根本的な問題が検出されたため、新しい方向で仮説生成からやり直す

各PIVOT/REFINEサイクルでは**以前の成果物がバージョン管理**され（`stage-08_v1/`、`stage-08_v2/`、...）、作業が失われることなく判定の推移を完全に追跡できます。

### 🤖 マルチエージェント討論

重要なステージでは、複数のLLM視点による構造化された討論プロトコルを使用します：

- **仮説生成** — 多様なエージェントがアイデアを提案し、互いに検証
- **結果分析** — 楽観派、懐疑派、実用派が結果を分析
- **査読** — 手法-証拠の一貫性チェック（論文が50回の試行を主張しているが、コードは5回しか実行していないかどうか）

### 🧬 進化：実行間の自己学習

パイプラインの各実行から詳細な教訓を抽出します — 単に「失敗した」だけでなく、*なぜ*失敗したかを記録：

- PIVOT/REFINE判定の根拠
- 実験のstderrからのランタイム警告（例: `RuntimeWarning: division by zero`）
- メトリクスの異常（NaN、Inf、同一の収束速度）

これらの教訓はJSONLストアに保持され、**30日の半減期による時間減衰重み付け**が適用され、将来の実行にプロンプトオーバーレイとして注入されます。パイプラインは文字通り自分のミスから学びます。

### 📚 知識ベース

各実行で6カテゴリの構造化された知識ベース（`docs/kb/`に保存）を構築します：

- **decisions/** — 実験設計、品質ゲート、研究判定、リソース計画、検索戦略、知識アーカイブ
- **experiments/** — コード生成ログ、実験実行、反復的改良
- **findings/** — 引用検証、結果分析、統合レポート
- **literature/** — 知識抽出、文献収集、スクリーニング結果
- **questions/** — 仮説生成、問題分解、トピック初期化
- **reviews/** — エクスポート/出版レポート、論文草稿、アウトライン、改訂、査読

### 🛡️ Sentinel Watchdog

メインパイプラインが見逃す可能性のある問題を検出するバックグラウンド品質モニター：

- **ランタイムバグ検出** — メトリクスのNaN/Inf、stderrの警告をLLMにフィードバックして的確な修復
- **論文-証拠の一貫性** — 実際の実験コード、実行結果、改良ログを査読に注入
- **引用関連性スコアリング** — 存在確認だけでなく、各参考文献のトピック関連性をLLMが評価
- **収束条件の強制** — 固定反復回数の実験を検出し、適切な早期終了を要求
- **アブレーション検証** — 重複/同一のアブレーション条件を検出し、壊れた比較にフラグ
- **捏造防止ガード** — 実験がメトリクスを生成しない場合、論文執筆を完全にブロック

---

## 🦞 OpenClaw統合

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClawは[OpenClaw](https://github.com/openclaw/openclaw)互換サービスです。** OpenClawにインストールして、メッセージ1つで自律研究を開始できます。CLI、Claude Code、その他のAIコーディングアシスタントを使ってスタンドアロンでも利用可能です。

</td>
</tr>
</table>

### 🚀 OpenClawで使う（推奨）

[OpenClaw](https://github.com/openclaw/openclaw)をすでにAIアシスタントとしてお使いの場合：

```
1️⃣  GitHubリポジトリのURLをOpenClawに共有
2️⃣  OpenClawがRESEARCHCLAW_AGENTS.mdを自動読み込み → パイプラインを理解
3️⃣  「Research [あなたのトピック]」と話しかける
4️⃣  完了 — OpenClawがクローン、インストール、設定、実行、結果の返却まですべて自動実行
```

**以上です。** OpenClawが`git clone`、`pip install`、設定、パイプライン実行を自動的に処理します。チャットするだけです。

<details>
<summary>💡 内部で何が起きているか</summary>

1. OpenClawが`RESEARCHCLAW_AGENTS.md`を読み取り → 研究オーケストレーターの役割を学習
2. OpenClawが`README.md`を読み取り → インストールとパイプライン構造を理解
3. OpenClawが`config.researchclaw.example.yaml` → `config.yaml`にコピー
4. LLMのAPIキーを要求（または環境変数を使用）
5. `pip install -e .` + `researchclaw run --topic "..." --auto-approve`を実行
6. 論文、LaTeX、実験、引用を返却

</details>

### 🔌 OpenClaw Bridge（上級）

より深い統合のために、AutoResearchClawには6つのオプション機能を備えた**ブリッジアダプターシステム**が含まれています：

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ スケジュール実行
  use_message: true           # 💬 進捗通知（Discord/Slack/Telegram）
  use_memory: true            # 🧠 セッション間の知識永続化
  use_sessions_spawn: true    # 🔀 並列サブセッションの生成
  use_web_fetch: true         # 🌐 文献レビュー中のライブWeb検索
  use_browser: false          # 🖥️ ブラウザベースの論文収集
```

各フラグは型付きアダプタープロトコルをアクティブにします。OpenClawがこれらの機能を提供する場合、アダプターはコード変更なしにそれらを利用します。詳細は[`integration-guide.md`](integration-guide.md)をご覧ください。

### 🛠️ その他の実行方法

| 方法 | 手順 |
|--------|-----|
| **スタンドアロンCLI** | `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | `RESEARCHCLAW_CLAUDE.md`を読み取り — *「Run research on [トピック]」*と言うだけ |
| **OpenCode** | `.claude/skills/`を読み取り — 同じ自然言語インターフェース |
| **任意のAI CLI** | `RESEARCHCLAW_AGENTS.md`をコンテキストとして提供 → エージェントが自動ブートストラップ |

---

## 🔬 パイプライン：23ステージ、8フェーズ

```
フェーズ A: 研究スコーピング          フェーズ E: 実験実行
  1. TOPIC_INIT                      12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE               13. ITERATIVE_REFINE  ← 自己修復

フェーズ B: 文献探索                フェーズ F: 分析と判定
  3. SEARCH_STRATEGY                 14. RESULT_ANALYSIS    ← マルチエージェント
  4. LITERATURE_COLLECT  ← 実API    15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [ゲート]
  6. KNOWLEDGE_EXTRACT               フェーズ G: 論文執筆
                                     16. PAPER_OUTLINE
フェーズ C: 知識統合                  17. PAPER_DRAFT
  7. SYNTHESIS                       18. PEER_REVIEW        ← 証拠チェック
  8. HYPOTHESIS_GEN    ← 討論        19. PAPER_REVISION

フェーズ D: 実験設計               フェーズ H: 最終処理
  9. EXPERIMENT_DESIGN   [ゲート]     20. QUALITY_GATE      [ゲート]
 10. CODE_GENERATION                 21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING               22. EXPORT_PUBLISH     ← LaTeX
                                     23. CITATION_VERIFY    ← 関連性チェック
```

> **ゲートステージ**（5, 9, 20）は人間の承認を待つか、`--auto-approve`で自動承認されます。却下時にはパイプラインがロールバックします。

> **判定ループ**: ステージ15はREFINE（→ ステージ13）またはPIVOT（→ ステージ8）をトリガーでき、成果物のバージョン管理が自動的に行われます。

<details>
<summary>📋 各フェーズの詳細</summary>

| フェーズ | 処理内容 |
|-------|-------------|
| **A: スコーピング** | LLMがトピックを研究質問を含む構造化された問題ツリーに分解 |
| **A+: ハードウェア** | GPU（NVIDIA CUDA / Apple MPS / CPUのみ）を自動検出、ローカルハードウェアが限定的な場合は警告、コード生成を適応 |
| **B: 文献** | マルチソース検索（arXiv優先、次にSemantic Scholar）で実際の論文を取得、関連性でスクリーニング、知識カードを抽出 |
| **C: 統合** | 発見事項をクラスタリング、研究ギャップを特定、マルチエージェント討論で検証可能な仮説を生成 |
| **D: 設計** | 実験計画を設計、ハードウェア対応の実行可能Python（GPUティア→パッケージ選択）を生成、リソース需要を推定 |
| **E: 実行** | サンドボックスで実験を実行、NaN/Infとランタイムバグを検出、LLMによる的確な修復で自己修復 |
| **F: 分析** | マルチエージェントによる結果分析；根拠付きの自律的PROCEED / REFINE / PIVOT判定 |
| **G: 執筆** | アウトライン → セクション別ドラフト（5,000〜6,500語）→ 査読（手法-証拠の一貫性付き）→ 文字数ガード付き改訂 |
| **H: 最終処理** | 品質ゲート、知識アーカイブ、学会テンプレート付きLaTeXエクスポート、引用の整合性＋関連性検証 |

</details>

---

## 🚀 クイックスタート

### 前提条件

- 🐍 Python 3.11+
- 🔑 OpenAI互換のLLM APIエンドポイント（GPT-4o、GPT-5.x、またはその他の互換プロバイダー）

### インストール

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 設定

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 最小限の必要設定</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"     # 任意のOpenAI互換エンドポイント
  api_key_env: "OPENAI_API_KEY"              # APIキーを含む環境変数名
  primary_model: "gpt-4o"                    # エンドポイントがサポートする任意のモデル
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                             # オプション: レート制限緩和のためのSemantic Scholar APIキー

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

### 実行

```bash
# APIキーを設定
export OPENAI_API_KEY="sk-..."

# 🚀 フルパイプラインを実行
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 トピックをインラインで指定
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ 設定を検証
researchclaw validate --config config.arc.yaml

# ⏩ 特定のステージから再開
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

出力先 → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/`（ステージごとに1つのサブディレクトリ）。

ユーザー向けの全成果物は自動的に1つの**`deliverables/`**フォルダに集約されます：

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # 最終論文（Markdown）
├── paper.tex                  # 学会対応LaTeX
├── references.bib             # 検証済みBibTeX参考文献（自動整理済み）
├── neurips_2025.sty           # 学会スタイルファイル（自動選択）
├── code/                      # 実験コード + requirements.txt
├── verification_report.json   # 引用整合性レポート
├── charts/                    # 結果の可視化（条件比較、誤差棒）
└── manifest.json              # メタデータ付き成果物インデックス
```

`deliverables/`フォルダは**コンパイル可能な状態**です — 学会の`.sty`と`.bst`ファイルが含まれているため、`pdflatex` + `bibtex`で`paper.tex`を直接コンパイルするか、追加ダウンロードなしでOverleafにアップロードできます。

---

## ✨ 主な機能

### 📚 マルチソース文献検索

ステージ4は**実際の学術API**に問い合わせます — LLMが幻覚した論文ではありません。Semantic Scholarのレート制限を回避するため、**arXiv優先**戦略を採用。

- **arXiv API**（プライマリ） — 実際のarXiv IDとメタデータを持つプレプリント、レート制限なし
- **Semantic Scholar API**（セカンダリ） — タイトル、アブストラクト、掲載誌、被引用数、DOIを持つ実際の論文
- **クエリ拡張** — 包括的なカバレッジ（30〜60件の参考文献）のため、より広いクエリ（サーベイ、ベンチマーク、比較バリアント）を自動生成
- **自動重複排除** — DOI → arXiv ID → ファジータイトルマッチング
- **BibTeX生成** — 実際のメタデータを含む有効な`@article{cite_key, ...}`エントリ
- **三状態サーキットブレーカー** — CLOSED → OPEN → HALF_OPEN の指数バックオフ冷却期間による回復（永久無効化なし）
- **グレースフルデグラデーション** — S2の障害がarXivの結果をブロックしない；すべてのAPIが失敗した場合はLLM拡張結果にフォールバック

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 引用検証（ステージ23）

論文執筆後、ステージ23がすべての参考文献の**整合性と関連性の両方をファクトチェック**します：

| レイヤー | 方法 | チェック内容 |
|-------|--------|----------------|
| L1 | arXiv API `id_list` | arXiv IDを持つ論文 — そのIDが実際に存在するか検証 |
| L2 | CrossRef `/works/{doi}` + DataCiteフォールバック | DOIを持つ論文 — DOIが解決でき、タイトルが一致するか検証（DataCiteがarXivの`10.48550` DOIに対応） |
| L3 | Semantic Scholar + arXivタイトル検索 | その他すべて — ファジータイトルマッチング（類似度≥0.80） |
| L4 | LLM関連性スコアリング | 検証済み全参考文献 — 研究に対するトピック関連性を評価 |

各参考文献 → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**自動クリーンアップ**: 幻覚された引用は論文テキストから静かに削除されます（`[HALLUCINATED]`タグなし）。引用されていない参考文献エントリは整理されます。最終的な`references.bib`には検証済みの引用された参考文献のみが含まれます。

### 🖥️ ハードウェア対応実行

ステージ1がローカルのGPU性能を自動検出し、パイプライン全体を適応させます：

| ティア | 検出方法 | 動作 |
|------|-----------|----------|
| **High** | 8GB以上のVRAMを持つNVIDIA GPU | フルPyTorch/GPUコード生成、torchが未インストールの場合は自動インストール |
| **Limited** | 8GB未満のNVIDIAまたはApple MPS | 軽量実験（パラメータ100万未満、20エポック以下）、ユーザーへ警告 |
| **CPU-only** | GPUが検出されない | NumPy/sklearnのみ、torchインポートなし、リモートGPU推奨の警告 |

ハードウェアプロファイルは`stage-01/hardware_profile.json`に保存され、コード生成、サンドボックスのインポート、プロンプトの制約に影響します。

### 🧪 サンドボックス実験実行

- **コード検証** — AST解析、インポートホワイトリスト、サンドボックス外のファイルI/O禁止
- **計算バジェットガード** — コード生成プロンプトに時間予算（設定可能、デフォルト600秒）を注入；LLMはサンドボックスのタイムアウト内に収まる実験を設計する必要あり
- **実験ハーネス** — `should_stop()`タイムガード、`report_metric()`のNaN/Inf拒否、`finalize()`の結果書き出しを含む不変の`experiment_harness.py`をサンドボックスに注入（karpathy/autoresearchの不変evalパターンに着想）
- **構造化出力** — 実験は型付きメトリクスを含む`results.json`を生成（stdout解析だけではない）
- **スマートメトリクス解析** — キーワード検出（`is_metric_name()`）を使用してログ行をメトリクスからフィルタリング
- **NaN/発散早期停止** — メトリクスからNaN/Inf値をフィルタリング；発散するloss（>100）を検出しフラグ
- **収束条件の強制** — 生成コードには早期終了条件を含める必要あり（固定反復回数ではない）
- **ランタイムバグ検出** — NaN/Infメトリクスとstderr警告（ゼロ除算、オーバーフロー）を自動検出
- **自己修復** — ランタイムの問題をLLMにフィードバックし、根本原因の修正を実施（応急措置のtry/exceptではない）
- **反復的改良** — ステージ13が結果を分析し、改良されたコード/パラメータで再実行（最大10回の反復、タイムアウト対応プロンプト付き）
- **部分結果の保持** — タイムアウトしたがメトリクスが取得できた実験は「failed」ではなく「partial」ステータスとなり、利用可能なデータを保持
- **トピック-実験の整合性** — LLMベースの生成後チェックにより、実験コードが指定された研究トピックを実際にテストしていることを確認

### 📝 学会グレードの論文執筆

執筆パイプラインはNeurIPS/ICML/ICLR基準（9ページ以上、5,000〜6,500語）を目標としています：

- **データ整合性の強制** — 実験がメトリクスを生成しない場合、論文執筆を完全にブロック（LLMによる結果の捏造を防止）；ドラフトと改訂の両方のプロンプトに捏造防止指示を注入
- **学会グレードのプロンプト** — 採択論文の分析から得た主要原則をシステムプロンプトに含む：新規性、ナラティブ、強力なベースライン、アブレーション、誠実さ、再現性；一般的な却下理由にフラグ
- **タイトルとフレーミングのガイドライン** — 新規性のシグナリング、ミーム性テスト、5文アブストラクト構造、汎用的なタイトルの検出と再生成
- **セクション別ドラフト** — 3回の逐次LLM呼び出し（序論+関連研究 → 手法+実験 → 結果+結論）で出力の切り詰めを回避
- **セクション別目標語数** — Abstract（150-250）、Introduction（800-1000）、Related Work（600-800）、Method（1000-1500）、Experiments（800-1200）、Results（600-800）、Discussion（400-600）
- **改訂時の文字数ガード** — 改訂版がドラフトより短い場合、より強い強制で自動リトライ；必要に応じてドラフト+注釈にフォールバック
- **免責事項の抑制** — 「計算リソースの制約により」を最大1回に制限；改訂プロンプトで繰り返しのヘッジを積極的に削除
- **統計的厳密性** — 結果テーブルに信頼区間、p値、効果量を必須化；壊れたアブレーションにフラグを立て、主張から除外
- **学会ルーブリック付き査読** — NeurIPS/ICMLルーブリック（新規性、ベースライン、アブレーション、主張vs証拠、限界）に従い1〜10で採点

### 📐 学会テンプレートの切り替え

```yaml
export:
  target_conference: "neurips_2025"   # または "iclr_2026" または "icml_2026"
```

| 学会 | スタイルパッケージ | カラム数 |
|------------|--------------|---------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

Markdown → LaTeXコンバーターが処理する内容：セクション見出し（自動番号振り重複排除付き）、インライン/ディスプレイ数式、太字/斜体、リスト、テーブル（`\caption`/`\label`付き）、図（`\includegraphics`）、コードブロック（Unicode安全）、相互参照、`\cite{}`参照。

### 🚦 品質ゲート

| ゲート | ステージ | 却下時のロールバック先 |
|------|-------|---------------------------|
| 文献スクリーニング | 5 | 文献の再収集（ステージ4） |
| 実験設計 | 9 | 仮説の再生成（ステージ8） |
| 品質ゲート | 20 | アウトラインからの論文再執筆（ステージ16） |

`--auto-approve`ですべてのゲートをスキップするか、`security.hitl_required_stages`で特定のステージを設定できます。

---

## ⚙️ 設定リファレンス

<details>
<summary>クリックして設定リファレンスの全体を展開</summary>

```yaml
# === プロジェクト ===
project:
  name: "my-research"              # プロジェクト識別子
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === 研究 ===
research:
  topic: "..."                     # 研究トピック（必須）
  domains: ["ml", "nlp"]           # 文献検索の研究ドメイン
  daily_paper_count: 8             # 検索クエリあたりの目標論文数
  quality_threshold: 4.0           # 論文の最小品質スコア

# === ランタイム ===
runtime:
  timezone: "America/New_York"     # タイムスタンプ用
  max_parallel_tasks: 3            # 同時実験数の上限
  approval_timeout_hours: 12       # ゲートステージのタイムアウト
  retry_limit: 2                   # ステージ失敗時のリトライ回数

# === LLM ===
llm:
  provider: "openai-compatible"    # プロバイダータイプ
  base_url: "https://..."          # APIエンドポイント（必須）
  api_key_env: "OPENAI_API_KEY"    # APIキーの環境変数（必須）
  api_key: ""                      # またはここにキーを直接記入
  primary_model: "gpt-4o"          # プライマリモデル
  fallback_models: ["gpt-4o-mini"] # フォールバックチェーン
  s2_api_key: ""                   # Semantic Scholar APIキー（オプション、レート制限緩和）

# === 実験 ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # 実行あたりの最大実行時間（デフォルト: 600秒）
  max_iterations: 10               # 最大最適化反復回数
  metric_key: "val_loss"           # プライマリメトリクス名
  metric_direction: "minimize"     # minimize | maximize
  sandbox:
    python_path: ".venv/bin/python"
    gpu_required: false
    allowed_imports: [math, random, json, csv, numpy, torch, sklearn]
    max_memory_mb: 4096
  docker:
    image: "researchclaw/experiment:latest"
    network_policy: "setup_only"   # none | setup_only | pip_only | full
    gpu_enabled: true
    memory_limit_mb: 8192
    auto_install_deps: true        # importを自動検出 → requirements.txt
  ssh_remote:
    host: ""                       # GPUサーバーのホスト名
    gpu_ids: []                    # 利用可能なGPU ID
    remote_workdir: "/tmp/researchclaw_experiments"

# === エクスポート ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === プロンプト ===
prompts:
  custom_file: ""                  # カスタムプロンプトYAMLのパス（空 = デフォルト）

# === セキュリティ ===
security:
  hitl_required_stages: [5, 9, 20] # 人間の承認が必要なステージ
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === 知識ベース ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === 通知 ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === OpenClaw Bridge ===
openclaw_bridge:
  use_cron: false                  # スケジュール研究実行
  use_message: false               # 進捗通知
  use_memory: false                # セッション間の知識永続化
  use_sessions_spawn: false        # 並列サブセッションの生成
  use_web_fetch: false             # ライブWeb検索
  use_browser: false               # ブラウザベースの論文収集
```

</details>

---

## 🙏 謝辞

以下のプロジェクトに着想を得ています：

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — 自動研究のパイオニア
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — エンドツーエンドの研究自動化
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — 完全自動研究システム

---

## 📄 ライセンス

MIT — 詳細は[LICENSE](../LICENSE)をご覧ください。

<p align="center">
  <sub>Built with 🦞 by the AutoResearchClaw team</sub>
</p>
