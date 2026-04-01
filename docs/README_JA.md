<p align="center">
  <img src="../image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>アイデアを話す。論文を手に入れる。完全自動 & 自己進化。</b></h2>



<p align="center">
  <b><i><font size="5"><a href="#openclaw-統合">OpenClaw</a> にチャットするだけ：「Xを研究して」→ 完了。</font></i></b>
</p>

<p align="center">
  <img src="../image/framework_v2.png" width="100%" alt="AutoResearchClaw Framework">
</p>


<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#テスト"><img src="https://img.shields.io/badge/Tests-1823%20passed-brightgreen?logo=pytest&logoColor=white" alt="1823 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-統合"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
  <a href="https://discord.gg/u4ksqW5P"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
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
  <a href="showcase/SHOWCASE.md">🏆 論文ショーケース</a> · <a href="integration-guide.md">📖 統合ガイド</a> · <a href="https://discord.gg/u4ksqW5P">💬 Discordコミュニティ</a>
</p>

---

<table>
<tr>
<td width="18%">
<a href="showcase/SHOWCASE.md"><img src="showcase/thumbnails/paper_I_random_matrix-01.png" width="120" alt="Sample Paper"/></a>
</td>
<td valign="middle">
<b>🏆 生成論文ショーケース</b><br><br>
<b>8つの分野にわたる8本の論文</b> — 数学、統計、生物学、コンピューティング、NLP、RL、ビジョン、ロバスト性 — 人間の介入なしに完全自律生成。<br><br>
<a href="showcase/SHOWCASE.md"><img src="https://img.shields.io/badge/View_Full_Showcase_→-All_8_Papers-d73a49?style=for-the-badge" alt="View Showcase"></a>
</td>
</tr>
</table>

---

> **🧪 テスターを募集しています！** あなた自身の研究アイデアで — どの分野からでも — パイプラインをお試しください。[ご意見をお聞かせください](TESTER_GUIDE.md)。あなたのフィードバックが次のバージョンに直接反映されます。 **[→ Testing Guide](TESTER_GUIDE.md)** | **[→ 中文测试指南](TESTER_GUIDE_CN.md)** | **[→ 日本語テストガイド](TESTER_GUIDE_JA.md)**

---

## 🔥 News
- **[03/22/2026]** [v0.3.2](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.2) — **クロスプラットフォーム対応 + 安定性大幅向上** — ACP互換AIエージェントバックエンド（Claude Code、Codex CLI、Copilot CLI、Gemini CLI、Kimi CLI）に対応し、OpenClawブリッジ経由でメッセージングプラットフォーム（Discord、Telegram、Lark、WeChat）もサポート。新しいCLIエージェントコード生成バックエンドにより、ステージ10と13を外部CLIエージェントに委任し、予算制御とタイムアウト管理に対応。反データ捏造システム（VerifiedRegistry + 実験診断・修復ループ）、100件以上のバグ修正、モジュラーexecutorリファクタリング、`--resume`自動検出、LLMリトライ強化、コミュニティ報告の修正を含む。
- **[03/18/2026]** [v0.3.1](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.1) — **OpenCode Beast Mode + Community Contributions** — New "Beast Mode" routes complex code generation to [OpenCode](https://github.com/anomalyco/opencode) with automatic complexity scoring and graceful fallback. Added Novita AI provider support, thread-safety hardening, improved LLM output parsing robustness, and 20+ bug fixes from community PRs and internal audit.
- **[03/17/2026]** [v0.3.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.0) — **MetaClaw Integration** — AutoResearchClaw now supports [MetaClaw](https://github.com/aiming-lab/MetaClaw) cross-run learning: pipeline failures → structured lessons → reusable skills, injected into all 23 stages. **+18.3%** robustness in controlled experiments. Opt-in (`metaclaw_bridge.enabled: true`), fully backward-compatible. See [Integration Guide](#-metaclaw-integration).
- **[03/16/2026]** [v0.2.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.2.0) — Three multi-agent subsystems (CodeAgent, BenchmarkAgent, FigureAgent), hardened Docker sandbox with network-policy-aware execution, 4-round paper quality audit (AI-slop detection, 7-dim review scoring, NeurIPS checklist), and 15+ bug fixes from production runs.
- **[03/15/2026]** [v0.1.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.1.0) — We release AutoResearchClaw: a fully autonomous 23-stage research pipeline that turns a single research idea into a conference-ready paper. No human intervention required.

---

## ⚡ ワンコマンド。ワンペーパー。

```bash
pip install -e . && researchclaw setup && researchclaw init && researchclaw run --topic "Your research idea here" --auto-approve
```


---

## 🤔 これは何？

**あなたが考える。AutoResearchClawが書く。**

研究トピックを入力するだけで — OpenAlex、Semantic Scholar、arXivからの実際の文献、ハードウェア対応のサンドボックス実験（GPU/MPS/CPUを自動検出）、統計分析、マルチエージェント査読、NeurIPS/ICML/ICLR対応の学会グレードLaTeXを含む完全な学術論文が得られます。監視不要。コピペ不要。幻覚された参考文献なし。

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>完全な学術論文（序論、関連研究、手法、実験、結果、結論）</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>学会対応LaTeX（NeurIPS / ICLR / ICMLテンプレート）</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>OpenAlex、Semantic Scholar、arXivからの実際のBibTeX参考文献 — 本文中の引用に合わせて自動整理</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4層の引用整合性 + 関連性検証（arXiv、CrossRef、DataCite、LLM）</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>生成されたコード + サンドボックス実行結果 + 構造化JSONメトリクス</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>誤差棒と信頼区間付きの条件比較チャートを自動生成</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>手法-証拠の一貫性チェック付きマルチエージェント査読</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>各実行から抽出された自己学習の教訓</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>すべての最終成果物を1フォルダに集約 — Overleafですぐにコンパイル可能</td></tr>
</table>

パイプラインは**人手の介入なしにエンドツーエンドで実行**されます。実験が失敗すれば自己修復します。仮説が成り立たなければ方向転換します。引用が偽物なら削除します。

🌍 **どこでも実行可能。** AutoResearchClaw は特定のプラットフォームに縛られません。CLI でスタンドアロン実行、[OpenClaw](https://github.com/openclaw/openclaw) に接続、または ACP 互換の AI エージェント —— 🤖 Claude Code、💻 Codex CLI、🐙 Copilot CLI、♊ Gemini CLI、🌙 Kimi CLI など —— と連携できます。さらに OpenClaw のメッセージブリッジにより、💬 Discord、✈️ Telegram、🐦 Lark（飛書）、💚 WeChat など、チームが普段使っているプラットフォームから研究を開始できます。トピックを入力すれば、論文が出力されます —— どこからでも。

---

## 🚀 クイックスタート

```bash
# 1. クローン & インストール
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. セットアップ（対話式 — OpenCode Beast Modeのインストール、Docker/LaTeXの確認）
researchclaw setup

# 3. 設定
researchclaw init          # 対話式：LLMプロバイダーを選択、config.arc.yamlを作成
# または手動：cp config.researchclaw.example.yaml config.arc.yaml

# 4. 実行
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

出力先 → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — コンパイル可能なLaTeX、BibTeX、実験コード、チャート。

<details>
<summary>📝 最小限の必要設定</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models: ["gpt-4o-mini"]

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

---

## 🧠 他と何が違うのか

| 機能 | 仕組み |
|------|--------|
| **🔄 PIVOT / REFINE ループ** | ステージ15が自律的に判定：PROCEED、REFINE（パラメータ調整）、またはPIVOT（新方向）。成果物は自動バージョン管理。 |
| **🤖 マルチエージェント討論** | 仮説生成、結果分析、査読のそれぞれで構造化された多視点討論を実施。 |
| **🧬 自己学習** | 各実行から教訓を抽出（判定根拠、ランタイム警告、メトリクス異常）、30日の時間減衰付き。将来の実行が過去のミスから学習。 |
| **📚 知識ベース** | 各実行で6カテゴリ（判定、実験、発見、文献、質問、レビュー）にわたる構造化知識ベースを構築。 |
| **🛡️ Sentinel Watchdog** | バックグラウンド品質モニター：NaN/Inf検出、論文-証拠の一貫性、引用関連性スコアリング、捏造防止ガード。 |

---

## 🦞 OpenClaw統合

<table>
<tr>

**AutoResearchClawは[OpenClaw](https://github.com/openclaw/openclaw)互換サービスです。** OpenClawにインストールして、メッセージ1つで自律研究を開始できます — CLI、Claude Code、その他のAIコーディングアシスタントを使ってスタンドアロンでも利用可能です。

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

### ACP (Agent Client Protocol)

AutoResearchClawは**任意のACP互換コーディングエージェント**をLLMバックエンドとして使用できます — APIキーは不要です。エージェントは[acpx](https://github.com/openclaw/acpx)を介して通信し、全23パイプラインステージにわたって単一の永続セッションを維持します。

| エージェント | コマンド | 備考 |
|-------------|---------|------|
| Claude Code | `claude` | Anthropic |
| Codex CLI | `codex` | OpenAI |
| Copilot CLI | `gh` | GitHub |
| Gemini CLI | `gemini` | Google |
| OpenCode | `opencode` | SST |
| Kimi CLI | `kimi` | Moonshot |

```yaml
# config.yaml — ACP例
llm:
  provider: "acp"
  acp:
    agent: "claude"   # 任意のACP互換エージェントCLIコマンド
    cwd: "."          # エージェントの作業ディレクトリ
  # base_urlやapi_keyは不要 — エージェントが独自の認証を処理します。
```

```bash
# そのまま実行 — エージェントは独自の認証情報を使用
researchclaw run --config config.yaml --topic "Your research idea" --auto-approve
```

### 🛠️ その他の実行方法

| 方法 | 手順 |
|------|------|
| **スタンドアロンCLI** | `researchclaw setup` → `researchclaw init` → `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | `RESEARCHCLAW_CLAUDE.md`を読み取り — *「Run research on [トピック]」*と言うだけ |
| **Copilot CLI** | `researchclaw run --topic "..."` で `llm.acp.agent: "gh"` を使用 |
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
|---------|----------|
| **A: スコーピング** | LLMがトピックを研究質問を含む構造化された問題ツリーに分解 |
| **A+: ハードウェア** | GPU（NVIDIA CUDA / Apple MPS / CPUのみ）を自動検出、ローカルハードウェアが限定的な場合は警告、コード生成を適応 |
| **B: 文献** | マルチソース検索（OpenAlex → Semantic Scholar → arXiv）で実際の論文を取得、関連性でスクリーニング、知識カードを抽出 |
| **C: 統合** | 発見事項をクラスタリング、研究ギャップを特定、マルチエージェント討論で検証可能な仮説を生成 |
| **D: 設計** | 実験計画を設計、ハードウェア対応の実行可能Python（GPUティア→パッケージ選択）を生成、リソース需要を推定 |
| **E: 実行** | サンドボックスで実験を実行、NaN/Infとランタイムバグを検出、LLMによる的確な修復で自己修復 |
| **F: 分析** | マルチエージェントによる結果分析；根拠付きの自律的PROCEED / REFINE / PIVOT判定 |
| **G: 執筆** | アウトライン → セクション別ドラフト（5,000〜6,500語）→ 査読（手法-証拠の一貫性付き）→ 文字数ガード付き改訂 |
| **H: 最終処理** | 品質ゲート、知識アーカイブ、学会テンプレート付きLaTeXエクスポート、引用の整合性 + 関連性検証 |

</details>

---

## ✨ 主な機能

| 機能 | 説明 |
|------|------|
| **📚 マルチソース文献** | OpenAlex、Semantic Scholar、arXivからの実際の論文 — クエリ拡張、重複排除、三状態サーキットブレーカーとグレースフルデグラデーション |
| **🔍 4層引用検証** | arXiv IDチェック → CrossRef/DataCite DOI → Semantic Scholarタイトルマッチ → LLM関連性スコアリング。幻覚された参考文献は自動削除。 |
| **🖥️ ハードウェア対応実行** | GPU（NVIDIA CUDA / Apple MPS / CPUのみ）を自動検出し、コード生成、インポート、実験スケールを適応 |
| **🦾 OpenCode Beast Mode** | 複雑な実験を自動的に[OpenCode](https://github.com/anomalyco/opencode)にルーティング — カスタムアーキテクチャ、トレーニングループ、アブレーション研究を含むマルチファイルプロジェクトを生成。`researchclaw setup`でインストール。 |
| **🧪 サンドボックス実験** | AST検証済みコード、不変ハーネス、NaN/Inf早期停止、自己修復、反復的改良（最大10ラウンド）、部分結果の保持 |
| **📝 学会グレード執筆** | NeurIPS/ICML/ICLRテンプレート、セクション別ドラフト（5,000〜6,500語）、捏造防止ガード、改訂文字数ガード、免責事項抑制 |
| **📐 テンプレート切り替え** | `neurips_2025`、`iclr_2026`、`icml_2026` — Markdown → LaTeX（数式、表、図、相互参照、`\cite{}`対応） |
| **🚦 品質ゲート** | 3つのHuman-in-the-loopゲート（ステージ5, 9, 20）、ロールバック対応。`--auto-approve`でスキップ。 |

---

## 🧠 MetaClaw統合

**AutoResearchClaw + [MetaClaw](https://github.com/aiming-lab/MetaClaw) = すべての実行から学習するパイプライン。**

MetaClawはAutoResearchClawに**クロスラン知識転移**を追加します。有効にすると、パイプラインは失敗や警告から自動的に教訓を抽出し、再利用可能なスキルに変換し、後続の実行で全23ステージに注入します — 同じ過ちを二度と繰り返しません。

### 仕組み

```
Run N executes → failures/warnings captured as Lessons
                      ↓
          MetaClaw Lesson → Skill conversion
                      ↓
          arc-* Skill files stored in ~/.metaclaw/skills/
                      ↓
Run N+1 → build_overlay() injects skills into every LLM prompt
                      ↓
          LLM avoids known pitfalls → higher quality, fewer retries
```

### クイックセットアップ

```bash
# 1. MetaClawをインストール（未インストールの場合）
pip install metaclaw

# 2. 設定で有効化
```

```yaml
# config.arc.yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"        # MetaClawプロキシ（オプション）
  skills_dir: "~/.metaclaw/skills"          # スキルの保存場所
  fallback_url: "https://api.openai.com/v1" # 直接LLMフォールバック
  fallback_api_key: ""                      # フォールバックURLのAPIキー
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # warning + errorを変換
    max_skills_per_run: 3
```

```bash
# 3. 通常通り実行 — MetaClawは透過的に動作
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve
```

各実行後、`~/.metaclaw/skills/arc-*/SKILL.md`を確認して、パイプラインが学習したスキルを確認できます。

### 実験結果

対照A/B実験（同じトピック、同じLLM、同じ設定）：

| メトリクス | ベースライン | MetaClaw使用時 | 改善 |
|-----------|------------|---------------|------|
| ステージリトライ率 | 10.5% | 7.9% | **-24.8%** |
| Refineサイクル数 | 2.0 | 1.2 | **-40.0%** |
| パイプラインステージ完了率 | 18/19 | 19/19 | **+5.3%** |
| 総合ロバスト性スコア（複合） | 0.714 | 0.845 | **+18.3%** |

> 複合ロバスト性スコアは、ステージ完了率（40%）、リトライ削減（30%）、Refineサイクル効率（30%）の加重平均です。

### 後方互換性

- **デフォルト: オフ。** `metaclaw_bridge`が存在しないか`enabled: false`の場合、パイプラインは以前と全く同じように動作します。
- **新しい依存関係なし。** MetaClawはオプションです — コアパイプラインはMetaClawなしで動作します。
- **既存の1,823テストすべてがパス**（統合コードを含む）。

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
  provider: "openai-compatible"    # openai | openrouter | deepseek | minimax | acp | openai-compatible
  base_url: "https://..."          # APIエンドポイント（openai-compatible必須）
  api_key_env: "OPENAI_API_KEY"    # APIキーの環境変数（openai-compatible必須）
  api_key: ""                      # またはここにキーを直接記入
  primary_model: "gpt-4o"          # プライマリモデル
  fallback_models: ["gpt-4o-mini"] # フォールバックチェーン
  s2_api_key: ""                   # Semantic Scholar APIキー（オプション、レート制限緩和）
  acp:                             # provider: "acp" の場合のみ使用
    agent: "claude"                # ACP Agent CLIコマンド（claude, codex, gemini等）
    cwd: "."                       # エージェントの作業ディレクトリ

# === 実験 ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 300             # 実行あたりの最大実行時間（デフォルト: 300秒）
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
  opencode:                          # OpenCode Beast Mode（`researchclaw setup`で自動インストール）
    enabled: true                    # マスタースイッチ（デフォルト: true）
    auto: true                       # 確認なしで自動トリガー（デフォルト: true）
    complexity_threshold: 0.2        # 0.0-1.0 — 高い = 複雑な実験のみトリガー
    model: ""                        # モデルのオーバーライド（空 = llm.primary_modelを使用）
    timeout_sec: 600                 # OpenCode生成の最大秒数
    max_retries: 1                   # 失敗時のリトライ回数
    workspace_cleanup: true          # 収集後に一時ワークスペースを削除

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

# === MetaClaw Bridge（オプション）===
metaclaw_bridge:
  enabled: false                   # trueに設定してクロスラン学習を有効化
  proxy_url: "http://localhost:30000"  # MetaClawプロキシURL
  skills_dir: "~/.metaclaw/skills" # arc-*スキルの保存場所
  fallback_url: ""                 # プロキシがダウン時の直接LLMフォールバック
  fallback_api_key: ""             # フォールバックエンドポイントのAPIキー
  lesson_to_skill:
    enabled: true                  # 教訓をスキルに自動変換
    min_severity: "warning"        # 変換する最小重大度
    max_skills_per_run: 3          # パイプラン実行あたりの最大新規スキル数

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

---

## 📌 引用

AutoResearchClawが役に立った場合は、以下を引用してください：

```bibtex
@misc{liu2026autoresearchclaw,
  author       = {Liu, Jiaqi and Xia, Peng and Han, Siwei and Qiu, Shi and Zhang, Letian and Chen, Guiming  and Tu, Haoqin and Yang, Xinyu and and Zhou, Jiawei and Zhu, Hongtu and Li, Yun and Zhou, Yuyin and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  title        = {AutoResearchClaw: Fully Autonomous Research from Idea to Paper},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/aiming-lab/AutoResearchClaw},
}
```

<p align="center">
  <sub>Built with 🦞 by the AutoResearchClaw team</sub>
</p>
