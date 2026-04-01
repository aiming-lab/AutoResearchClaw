<p align="center">
  <img src="../image/logo.png" width="500" alt="AutoResearchClaw Logo">
</p>

<h2 align="center">🧪 コミュニティテストガイド</h2>

<p align="center">
  <b>世界初の完全自律型研究パイプラインを、あらゆる分野でストレステストするためにご協力ください。</b>
</p>

<p align="center">
  <a href="https://github.com/aiming-lab/AutoResearchClaw">⭐ リポジトリにスターを付ける</a> ·
  <a href="#-クイックスタート">🚀 クイックスタート</a> ·
  <a href="#-フィードバックテンプレート">📋 フィードバックテンプレート</a> ·
  <a href="TESTER_GUIDE.md">🇺🇸 English Testing Guide</a> ·
  <a href="TESTER_GUIDE_CN.md">🇨🇳 中文测试指南</a>
</p>

---

## 👋 テスターの皆さんへ

**AutoResearchClaw** は、完全自律型の学術論文生成パイプラインです。研究アイデアを入力するだけで、文献検索、実験設計、コード生成、実験実行、論文執筆、査読、最終成果物の作成まで、すべてを自動で処理します。**23ステージ、人手介入ゼロ。**

**あらゆる分野・バックグラウンド**のテスターを募集しています — 機械学習、NLP、コンピュータビジョン、強化学習、バイオインフォマティクス、物理学、社会科学など。テストが多様であるほど、パイプラインの改善に繋がります。

**あなたのミッション：** 自分の研究アイデアでパイプラインを実行し、出力を検査して、詳細なフィードバックレポートを提出してください。それだけです。すべてのフィードバックが次のバージョンに直接反映されます。

---

## 📋 目次

1. [前提条件](#-前提条件)
2. [インストールとセットアップ](#-インストールとセットアップ)
3. [パイプラインの実行](#-パイプラインの実行)
4. [出力の確認](#-出力の確認)
5. [フィードバックレポートの要件](#-フィードバックレポートの要件)
6. [フィードバックテンプレート](#-フィードバックテンプレート)
7. [FAQ](#-faq)

---

## 📦 前提条件

| 項目 | 最小要件 | 推奨 |
|------|---------|------|
| OS | macOS / Linux / WSL2 | Linux (Ubuntu 22.04+) |
| Python | 3.11+ | 3.11 または 3.12 |
| ディスク | 500 MB | 2 GB+ |
| RAM | 8 GB | 16 GB+ |
| GPU | 不要（sandboxモード） | NVIDIA GPU + CUDA 12.x（dockerモード） |
| ネットワーク | 必要（LLM API + 文献検索） | 安定した接続 |
| LLM APIキー | **必須** | OpenAI または Anthropic |

### 🔑 APIキーについて

パイプラインは、執筆、コーディング、レビューなど、すべてのステージで大規模言語モデル（LLM）を呼び出します。**OpenAI** または **Anthropic** のAPIキーが必要です。

> **最良の結果を得るために、利用可能な最も高性能なモデルの使用を強く推奨します：**
>
> | プロバイダー | 推奨モデル | フォールバック |
> |-------------|-----------|--------------|
> | **OpenAI** | **GPT-5.4**（最良） | GPT-5.1 または GPT-4.1 |
> | **Anthropic** | **Claude Opus 4.6**（最良） | Claude Sonnet 4.6 |
>
> トップティアのモデルを使用することで、論文の品質、コードの正確性、実験設計が大幅に向上します。古いモデル（例：GPT-4o）では、出力品質が著しく低下する可能性があります。

---

## 🛠 インストールとセットアップ

### ⚠️ 常に最新バージョンを使用してください

> **このプロジェクトは活発に開発中です。** コードベースは頻繁に更新され、バージョンによって結果が大きく異なる場合があります。
>
> **テスト実行の前に、必ず最新のコードをプルしてください：**
>
> ```bash
> cd AutoResearchClaw
> git pull origin main
> pip install -e .    # 変更を反映するために再インストール
> ```
>
> フィードバックレポート用にバージョンを記録してください：
> ```bash
> git log --oneline -1
> ```

---

### オプションA：Claude Code（最速 — 推奨 ⚡）

[Claude Code](https://claude.ai/claude-code)（AnthropicのCLIツール）をお持ちの場合、以下を貼り付けるだけです：

```
Please clone and install AutoResearchClaw:
https://github.com/aiming-lab/AutoResearchClaw.git

If already cloned, run git pull origin main to update to the latest version first.

Then create a config file with:
- LLM: OpenAI with gpt-5.4 (or Anthropic Claude Opus 4.6)
- Experiment mode: sandbox (local execution)
- Research topic: "<ここに研究アイデアを入力>"
- Auto-approve all gate stages

My API key is: sk-xxxx (set it as an environment variable, don't hardcode it)
```

Claude Codeがクローン、依存関係、設定、実行をすべて自動で処理します。

### オプションB：手動インストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw

# 2. 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows（WSL2推奨）

# 3. インストール
pip install -e .

# 4. 動作確認
researchclaw --help
```

### ⚙️ 設定

```bash
cp config.researchclaw.example.yaml config.yaml
```

`config.yaml` を編集してください — 主要なフィールドは以下の通りです：

```yaml
# === プロジェクト ===
project:
  name: "my-test"
  mode: "full-auto"

# === 研究トピック — アイデアを英語で記述してください ===
research:
  topic: "Your research idea in 1-2 sentences"
  domains:
    - "machine-learning"     # 選択肢: nlp, cv, rl, graph-learning など

# === LLM — 利用可能な最も高性能なモデルを使用してください！ ===
#
# オプション1: OpenAI（GPT-5.4推奨）
llm:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-5.4"              # 最良のモデル
  fallback_models:
    - "gpt-5.1"
    - "gpt-4.1"

# オプション2: Anthropic Claude（Claude Opus 4.6推奨）
# llm:
#   provider: "openai-compatible"
#   base_url: "https://api.anthropic.com/v1"
#   api_key_env: "ANTHROPIC_API_KEY"
#   primary_model: "claude-opus-4-6"
#   fallback_models:
#     - "claude-sonnet-4-6"

# === 実験 ===
experiment:
  mode: "sandbox"                # sandbox = ローカル実行（推奨）
  time_budget_sec: 600           # 実験実行あたりの最大秒数
  max_iterations: 10
  metric_key: "primary_metric"
  metric_direction: "minimize"   # または "maximize"
```

### 🔐 APIキーの設定

```bash
# OpenAIユーザー：
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Anthropicユーザー：
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx"

# オプション：Semantic Scholar APIキー（文献検索を高速化）
export S2_API_KEY="your-s2-key"
```

> **🔒 セキュリティ：** APIキーをファイルにハードコードしないでください。設定ファイルの `api_key_env` を使用して環境変数を参照してください。

---

## 🚀 パイプラインの実行

### クイックスタート

```bash
source .venv/bin/activate
export OPENAI_API_KEY="sk-xxxx"       # または ANTHROPIC_API_KEY

researchclaw run --config config.yaml --auto-approve
```

### 特定のトピックを指定する場合

```bash
researchclaw run \
  --config config.yaml \
  --topic "Investigating the effect of curriculum learning on image classification with adaptive difficulty scheduling" \
  --auto-approve
```

### ⏱ 想定実行時間

| モード | 推定時間 | 備考 |
|--------|---------|------|
| sandbox | 30分 〜 2時間 | 実験の複雑さとAPIの速度に依存 |
| docker (GPU) | 1 〜 4時間 | より大規模なディープラーニング実験向け |

ターミナルにリアルタイムで進捗が表示されます。**手動介入は不要です** — あとは実行完了を待つだけです。

### ✅ 完了の確認方法

以下のような出力が表示されます：

```
[Stage 23/23] ✓ Deliverables packaged
Pipeline complete — deliverables at: artifacts/rc-20260315-XXXXXX-YYYY/deliverables/
```

### 🔄 中断された場合

パイプラインはチェックポイントをサポートしています — 再開するだけです：

```bash
researchclaw run --config config.yaml --resume
```

---

## 🔍 出力の確認

完了後、結果は `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` に格納されます。

### 📂 成果物

| ファイル / ディレクトリ | 説明 |
|------------------------|------|
| `paper_final.md` | Markdown形式の最終論文（5,000〜6,500語） |
| `paper.tex` | 学会投稿可能なLaTeXソース（直接コンパイル可能） |
| `references.bib` | BibTeX参考文献（検証済みの引用） |
| `code/main.py` | 自動生成された実験コード |
| `code/requirements.txt` | 実験用のPython依存関係 |
| `charts/` | 結果の可視化チャート（PNG） |
| `verification_report.json` | 引用整合性の検証レポート |
| `manifest.json` | メタデータ付きの成果物マニフェスト |

### 🔎 確認すべきポイント

1. **論文の内容** (`paper_final.md` または `paper.tex`)
   - タイトルはトピックに関連しているか？
   - アブストラクトは問題、手法、結果を明確に述べているか？
   - 関連研究はその分野の主要な論文を引用しているか？
   - 手法の記述は技術的に正確か？
   - 実験設計は妥当か（データセット、ベースライン、指標）？
   - 結果は有意義か（すべてゼロやNaNではないか）？
   - 結論は実験結果と一貫しているか？

2. **実験コード** (`code/main.py`)
   - 単独で実行できるか？
   - 実際のデータセットを使用しているか（ランダム生成の偽データではないか）？
   - 論文に記述された内容を実装しているか？
   - ハイパーパラメータは妥当か？

3. **チャート** (`charts/`)
   - 読みやすく整理されているか？
   - 軸ラベルは正しいか？
   - データは論文の主張と一致しているか？

4. **参考文献** (`references.bib`)
   - 引用された論文は実在するか？
   - 引用は議論に関連しているか？

### 📊 自動生成品質レポート

パイプラインは `stage-20/quality_report.json` に品質評価を出力します。内容は以下の通りです：

- `score_1_to_10` — 自動品質スコア
- `verdict` — 受理 / 却下の推奨
- `strengths` — 良かった点
- `weaknesses` — 特定された問題点
- `required_actions` — 改善提案

フィードバックでこれを参照し、ご自身の専門的な判断も加えてください。

---

## 📝 フィードバックレポートの要件

**あなたのフィードバックは、このプロジェクトを改善するための最も重要なインプットです。** 徹底的かつ正直に記述してください — 批判的なフィードバックも称賛と同様に価値があります。

### 提出物

| # | 項目 | 詳細 |
|---|------|------|
| F1 | **フィードバックレポート**（以下のテンプレートを使用） | Markdown形式、ファイル名は `feedback_<your-name>.md` |
| F2 | **出力ディレクトリ一式** | `artifacts/rc-XXXXXX/` ディレクトリ全体をZip圧縮 |
| F3 | **設定ファイル** | `config.yaml`（**APIキーを事前に削除してください！**） |
| F4 | **ターミナルログ**（任意だが推奨） | 実行中のターミナル出力のコピー |

### フィードバックの4つの観点

#### 🎯 (a) 品質評価

あなたの専門知識から：

- この論文があなたの分野で発表されたとしたら、どのレベルに達するか？（トップ会議 / 中堅 / ワークショップ / 出版不可）
- 普段読む論文と比較して、文章の質はどうか？
- 手法は技術的に正確か？明らかな誤りはないか？
- 実験設計は妥当か？

#### 💡 (b) 改善提案

- どのステージの出力が最も弱いか？（文献検索 / 実験設計 / コード生成 / 論文執筆）
- 明らかなコードエラーや設計上の問題はないか？
- 論文の構成や執筆の改善に関する具体的な提案は？

#### ⚖️ (c) パイプライン設計の評価

- 23ステージの設計は適切か？冗長または不足しているステップはないか？
- 反復的な実験改善は効果的か？
- 各ステージでのLLMの指示は適切か？

#### 🐛 (d) バグ報告

発見した問題をできるだけ具体的に報告してください：

- **文章のバグ：** 文法エラー、段落の繰り返し、矛盾、存在しない図への参照
- **コードのバグ：** ランタイムエラー、ロジックエラー、データ処理の問題
- **結果のバグ：** すべてゼロの結果、NaN値、不合理な指標
- **パイプラインのバグ：** ステージの停止、予期しないクラッシュ、リソース枯渇

---

## 📋 フィードバックテンプレート

以下のテンプレートをコピーし、記入して `feedback_<your-name>.md` として保存してください：

````markdown
# AutoResearchClaw — テストフィードバックレポート

## 基本情報

- **テスター名：**
- **専門分野：** （例：コンピュータビジョン / NLP / 強化学習 / バイオインフォマティクス / ...）
- **テスト日：**
- **コードバージョン：** （`git log --oneline -1` の出力、例：`44151b1 fix: Phase 3 regression test findings`）
- **研究トピック（英語）：**
- **使用したLLMモデル：** （例：gpt-5.4 / gpt-5.1 / claude-opus-4-6 / claude-sonnet-4-6）
- **実験モード：** （sandbox / docker）
- **合計実行時間：** （約X分）
- **全23ステージ完了？：** はい / いいえ（いいえの場合、どのステージで失敗？）

---

## 1. 品質評価（スコア：1〜10）

**私のスコア：** X / 10

### 1.1 論文全体の品質
- この論文はどのレベルに相当するか？（トップ会議 / 中堅 / ワークショップ / 出版不可）
- スコアの理由：

### 1.2 セクション別評価

| セクション | スコア (1-10) | コメント |
|-----------|-------------|---------|
| タイトル | | |
| アブストラクト | | |
| イントロダクション | | |
| 関連研究 | | |
| 手法 | | |
| 実験設計 | | |
| 結果と分析 | | |
| 結論 | | |
| 参考文献 | | |
| チャート / 図表 | | |
| コード品質 | | |

### 1.3 人間が書いた論文との比較
- 普段読み書きする論文と比較して、どこにギャップがあるか？
- 意外に良かった点は？

---

## 2. 改善提案

### 2.1 主要な問題点（優先順位で3〜5つ）

1.
2.
3.

### 2.2 コードの問題
- コードは単独で実行できるか？
- 実際のデータセットとベースラインを使用しているか？
- 具体的なコードの問題（もしあれば）：

### 2.3 文章の問題
- 論文の構成は妥当か？
- 技術的な記述は正確か？
- 具体的な文章の問題（もしあれば）：

---

## 3. パイプライン設計の評価

### 3.1 パイプラインフロー
- 23ステージの設計は妥当か？
- 冗長または不足しているステップはないか？

### 3.2 実験実行
- 実験設計は妥当か？（データセットの選択、比較手法、指標）
- 反復的な改善は効果的か？

### 3.3 LLMの使用
- 各ステージでのLLMのパフォーマンスはどうか？
- 明らかな「ハルシネーション」や不合理な出力はないか？

---

## 4. バグ報告

### 4.1 文章のバグ
| # | 場所（セクション/段落） | 説明 | 重要度（高/中/低） |
|---|------------------------|------|-------------------|
| W1 | | | |
| W2 | | | |

### 4.2 コードのバグ
| # | ファイル / 行 | 説明 | 重要度（高/中/低） |
|---|--------------|------|-------------------|
| C1 | | | |
| C2 | | | |

### 4.3 結果のバグ
| # | 説明 | 影響を受ける指標/チャート | 重要度（高/中/低） |
|---|------|--------------------------|-------------------|
| R1 | | | |
| R2 | | | |

### 4.4 パイプラインのバグ
| # | ステージ | 説明 | 重要度（高/中/低） |
|---|---------|------|-------------------|
| P1 | | | |
| P2 | | | |

---

## 5. その他のコメント

（自由記述：有益と思われる観察、アイデア、提案など）

---

## 添付チェックリスト

- [ ] フィードバックレポート (`feedback_<name>.md`)
- [ ] 出力ディレクトリ一式 (`artifacts/rc-XXXXXX.zip`)
- [ ] 設定ファイル (`config.yaml`、APIキー削除済み)
- [ ] ターミナルログ（任意）
````

---

## ❓ FAQ

### Q1: GPUなしでテストできますか？

**はい！** `experiment.mode: "sandbox"` を使用してください — パイプラインはCPU上で実験を実行します。実験はシンプルになりますが、エンドツーエンドの完全なテストには十分です。

### Q2: API呼び出しの費用はどのくらいですか？

パイプラインの完全な実行は、モデル、修正反復回数、実験の複雑さに応じて、APIの費用が約**$5〜15**かかります。トップティアのモデル（GPT-5.4、Claude Opus 4.6）はやや高価ですが、大幅に良い結果を生成します。

### Q3: パイプラインが実行中にクラッシュした場合は？

チェックポイントから再開してください：

```bash
researchclaw run --config config.yaml --resume
```

### Q4: 英語以外の研究トピックを使用できますか？

トピックは**英語**で記述することを推奨します。パイプラインのプロンプト、文献検索、論文生成はすべて英語ベースです。アイデアが他の言語の場合は、事前に翻訳してください。

### Q5: どのような研究トピックを選べばよいですか？

**自分がよく知っている分野の具体的な研究課題**を選んでください — そうすることで、出力が技術的に正確かどうかを意味のある形で評価できます。ヒント：

- ✅ 明確な実験的検証があるトピックを選ぶ（分類、回帰、強化学習タスクなど）
- ❌ 過度に広範または抽象的なトピックは避ける（例：「AGI」、「汎用知能」）
- ✅ 具体的に：*"医用画像分類におけるFew-shot学習に対するデータ拡張戦略の効果の調査"*

### Q6: Dockerモードの使用方法は？（上級者向け）

NVIDIA GPUとDocker + NVIDIA Container Toolkitがある場合：

```bash
# 1. 実験用イメージをビルド
docker build -t researchclaw/experiment:latest researchclaw/docker/

# 2. config.yamlを更新：
#   experiment:
#     mode: "docker"
#     docker:
#       gpu_enabled: true
#       memory_limit_mb: 8192
#       network_policy: "setup_only"  # 推奨デフォルト

# 3. 実行
researchclaw run --config config.yaml --auto-approve
```

Dockerモードは3フェーズの実行モデルを使用します：pip install（ネットワーク有効）→ setup.py（ネットワーク有効）→ 実験（ネットワーク無効）。イメージにはプリキャッシュされたデータセット（CIFAR-10/100、MNIST、FashionMNIST、STL-10、SVHN）が含まれているため、標準的なベンチマークはネットワークアクセスなしで動作します。

### Q7: 以前テストしましたが、再テストの場合はどうすればよいですか？

テストの前に**必ず最新のコードをプル**してください：

```bash
cd AutoResearchClaw
git pull origin main
pip install -e .
```

バージョンを確認してください：

```bash
git log --oneline -1
```

バージョンが異なると、結果が大きく変わる可能性があります。フィードバックレポートには必ずコミットハッシュを記載してください。

### Q8: フィードバックはどこに提出しますか？

フィードバックレポートと添付ファイルは、以下のいずれかの方法で提出してください：

- **GitHub Issues：** [Issueを作成](https://github.com/aiming-lab/AutoResearchClaw/issues)し、`feedback` ラベルを付ける
- **Pull Request：** `feedback_<name>.md` を `community-feedback/` ディレクトリに提出
- **メール：** プロジェクトのメンテナーに連絡（詳細はリポジトリを参照）

---

## 🌍 あらゆる分野のテスターを募集しています

パイプラインはこれまで主にML関連のトピックでテストされてきました。特に以下の分野のテスターを歓迎します：

- 🧬 **バイオインフォマティクス・計算生物学**
- 🧪 **化学・材料科学**
- 📊 **統計学・応用数学**
- 🤖 **ロボティクス・制御システム**
- 🗣️ **NLP・計算言語学**
- 👁️ **コンピュータビジョン・グラフィックス**
- 🎮 **強化学習・ゲーム理論**
- 🏥 **医療AI・ヘルスケア**
- 🌐 **グラフ学習・ネットワーク科学**
- 💹 **金融ML・計量経済学**
- 🛰️ **リモートセンシング・地理空間AI**

...その他、計算実験が関わるあらゆる分野！

---

## 🙏 ありがとうございます

大小問わず、すべてのフィードバックがAutoResearchClawの改善に直接つながります。この取り組みに参加していただき、ありがとうございます。

<p align="center">
  <b>⭐ このプロジェクトに興味を持たれたら、<a href="https://github.com/aiming-lab/AutoResearchClaw">GitHub</a>でスターをお願いします！</b>
</p>
