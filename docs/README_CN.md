<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>聊一个想法。出一篇论文。全自动。</b></h2>



<p align="center">
  <i>直接与 <a href="#openclaw-集成">OpenClaw</a> 对话："研究 X" → 搞定。</i>
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#测试"><img src="https://img.shields.io/badge/Tests-1284%20passed-brightgreen?logo=pytest&logoColor=white" alt="1284 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-集成"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
</p>

<p align="center">
  <a href="../README.md">🇺🇸 English</a> ·
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
  <a href="integration-guide.md">📖 集成指南</a>
</p>

---

## ⚡ 一行启动

```bash
pip install -e . && researchclaw run --topic "你的研究想法" --auto-approve
```

---

## 🤔 这是什么？

你有一个灵感，你想要一篇论文。**就这么简单。**

AutoResearchClaw 接收一个研究主题，自主产出完整学术论文——真实文献来自 arXiv 和 Semantic Scholar（多源搜索、arXiv 优先以避免限流），硬件感知沙箱实验（自动检测 GPU/MPS/CPU），包含统计分析、同行评审和顶会级 LaTeX 排版（目标 5,000-6,500 词，符合 NeurIPS/ICML/ICLR 标准）。不用盯着，不用在工具间来回复制。

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>完整学术论文（引言、相关工作、方法、实验、结果、结论）</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>适配顶会模板的 LaTeX 文件（NeurIPS / ICLR / ICML）</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>来自 Semantic Scholar 和 arXiv 的真实 BibTeX 引用——自动精简至与正文引用一致</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>四层引用核查（arXiv、CrossRef、DataCite、LLM 相关性评分）</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>生成的代码 + 沙箱结果 + 结构化 JSON 指标</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>自动生成的条件对比图（含误差线和置信区间）</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>多 Agent 同行评审（含方法论-证据一致性检查）</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>从每次运行中提取的自学习教训</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>所有最终产出集中在一个文件夹——可直接上传 Overleaf 编译</td></tr>
</table>

流水线**端到端无需人工介入**运行（除非你配置了门控阶段）。实验失败时自动修复，假设不成立时自主转向。

### 🎯 试试看

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 有什么不同

### 🔄 PIVOT / REFINE 决策循环

流水线不只是线性运行。第 15 阶段（RESEARCH_DECISION）根据实验结果评估假设，做出自主决策：

- **PROCEED** — 结果支持假设，继续写论文
- **REFINE** — 结果有前景但需改进，回到代码/参数优化
- **PIVOT** — 发现根本性问题，从假设生成重新开始

每次 PIVOT/REFINE 都会**版本化之前的产物**（`stage-08_v1/`、`stage-08_v2/`……），确保工作不丢失，决策演化完全可追溯。

### 🤖 多 Agent 辩论

关键阶段使用结构化辩论协议，汇集多个 LLM 视角：

- **假设生成** — 多个 Agent 提出和挑战创意
- **结果分析** — 乐观者、怀疑者、实用者多角度分析
- **同行评审** — 方法论-证据一致性审查（论文声称跑了 50 次实验，代码只跑了 5 次？）

### 🧬 Evolution：跨运行自学习

每次运行提取细粒度教训——不只是"失败了"，而是*为什么*：

- PIVOT/REFINE 决策的具体理由
- 实验 stderr 中的运行时警告（如 `RuntimeWarning: division by zero`）
- 指标异常（NaN、Inf、所有算法收敛速度相同）

这些教训持久化存储在 JSONL 中，使用 **30 天半衰期时间衰减加权**，作为 prompt overlay 注入未来运行。流水线真正从错误中学习。

### 📚 知识库

每次运行自动构建结构化知识库（存储在 `docs/kb/` 中），包含 6 个类别：

- **decisions/** — 实验设计、质量门控、研究决策、资源规划、搜索策略、知识归档
- **experiments/** — 代码生成日志、实验运行记录、迭代优化过程
- **findings/** — 引用核查报告、结果分析、综合报告
- **literature/** — 知识提取、文献采集、筛选结果
- **questions/** — 假设生成、问题分解、主题初始化
- **reviews/** — 导出/发布报告、论文草稿、大纲、修订、同行评审

### 🛡️ Sentinel 看门狗

后台质量监控，捕获主流水线可能遗漏的问题：

- **运行时 Bug 检测** — 指标中的 NaN/Inf、stderr 警告反馈给 LLM 进行定向修复
- **论文-证据一致性** — 实际实验代码、运行结果、迭代日志注入同行评审
- **引用相关性评分** — 不仅验证引用存在性，还用 LLM 评估与研究主题的相关性
- **收敛判据强制** — 检测固定迭代实验，要求实现正确的 early stopping
- **消融验证** — 检测重复/相同的消融条件，标记失效的对比实验
- **反数据捏造守卫** — 实验无指标时硬性阻止论文撰写

---

## 🦞 OpenClaw 集成

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw 是 [OpenClaw](https://github.com/openclaw/openclaw) 兼容服务。** 在 OpenClaw 中安装后，一句话即可启动自动研究——也可通过 CLI、Claude Code 或其他 AI 编码助手独立使用。

</td>
</tr>
</table>

### 🚀 通过 OpenClaw 使用（推荐）

如果你已经在使用 [OpenClaw](https://github.com/openclaw/openclaw) 作为 AI 助手：

```
1️⃣  把 GitHub 仓库地址分享给 OpenClaw
2️⃣  OpenClaw 自动读取 RESEARCHCLAW_AGENTS.md → 理解流水线
3️⃣  对它说："帮我研究 [你的主题]"
4️⃣  完成 — OpenClaw 自动克隆、安装、配置、运行，然后返回结果
```

**就这么简单。** OpenClaw 自动处理 `git clone`、`pip install`、配置和流水线执行。你只需聊天。

<details>
<summary>💡 底层发生了什么</summary>

1. OpenClaw 读取 `RESEARCHCLAW_AGENTS.md` → 学习研究编排器角色
2. OpenClaw 读取 `README.md` → 理解安装方式和流水线结构
3. OpenClaw 复制 `config.researchclaw.example.yaml` → `config.yaml`
4. 向你询问 LLM API Key（或使用环境变量）
5. 运行 `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. 返回论文、LaTeX、实验结果和引用

</details>

### 🔌 OpenClaw Bridge（高级功能）

AutoResearchClaw 内置了 **Bridge 适配器系统**，提供 6 个可选集成能力：

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ 定时研究任务
  use_message: true           # 💬 进度通知（Discord/Slack/Telegram）
  use_memory: true            # 🧠 跨会话知识持久化
  use_sessions_spawn: true    # 🔀 为并行阶段派生子会话
  use_web_fetch: true         # 🌐 文献检索中的实时网络搜索
  use_browser: false          # 🖥️ 基于浏览器的论文采集
```

每个标志激活一个类型化适配器协议。当 OpenClaw 提供对应能力时，适配器无需改代码即可消费。详见 [`integration-guide.md`](integration-guide.md)。

### 🛠️ 其他运行方式

| 方式 | 怎么用 |
|------|--------|
| **独立 CLI** | `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | 读取 `RESEARCHCLAW_CLAUDE.md` — 直接说 *"帮我研究 [主题]"* |
| **OpenCode** | 读取 `.claude/skills/` — 同样的自然语言交互 |
| **任何 AI CLI** | 提供 `RESEARCHCLAW_AGENTS.md` 作为上下文 → agent 自动引导 |

---

## 🔬 流水线：23 个阶段，8 个阶段组

```
阶段组 A：研究定义                阶段组 E：实验执行
  1. TOPIC_INIT                    12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE             13. ITERATIVE_REFINE  ← 自修复

阶段组 B：文献发现                阶段组 F：分析与决策
  3. SEARCH_STRATEGY               14. RESULT_ANALYSIS    ← 多Agent
  4. LITERATURE_COLLECT ← 真实API  15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN  [门控]
  6. KNOWLEDGE_EXTRACT             阶段组 G：论文撰写
                                   16. PAPER_OUTLINE
阶段组 C：知识综合                 17. PAPER_DRAFT
  7. SYNTHESIS                     18. PEER_REVIEW        ← 证据审查
  8. HYPOTHESIS_GEN   ← 辩论      19. PAPER_REVISION

阶段组 D：实验设计                阶段组 H：终稿
  9. EXPERIMENT_DESIGN  [门控]     20. QUALITY_GATE     [门控]
 10. CODE_GENERATION               21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING             22. EXPORT_PUBLISH    ← LaTeX
                                   23. CITATION_VERIFY   ← 相关性审查
```

> **门控阶段**（5、9、20）可暂停等待人工审批，也可用 `--auto-approve` 自动通过。

> **决策循环**：第 15 阶段可触发 REFINE（→ 第 13 阶段）或 PIVOT（→ 第 8 阶段），自动版本化之前的产物。

<details>
<summary>📋 各阶段组职责</summary>

| 阶段组 | 做什么 |
|--------|--------|
| **A：定义** | LLM 将主题分解为结构化问题树和研究问题 |
| **A+：硬件检测** | 自动检测 GPU（NVIDIA CUDA / Apple MPS / 纯 CPU），性能不足时警告用户，据此调整代码生成策略 |
| **B：文献** | 多源搜索（arXiv 优先，Semantic Scholar 补充）获取真实论文，按相关性筛选，提取知识卡片 |
| **C：综合** | 聚类研究发现，识别研究空白，通过多 Agent 辩论生成可验证假设 |
| **D：设计** | 设计实验方案，生成硬件感知的可运行 Python 代码（GPU 等级 → 包选择），估算资源需求 |
| **E：执行** | 在沙箱中运行实验，检测 NaN/Inf 和运行时 Bug，通过定向 LLM 修复自愈代码 |
| **F：分析** | 多 Agent 分析实验结果；自主 PROCEED / REFINE / PIVOT 决策并附理由 |
| **G：写作** | 大纲 → 分段撰写初稿（5,000-6,500 词）→ 同行评审（含方法论-证据一致性）→ 带长度保障的修订 |
| **H：终稿** | 质量门控，知识归档，LaTeX 导出（适配顶会模板），引用完整性 + 相关性核查 |

</details>

---

## 🚀 快速开始

### 前置条件

- 🐍 Python 3.11+
- 🔑 一个 OpenAI 兼容的 LLM API（GPT-4o、GPT-5.x，或任何兼容接口）

### 安装

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 配置

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 最小必要配置</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "你的研究主题"

llm:
  base_url: "https://api.openai.com/v1"     # 任何 OpenAI 兼容端点
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                             # 可选：Semantic Scholar API key（更高速率限制）

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

### 运行

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-..."

# 🚀 运行完整流水线
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 指定研究主题
researchclaw run --config config.arc.yaml --topic "Transformer 注意力机制在时间序列中的应用" --auto-approve

# ✅ 仅验证配置
researchclaw validate --config config.arc.yaml

# ⏩ 从指定阶段恢复
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

输出保存在 `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/`，每个阶段一个子目录。

所有用户可交付成果自动汇集到 **`deliverables/`** 文件夹：

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # 终稿（Markdown）
├── paper.tex                  # 适配顶会模板的 LaTeX
├── references.bib             # 已验证的 BibTeX 参考文献（自动精简）
├── neurips_2025.sty           # 会议样式文件（自动选择）
├── code/                      # 实验代码 + requirements.txt
├── verification_report.json   # 引用完整性报告
├── charts/                    # 结果可视化（条件对比图、误差线）
└── manifest.json              # 交付清单（含元数据）
```

`deliverables/` 文件夹**可直接编译**——包含会议 `.sty` 和 `.bst` 文件，可直接用 `pdflatex` + `bibtex` 编译，或上传 Overleaf 无需额外下载。

---

## ✨ 核心功能

### 📚 多源文献搜索

第 4 阶段调用**真实学术 API**，而非依赖 LLM 凭记忆编造论文。采用 **arXiv 优先**策略以避免 Semantic Scholar 限流。

- **arXiv API**（主源）— 预印本（真实 arXiv ID 和元数据），无速率限制
- **Semantic Scholar API**（副源）— 真实论文（标题、摘要、期刊、引用次数、DOI）
- **查询扩展** — 自动生成更广泛的搜索词（综述、基准、对比变体），目标覆盖 30-60 篇参考文献
- **自动去重** — DOI → arXiv ID → 模糊标题匹配
- **BibTeX 生成** — 有效的 `@article{cite_key, ...}` 条目
- **三态熔断器** — CLOSED → OPEN → HALF_OPEN 自动恢复，指数退避冷却（永不永久禁用）
- **优雅降级** — S2 失败不阻塞 arXiv 结果；所有 API 均失败时回退到 LLM 增强结果

### 🔍 引用核查（第 23 阶段）

论文写完后，自动对每条引用进行**完整性和相关性核查**：

| 层级 | 方法 | 检查内容 |
|------|------|----------|
| L1 | arXiv API `id_list` | 有 arXiv ID 的论文 — 验证 ID 是否真实存在 |
| L2 | CrossRef `/works/{doi}` + DataCite 回退 | 有 DOI 的论文 — 验证 DOI 能否解析且标题匹配（DataCite 处理 arXiv `10.48550` DOI） |
| L3 | Semantic Scholar + arXiv 标题搜索 | 其他所有论文 — 模糊标题匹配（≥0.80 相似度） |
| L4 | LLM 相关性评分 | 所有已验证引用 — 评估与研究主题的相关性 |

每条引用 → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**自动清理**：幻觉引用从论文文本中静默移除（不留标记）。未被引用的参考文献条目自动精简。最终的 `references.bib` 仅包含已验证的、被引用的参考文献。

### 🖥️ 硬件感知执行

第 1 阶段自动检测本地 GPU 能力，据此调整整条流水线：

| 等级 | 检测方式 | 行为 |
|------|----------|------|
| **高性能** | NVIDIA GPU ≥8 GB 显存 | 完整 PyTorch/GPU 代码生成，缺少 torch 时自动安装 |
| **受限** | NVIDIA <8 GB 或 Apple MPS | 轻量级实验（<1M 参数、≤20 epochs），向用户发出警告 |
| **纯 CPU** | 未检测到 GPU | 仅使用 NumPy/sklearn，不导入 torch，建议用户使用远程 GPU 服务器 |

### 🧪 沙箱实验执行

- **代码验证** — AST 解析、import 白名单、禁止沙箱外文件操作
- **计算预算守卫** — 时间预算（可配置，默认 600 秒）注入代码生成提示；LLM 必须设计在沙箱超时内可完成的实验
- **实验 Harness** — 不可变的 `experiment_harness.py` 注入沙箱，提供 `should_stop()` 时间守卫、`report_metric()` NaN/Inf 拒绝、`finalize()` 结果写入
- **结构化输出** — 实验产出 `results.json`（类型化指标，非仅 stdout 解析）
- **NaN/发散快速失败** — NaN/Inf 值从指标中过滤；发散损失（>100）被检测并标记
- **收敛判据强制** — 生成的代码必须包含 early stopping，不允许固定迭代次数
- **运行时 Bug 检测** — 自动发现指标中的 NaN/Inf 和 stderr 警告
- **自修复** — 运行时问题反馈给 LLM 进行根因诊断修复
- **迭代优化** — 第 13 阶段分析结果并改进代码/参数后重新运行（最多 10 次迭代）
- **部分结果捕获** — 超时但已捕获指标的实验状态设为 "partial"，保留可用数据
- **主题-实验对齐** — 基于 LLM 的代码生成后验证，确保实验代码真正测试所声明的研究主题

### 📝 顶会级论文撰写

写作流水线对标 NeurIPS/ICML/ICLR 标准（9+ 页，5,000-6,500 词）：

- **数据完整性执行** — 实验无指标时硬性阻止论文撰写（防止 LLM 捏造结果）
- **顶会级提示** — 系统提示包含已接收论文分析的关键原则：新颖性、叙事性、强基线、消融实验、诚实性、可复现性
- **标题与框架指南** — 新颖性信号、"可传播性测试"、5 句式摘要结构
- **分段撰写** — 3 次顺序 LLM 调用，避免输出截断
- **逐节字数目标** — 摘要（150-250）、引言（800-1000）、相关工作（600-800）、方法（1000-1500）、实验（800-1200）、结果（600-800）、讨论（400-600）
- **修订长度保障** — 修订稿若短于初稿，自动重试
- **反免责声明强制** — "due to computational constraints" 最多出现 1 次；修订提示主动删除重复的模糊表述
- **统计严谨性** — 结果表格要求包含置信区间、p 值和效应量；失效消融被标记并排除在声明之外
- **顶会级同行评审** — 审稿人按 NeurIPS/ICML 评分标准打分 1-10

### 📐 会议模板切换

```yaml
export:
  target_conference: "neurips_2025"   # 或 "iclr_2026" 或 "icml_2026"
```

| 会议 | 样式包 | 分栏 |
|------|--------|------|
| NeurIPS 2025 | `neurips_2025` | 单栏 |
| ICLR 2026 | `iclr2026_conference` | 单栏 |
| ICML 2026 | `icml2026` | 双栏 |
| NeurIPS 2024 | `neurips_2024` | 单栏 |
| ICLR 2025 | `iclr2025_conference` | 单栏 |
| ICML 2025 | `icml2025` | 双栏 |

Markdown → LaTeX 转换器自动处理：章节标题（含自动编号去重）、行内/行间数学公式、粗体/斜体、列表、表格（含 `\caption`/`\label`）、图片（`\includegraphics`）、代码块（Unicode 安全）、交叉引用和 `\cite{}` 引用。

### 🚦 质量门控

| 门控 | 阶段 | 拒绝后回退到 |
|------|------|-------------|
| 文献筛选 | 5 | 重新采集文献（阶段 4） |
| 实验设计 | 9 | 重新生成假设（阶段 8） |
| 质量门控 | 20 | 从大纲重新写论文（阶段 16） |

用 `--auto-approve` 跳过所有门控，或在 `security.hitl_required_stages` 中配置特定阶段。

---

## ⚙️ 配置参考

<details>
<summary>点击展开完整配置参考</summary>

```yaml
# === 项目 ===
project:
  name: "my-research"
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === 研究 ===
research:
  topic: "..."                     # 研究主题（必填）
  domains: ["ml", "nlp"]
  daily_paper_count: 8
  quality_threshold: 4.0

# === 运行时 ===
runtime:
  timezone: "Asia/Shanghai"
  max_parallel_tasks: 3
  approval_timeout_hours: 12
  retry_limit: 2

# === LLM ===
llm:
  provider: "openai-compatible"
  base_url: "https://..."
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                   # Semantic Scholar API key（可选，更高速率限制）

# === 实验 ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # 每次运行最大执行时间（默认：600 秒）
  max_iterations: 10
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
    auto_install_deps: true        # 自动检测 import → requirements.txt

# === 导出 ===
export:
  target_conference: "neurips_2025"
  authors: "匿名"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # 自定义 Prompt YAML 路径（空 = 使用默认）

# === 安全 ===
security:
  hitl_required_stages: [5, 9, 20]
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === 知识库 ===
knowledge_base:
  backend: "markdown"
  root: "docs/kb"

# === 通知 ===
notifications:
  channel: "console"
  target: ""

# === OpenClaw Bridge ===
openclaw_bridge:
  use_cron: false
  use_message: false
  use_memory: false
  use_sessions_spawn: false
  use_web_fetch: false
  use_browser: false
```

</details>

---

## 🙏 致谢

灵感来源：

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist)（Sakana AI）— 自动化研究先驱
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch)（Andrej Karpathy）— 端到端研究自动化
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/)（Analemma）— 全自动研究系统

---

## 📄 许可证

MIT — 详见 [LICENSE](../LICENSE)。

<p align="center">
  <sub>Built with 🦞 by the AutoResearchClaw team</sub>
</p>
