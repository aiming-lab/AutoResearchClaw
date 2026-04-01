<p align="center">
  <img src="../image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>聊一个想法。出一篇论文。全自动 & 自演化。</b></h2>



<p align="center">
  <b><i><font size="5">直接与 <a href="#openclaw-集成">OpenClaw</a> 对话："研究 X" → 搞定。</font></i></b>
</p>

<p align="center">
  <img src="../image/framework_v2.png" width="100%" alt="AutoResearchClaw Framework">
</p>


<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#测试"><img src="https://img.shields.io/badge/Tests-1823%20passed-brightgreen?logo=pytest&logoColor=white" alt="1823 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-集成"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="showcase/SHOWCASE.md">🏆 论文展示</a> · <a href="integration-guide.md">📖 集成指南</a> · <a href="https://discord.gg/u4ksqW5P">💬 Discord 社区</a>
</p>

---

<table>
<tr>
<td width="18%">
<a href="showcase/SHOWCASE.md"><img src="showcase/thumbnails/paper_I_random_matrix-01.png" width="120" alt="Sample Paper"/></a>
</td>
<td valign="middle">
<b>🏆 生成论文展示</b><br><br>
<b>8 篇论文覆盖 8 个领域</b> — 数学、统计、生物、计算、NLP、RL、视觉、鲁棒性 — 完全自主生成，零人工干预。<br><br>
<a href="showcase/SHOWCASE.md"><img src="https://img.shields.io/badge/View_Full_Showcase_→-All_8_Papers-d73a49?style=for-the-badge" alt="View Showcase"></a>
</td>
</tr>
</table>

---

> **🧪 我们正在寻找测试者！** 用你自己的研究想法试试这个流水线 — 任何领域 — 然后 [告诉我们你的反馈](TESTER_GUIDE.md)。你的反馈将直接影响下一个版本。 **[→ Testing Guide](TESTER_GUIDE.md)** | **[→ 中文测试指南](TESTER_GUIDE_CN.md)** | **[→ 日本語テストガイド](TESTER_GUIDE_JA.md)**

---

## 🔥 News
- **[03/22/2026]** [v0.3.2](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.2) — **跨平台支持 + 重大稳定性更新** — AutoResearchClaw 现已支持任何 ACP 兼容的 AI 代理后端（Claude Code、Codex CLI、Copilot CLI、Gemini CLI、Kimi CLI），并通过 OpenClaw 桥接支持消息平台（Discord、Telegram、飞书、微信）。新增 CLI-agent 代码生成后端，将 Stage 10 和 13 委托给外部 CLI agent，支持预算控制和超时管理。同时包含反数据捏造系统（VerifiedRegistry + 实验诊断与修复循环），100+ 个 bug 修复，模块化 executor 重构，`--resume` 自动检测，LLM 重试加固，以及社区反馈修复。
- **[03/18/2026]** [v0.3.1](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.1) — **OpenCode Beast Mode + Community Contributions** — New "Beast Mode" routes complex code generation to [OpenCode](https://github.com/anomalyco/opencode) with automatic complexity scoring and graceful fallback. Added Novita AI provider support, thread-safety hardening, improved LLM output parsing robustness, and 20+ bug fixes from community PRs and internal audit.
- **[03/17/2026]** [v0.3.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.0) — **MetaClaw Integration** — AutoResearchClaw now supports [MetaClaw](https://github.com/aiming-lab/MetaClaw) cross-run learning: pipeline failures → structured lessons → reusable skills, injected into all 23 stages. **+18.3%** robustness in controlled experiments. Opt-in (`metaclaw_bridge.enabled: true`), fully backward-compatible. See [Integration Guide](#-metaclaw-integration).
- **[03/16/2026]** [v0.2.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.2.0) — Three multi-agent subsystems (CodeAgent, BenchmarkAgent, FigureAgent), hardened Docker sandbox with network-policy-aware execution, 4-round paper quality audit (AI-slop detection, 7-dim review scoring, NeurIPS checklist), and 15+ bug fixes from production runs.
- **[03/15/2026]** [v0.1.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.1.0) — We release AutoResearchClaw: a fully autonomous 23-stage research pipeline that turns a single research idea into a conference-ready paper. No human intervention required.

---

## ⚡ 一行命令。一篇论文。

```bash
pip install -e . && researchclaw setup && researchclaw init && researchclaw run --topic "Your research idea here" --auto-approve
```


---

## 🤔 这是什么？

**你有一个灵感，AutoResearchClaw 把它写出来。**

输入一个研究主题——获得一篇完整的学术论文，包含来自 OpenAlex、Semantic Scholar 和 arXiv 的真实文献，硬件感知沙箱实验（自动检测 GPU/MPS/CPU），统计分析，多 Agent 同行评审，以及面向 NeurIPS/ICML/ICLR 的顶会级 LaTeX。不需要盯着看。不需要复制粘贴。不会出现幻觉引用。

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>完整学术论文（引言、相关工作、方法、实验、结果、结论）</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>适配顶会模板的 LaTeX 文件（NeurIPS / ICLR / ICML）</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>来自 OpenAlex、Semantic Scholar 和 arXiv 的真实 BibTeX 引用——自动精简至与正文引用一致</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>四层引用完整性 + 相关性核查（arXiv、CrossRef、DataCite、LLM）</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>生成的代码 + 沙箱结果 + 结构化 JSON 指标</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>自动生成的条件对比图（含误差线和置信区间）</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>多 Agent 同行评审（含方法论-证据一致性检查）</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>从每次运行中提取的自学习教训</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>所有最终产出集中在一个文件夹——可直接上传 Overleaf 编译</td></tr>
</table>

流水线**端到端无需人工介入**运行。实验失败时自动修复。假设不成立时自主转向。引用是假的？自动删除。

🌍 **随处可用。** AutoResearchClaw 不绑定任何单一平台。你可以通过 CLI 独立运行，接入 [OpenClaw](https://github.com/openclaw/openclaw)，或对接任何 ACP 兼容的 AI 代理 —— 🤖 Claude Code、💻 Codex CLI、🐙 Copilot CLI、♊ Gemini CLI、🌙 Kimi CLI，应有尽有。而且，借助 OpenClaw 的消息桥接能力，你还可以从 💬 Discord、✈️ Telegram、🐦 飞书、💚 微信，或任何你团队日常使用的平台发起一次完整的研究。输入一个课题，输出一篇论文 —— 无论你在哪里输入。

---

## 🚀 快速开始

```bash
# 1. 克隆 & 安装
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. 初始化（交互式 — 安装 OpenCode Beast Mode，检查 Docker/LaTeX）
researchclaw setup

# 3. 配置
researchclaw init          # 交互式：选择 LLM 提供商，创建 config.arc.yaml
# 或手动：cp config.researchclaw.example.yaml config.arc.yaml

# 4. 运行
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

输出 → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — 可编译的 LaTeX、BibTeX、实验代码、图表。

<details>
<summary>📝 最小必要配置</summary>

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

## 🧠 有什么不同

| 能力 | 工作原理 |
|------|----------|
| **🔄 PIVOT / REFINE 循环** | 第 15 阶段自主决策：PROCEED、REFINE（调参）或 PIVOT（新方向）。产物自动版本化。 |
| **🤖 多 Agent 辩论** | 假设生成、结果分析、同行评审均使用结构化的多视角辩论。 |
| **🧬 自学习** | 每次运行提取教训（决策理由、运行时警告、指标异常），30 天时间衰减。未来运行从过去的错误中学习。 |
| **📚 知识库** | 每次运行在 6 个类别（决策、实验、发现、文献、问题、评审）中构建结构化知识库。 |
| **🛡️ Sentinel 看门狗** | 后台质量监控：NaN/Inf 检测、论文-证据一致性、引用相关性评分、反数据捏造守卫。 |

---

## 🦞 OpenClaw 集成

<table>
<tr>

**AutoResearchClaw 是 [OpenClaw](https://github.com/openclaw/openclaw) 兼容服务。** 在 OpenClaw 中安装后，一句话即可启动自主研究——也可通过 CLI、Claude Code 或其他 AI 编码助手独立使用。

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

### ACP (Agent Client Protocol)

AutoResearchClaw 可以使用**任何 ACP 兼容的编码 Agent** 作为其 LLM 后端——无需 API 密钥。Agent 通过 [acpx](https://github.com/openclaw/acpx) 通信，在全部 23 个流水线阶段中维持单个持久会话。

| Agent | 命令 | 备注 |
|-------|------|------|
| Claude Code | `claude` | Anthropic |
| Codex CLI | `codex` | OpenAI |
| Copilot CLI | `gh` | GitHub |
| Gemini CLI | `gemini` | Google |
| OpenCode | `opencode` | SST |
| Kimi CLI | `kimi` | Moonshot |

```yaml
# config.yaml — ACP 示例
llm:
  provider: "acp"
  acp:
    agent: "claude"   # 任何 ACP 兼容的 Agent CLI 命令
    cwd: "."          # Agent 的工作目录
  # 无需 base_url 或 api_key — Agent 自行处理认证。
```

```bash
# 直接运行 — Agent 使用自己的凭据
researchclaw run --config config.yaml --topic "Your research idea" --auto-approve
```

### 🛠️ 其他运行方式

| 方式 | 怎么用 |
|------|--------|
| **独立 CLI** | `researchclaw setup` → `researchclaw init` → `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | 读取 `RESEARCHCLAW_CLAUDE.md` — 直接说 *"Run research on [主题]"* |
| **Copilot CLI** | `researchclaw run --topic "..."` 配合 `llm.acp.agent: "gh"` |
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

> **门控阶段**（5、9、20）可暂停等待人工审批，也可用 `--auto-approve` 自动通过。拒绝后流水线回滚。

> **决策循环**：第 15 阶段可触发 REFINE（→ 第 13 阶段）或 PIVOT（→ 第 8 阶段），自动版本化之前的产物。

<details>
<summary>📋 各阶段组职责</summary>

| 阶段组 | 做什么 |
|--------|--------|
| **A：定义** | LLM 将主题分解为结构化问题树和研究问题 |
| **A+：硬件检测** | 自动检测 GPU（NVIDIA CUDA / Apple MPS / 纯 CPU），性能不足时警告用户，据此调整代码生成策略 |
| **B：文献** | 多源搜索（OpenAlex → Semantic Scholar → arXiv）获取真实论文，按相关性筛选，提取知识卡片 |
| **C：综合** | 聚类研究发现，识别研究空白，通过多 Agent 辩论生成可验证假设 |
| **D：设计** | 设计实验方案，生成硬件感知的可运行 Python 代码（GPU 等级 → 包选择），估算资源需求 |
| **E：执行** | 在沙箱中运行实验，检测 NaN/Inf 和运行时 Bug，通过定向 LLM 修复自愈代码 |
| **F：分析** | 多 Agent 分析实验结果；自主 PROCEED / REFINE / PIVOT 决策并附理由 |
| **G：写作** | 大纲 → 分段撰写初稿（5,000-6,500 词）→ 同行评审（含方法论-证据一致性）→ 带长度保障的修订 |
| **H：终稿** | 质量门控，知识归档，LaTeX 导出（适配顶会模板），引用完整性 + 相关性核查 |

</details>

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| **📚 多源文献** | 来自 OpenAlex、Semantic Scholar 和 arXiv 的真实论文——查询扩展、去重、三态熔断器与优雅降级 |
| **🔍 四层引用核查** | arXiv ID 校验 → CrossRef/DataCite DOI → Semantic Scholar 标题匹配 → LLM 相关性评分。幻觉引用自动删除。 |
| **🖥️ 硬件感知执行** | 自动检测 GPU（NVIDIA CUDA / Apple MPS / 纯 CPU），据此调整代码生成、import 和实验规模 |
| **🦾 OpenCode Beast Mode** | 复杂实验自动路由至 [OpenCode](https://github.com/anomalyco/opencode)——生成多文件项目，含自定义架构、训练循环和消融实验。通过 `researchclaw setup` 安装。 |
| **🧪 沙箱实验** | AST 验证代码、不可变 harness、NaN/Inf 快速失败、自修复、迭代优化（最多 10 轮）、部分结果捕获 |
| **📝 顶会级写作** | NeurIPS/ICML/ICLR 模板，分段撰写（5,000-6,500 词），反数据捏造守卫、修订长度保障、反免责声明强制 |
| **📐 模板切换** | `neurips_2025`、`iclr_2026`、`icml_2026` — Markdown → LaTeX，含数学公式、表格、图片、交叉引用、`\cite{}` |
| **🛡️ 反数据捏造** | VerifiedRegistry 强制论文中使用经过验证的实验数据。自动诊断失败实验并在写作前修复。未验证数字被清理。 |
| **🚦 质量门控** | 3 个人工审批门控（阶段 5、9、20），支持回滚。用 `--auto-approve` 跳过。 |

---

## 🧠 MetaClaw 集成

**AutoResearchClaw + [MetaClaw](https://github.com/aiming-lab/MetaClaw) = 一个能从每次运行中学习的流水线。**

MetaClaw 为 AutoResearchClaw 添加了**跨运行知识迁移**。启用后，流水线会自动从失败和警告中提取教训，将其转化为可复用的技能，并在后续运行中注入到全部 23 个阶段——让同样的错误不再重犯。

### 工作原理

```
运行 N 执行 → 失败/警告被捕获为 Lessons
                      ↓
          MetaClaw Lesson → Skill 转换
                      ↓
          arc-* Skill 文件存储在 ~/.metaclaw/skills/
                      ↓
运行 N+1 → build_overlay() 将技能注入每个 LLM 提示
                      ↓
          LLM 规避已知陷阱 → 更高质量，更少重试
```

### 快速配置

```bash
# 1. 安装 MetaClaw（如未安装）
pip install metaclaw

# 2. 在配置中启用
```

```yaml
# config.arc.yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"        # MetaClaw 代理（可选）
  skills_dir: "~/.metaclaw/skills"          # 技能存储位置
  fallback_url: "https://api.openai.com/v1" # 直连 LLM 回退
  fallback_api_key: ""                      # 回退 URL 的 API key
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # 转换 warning + error
    max_skills_per_run: 3
```

```bash
# 3. 照常运行 — MetaClaw 透明运作
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve
```

每次运行后，查看 `~/.metaclaw/skills/arc-*/SKILL.md` 以了解流水线学到了哪些技能。

### 实验结果

在对照 A/B 实验中（相同主题、相同 LLM、相同配置）：

| 指标 | 基线 | 使用 MetaClaw | 改善 |
|------|------|---------------|------|
| 阶段重试率 | 10.5% | 7.9% | **-24.8%** |
| Refine 循环次数 | 2.0 | 1.2 | **-40.0%** |
| 流水线阶段完成率 | 18/19 | 19/19 | **+5.3%** |
| 整体鲁棒性得分（综合） | 0.714 | 0.845 | **+18.3%** |

> 综合鲁棒性得分是阶段完成率（40%）、重试减少（30%）和 Refine 循环效率（30%）的加权平均。

### 向后兼容性

- **默认：关闭。** 如果 `metaclaw_bridge` 不存在或 `enabled: false`，流水线行为与之前完全一致。
- **无新依赖。** MetaClaw 是可选的——核心流水线无需它即可运行。
- **所有 1,823 项现有测试通过**（包含集成代码）。

---

## ⚙️ 配置参考

<details>
<summary>点击展开完整配置参考</summary>

```yaml
# === 项目 ===
project:
  name: "my-research"              # 项目标识符
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === 研究 ===
research:
  topic: "..."                     # 研究主题（必填）
  domains: ["ml", "nlp"]           # 文献搜索的研究领域
  daily_paper_count: 8             # 每个搜索查询的目标论文数
  quality_threshold: 4.0           # 论文最低质量分

# === 运行时 ===
runtime:
  timezone: "America/New_York"     # 用于时间戳
  max_parallel_tasks: 3            # 并发实验限制
  approval_timeout_hours: 12       # 门控阶段超时
  retry_limit: 2                   # 阶段失败重试次数

# === LLM ===
llm:
  provider: "openai-compatible"    # openai | openrouter | deepseek | minimax | acp | openai-compatible
  base_url: "https://..."          # API 端点（openai-compatible 必填）
  api_key_env: "OPENAI_API_KEY"    # API key 环境变量（openai-compatible 必填）
  api_key: ""                      # 或直接填写 key
  primary_model: "gpt-4o"          # 主模型
  fallback_models: ["gpt-4o-mini"] # 回退链
  s2_api_key: ""                   # Semantic Scholar API key（可选，更高速率限制）
  acp:                             # 仅在 provider: "acp" 时使用
    agent: "claude"                # ACP Agent CLI 命令（claude, codex, gemini 等）
    cwd: "."                       # Agent 的工作目录

# === 实验 ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 300             # 每次运行最大执行时间（默认：300 秒）
  max_iterations: 10               # 最大优化迭代次数
  metric_key: "val_loss"           # 主指标名称
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
    auto_install_deps: true        # 自动检测 import → requirements.txt
  ssh_remote:
    host: ""                       # GPU 服务器主机名
    gpu_ids: []                    # 可用 GPU ID
    remote_workdir: "/tmp/researchclaw_experiments"
  opencode:                          # OpenCode Beast Mode（通过 `researchclaw setup` 自动安装）
    enabled: true                    # 主开关（默认：true）
    auto: true                       # 无需确认自动触发（默认：true）
    complexity_threshold: 0.2        # 0.0-1.0 — 越高 = 仅在复杂实验时触发
    model: ""                        # 覆盖模型（空 = 使用 llm.primary_model）
    timeout_sec: 600                 # OpenCode 生成最大秒数
    max_retries: 1                   # 失败重试次数
    workspace_cleanup: true          # 采集后清理临时工作区
  code_agent:                        # CodeAgent v2 — 多阶段代码生成
    enabled: true                    # 使用 CodeAgent 替代传统单 prompt 代码生成
    architecture_planning: true      # 生成代码前先生成深度实现蓝图
    sequential_generation: true      # 按依赖 DAG 逐文件生成
    hard_validation: true            # 基于 AST 的验证门控（拦截相同消融、硬编码指标）
    hard_validation_max_repairs: 2   # 验证失败时最大修复次数
    exec_fix_max_iterations: 3       # 执行修复循环最大次数
    exec_fix_timeout_sec: 60         # 每次执行修复超时（秒）
  benchmark_agent:                   # BenchmarkAgent — 自动数据集和基线选择
    enabled: true                    # 启用 4-agent 基准测试流水线（Surveyor→Selector→Acquirer→Validator）
    enable_hf_search: true           # 搜索 HuggingFace Datasets
    enable_web_search: true          # 搜索 Google Scholar 获取基准
    tier_limit: 2                    # 数据集级别过滤（1=小型/已缓存，2=中型，3=大型）
    min_benchmarks: 1                # 最少需要的数据集数量
    min_baselines: 2                 # 最少需要的基线方法数量
  figure_agent:                      # FigureAgent — 学术图表生成
    enabled: true                    # 启用 5-agent 图表流水线（Planner→CodeGen→Renderer→Critic→Integrator）
    min_figures: 3                   # 最少生成图表数
    max_figures: 8                   # 最多生成图表数
    max_iterations: 3                # Critic 驱动的迭代优化次数
    dpi: 300                         # 输出分辨率
    strict_mode: false               # 图表生成失败时是否阻塞流水线
  repair:                            # 反数据捏造实验修复
    enabled: true                    # 自动诊断并修复失败的实验
    max_cycles: 3                    # 修复重试循环数
    min_completion_rate: 0.5         # >=50% 条件必须完成才可继续
    min_conditions: 2                # 有效实验至少需要 2 个条件
    use_opencode: true               # 通过 OpenCode Beast Mode 进行修复

# === 网络搜索（可选）===
web_search:
  enabled: true                      # 启用网络增强文献搜索
  tavily_api_key_env: "TAVILY_API_KEY"  # Tavily API key 环境变量（可选）
  enable_scholar: true               # Google Scholar 搜索
  enable_pdf_extraction: true        # 从 PDF 中提取文本
  max_web_results: 10                # 每次查询最大网络结果数

# === 导出 ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # 自定义 Prompt YAML 路径（空 = 使用默认）

# === 安全 ===
security:
  hitl_required_stages: [5, 9, 20] # 需要人工审批的阶段
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === 知识库 ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === 通知 ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === MetaClaw Bridge（可选）===
metaclaw_bridge:
  enabled: false                   # 设为 true 以启用跨运行学习
  proxy_url: "http://localhost:30000"  # MetaClaw 代理 URL
  skills_dir: "~/.metaclaw/skills" # arc-* 技能的存储位置
  fallback_url: ""                 # 代理不可用时的直连 LLM 回退
  fallback_api_key: ""             # 回退端点的 API key
  lesson_to_skill:
    enabled: true                  # 自动将教训转换为技能
    min_severity: "warning"        # 转换的最低严重级别
    max_skills_per_run: 3          # 每次流水线运行的最大新技能数
  prm:                             # 过程奖励模型质量门控（可选）
    enabled: false                 # 使用 LLM-as-judge 评分阶段产出
    model: "gpt-5.4"              # PRM 评判模型
    votes: 3                       # 多数投票次数
    gate_stages: [5, 9, 15, 20]   # 应用 PRM 门控的阶段

# === OpenClaw Bridge ===
openclaw_bridge:
  use_cron: false                  # 定时研究运行
  use_message: false               # 进度通知
  use_memory: false                # 跨会话知识持久化
  use_sessions_spawn: false        # 派生并行子会话
  use_web_fetch: false             # 实时网络搜索
  use_browser: false               # 基于浏览器的论文采集
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

---

## 📌 引用

如果你觉得 AutoResearchClaw 有用，请引用：

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
