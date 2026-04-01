<p align="center">
  <img src="../image/logo.png" width="500" alt="AutoResearchClaw Logo">
</p>

<h2 align="center">🧪 社区测试指南</h2>

<p align="center">
  <b>欢迎来自各个领域的你，一起测试全球首个全自动学术论文生成 Pipeline。</b>
</p>

<p align="center">
  <a href="https://github.com/aiming-lab/AutoResearchClaw">⭐ Star 项目</a> ·
  <a href="#-快速开始">🚀 快速开始</a> ·
  <a href="#-反馈报告模板">📋 反馈模板</a> ·
  <a href="TESTER_GUIDE.md">🇬🇧 English</a> ·
  <a href="TESTER_GUIDE_JA.md">🇯🇵 日本語テストガイド</a>
</p>

---

## 👋 你好，测试者！

**AutoResearchClaw** 是一个全自动学术论文生成 Pipeline。你只需提供一个研究 idea，系统就会自动完成文献检索、实验设计、代码生成、实验执行、论文撰写、同行评审到最终交付的全部 **23 个阶段**——无需任何人工干预。

我们正在寻找来自**各个学科和领域**的测试者——机器学习、NLP、计算机视觉、强化学习、生物信息学、物理学、社会科学……领域越多样，Pipeline 就能变得越好。

**你的任务：** 用你自己的研究 idea 运行一次完整的 Pipeline，检查输出质量，然后向我们提交一份详细的反馈报告。就这么简单——你的每一条反馈都会直接推动下一个版本的改进。

---

## 📋 目录

1. [环境要求](#-环境要求)
2. [安装与配置](#-安装与配置)
3. [运行测试](#-运行测试)
4. [查看交付结果](#-查看交付结果)
5. [反馈报告要求](#-反馈报告要求)
6. [反馈报告模板](#-反馈报告模板)
7. [常见问题](#-常见问题)

---

## 📦 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | macOS / Linux / WSL2 | Linux (Ubuntu 22.04+) |
| Python | 3.11+ | 3.11 或 3.12 |
| 磁盘空间 | 500 MB | 2 GB+ |
| 内存 | 8 GB | 16 GB+ |
| GPU | 非必须（sandbox 模式） | NVIDIA GPU + CUDA 12.x（docker 模式） |
| 网络 | 需要（调用 LLM API + 文献检索） | 稳定的网络连接 |
| LLM API Key | **必须** | OpenAI 或 Anthropic |

### 🔑 关于 API Key

Pipeline 在每个阶段都会调用大语言模型（LLM）来完成写作、编码、评审等任务。你需要准备一个 **OpenAI** 或 **Anthropic** 的 API Key。

> **强烈建议使用最新、最强的模型以获得最佳效果：**
>
> | 提供商 | 推荐模型 | 备选 |
> |--------|---------|------|
> | **OpenAI** | **GPT-5.4**（首选） | GPT-5.1 或 GPT-4.1 |
> | **Anthropic** | **Claude Opus 4.6**（首选） | Claude Sonnet 4.6 |
>
> 使用顶级模型会显著提升论文写作质量、代码生成准确性和实验设计合理性。较低版本的模型（如 gpt-4o）可能导致输出质量明显下降。

---

## 🛠 安装与配置

### ⚠️ 请务必使用最新版本

> **本项目处于快速迭代阶段，** 代码更新频繁，不同版本之间的生成效果可能存在较大差异。
>
> **每次测试前，请务必拉取最新代码：**
>
> ```bash
> cd AutoResearchClaw
> git pull origin main
> pip install -e .    # 重新安装以确保更新生效
> ```
>
> 记录你的版本号，方便填写反馈报告：
> ```bash
> git log --oneline -1
> ```

---

### 方式 A：使用 Claude Code（最快 ⚡ 推荐）

如果你正在使用 [Claude Code](https://claude.ai/claude-code)（Anthropic 的 CLI 工具），直接粘贴以下内容即可：

```
请帮我克隆并安装 AutoResearchClaw 项目：
https://github.com/aiming-lab/AutoResearchClaw.git

如果已经克隆过，请先 git pull origin main 更新到最新版本。

安装完成后，帮我创建一个配置文件，使用以下参数：
- LLM: OpenAI，模型选择 gpt-5.4（或 Anthropic Claude Opus 4.6）
- 实验模式: sandbox（本地沙盒执行）
- 研究主题: "<在这里填入你的研究 idea>"
- 自动审批所有 gate stage

我的 API Key 是: sk-xxxx（请设为环境变量，不要写在配置文件里）
```

Claude Code 会自动完成克隆、安装依赖、创建配置文件、运行 Pipeline 的全部步骤。

### 方式 B：手动安装

```bash
# 1. 克隆项目
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw

# ⚠️ 如果已经克隆过，务必先更新！
# git pull origin main

# 2. 创建 Python 虚拟环境
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows（推荐使用 WSL2）

# 3. 安装项目
pip install -e .

# 4. 验证安装成功
researchclaw --help
```

### ⚙️ 配置文件

```bash
cp config.researchclaw.example.yaml config.yaml
```

编辑 `config.yaml`，修改以下关键字段：

```yaml
# === 项目设置 ===
project:
  name: "my-test"
  mode: "full-auto"

# === 研究主题——用英文描述你的 idea ===
research:
  topic: "你的研究 idea，用英文描述，一两句话即可"
  domains:
    - "machine-learning"    # 可选: nlp, cv, rl, graph-learning, etc.

# === LLM 配置——请使用最强模型！ ===
#
# 方案一：OpenAI（推荐 GPT-5.4）
llm:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-5.4"              # 首选最强模型
  fallback_models:
    - "gpt-5.1"
    - "gpt-4.1"

# 方案二：Anthropic Claude（推荐 Claude Opus 4.6）
# llm:
#   provider: "openai-compatible"
#   base_url: "https://api.anthropic.com/v1"
#   api_key_env: "ANTHROPIC_API_KEY"
#   primary_model: "claude-opus-4-6"
#   fallback_models:
#     - "claude-sonnet-4-6"

# === 实验模式 ===
experiment:
  mode: "sandbox"                # sandbox = 本地执行（推荐）
  time_budget_sec: 600           # 每次实验最长运行时间（秒）
  max_iterations: 10
  metric_key: "primary_metric"
  metric_direction: "minimize"   # 或 "maximize"
```

### 🔐 设置 API Key

```bash
# OpenAI 用户：
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Anthropic 用户：
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx"

# 可选：Semantic Scholar API Key（可加快文献检索）
export S2_API_KEY="your-s2-key"
```

> **🔒 安全提醒：** 请勿将 API Key 硬编码在任何文件中。使用 `api_key_env` 指定环境变量名即可。

---

## 🚀 运行测试

### 快速开始

```bash
source .venv/bin/activate
export OPENAI_API_KEY="sk-xxxx"       # 或 ANTHROPIC_API_KEY

researchclaw run --config config.yaml --auto-approve
```

### 指定研究主题运行

```bash
researchclaw run \
  --config config.yaml \
  --topic "Investigating the effect of curriculum learning on image classification with adaptive difficulty scheduling" \
  --auto-approve
```

### ⏱ 预估运行时间

| 实验模式 | 预估时间 | 说明 |
|---------|---------|------|
| sandbox | 30 分钟 ~ 2 小时 | 取决于实验复杂度和 API 响应速度 |
| docker (GPU) | 1 ~ 4 小时 | 可运行更复杂的深度学习实验 |

运行过程中终端会实时显示当前阶段和进度。**无需任何手动操作**，安心等待即可。

### ✅ 如何知道运行结束

当看到类似以下输出时，表示 Pipeline 已成功完成：

```
[Stage 23/23] ✓ Deliverables packaged
Pipeline complete — deliverables at: artifacts/rc-20260315-XXXXXX-YYYY/deliverables/
```

### 🔄 如果运行中断

Pipeline 支持断点续跑：

```bash
researchclaw run --config config.yaml --resume
```

---

## 🔍 查看交付结果

运行结束后，输出文件位于 `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` 目录下。

### 📂 交付物清单

| 文件/目录 | 内容 |
|----------|------|
| `paper_final.md` | 最终论文（Markdown 格式，5,000–6,500 词） |
| `paper.tex` | 会议格式 LaTeX 源文件（可直接编译为 PDF） |
| `references.bib` | BibTeX 参考文献（经过引用验证） |
| `code/main.py` | 自动生成的实验代码 |
| `code/requirements.txt` | 实验代码的 Python 依赖 |
| `charts/` | 实验结果可视化图表（PNG 格式） |
| `verification_report.json` | 引用完整性验证报告 |
| `manifest.json` | 交付物清单及元信息 |

### 🔎 重点检查项

1. **论文内容**（`paper_final.md` 或 `paper.tex`）
   - 标题是否合理、与主题相关
   - 摘要是否清晰概述了问题、方法、结果
   - 相关工作是否引用了该领域的关键文献
   - 方法描述是否清晰、技术上正确
   - 实验设计是否合理（数据集、baselines、评估指标）
   - 结果是否有意义（不是全零、不是 NaN）
   - 结论是否与实验结果一致

2. **实验代码**（`code/main.py`）
   - 代码是否能独立运行
   - 是否使用了真实数据集（而非随机生成的假数据）
   - 是否实现了论文中描述的方法
   - 是否包含合理的超参数设置

3. **图表**（`charts/`）
   - 图表是否清晰可读
   - 坐标轴标签是否正确
   - 数据是否与论文描述一致

4. **引用**（`references.bib`）
   - 引用的论文是否真实存在
   - 引用是否与论文讨论的内容相关

### 📊 自动质量评估报告

Pipeline 会自动生成一份质量评估报告，位于 `stage-20/quality_report.json`，其中包含：

- `score_1_to_10` — 自动评分
- `verdict` — 接收/拒绝建议
- `strengths` — 优点列表
- `weaknesses` — 缺点列表
- `required_actions` — 建议的改进事项

请在你的反馈报告中参考此评估，并补充你自己的专业判断。

---

## 📝 反馈报告要求

**你的反馈是本项目改进的核心依据。** 无论是批评还是肯定，对我们都同样重要——请务必认真、详细地填写。

### 需要提交的内容

| # | 提交内容 | 说明 |
|---|---------|------|
| F1 | **反馈报告**（按下方模板填写） | Markdown 格式，命名为 `feedback_<你的名字>.md` |
| F2 | **完整输出目录** | 将整个 `artifacts/rc-XXXXXX/` 目录打包提交（`.zip` 或 `.tar.gz`） |
| F3 | **配置文件** | 你使用的 `config.yaml`（**删除 API Key 后**提交） |
| F4 | **终端日志**（可选但推荐） | 运行时的终端输出，便于我们排查问题 |

### 反馈的四个维度

#### 🎯 (a) 质量评价

请从你的专业领域角度评价产出论文的质量：

- 如果这是你所在领域的论文，它能达到什么水平？（顶会 / 一般会议 / 无法发表）
- 与你读过的该领域论文相比，写作质量如何？
- 方法的技术正确性如何？有无明显错误？
- 实验设计的合理性如何？

#### 💡 (b) 优化建议

请指出你认为可以改进的地方：

- 哪个阶段的输出质量最差？（文献检索 / 实验设计 / 代码生成 / 论文撰写）
- 代码中有没有明显写错或不合理的地方？
- 论文结构或表述有什么具体的改进建议？

#### ⚖️ (c) 合理性评估

请评估 Pipeline 流程的合理性：

- 23 个阶段的设计是否合理？有没有多余或缺失的步骤？
- 实验迭代优化的过程是否有效？
- LLM 生成内容的引导方式是否合理？

#### 🐛 (d) Bug 报告

请尽可能详细地报告你发现的任何问题：

- **写作 Bug**：语法错误、重复段落、前后矛盾、引用不存在的图表
- **代码 Bug**：运行报错、逻辑错误、数据处理问题
- **结果 Bug**：全零结果、NaN 值、指标不合理
- **流程 Bug**：阶段卡住、异常中断、资源耗尽

---

## 📋 反馈报告模板

请复制以下模板，填写后保存为 `feedback_<你的名字>.md`：

````markdown
# AutoResearchClaw 测试反馈报告

## 基本信息

- **测试人员**：
- **所属领域**：（例如：计算机视觉 / 自然语言处理 / 强化学习 / 生物信息 / ...）
- **测试日期**：
- **代码版本**：（运行 `git log --oneline -1` 的输出，例如：`44151b1 fix: Phase 3 regression test findings`）
- **研究主题（英文）**：
- **使用的 LLM 模型**：（例如：gpt-5.4 / gpt-5.1 / claude-opus-4-6 / claude-sonnet-4-6）
- **实验模式**：（sandbox / docker）
- **运行总时长**：（约 X 分钟）
- **是否成功完成 23 个阶段**：是 / 否（如否，请说明卡在哪个阶段）

---

## 一、质量评价（总分 1-10）

**我的评分**：X / 10

### 1.1 论文整体质量
- 相当于什么级别的论文？（顶会 / 一般会议 / workshop / 无法发表）
- 简要说明评分理由：

### 1.2 各部分质量评价

| 部分 | 评分 (1-10) | 评价说明 |
|------|-----------|---------|
| 标题 | | |
| 摘要 | | |
| 引言 | | |
| 相关工作 | | |
| 方法 | | |
| 实验设计 | | |
| 结果与分析 | | |
| 结论 | | |
| 参考文献 | | |
| 图表质量 | | |
| 代码质量 | | |

### 1.3 与人工撰写论文的对比
- 与你平时阅读/撰写的论文相比，差距在哪里？
- 有哪些方面出乎意料地好？

---

## 二、优化建议

### 2.1 最需要改进的环节
（请列出 3-5 个最需要改进的具体问题，按优先级排序）

1.
2.
3.

### 2.2 代码问题
- 代码是否能独立运行？
- 是否使用了真实数据集和基线方法？
- 具体代码问题（如有）：

### 2.3 写作问题
- 论文结构是否合理？
- 技术描述是否准确？
- 具体写作问题（如有）：

---

## 三、合理性评估

### 3.1 Pipeline 流程评价
- 23 个阶段的流程设计是否合理？
- 有没有你认为多余或缺失的步骤？

### 3.2 实验执行评价
- 实验设计是否合理？（数据集选择、对比方法、评估指标）
- 迭代优化过程是否有效？

### 3.3 LLM 使用评价
- LLM 在各阶段的表现如何？
- 有没有明显的"幻觉"或不合理的生成内容？

---

## 四、Bug 报告

### 4.1 写作 Bug
| 编号 | 位置（章节/段落） | 描述 | 严重程度 (高/中/低) |
|------|-----------------|------|-------------------|
| W1 | | | |
| W2 | | | |

### 4.2 代码 Bug
| 编号 | 文件/行号 | 描述 | 严重程度 (高/中/低) |
|------|----------|------|-------------------|
| C1 | | | |
| C2 | | | |

### 4.3 结果 Bug
| 编号 | 描述 | 涉及指标/图表 | 严重程度 (高/中/低) |
|------|------|-------------|-------------------|
| R1 | | | |
| R2 | | | |

### 4.4 流程 Bug
| 编号 | 阶段 | 描述 | 严重程度 (高/中/低) |
|------|------|------|-------------------|
| P1 | | | |
| P2 | | | |

---

## 五、其他建议

（自由发挥：任何你觉得有价值的观察、建议或想法）

---

## 附件清单

- [ ] 反馈报告 (`feedback_<名字>.md`)
- [ ] 完整输出目录 (`artifacts/rc-XXXXXX.zip`)
- [ ] 配置文件 (`config.yaml`，已删除 API Key)
- [ ] 终端日志（可选）
````

---

## ❓ 常见问题

### Q1: 没有 GPU 能测试吗？

**当然可以！** 使用 `experiment.mode: "sandbox"` 模式，Pipeline 会在本地 CPU 上运行实验。虽然实验规模会受限，但足以完成一次完整的端到端测试。

### Q2: API 调用大概要花多少钱？

一次完整的 Pipeline 运行约消耗 **$5–15** 的 API 费用，取决于所选模型、论文修订次数和实验复杂度。顶级模型（GPT-5.4、Claude Opus 4.6）费用稍高，但产出质量显著更好，推荐优先使用。

### Q3: Pipeline 运行中断了怎么办？

从断点继续即可：

```bash
researchclaw run --config config.yaml --resume
```

### Q4: 可以用中文主题吗？

建议使用 **英文** 描述你的研究主题。Pipeline 的提示词、文献检索和论文生成均以英文为主。如果你的 idea 原始语言是中文，请先翻译成英文。

### Q5: 我应该选什么样的研究主题？

选择你**熟悉的领域内的一个具体研究问题**——这样你才能有效评估论文的技术正确性。建议：

- ✅ 选择有明确实验验证方法的主题（分类、回归、强化学习任务等）
- ❌ 避免过于宏大或抽象的主题（如 "AGI" 或 "通用人工智能"）
- ✅ 描述要具体，例如：*"Investigating the effect of data augmentation strategies on few-shot learning for medical image classification"*

### Q6: 如何使用 Docker 模式？（进阶）

如果你有 NVIDIA GPU 并安装了 Docker + NVIDIA Container Toolkit：

```bash
# 1. 构建实验镜像
docker build -t researchclaw/experiment:latest researchclaw/docker/

# 2. 修改 config.yaml:
#   experiment:
#     mode: "docker"
#     docker:
#       gpu_enabled: true
#       memory_limit_mb: 8192
#       network_policy: "setup_only"  # 推荐默认值

# 3. 运行
researchclaw run --config config.yaml --auto-approve
```

Docker 模式采用三阶段执行：pip install（联网）→ setup.py（联网）→ 实验代码（断网）。镜像已预缓存常用数据集（CIFAR-10/100、MNIST、FashionMNIST、STL-10、SVHN），标准基准测试无需网络。

### Q7: 我之前已经测试过了，再次测试需要注意什么？

**每次测试前务必拉取最新代码：**

```bash
cd AutoResearchClaw
git pull origin main
pip install -e .
```

然后确认版本号：

```bash
git log --oneline -1
```

不同版本的生成效果可能差异很大，请在反馈报告中注明你使用的 commit hash。

### Q8: 反馈提交到哪里？

你可以通过以下任一渠道提交反馈：

- **GitHub Issues：** [提交 Issue](https://github.com/aiming-lab/AutoResearchClaw/issues)，添加 `feedback` 标签
- **Pull Request：** 将 `feedback_<名字>.md` 提交到 `community-feedback/` 目录
- **邮件：** 联系项目维护者（详见仓库主页）

---

## 🌍 我们需要来自各个领域的测试者

目前 Pipeline 主要在机器学习领域进行了测试，我们特别欢迎来自以下领域的测试者：

- 🧬 **生物信息学与计算生物学**
- 🧪 **化学与材料科学**
- 📊 **统计学与应用数学**
- 🤖 **机器人学与控制系统**
- 🗣️ **NLP 与计算语言学**
- 👁️ **计算机视觉与图形学**
- 🎮 **强化学习与博弈论**
- 🏥 **医学 AI 与医疗健康**
- 🌐 **图学习与网络科学**
- 💹 **金融 ML 与计量经济学**
- 🛰️ **遥感与地理空间 AI**

……以及任何涉及计算实验的领域！

---

## 🙏 感谢你的参与

你的每一条反馈——无论大小——都在直接推动 AutoResearchClaw 变得更好。感谢你成为这段旅程的一部分。

<p align="center">
  <b>⭐ 如果你觉得这个项目有趣，请在 <a href="https://github.com/aiming-lab/AutoResearchClaw">GitHub</a> 上给我们一颗 Star！</b>
</p>
