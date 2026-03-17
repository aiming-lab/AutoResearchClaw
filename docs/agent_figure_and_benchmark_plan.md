# Multi-Agent Figure Generation & Benchmark Selection — Task Requirements

> **Created**: 2026-03-15
> **Updated**: 2026-03-15
> **Status**: BenchmarkAgent IMPLEMENTED, FigureAgent IMPLEMENTED
> **Scope**: Two new multi-agent subsystems for AutoResearchClaw pipeline
>
> **Implementation Progress**:
> - [x] Part B: BenchmarkAgent — fully implemented (4 agents + orchestrator + config + pipeline integration + 43 tests)
> - [x] Part A: FigureAgent — fully implemented (5 agents + orchestrator + config + pipeline integration + 45 tests)
>
> **Key Research Findings (supplemental)**:
> - Papers With Code was shut down by Meta in July 2025; HuggingFace Hub API is now the primary dataset discovery source
> - AI Scientist v2 and MLR-Copilot both use pure LLM-driven dataset selection (no API search) — our API-based approach is more structured
> - MLE-bench (OpenAI) validates the pre-download + container-mount pattern (matches our `setup_only` network policy)
> - CodeSOTA (codesota.com) provides a lighter-weight benchmark database as an alternative to Papers With Code

---

## Executive Summary

当前 Pipeline 的图表生成和数据集/基准选择存在根本性缺陷：

**图表问题**（实测产出）：
- 每次固定只生成 2 张图（`method_comparison.png` + `experiment_comparison.png`）
- 图表类型单一：只有柱状图，无训练曲线、热力图、消融分析图等
- 数据无差异化：所有方法都显示 1.000，完全无信息量
- 样式简陋：默认 matplotlib 风格，远低于 AI 顶会标准
- 不适应实验内容：无论做什么研究都画一样的图
- DPI=150，不满足出版要求（300+ DPI）

**数据集/基准问题**：
- 当前仅通过 `dataset_guidance` 提示词列出预缓存数据集
- 无法根据研究领域动态搜索和选择最合适的 benchmark
- 无法自动下载非预缓存数据集
- 缺乏 baseline 方法的自动复现能力

**解决方案**：设计两个独立的多 Agent 子系统：
1. **FigureAgent** — 智能图表生成系统（6 个子 Agent 协作）
2. **BenchmarkAgent** — 数据集与基准选择系统（4 个子 Agent 协作）

---

## Part A: FigureAgent — 多 Agent 图表生成系统

### A.1 问题分析

#### 当前架构缺陷

```
现状：Stage 14 → visualize.py (5 个硬编码函数) → 固定 2 张图 → Stage 17/22 嵌入论文
```

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 图表类型固定 | Critical | 只有 bar chart 和 line chart，缺少 heatmap、scatter、violin、architecture diagram 等 |
| 不适应实验内容 | Critical | 知识蒸馏实验和 RL 实验画的图完全一样 |
| 无智能决策 | Critical | 不分析"应该画什么"，直接调用固定函数 |
| 数据正确性无验证 | High | 不验证图中数据是否与实验结果一致 |
| 样式不达标 | High | 默认 matplotlib，不符合学术论文视觉标准 |
| 无架构图能力 | High | 不能生成方法流程图 / 模型架构图（顶会 Figure 1 必备） |
| DPI 不足 | Medium | 150 DPI，出版要求 300+ |
| 无 VLM 审查 | Medium | 生成后不检查质量，直接用 |

#### 业界参考方案

| 项目 | 图表策略 | 核心创新 |
|------|---------|---------|
| AI Scientist v1 (Sakana) | 人工编写 `plot.py` 模板，LLM 不参与 | 可靠但不灵活 |
| AI Scientist v2 (Sakana) | LLM 自主生成画图代码 + VLM 审查反馈循环 | **VLM-as-critic**，首篇通过 ICLR workshop 审稿 |
| PlotGen (Adobe) | 三模态反馈：数值准确性 + 文本正确性 + 视觉质量 | **Tri-modal feedback**，MatPlotBench 最优 |
| PaperBanana (Google) | 3 阶段 pipeline：Caption 精炼 → 参考检索 → 迭代渲染 | **Caption sharpening** + 参考图库 |

### A.2 目标架构

```
                          ┌─────────────────────┐
                          │   FigureAgent        │
                          │   (Orchestrator)     │
                          └──────────┬──────────┘
                                     │
              ┌──────────┬───────────┼───────────┬──────────┐
              ▼          ▼           ▼           ▼          ▼
        ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
        │ Planner  ││ CodeGen  ││ Renderer ││ Critic   ││ Integra- │
        │ Agent    ││ Agent    ││ Agent    ││ Agent    ││ tor Agent│
        └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘
              │          │           │           │          │
              ▼          ▼           ▼           ▼          ▼
         图表规划     代码生成     执行渲染     质量审查    论文嵌入
```

#### Agent 职责定义

**1. Orchestrator（编排器）**
- 接收：实验结果 JSON、论文草稿 markdown、研究主题描述
- 协调所有子 Agent 的执行顺序
- 管理迭代循环（Critic 不满意时回到 CodeGen）
- 输出：最终图表集合 + 嵌入指令

**2. Planner Agent（图表规划）**
- 输入：实验结果数据结构、论文 idea、研究领域
- 职责：
  - 分析实验数据，确定需要哪些图、每张图展示什么
  - 为每张图生成精确的 caption specification（非模糊描述）
  - 确定图表类型（bar / line / heatmap / scatter / architecture / ablation 等）
  - 确定布局（single / subplot / multi-panel）
  - 输出图表规划清单（JSON 格式）
- 关键规则：
  - 至少规划 4 张图：1 架构图 + 1 主结果图 + 1 消融图 + 1 分析图
  - 根据研究领域自动选择合适的图表类型
  - Caption sharpening：将模糊描述转化为精确视觉规范

**3. CodeGen Agent（代码生成）**
- 输入：Planner 输出的图表规划 + 实验数据
- 职责：
  - 为每张图生成独立的 Python 绘图脚本
  - 使用 SciencePlots 学术样式 (`plt.style.use(['science', 'ieee'])`)
  - 确保 colorblind-safe 配色
  - 300+ DPI 输出
  - 代码保存到 `charts/scripts/` 供复现
- 代码模板库：
  - 内置常用学术图表模板（training curve, bar comparison, heatmap, confusion matrix 等）
  - 新图表可基于模板扩展

**4. Renderer Agent（渲染执行）**
- 输入：CodeGen 生成的 Python 脚本
- 职责：
  - 在 Docker sandbox 中执行绘图脚本
  - 捕获执行错误并反馈给 CodeGen 修复
  - 验证输出文件存在且可读
  - 检查图像尺寸和分辨率

**5. Critic Agent（质量审查 — 三模态反馈）**
- 输入：渲染后的图像 + 源数据 + caption 规范
- 职责（三维度审查，参考 PlotGen）：
  - **数值准确性**：验证图中呈现的数值与源数据一致（读取 JSON → 对比图中数据点）
  - **文本正确性**：检查标题、坐标轴标签、图例是否准确完整
  - **视觉质量**：通过 VLM（如 GPT-4o vision）审查整体美观度、可读性、学术规范
- 输出：pass / fail + 具体修改建议
- 如果 fail：将反馈发回 CodeGen Agent，最多迭代 3 次

**6. Integrator Agent（论文嵌入）**
- 输入：通过审查的图表集合 + 论文草稿
- 职责：
  - 确定每张图在论文中的最佳位置
  - 生成 LaTeX figure 环境代码（支持 subfigure 多面板）
  - 生成交叉引用（`\ref{fig:xxx}`）
  - 确保图表在正确的 section（架构图在 Method，结果图在 Results）
  - 更新论文文本中的图表引用语句

### A.3 图表类型矩阵

根据研究领域和实验类型，Planner Agent 应遵循以下决策矩阵：

| 实验类型 | 必须包含的图表 | 可选图表 |
|---------|--------------|---------|
| **分类任务** | 精度对比 bar chart、confusion matrix | ROC 曲线、t-SNE 可视化 |
| **生成模型** | 生成样本 grid、FID/IS 曲线 | 插值可视化、attention map |
| **强化学习** | reward curve (mean±std shading)、episode length | 策略可视化、环境截图 |
| **知识蒸馏** | teacher-student 精度对比、知识迁移效率曲线 | 特征对齐热力图 |
| **NLP** | BLEU/ROUGE 对比表、attention heatmap | 样本输出对比 |
| **图神经网络** | 节点分类精度、图可视化 | 消息传递可视化 |
| **元学习** | few-shot 精度 vs shot 数曲线 | 任务适应速度 |
| **持续学习** | 遗忘率曲线、任务精度矩阵 | 表征漂移可视化 |
| **所有类型** | 消融分析 (grouped bar)、训练 loss 曲线 | 超参敏感性热力图 |

### A.4 样式规范

所有图表必须遵循以下学术出版标准：

```python
# 全局样式配置 (charts/style_config.py)
STYLE_CONFIG = {
    "matplotlib_style": ["science", "ieee"],   # SciencePlots
    "dpi": 300,                                 # 出版级
    "font_size": {"title": 12, "axis": 10, "tick": 8, "legend": 9},
    "figure_width": {
        "single_column": 3.5,   # IEEE single column (inches)
        "double_column": 7.0,   # IEEE double column
        "full_page": 7.0,       # Full width
    },
    "colors": "bright",          # colorblind-safe (Paul Tol)
    "line_styles": ["-", "--", "-.", ":"],  # 配合 B&W 打印
    "marker_styles": ["o", "s", "^", "D", "v", "P"],
    "error_bar_style": "shading",  # mean ± std 用阴影而非 error bar
    "format": "pdf",               # 矢量格式优先
    "fallback_format": "png",      # PNG 备用
}
```

### A.5 实现计划

#### 文件结构

```
researchclaw/
├── agents/
│   └── figure_agent/
│       ├── __init__.py
│       ├── orchestrator.py      # FigureAgent 主编排器
│       ├── planner.py           # Planner Agent
│       ├── codegen.py           # CodeGen Agent
│       ├── renderer.py          # Renderer Agent
│       ├── critic.py            # Critic Agent (三模态审查)
│       ├── integrator.py        # Integrator Agent
│       ├── templates/           # 图表代码模板库
│       │   ├── bar_comparison.py
│       │   ├── training_curve.py
│       │   ├── heatmap.py
│       │   ├── confusion_matrix.py
│       │   ├── scatter_plot.py
│       │   ├── ablation_grouped.py
│       │   ├── violin_box.py
│       │   └── multi_panel.py
│       └── style_config.py      # 全局样式配置
```

#### 开发任务清单

| ID | 任务 | 依赖 | 估计改动量 |
|----|------|------|-----------|
| FA-01 | 创建 `agents/figure_agent/` 目录结构和基础类 | 无 | 新建 |
| FA-02 | 实现 Planner Agent：图表规划逻辑 + 类型决策矩阵 | FA-01 | ~300 行 |
| FA-03 | 实现 CodeGen Agent：代码生成 + 模板库 | FA-01 | ~500 行 |
| FA-04 | 实现 Renderer Agent：sandbox 执行 + 错误处理 | FA-01, FA-03 | ~200 行 |
| FA-05 | 实现 Critic Agent：三模态审查（数值 / 文本 / VLM） | FA-01, FA-04 | ~400 行 |
| FA-06 | 实现 Integrator Agent：论文嵌入 + LaTeX subfigure 支持 | FA-01 | ~250 行 |
| FA-07 | 实现 Orchestrator：编排循环 + 最大迭代控制 | FA-02 ~ FA-06 | ~300 行 |
| FA-08 | 添加 SciencePlots 到 Docker 镜像 + 样式配置 | 无 | ~50 行 |
| FA-09 | 修改 executor.py：Stage 14 调用 FigureAgent 替代 `visualize.py` | FA-07 | ~100 行 |
| FA-10 | 修改 executor.py：Stage 17/22 使用 Integrator 输出 | FA-07 | ~100 行 |
| FA-11 | 修改 converter.py：支持 subfigure 和 PDF 格式 | FA-06 | ~80 行 |
| FA-12 | 添加图表代码模板库（8+ 模板） | FA-03 | ~600 行 |
| FA-13 | 测试：单元测试 + 集成测试 | FA-01 ~ FA-12 | ~400 行 |
| FA-14 | 向后兼容：保留 `visualize.py` 作为 fallback | FA-09 | ~30 行 |

#### Pipeline 集成点

```
Stage 12-13: 实验执行完成，生成 results.json
      │
      ▼
Stage 14: Result Analysis
      │── 调用 FigureAgent.orchestrate()
      │   ├── Planner: 分析 results.json → 图表规划
      │   ├── CodeGen: 生成绘图脚本 → charts/scripts/
      │   ├── Renderer: 执行脚本 → charts/*.pdf + charts/*.png
      │   ├── Critic: 审查图表质量 (max 3 iterations)
      │   └── 输出: charts/ 目录 + figure_manifest.json
      │
      ▼
Stage 17: Paper Draft
      │── Integrator: 读取 figure_manifest.json
      │   ├── 确定每张图的论文位置
      │   ├── 注入 markdown 图片引用 + caption
      │   └── 更新交叉引用文本
      │
      ▼
Stage 22: Paper Export
      │── 复制 charts/ 到 submission/
      │── converter.py 处理 subfigure 环境
      └── 最终 LaTeX 编译验证
```

---

## Part B: BenchmarkAgent — 多 Agent 数据集与基准选择系统

### B.1 问题分析

#### 当前架构缺陷

```
现状：dataset_guidance 提示词 (硬编码列表) + dataset_registry.yaml (静态清单) → LLM 自行选择
```

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 数据集选择不智能 | Critical | 仅列出预缓存数据集，LLM 可能选择不合适的 benchmark |
| 无领域适配 | Critical | 不根据研究领域搜索该领域的标准 benchmark |
| 无最新性保证 | High | 不检查是否有更新、更好的 benchmark 可用 |
| baseline 无法复现 | High | 不提供已有方法的参考实现 / 预训练权重 |
| 下载路径硬编码 | Medium | 非预缓存数据集无法自动获取 |
| 无数据集验证 | Medium | 不验证下载的数据集是否完整、格式正确 |

#### 理想工作流

一个好的数据集/基准选择流程应该：
1. **理解研究问题** → 确定评估维度（分类精度？生成质量？推理速度？）
2. **搜索领域标准** → 查找该领域顶会论文常用的 benchmark
3. **评估适用性** → 数据集大小、难度、License、可获取性
4. **获取数据** → 自动下载或生成下载脚本
5. **获取 baseline** → 找到对比方法的开源实现或预训练权重
6. **验证完整性** → 确认数据集可正常加载和使用

### B.2 目标架构

```
                          ┌─────────────────────┐
                          │  BenchmarkAgent      │
                          │  (Orchestrator)      │
                          └──────────┬──────────┘
                                     │
              ┌──────────┬───────────┼───────────┐
              ▼          ▼           ▼           ▼
        ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
        │ Surveyor ││ Selector ││ Acquirer ││ Validator│
        │ Agent    ││ Agent    ││ Agent    ││ Agent    │
        └──────────┘└──────────┘└──────────┘└──────────┘
              │          │           │           │
              ▼          ▼           ▼           ▼
         领域调研     选择决策     数据获取     验证确认
```

#### Agent 职责定义

**1. Orchestrator（编排器）**
- 接收：研究主题、假设、实验设计方案
- 协调 4 个子 Agent 的执行
- 输出：`benchmark_plan.json`（包含数据集列表、下载脚本、baseline 方案）

**2. Surveyor Agent（领域调研）**
- 输入：研究主题关键词、相关文献列表
- 职责：
  - 搜索 Papers With Code 的领域 benchmark 排行榜
  - 搜索 HuggingFace Datasets 的相关数据集
  - 搜索 OpenML、Kaggle 的相关 benchmark
  - 分析近 2 年顶会论文（ICML、NeurIPS、ICLR）使用的数据集
  - 汇总领域标准 benchmark 清单（含引用频次、数据规模、难度级别）
- 输出：`survey_results.json` — 候选 benchmark 列表（按推荐度排序）
- 数据源优先级：
  1. Papers With Code (Benchmarks API)
  2. HuggingFace Datasets Hub
  3. torchvision / torchaudio / torchtext 内置
  4. 顶会论文附录中的数据集描述

**3. Selector Agent（选择决策）**
- 输入：survey_results.json + 实验约束（GPU 内存、时间预算、网络可用性）
- 职责：
  - 根据约束过滤不可行的数据集（太大 / 需要申请 / License 不兼容）
  - 考虑 Docker sandbox 已缓存的数据集（优先使用）
  - 选择 primary benchmark（必须是领域标准）+ secondary benchmarks（补充验证）
  - 选择 baseline 方法（至少 2 个有开源实现的对比方法）
  - 生成选择理由文档（供论文 Experimental Setup section 使用）
- 约束规则：
  - Tier 1（已缓存）：无网络需求，最优先
  - Tier 2（torchvision/HF datasets 可直接下载）：需 setup 阶段网络
  - Tier 3（需自定义下载脚本）：仅在 `network_policy: full` 时可用
- 输出：`selected_benchmarks.json` + `baseline_methods.json`

**4. Acquirer Agent（数据获取）**
- 输入：selected_benchmarks.json
- 职责：
  - 生成 `setup.py` 中的数据集下载代码
  - 为每个数据集生成加载 boilerplate 代码
  - 为 baseline 方法生成安装和调用代码
  - 处理 HuggingFace `datasets.load_dataset()` / `torchvision.datasets` 等接口
  - 生成 `requirements.txt` 中需要额外安装的包
- 输出：
  - `data_loading_snippets.py` — 数据加载代码片段（注入 CodeAgent）
  - `baseline_snippets.py` — baseline 调用代码片段
  - `setup.py` 追加内容 — 下载脚本

**5. Validator Agent（验证确认）**
- 输入：Acquirer 生成的下载/加载代码
- 职责：
  - 验证数据集 API 调用语法正确
  - 验证数据集分割（train/val/test）存在
  - 验证数据格式与实验代码兼容
  - 验证 baseline 方法可运行
  - 如果验证失败，反馈给 Acquirer 修复
- 输出：validation_report.json

### B.3 知识库设计

BenchmarkAgent 需要一个结构化知识库来支持决策：

```yaml
# researchclaw/data/benchmark_knowledge.yaml

domains:
  image_classification:
    standard_benchmarks:
      - name: CIFAR-10/100
        source: torchvision
        tier: 1  # 已缓存
        difficulty: easy/medium
        use_when: "小规模验证、快速原型"
      - name: ImageNet-1K
        source: torchvision
        tier: 3  # 需要下载 ~150GB
        difficulty: hard
        use_when: "大规模验证、与 SOTA 对比"
    common_baselines:
      - name: ResNet-50
        source: "torchvision.models.resnet50(pretrained=True)"
        paper: "He et al., 2016"
      - name: ViT-B/16
        source: "timm.create_model('vit_base_patch16_224', pretrained=True)"
        paper: "Dosovitskiy et al., 2021"

  reinforcement_learning:
    standard_benchmarks:
      - name: Gymnasium (MuJoCo)
        source: "gymnasium[mujoco]"
        tier: 2
      - name: Atari
        source: "gymnasium[atari]"
        tier: 2
    common_baselines:
      - name: PPO
        source: "stable-baselines3"
        paper: "Schulman et al., 2017"

  # ... 更多领域
```

### B.4 实现计划

#### 文件结构

```
researchclaw/
├── agents/
│   └── benchmark_agent/
│       ├── __init__.py
│       ├── orchestrator.py      # BenchmarkAgent 主编排器
│       ├── surveyor.py          # Surveyor Agent (领域调研)
│       ├── selector.py          # Selector Agent (选择决策)
│       ├── acquirer.py          # Acquirer Agent (数据获取)
│       ├── validator.py         # Validator Agent (验证确认)
│       └── knowledge_base.py    # 知识库加载和查询
├── data/
│   ├── benchmark_knowledge.yaml # 领域 benchmark 知识库
│   └── dataset_registry.yaml    # 已有数据集注册表 (保留)
```

#### 开发任务清单

| ID | 任务 | 依赖 | 估计改动量 |
|----|------|------|-----------|
| BA-01 | 创建 `agents/benchmark_agent/` 目录结构和基础类 | 无 | 新建 |
| BA-02 | 编写 `benchmark_knowledge.yaml` 知识库（覆盖 10+ 领域） | 无 | ~500 行 YAML |
| BA-03 | 实现 Surveyor Agent：Papers With Code API + HF Datasets 搜索 | BA-01 | ~350 行 |
| BA-04 | 实现 Selector Agent：约束过滤 + Tier 匹配 + 选择逻辑 | BA-01, BA-02 | ~300 行 |
| BA-05 | 实现 Acquirer Agent：代码生成 + 下载脚本 | BA-01, BA-04 | ~350 行 |
| BA-06 | 实现 Validator Agent：语法/可用性验证 | BA-01, BA-05 | ~250 行 |
| BA-07 | 实现 Orchestrator：编排 + 迭代修复 | BA-02 ~ BA-06 | ~250 行 |
| BA-08 | 修改 executor.py：Stage 6/7 调用 BenchmarkAgent | BA-07 | ~150 行 |
| BA-09 | 修改 executor.py：将 benchmark_plan 注入 CodeAgent | BA-07 | ~100 行 |
| BA-10 | 更新 prompts.py：基于 BenchmarkAgent 输出动态构建提示词 | BA-07 | ~100 行 |
| BA-11 | 测试：单元测试 + 集成测试 | BA-01 ~ BA-10 | ~300 行 |
| BA-12 | 向后兼容：保留 `dataset_registry.yaml` 作为 fallback | BA-08 | ~30 行 |

#### Pipeline 集成点

```
Stage 3: Topic Initialization
      │── 研究主题确定
      ▼
Stage 4-5: Literature Collection & Screening
      │── 文献列表生成
      ▼
Stage 6: Hypothesis Generation
      │── 调用 BenchmarkAgent.orchestrate()
      │   ├── Surveyor: 搜索领域标准 benchmark
      │   ├── Selector: 根据约束选择最优 benchmark + baseline
      │   ├── Acquirer: 生成下载/加载代码
      │   └── Validator: 验证代码可执行
      │── 输出: benchmark_plan.json
      ▼
Stage 7: Experiment Design
      │── benchmark_plan.json 注入实验设计
      │── 实验方案明确使用哪些数据集和 baseline
      ▼
Stage 8-9: Code Generation (CodeAgent)
      │── data_loading_snippets 注入生成代码
      │── baseline_snippets 注入对比方法
      ▼
Stage 10-11: Experiment Execution
      │── setup.py 执行数据集下载
      │── main.py 使用生成的数据加载代码
      ▼
Stage 14: Result Analysis
      │── 对比结果基于真实 baseline，可信度高
```

---

## Part C: 共同基础设施

### C.1 Agent 基类

两个多 Agent 系统共享同一套基础设施：

```python
# researchclaw/agents/base.py

class BaseAgent:
    """所有子 Agent 的基类"""
    def __init__(self, llm_client, config):
        self.llm = llm_client
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(self, context: dict) -> dict:
        """执行 Agent 任务，返回结果"""
        raise NotImplementedError

    def _call_llm(self, system_prompt, user_prompt, **kwargs):
        """统一 LLM 调用接口"""
        return self.llm.chat(system_prompt, user_prompt, **kwargs)


class AgentOrchestrator:
    """Agent 编排器基类"""
    def __init__(self, agents: list[BaseAgent], max_iterations=3):
        self.agents = agents
        self.max_iterations = max_iterations

    async def orchestrate(self, context: dict) -> dict:
        """执行多 Agent 编排流程"""
        raise NotImplementedError
```

### C.2 与现有 LLM Client 的集成

两个系统都通过现有的 `researchclaw/llm/client.py` 调用 LLM：
- Planner / Selector / Critic 等决策类 Agent → 使用 `gpt-4.1` 或 `gpt-4o`
- CodeGen 类 Agent → 使用 `gpt-4.1`（代码生成能力最强）
- VLM Critic → 使用 `gpt-4o`（支持 vision）

### C.3 配置扩展

```yaml
# config.yaml 新增配置
agents:
  figure_agent:
    enabled: true
    max_iterations: 3          # Critic 反馈最大迭代次数
    min_figures: 4             # 最少图表数
    style: "science+ieee"      # matplotlib 样式
    dpi: 300
    format: "pdf"              # 优先格式
    vlm_review: true           # 是否启用 VLM 视觉审查
  benchmark_agent:
    enabled: true
    max_search_results: 20     # Papers With Code 最大搜索结果
    prefer_cached: true        # 优先使用已缓存数据集
    tier_limit: 2              # 最高允许的 Tier 级别 (1=缓存, 2=可下载, 3=大型)
    min_baselines: 2           # 最少 baseline 方法数
```

---

## Part D: 风险与兜底

### D.1 向后兼容

| 组件 | 兜底策略 |
|------|---------|
| FigureAgent 失败 | 回退到现有 `visualize.py` 生成基础图表 |
| BenchmarkAgent 失败 | 回退到 `dataset_registry.yaml` + `dataset_guidance` 提示词 |
| VLM 审查不可用 | 跳过视觉审查，仅做数值 + 文本验证 |
| SciencePlots 未安装 | 使用 `seaborn-v0_8-whitegrid` 样式 |
| 网络不可用 | Surveyor 使用本地 `benchmark_knowledge.yaml` |

### D.2 Token 成本控制

| 操作 | 预估 Token 消耗 | 控制策略 |
|------|----------------|---------|
| Planner (1 次) | ~2K input + ~1K output | 固定 |
| CodeGen (4 图 × 最多 3 次迭代) | ~3K × 12 = ~36K | 迭代次数上限 |
| Critic (4 图 × 最多 3 次) | ~2K × 12 = ~24K | 迭代次数上限 |
| VLM 审查 (4 图) | ~4K × 4 = ~16K | 仅终轮审查 |
| Surveyor (1 次) | ~2K input + ~2K output | API 调用为主 |
| Selector (1 次) | ~3K input + ~1K output | 固定 |
| **总增量** | **~80K tokens** | 约增加 $0.30-0.50/run |

### D.3 测试策略

1. **单元测试**：每个 Agent 独立测试（mock LLM 响应）
2. **集成测试**：使用固定 results.json 测试 FigureAgent 完整流程
3. **回归测试**：确认 fallback 到旧系统仍可正常工作
4. **端到端测试**：Run 14+ 完整 Pipeline 运行，对比图表质量

---

## Part E: 执行优先级

建议按以下顺序实施：

### Phase 1: FigureAgent 核心（优先级最高）
1. FA-01 ~ FA-03: 基础类 + Planner + CodeGen
2. FA-04 ~ FA-05: Renderer + Critic
3. FA-08: SciencePlots 集成
4. FA-12: 模板库

### Phase 2: FigureAgent 集成
5. FA-06 ~ FA-07: Integrator + Orchestrator
6. FA-09 ~ FA-11: Pipeline 集成
7. FA-13 ~ FA-14: 测试 + 向后兼容

### Phase 3: BenchmarkAgent 核心
8. BA-01 ~ BA-02: 基础类 + 知识库
9. BA-03 ~ BA-06: 4 个子 Agent
10. BA-07: Orchestrator

### Phase 4: BenchmarkAgent 集成
11. BA-08 ~ BA-10: Pipeline 集成
12. BA-11 ~ BA-12: 测试 + 向后兼容

### Phase 5: 端到端验证
13. 完整 Pipeline 运行（Run 14+）
14. 对比图表质量和数据集选择质量
15. 根据结果调优

---

## Appendix: 参考资料

| 来源 | 关键收获 |
|------|---------|
| [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | VLM-as-critic, 首篇通过 ICLR workshop 审稿 |
| [PlotGen (Adobe)](https://arxiv.org/abs/2502.00988) | 三模态反馈：数值 + 文本 + 视觉 |
| [PaperBanana (Google)](https://github.com/llmsresearch/paperbanana) | Caption sharpening + 参考图库检索 |
| [SciencePlots](https://github.com/garrettj403/SciencePlots) | 学术论文 matplotlib 样式库 |
| [VLM-Enhanced Discovery](https://arxiv.org/html/2511.14631) | Correction mode + Discovery mode |
| [Papers With Code API](https://paperswithcode.com/api/v1/) | 领域 benchmark 排行榜搜索 |
| [HuggingFace Datasets](https://huggingface.co/docs/datasets/) | 数据集搜索和加载 API |
