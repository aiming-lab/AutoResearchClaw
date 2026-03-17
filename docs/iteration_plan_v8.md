# AutoResearchClaw Pipeline — 持续迭代改进方案 V8

> 创建日期: 2026-03-15
> 基于: V7 质量修复 (P1-P14) + Run 1-7 测试反馈
> 目标: 将 pipeline 从 3/10 提升至 7+/10 审稿人评分

---

## 一、当前问题总览

### 1.1 已确认的核心问题

| ID | 问题 | 严重程度 | 类别 |
|----|------|----------|------|
| Q1 | **代码过于简单/偷懒** — LLM 生成的实验代码复杂度不足，缺乏真正的算法实现深度 | 🔴 Critical | 代码质量 |
| Q2 | **不支持 LLM 微调任务** — 无法使用 Llama-Factory/TRL/Axolotl 等框架进行模型训练 | 🔴 Critical | 能力缺失 |
| Q3 | **Docker 环境缺失关键包** — transformers, PEFT, TRL, datasets, accelerate 未预装 | 🔴 Critical | 基础设施 |
| Q4 | **计算预算不匹配** — 默认 600s 完全不够 LLM 微调/复杂训练任务 | 🟡 High | 配置 |
| Q5 | **数据集指导不全** — 只覆盖图像分类(CIFAR-10/FashionMNIST)，缺少 NLP/多模态数据集 | 🟡 High | 提示工程 |
| Q6 | **缺少先进训练技巧指导** — 无混合精度、梯度累积、LoRA/QLoRA 等指导 | 🟡 High | 提示工程 |
| Q7 | **选题缺乏前沿性验证** — topic_init 阶段无法确保选题与最新会议趋势对齐 | 🟡 High | 提示工程 |
| Q8 | **实验设计与代码脱节** — experiment_design 阶段产出的方案过于抽象，代码难以还原 | 🟠 Medium | 流程 |
| Q9 | **消融实验质量低** — 消融 variant 经常与 baseline 结果相同（代码偷懒） | 🟠 Medium | 代码质量 |
| Q10 | **论文写作质量待提升** — 数字重复、结构松散、结论与实验脱节 | 🟠 Medium | 写作 |

### 1.2 硬件环境

| 资源 | 配置 |
|------|------|
| GPU | NVIDIA RTX 6000 Ada (49GB VRAM) |
| 可训练模型 | Full FT: ≤3B; LoRA 16-bit: ≤14B; QLoRA 4-bit: ≤72B |
| 最优甜点 | Qwen-2.5-14B + QLoRA (rank 32, batch 2-4, seq 4096) |
| 极限模型 | Qwen-2.5-72B + QLoRA (rank 8-16, batch 1, seq 1024) |
| 推荐框架 | Llama-Factory (内置 Unsloth 加速)、TRL (RLHF/DPO) |

---

## 二、迭代方案概览

### 总体路线

```
Phase 0: 诊断测试 (Run 8-10)     ← 当前阶段
    ↓ 发现问题
Phase 1: 代码质量根本性改进
    ↓ 修复后
Phase 2: LLM 微调能力扩展
    ↓ 新能力
Phase 3: 回归测试 (Run 11-13)
    ↓ 验证
Phase 4: 高级特性 & 持续迭代
    ↓ 长期
Phase N: 持续监控与改进
```

---

## 三、Phase 0: 诊断测试 (Run 8-10)

### 目标
并行运行 3 个精心选择的主题，覆盖不同类型的研究任务，以审稿人视角全面评估代码和论文质量。

### 3 个测试主题

#### Run 8: 经典 ML + 视觉任务
**主题**: "Adaptive Per-Layer LoRA Rank Allocation for Memory-Optimal Fine-Tuning of Vision Transformers"
- **为什么选这个**: 测试 pipeline 能否生成涉及 LoRA、ViT、多实验对比的复杂代码
- **预期难点**: 需要 transformers + PEFT 库，需要多层分析逻辑
- **关注指标**: 代码是否真正实现了逐层 rank 分配，还是偷懒用了统一 rank

#### Run 9: 强化学习 + 策略优化
**主题**: "Comparing Flow Matching, Diffusion, and Consistency Models as Generative Trajectory Policies for Offline Reinforcement Learning"
- **为什么选这个**: 测试 pipeline 能否正确实现 3 种不同的生成模型并进行公平对比
- **预期难点**: 需要 D4RL 数据集，三种算法各有复杂实现
- **关注指标**: 每种算法是否有独立的完整实现，还是共享同一套代码换个名字

#### Run 10: LLM 推理 + 计算效率
**主题**: "First-Token Reasoning Quality as a Predictor for Adaptive Test-Time Compute Allocation in Language Models"
- **为什么选这个**: 测试 pipeline 能否处理 LLM 推理/效率优化类任务
- **预期难点**: 需要加载 Qwen-2.5-7B/14B，需要 token-level 分析
- **关注指标**: 是否真正加载了模型进行推理，还是用假数据模拟

### 评审清单

对每个 Run 的输出，按以下维度打分（1-10）：

| 维度 | 评审要点 |
|------|----------|
| **代码完整性** | 是否实现了实验设计中描述的所有算法？有无偷懒/跳过？ |
| **代码复杂度** | 代码是否达到了论文级别的复杂度？是否有非平凡的算法实现？ |
| **框架使用** | 是否正确使用了所需的框架/库？调用方式是否正确？ |
| **实验公平性** | 对比实验是否使用了相同的随机种子、数据划分、评估协议？ |
| **结果可信度** | 结果是否合理？是否有明显的造假/随机数伪造痕迹？ |
| **消融有效性** | 消融实验是否真正去除了关键组件？结果是否有区分度？ |
| **论文与代码一致性** | 论文中描述的方法是否与代码实现一致？ |
| **写作质量** | 论文结构、数字使用、引用质量是否达标？ |

### 期望产出
- 每个 Run 的详细评审报告
- 发现的新问题列表（追加到本文档）
- 更新 Phase 1 的优先级排序

---

## 四、Phase 1: 代码质量根本性改进

### P1.1 代码复杂度强制要求

**文件**: `researchclaw/prompts.py`
**问题**: 当前 code_generation prompt 虽然有很多规则，但缺乏对算法实现深度的硬性要求

**改进方案**:
1. 添加 `code_complexity` block:
   - 每个算法/方法必须有独立的 class 实现（不能是函数别名）
   - 每个 class 必须有 `__init__`, `forward/predict`, `train_step` 三个核心方法
   - 主要算法 class 不少于 50 行有效代码
   - 消融变体必须通过修改算法逻辑实现，不能仅改超参数
   - 禁止 `class MethodB(MethodA): pass` 这种空继承

2. 添加 `implementation_depth` 检查:
   - 在 validator.py 中新增复杂度评分
   - 检查每个 class 的方法数量和代码行数
   - 检查是否存在 "名不副实" 的类（如 BayesianOpt 但没有 acquisition function）

**状态**: ⬜ 待实施

### P1.2 算法实现正确性验证

**文件**: `researchclaw/pipeline/executor.py`
**问题**: 当前仅做语法检查和安全扫描，不验证算法是否正确实现

**改进方案**:
1. 在代码生成后增加 `_verify_algorithm_implementation()` 阶段:
   - 用 LLM 审查代码，逐条检查实验设计中的每个组件是否在代码中实现
   - 生成 checklist: `✅ PPO clipped surrogate objective implemented` / `❌ Missing value function baseline`
   - 如果有 ❌ 项，触发代码修复循环

2. 添加 `_verify_condition_independence()`:
   - 解析代码 AST，检查每个实验条件/方法的 class 是否有独立的逻辑
   - 如果两个 class 的方法体完全相同（hash 匹配），标记为 "identical implementation"
   - 注入警告到修复 prompt 中

**状态**: ⬜ 待实施

### P1.3 实验设计→代码 桥接增强

**文件**: `researchclaw/prompts.py`, `researchclaw/pipeline/executor.py`
**问题**: experiment_design 产出的方案过于抽象，code_generation 难以还原

**改进方案**:
1. 在 experiment_design prompt 中增加 "Pseudocode" 要求:
   - 每个方法/算法必须给出伪代码级别的描述
   - 明确输入输出 tensor shape
   - 明确 loss function 公式
   - 明确 training loop 结构

2. 在 code_generation 中注入 pseudocode 上下文:
   - 从 experiment_design 输出中提取伪代码部分
   - 作为 code_generation prompt 的一部分传入
   - 明确要求 "代码必须与伪代码逻辑一致"

**状态**: ⬜ 待实施

### P1.4 代码审查自动化（LLM-as-Reviewer）

**文件**: `researchclaw/pipeline/executor.py` (新阶段)
**问题**: 代码生成后无系统性审查

**改进方案**:
1. 在 Stage 10 (code_generation) 和 Stage 11 (experiment_run) 之间插入 Stage 10.5:
   - `_execute_code_review()` 方法
   - 用 LLM 以审稿人视角审查代码
   - 生成审查报告: 实现完整性、算法正确性、代码质量
   - 如果审查不通过，返回 Stage 10 重新生成（最多 2 次）

2. 审查 prompt 关注点:
   - 算法命名是否与实现一致？
   - 是否存在 "假实现"（名字是 X 但代码是 Y）？
   - 数据处理是否合理？
   - 损失函数是否正确？
   - 评估协议是否科学？

**状态**: ⬜ 待实施

---

## 五、Phase 2: LLM 微调能力扩展

### P2.1 Docker 环境升级

**文件**: `researchclaw/docker/Dockerfile`

**新增包**:
```dockerfile
# LLM Training Stack
RUN pip install --no-cache-dir \
    transformers>=4.46.0 \
    datasets>=3.0.0 \
    accelerate>=1.0.0 \
    peft>=0.13.0 \
    trl>=0.12.0 \
    bitsandbytes>=0.44.0 \
    sentencepiece \
    protobuf \
    tokenizers \
    safetensors \
    flash-attn --no-build-isolation \
    wandb

# Optional: Llama-Factory
RUN pip install --no-cache-dir llamafactory>=0.9.0
```

**状态**: ⬜ 待实施

### P2.2 LLM 微调 Prompt 体系

**文件**: `researchclaw/prompts.py`

新增 prompt blocks:

1. **`llm_training_guidance`** block:
   - 何时使用 LoRA vs QLoRA vs Full FT
   - GPU 内存估算公式
   - 推荐框架选择指南（Llama-Factory / TRL / 原生 transformers）
   - 训练超参数模板（lr, warmup, scheduler, gradient accumulation）
   - 模型加载方式 (AutoModelForCausalLM + BitsAndBytesConfig)

2. **`llm_eval_guidance`** block:
   - 标准评估基准 (MMLU, MT-Bench, AlpacaEval, HumanEval)
   - 评估框架 (lm-eval-harness, vllm 推理加速)
   - 评估指标定义

3. **`llm_data_guidance`** block:
   - 指令微调数据格式 (Alpaca, ShareGPT, OpenAI chat)
   - HuggingFace datasets 加载方式
   - 数据预处理 pipeline (tokenization, padding, truncation)
   - 常用数据集列表 (Alpaca, ShareGPT, MetaMathQA, CodeAlpaca)

**状态**: ⬜ 待实施

### P2.3 计算预算自适应

**文件**: `researchclaw/pipeline/executor.py`, `researchclaw/config.py`

**改进方案**:
1. 根据研究主题自动调整 time_budget:
   - 经典 ML (CIFAR 级): 600s
   - 中等 (ViT/ResNet 训练): 1800s
   - LLM 微调 (7B LoRA): 7200s (2h)
   - LLM 微调 (14B QLoRA): 14400s (4h)
   - 大规模训练 (72B): 43200s (12h)

2. 在 experiment_design 阶段估算所需计算量:
   - 根据模型大小、数据量、训练 epoch 预估时间
   - 自动设置合理的 time_budget

**状态**: ⬜ 待实施

### P2.4 模型缓存与下载管理

**文件**: `researchclaw/experiment/docker_sandbox.py`

**改进方案**:
1. 支持 HuggingFace Hub 模型缓存目录挂载:
   - 宿主机 `~/.cache/huggingface` → 容器 `/root/.cache/huggingface`
   - 避免每次运行重新下载模型

2. 网络策略调整:
   - LLM 微调任务: `network_policy: "huggingface_only"` (仅允许 HF Hub 下载)
   - 传统 ML 任务: `network_policy: "pip_only"` 或 `"none"`

**状态**: ⬜ 待实施

---

## 六、Phase 3: 回归测试 (Run 11-13)

### 测试主题（Phase 1-2 完成后执行）

#### Run 11: LLM 微调任务
**主题**: "QLoRA Rank Allocation: Adaptive Per-Layer Rank Selection for Memory-Optimal Fine-Tuning of Qwen-2.5"
- **目的**: 验证 P2 (LLM 微调能力) 是否正确工作
- **验证点**: 能否正确调用 PEFT/QLoRA，能否加载 Qwen-2.5 模型

#### Run 12: VLM 推理分析
**主题**: "Modular Causal Attribution for Hallucination Mitigation in Vision-Language Models via MHA Intervention"
- **目的**: 验证 pipeline 能否处理多模态任务
- **验证点**: 代码复杂度是否达标，分析方法是否正确

#### Run 13: 经典 RL 复杂实验
**主题**: "Generative Trajectory Policies: Flow Matching vs Diffusion vs Consistency Models for Offline RL on D4RL"
- **目的**: 验证 P1 (代码质量改进) 是否有效
- **验证点**: 三种算法是否有独立完整实现

### 评分标准
- 每个 Run 使用 Phase 0 的评审清单打分
- 目标: 所有维度 ≥ 6/10，平均 ≥ 7/10
- 如果不达标，回到 Phase 1/2 继续修复

---

## 七、Phase 4: 高级特性 & 持续迭代

### P4.1 基准发现系统 (Benchmark Discovery)
- 在 experiment_design 阶段新增 LLM 调用，自动推荐相关基准和 SOTA 基线
- 已测试: LLM 知识法（Plan 2）效果极佳，可找到 40+ 基准

### P4.2 实验复现性保障
- 记录完整的环境信息 (pip freeze, CUDA version, GPU type)
- 自动生成 requirements.txt
- 支持实验结果复现

### P4.3 多 GPU 分布式训练支持
- DeepSpeed / FSDP 集成
- 多节点训练配置

### P4.4 论文质量进一步提升
- LaTeX 格式化增强
- 图表自动优化（配色、字体、分辨率）
- 引用格式严格化

### P4.5 端到端自动评估
- 集成 LLM-as-Judge 对生成论文自动打分
- 与人工审稿打分对比校准
- 建立质量基线

---

## 八、跟踪记录

### 测试运行记录

| Run | 日期 | 主题 | 模式 | 代码评分 | 论文评分 | 发现的问题 |
|-----|------|------|------|----------|----------|------------|
| 1 | 2026-03-xx | Continual Learning | sandbox | - | - | Bug 1-4 |
| 2 | 2026-03-xx | RIM Agents | sandbox | - | - | Bug 1-4 |
| 3 | 2026-03-xx | (与 Run 1 同主题) | sandbox | - | - | Bug 1-4 |
| 4 | 2026-03-xx | RL for AI4Science | sandbox | 4/10 | - | Bug 5-8, 变量作用域, 5/7条件崩溃 |
| 5 | 2026-03-xx | Graph Neural ODE | sandbox | 4/10 | - | Bug 5-8, nn.Linear in forward, no-op ablation |
| 6 | 2026-03-xx | Meta-Learning | sandbox | - | - | Bug 5-8 |
| 7 | 2026-03-14 | Normalization Techniques | docker | 3/10 | 3/10 | P1-P14 |
| 8 | 2026-03-15 | KD Comparison (CIFAR-10) | docker | 5/10 | - | Q13-Q15, 随机水平结果 |
| 9 | 2026-03-15 | PPO/SAC/TD3+PER | docker | 7/10 | - | Q11, MuJoCo缺失致完全失败 |
| 10 | 2026-03-15 | Neural ODE Robustness | docker | 7/10 | - | Q12/Q16, CIFAR-10挂载失败 |
| 11 | 2026-03-15 | QLoRA Rank Allocation | docker | 4/10 | 7/10 | Q17-Q20, 合成模拟非真实训练 |
| 12 | 2026-03-15 | VLM Hallucination | docker | 3/10 | TBD | Q21-Q23, KeyError崩溃, 训练/验证数据重叠 |
| 13 | 2026-03-15 | PPO/SAC/TD3 MuJoCo | docker | 6/10 | TBD | Q24-Q26, 60k步不够收敛, PPO容量不公平 |

### 问题追踪

| 问题 ID | 描述 | Phase | 状态 | 修复 Commit |
|---------|------|-------|------|-------------|
| Q1 | 代码过于简单/偷懒 | P1 | ✅ 已修复 | cb4af26 |
| Q2 | 不支持 LLM 微调 | P2 | ✅ 已修复 | e72a818 |
| Q3 | Docker 缺关键包 | P2 | ✅ 已修复 | e72a818 |
| Q4 | 计算预算不匹配 | P2 | ✅ 已修复 | e72a818 |
| Q5 | 数据集指导不全 | P1/P2 | ✅ 已修复 | (本次) |
| Q6 | 缺先进训练技巧 | P2 | ✅ 已修复 | e72a818 |
| Q7 | 选题前沿性验证 | P4 | ⬜ 待实施 | - |
| Q8 | 实验设计与代码脱节 | P1 | ✅ 已修复 | cb4af26 |
| Q9 | 消融实验质量低 | P1 | ✅ 已修复 | cb4af26 |
| Q10 | 论文写作质量 | P4 | 🟡 V7已部分修复 | - |
| Q11 | Docker 缺 MuJoCo | P0 | ✅ 已修复 | (本次) |
| Q12 | CIFAR-10 挂载失效 | P0 | ✅ 已修复 | (本次,重建镜像) |
| Q13 | 训练 epoch 过少 | P0 | ✅ 已修复 | (本次) |
| Q14 | Feature KD 维度不匹配 | P1 | 🟡 P1代码审查会捕获 | - |
| Q15 | 消融与 baseline 重复 | P1 | 🟡 P1深度验证会捕获 | - |
| Q16 | 缺少关键实验条件 | P1 | 🟡 P1代码审查会捕获 | - |
| Q17 | Docker HF缓存重复挂载 | P3 | ✅ 已修复 | (本次) |
| Q18 | LLM代码审查JSON解析失败 | P3 | ✅ 已修复 | (本次) |
| Q19 | LLM任务用合成模拟代替真实训练 | P3 | ✅ 已修复(提示) | (本次) |
| Q20 | ndarray.ptp()等NumPy 2.0移除API | P3 | ✅ 已修复(检测+提示) | (本次) |
| Q21 | dict[key]无默认值致KeyError | P3 | ✅ 已修复(提示) | (本次) |
| Q22 | 训练/验证数据集重叠 | P4 | ⬜ 待实施 | - |
| Q23 | 损失函数方向错误(鼓励而非惩罚) | P4 | ⬜ 待实施 | - |
| Q24 | RL训练步数不足(60k vs 需1M) | P3 | 🟡 已有epoch指导,需扩展到RL | - |
| Q25 | 实验条件间模型容量不公平 | P4 | ⬜ 待实施 | - |
| Q26 | proposed_method_variant与主方法相同 | P1 | 🟡 P1深度验证会捕获 | - |

---

## 九、测试选题库

### 优先级 A — Phase 0 诊断测试用

| # | 主题 | 类型 | 预期复杂度 | GPU 时间 |
|---|------|------|-----------|----------|
| A1 | Adaptive Per-Layer LoRA Rank Allocation for ViT | 高效训练 | 高 | 4-8h |
| A2 | Flow Matching vs Diffusion vs Consistency for Offline RL | 强化学习 | 高 | 6-10h |
| A3 | First-Token Reasoning Quality for Compute Allocation | LLM 推理 | 中 | 3-6h |

### 优先级 B — Phase 2 后 LLM 微调测试

| # | 主题 | 类型 | 预期复杂度 | GPU 时间 |
|---|------|------|-----------|----------|
| B1 | QLoRA Fine-Tuning of Qwen-2.5-14B for Medical QA | LLM 微调 | 高 | 4-8h |
| B2 | GainLoRA++ for LLM Continual Learning | 持续学习 | 高 | 6-10h |
| B3 | Spurious Forgetting Analysis in Instruction-Tuned LLMs | LLM 分析 | 中 | 4-8h |

### 优先级 C — 多样性覆盖测试

| # | 主题 | 类型 | 预期复杂度 | GPU 时间 |
|---|------|------|-----------|----------|
| C1 | Modular Causal Attribution for VLM Hallucination | VLM 分析 | 中 | 3-6h |
| C2 | Neural Operator Downscaling for Weather Prediction | AI4Science | 中 | 4-8h |
| C3 | Meta-Learned LoRA Initialization for Few-Shot Adaptation | Meta-Learning | 中 | 4-8h |
| C4 | Prune-Then-LoRA for Parameter-Efficient Fine-Tuning | 高效训练 | 中 | 4-8h |
| C5 | Decomposition-of-Thought for VLM Reasoning | VLM 推理 | 低 | 2-4h |

---

## 十、执行计划

### 执行进度

#### Phase 0: 诊断测试 ✅ 完成
1. ✅ 调研热门主题，筛选测试用 idea
2. ✅ 为 Run 8/9/10 创建配置文件
3. ✅ 并行启动 3 个 Run
4. ✅ 监控中间输出，特别关注 Stage 10 (代码生成) 产出
5. ✅ 以审稿人视角评审代码 + 论文
6. ✅ 汇总发现的问题 (Q11-Q16)
7. ✅ 确定 Phase 1 优先级

#### Phase 1: 代码质量改进 ✅ 完成 (commit cb4af26)
- P1.1: 深度代码质量检查 (AST分析: 类质量, 变量作用域, API正确性)
- P1.2: 自动修复循环 (深度验证 → LLM修复 → 重验证)
- P1.3: 实验设计增加 implementation_spec (伪代码级描述)
- P1.4: LLM代码审查 (Stage 10.5, 评分1-10, 严重问题触发修复)

#### Phase 2: LLM微调能力 ✅ 完成 (commit e72a818)
- P2.1: Docker新增transformers/peft/trl/bitsandbytes/datasets
- P2.2: llm_training_guidance + llm_eval_guidance 提示块
- P2.3: 自动检测LLM主题注入指导; time_budget警告
- P2.4: HuggingFace缓存挂载 + HF_TOKEN透传

#### Phase 3: 回归测试 🔄 进行中
Run 11-13 结果分析:
- **Run 11 (QLoRA)**: 代码4/10, 论文7/10 — 合成模拟非真实训练, 但论文质量达标
- **Run 12 (VLM)**: 代码3/10 — KeyError崩溃, 训练/验证重叠, 损失方向错误
- **Run 13 (RL)**: 代码6/10 — MuJoCo成功! 但60k步不够收敛, PPO容量不公平

Phase 3 修复 (本次commit):
- Q17: Docker HF缓存重复挂载 → 优先HF_HOME, 避免重复
- Q18: LLM代码审查JSON解析失败 → 正确提取LLMResponse.content + 去除markdown fence
- Q19: LLM任务合成模拟 → 添加"CRITICAL — NO SIMULATION"规则
- Q20: NumPy 2.0移除API → 检测器 + 禁止模式更新
- Q21: dict[key]无默认值 → 禁止模式更新

### 注意事项
- 每次迭代结束后更新本文档
- 新发现的问题立即追加到问题追踪表
- 修复后必须有对应的回归测试 Run
- 配置文件中 API key 已在 .gitignore 中排除
