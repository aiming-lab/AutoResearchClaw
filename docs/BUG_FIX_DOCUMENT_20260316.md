# Bug Fix Document — AutoResearchClaw Pipeline

> 生成日期：2026-03-16
> 反馈来源：2 位测试者（user1: CV 方向 / GPU 环境, user2: Windows 环境）
> 总计问题：9 个

## 📊 总览

| 分类 | 数量 |
|------|------|
| 🔴 确认的 Bug（需修复） | **4** |
| 🟠 架构改进（强烈建议） | **2** |
| 🔵 功能需求 | **3** |

## 🔥 修复优先级

| 优先级 | ID | 问题 | 阶段 | 涉及文件 |
|--------|----|------|------|----------|
| 🔴 CRITICAL | BUG-001 | 论文硬件信息与实际不一致 | PAPER_DRAFT (17) | `executor.py`, `prompts.py` |
| 🔴 CRITICAL | BUG-002 | Windows 环境 Docker 不可用导致实验链式失败 | EXPERIMENT_RUN (12) | `factory.py`, `docker_sandbox.py` |
| 🔴 HIGH | BUG-003 | 论文内容自相矛盾（承诺评测但未执行） | PAPER_DRAFT (17), PEER_REVIEW (18) | `executor.py`, `prompts.py` |
| 🔴 HIGH | BUG-004 | 生成代码缺少数值稳定性防护（NaN/Inf） | CODE_GENERATION (10) | `code_agent.py`, `prompts.py` |
| 🟠 HIGH | ARCH-001 | Stage 17 过于严格的 hard block 策略 | PAPER_DRAFT (17) | `executor.py` |
| 🟠 HIGH | ARCH-002 | Idea 降级时不询问用户确认 | EXPERIMENT_DESIGN (9), RESEARCH_DECISION (15) | `executor.py`, `stages.py` |

---

## 确认的 Bug — 详细修复方案

### 🔴 `BUG-001` — 论文硬件信息与实际机器不一致

| 字段 | 内容 |
|------|------|
| **严重程度** | CRITICAL |
| **所属阶段** | PAPER_DRAFT (Stage 17) |
| **报告者** | user1 |

**问题描述：**
论文中声称使用 A100 GPU 训练，但测试者实际机器上是 A5000。Pipeline 在 Stage 1 检测了硬件并保存到 `hardware_profile.json`，但在论文生成阶段完全没有利用这个信息来约束 LLM 输出。

**根因分析：**
- `executor.py` 第 1226-1233 行：Stage 1 (TOPIC_INIT) 检测硬件，保存 `hardware_profile.json`，包含 `gpu_name`、`vram_gb` 等
- `executor.py` 第 2352-2391 行：硬件信息 **仅** 用于 CODE_GENERATION 阶段的代码生成 hints
- `executor.py` 第 5776-5848 行：PAPER_DRAFT 阶段构建 prompt 时，**没有注入硬件 profile 信息**
- LLM 在缺少约束的情况下会「幻觉」出常见的高端硬件名称（如 A100）

**涉及文件：**
- `researchclaw/pipeline/executor.py`（PAPER_DRAFT 阶段的 prompt 构建部分，约第 5776-5960 行）
- `researchclaw/prompts.py`（paper writing prompt 模板）

**修复方案：**
1. 在 PAPER_DRAFT 阶段的 prompt 构建中，读取 `stage-01/hardware_profile.json`
2. 将实际硬件信息（GPU 型号、VRAM、CPU 等）作为 **硬性约束** 注入 prompt，例如：
   ```
   HARDWARE CONSTRAINT: The experiments were run on the following hardware:
   - GPU: {gpu_name} ({vram_gb} GB VRAM)
   - CPU: {cpu_info}
   You MUST use this exact hardware specification in the paper. Do NOT substitute with other GPU models.
   ```
3. 在 PEER_REVIEW (Stage 18) 的 prompt 中增加一条审核规则：验证 paper 中提到的硬件是否与 `hardware_profile.json` 一致

**修复后预期行为：**
论文中的硬件描述必须与实际运行环境一致。

<details>
<summary>原始反馈证据</summary>

> 然后就是paper和实验中有一些misalign的地方，比如paper里写说用的A100，实际上机器里的是A5000
</details>

---

### 🔴 `BUG-002` — Windows 环境下 Docker 不可用导致实验链式失败

| 字段 | 内容 |
|------|------|
| **严重程度** | CRITICAL |
| **所属阶段** | EXPERIMENT_RUN (Stage 12) → 链式影响到 Stage 13, 14, 17 |
| **报告者** | user2 |

**问题描述：**
在 Windows 环境下，Docker 不可用时 Pipeline 直接崩溃（`[WinError 2] The system cannot find the file specified`），导致所有后续阶段连锁失败。用户最终看到的是 Stage 17 的误导性错误「没有实验数据无法写论文」，完全看不到真正的根因。

**根因分析：**
- `experiment/factory.py` 第 25-29 行：当 `config.experiment.mode == "docker"` 时调用 `DockerSandbox.check_docker_available()`，如果 Docker 不可用直接 raise `RuntimeError`，**没有自动 fallback 到 subprocess sandbox**
- `docker_sandbox.py` 第 337、366 行：Docker volume mount 使用 POSIX 风格路径（如 `{staging_dir}:/workspace`），在 Windows 上可能导致挂载失败
- **链式失败：** Stage 12 crash → 无 metrics → Stage 13 空跑（`refine_sandbox_v1` 到 `v9` 都失败） → Stage 14 空 `experiment_summary.json` → Stage 17 hard block
- 用户看到的错误完全不提 Docker，只说「no metrics」，非常误导

**涉及文件：**
- `researchclaw/experiment/factory.py`（第 25-29 行，sandbox 创建逻辑）
- `researchclaw/experiment/docker_sandbox.py`（第 337、366、384 行，路径和命令构建）
- `researchclaw/pipeline/executor.py`（第 6000-6020 行，Stage 17 hard block）

**修复方案：**
1. `factory.py`：当 Docker 不可用时，自动 fallback 到 subprocess sandbox 模式，而不是 raise RuntimeError。增加日志 warning 告知用户：
   ```python
   if not DockerSandbox.check_docker_available():
       logger.warning("Docker not available, falling back to subprocess sandbox mode")
       return SubprocessSandbox(...)
   ```
2. `docker_sandbox.py`：修复 Windows 路径兼容性问题，使用 `pathlib.PureWindowsPath` 或 `os.path` 正确处理跨平台路径
3. 在 Stage 12 的错误信息中明确指出是 Docker 问题，而不是让错误沿链传播变成「no metrics」

**修复后预期行为：**
Windows 用户即使没有 Docker，Pipeline 也能通过 subprocess sandbox 完成实验。即使实验部分失败，错误信息应清晰指向根因。

<details>
<summary>原始反馈证据</summary>

> 我跑了两次 两次都有stage fail 最后没有生成报告

压缩包中 `experiment_summary.json` stderr: `[WinError 2] The system cannot find the file specified`
`pipeline_summary.json`: `"final_status": "failed"`, `"stages_failed": 1`
`stage-17/paper_draft.md`: `Experiment stage produced no metrics (status: failed/timeout). Cannot write a paper without real experimental data.`
</details>

---

### 🔴 `BUG-003` — 论文内容自相矛盾（承诺评测数据集但未实际执行）

| 字段 | 内容 |
|------|------|
| **严重程度** | HIGH |
| **所属阶段** | PAPER_DRAFT (Stage 17), PEER_REVIEW (Stage 18) |
| **报告者** | user1 |

**问题描述：**
论文前半部分按照用户的 topic 描述声称会在 MME、DocVQA、TextVQA 等数据集上评测，但实际实验阶段因为环境原因未能完成这些评测。论文后半部分在 Limitation 中又说「没有在这些数据集上评估」，形成自相矛盾。

**根因分析：**
- `prompts.py` 第 2006-2018 行：有 EVIDENCE-BOUNDING RULES（Rule 7-9），但这些只是 prompt 中的 **建议**，LLM 可以忽略
- `executor.py` 第 5647-5715 行：`_detect_result_contradictions()` 函数检测 null/negative results，但只生成 advisory text 注入 prompt，**不做硬性阻断**
- `executor.py` 第 6432-6443 行：PEER_REVIEW 阶段收集 `actual_run_count` 作为 evidence，但 **没有自动扫描 paper 文本提取声称的数据集列表并与实际评测记录对比**
- 核心问题：**缺少 claim-evidence 的自动对齐验证**

**涉及文件：**
- `researchclaw/pipeline/executor.py`（第 5647-5715 行、5944-5956 行、6432-6443 行）
- `researchclaw/prompts.py`（第 2006-2049 行、2124-2138 行）

**修复方案：**
1. 在 PAPER_DRAFT 阶段的 prompt 中，**明确列出** 实际完成评测的数据集和指标（从 `experiment_summary.json` 提取），硬性要求 LLM **只能**声称在这些数据集上进行了评测：
   ```
   ACTUAL EVALUATED DATASETS: [ImageNet-val (reconstruction)]
   You MUST NOT claim evaluation on any dataset not listed above.
   If the original research plan included additional datasets that were not evaluated,
   explain this honestly in the Limitations section WITHOUT first claiming you did evaluate them.
   ```
2. 在 PEER_REVIEW (Stage 18) 增加一个专项检查：自动提取 paper 中所有提到的 benchmark/dataset 名称，与 `experiment_summary.json` 中的实际 metrics keys 对比，不一致则标记为 CRITICAL discrepancy
3. 在 PAPER_REVISION (Stage 19) 中把这些 discrepancy 作为必须修改的 reviewer comment

**修复后预期行为：**
论文中不会出现「前面说评测了 X，后面说没评测 X」的自相矛盾。所有评测声明必须有实验数据支撑。

<details>
<summary>原始反馈证据</summary>

> 以及就是paper中有一些自相矛盾的地方，比如前面按照我的要求，说会在哪几个数据集上面进行评估，后面又没有测，然后在limitation说我们没有在这几个数据集上评估
</details>

---

### 🔴 `BUG-004` — 生成代码缺少数值稳定性防护（NaN/Inf 导致实验提前终止）

| 字段 | 内容 |
|------|------|
| **严重程度** | HIGH |
| **所属阶段** | CODE_GENERATION (Stage 10), ITERATIVE_REFINE (Stage 13) |
| **报告者** | user1 |

**问题描述：**
实验训练过程中出现 `loss = inf` → `loss = nan` 的数值爆炸，触发 harness 的 NaN 检测后实验提前终止。代码生成阶段没有在生成的训练代码中加入数值稳定性保护。

**根因分析：**
- `code_agent.py`：**完全没有** 关于数值稳定性的 prompt 指令。4 个阶段（Planning → Code Generation → Execution-in-the-Loop → Multi-Agent Review）都不检查 NaN guard
- `experiment/harness_template.py` 第 45-62 行：有 `check_value()` 做 NaN/Inf 检测，但这是 **opt-in 机制**——只有生成代码主动调用 `self.check_value(loss, "loss")` 才有效
- `executor.py` 第 779-900 行：`_detect_runtime_issues()` 在运行 **之后** 检测 NaN，但此时实验已经失败了
- `executor.py` 第 3915-3956 行：Stage 13 检测到 NaN 后调用 LLM 做 `iterative_repair`，但修复质量不稳定

**涉及文件：**
- `researchclaw/pipeline/code_agent.py`（prompt 构建，所有阶段）
- `researchclaw/prompts.py`（代码生成相关 prompt）
- `researchclaw/experiment/harness_template.py`（第 45-62 行）

**修复方案：**
1. 在 `code_agent.py` 的代码生成 prompt 中，增加 **强制性** 数值稳定性要求：
   ```
   NUMERICAL STABILITY REQUIREMENTS (MANDATORY):
   - Add gradient clipping (max_norm=1.0) to all optimizer steps
   - Check loss for NaN/Inf before backward pass: if not math.isfinite(loss): skip this batch
   - Use torch.amp.GradScaler for mixed precision training if applicable
   - Add learning rate warmup for the first 5-10% of training steps
   - Use self.check_value(loss, "loss") from experiment harness for NaN tracking
   ```
2. 在 `harness_template.py` 中，将 `check_value()` 改为 **自动 hook** 而非 opt-in——在 `finalize()` 中自动检查 metrics 是否为 finite
3. 在 Multi-Agent Review 阶段（`code_agent.py` Phase 4）增加数值稳定性作为必审项

**修复后预期行为：**
生成的训练代码默认包含 gradient clipping 和 NaN guard，训练过程中数值爆炸能被及时 catch 并恢复，而不是直接终止。

<details>
<summary>原始反馈证据</summary>

> 好像是他的代码写错了之类的

压缩包中 `experiment_summary.json` stderr:
```
WARNING: loss = inf (non-finite, skipped)
WARNING: loss = nan (non-finite, skipped)
WARNING: loss = nan (non-finite, skipped)
WARNING: loss = nan (non-finite, skipped)
WARNING: loss = nan (non-finite, skipped)
FAIL: Too many NaN/Inf values detected. Stopping experiment early.
```
</details>

---

## 架构改进 — 强烈建议

### 🟠 `ARCH-001` — Stage 17 (PAPER_DRAFT) 过于严格的 hard block 策略

| 字段 | 内容 |
|------|------|
| **严重程度** | HIGH |
| **所属阶段** | PAPER_DRAFT (Stage 17) |
| **报告者** | user2（链式影响） |

**问题描述：**
当实验阶段没有产出完整 metrics 时，Stage 17 直接 FAILED，不尝试用已有数据写论文。这导致前面 1-16 阶段的全部成果被浪费。

**根因分析：**
- `executor.py` 第 6000-6020 行：当 `has_real_metrics == False` 且 domain 为 empirical 时，直接返回 `StageStatus.FAILED`
- Stage 13 (ITERATIVE_REFINE) 的中间迭代可能产出了部分有效 metrics，但 Stage 17 只看 `experiment_summary.json` 的 final best_run

**涉及文件：**
- `researchclaw/pipeline/executor.py`（第 6000-6020 行）

**修复方案：**
将 hard block 改为 soft degradation：
1. 如果有部分 metrics（即使不完整），用已有数据写论文
2. 在 prompt 中明确告知 LLM 数据不完整，要求在 Abstract 和 Limitations 中如实说明
3. 只有在 **完全没有任何数据**（甚至没有 stage-07 synthesis 和 stage-08 hypotheses）的极端情况下才 hard block
4. 在输出的 `paper_draft.md` 头部加 warning 标记，方便后续阶段识别

**修复后预期行为：**
实验部分失败时，Pipeline 仍能生成一篇带有诚实 Limitations 的论文，用户至少得到有价值的输出。

---

### 🟠 `ARCH-002` — Idea 被降级到弱版本时不询问用户

| 字段 | 内容 |
|------|------|
| **严重程度** | HIGH |
| **所属阶段** | EXPERIMENT_DESIGN (Stage 9), RESEARCH_DECISION (Stage 15) |
| **报告者** | user1 |

**问题描述：**
用户给了一个复杂的 strong idea（如 VAE+ViT 统一编码器 + 多数据集评测），Pipeline 因资源限制（数据集不可用、GPU 不够、环境配不好）自动降级到 weaker 版本，但不通知或征求用户意见。用户认为降级后的研究「变得没啥意义」。

**根因分析：**
- `executor.py` 第 2220-2236 行：LLM 生成的实验计划无效时，使用 topic-derived fallback，**不询问用户**
- `executor.py` 第 4618-4640 行：RESEARCH_DECISION 检测 degenerate cycle 时只给 LLM advisory，**不暂停**
- `stages.py` 第 109-115 行：GATE_STAGES 只包含 Stage 5、9、20，不包含 Stage 15
- `agents/benchmark_agent/orchestrator.py` 第 314-322 行：BenchmarkAgent 验证失败时 silent retry，最终 silent proceed

**涉及文件：**
- `researchclaw/pipeline/executor.py`（第 2220-2236 行、4618-4640 行）
- `researchclaw/pipeline/stages.py`（GATE_STAGES 定义）
- `researchclaw/agents/benchmark_agent/orchestrator.py`（第 314-322 行）

**修复方案：**
1. 在 EXPERIMENT_DESIGN (Stage 9) 中，当检测到 significant downgrade（如：用户要求的数据集不可用、GPU 不满足要求、关键组件被简化）时，生成一个 **downgrade summary** 并暂停等待用户确认
2. 在 RESEARCH_DECISION (Stage 15) 中，将 REFINE → weaker idea 的决策标记为 GATE，需要用户 approve
3. 可以通过 `auto_approve` 参数让用户选择是否跳过这些确认（保持向后兼容）

**修复后预期行为：**
Pipeline 在降级研究方案前通知用户，用户可以选择：接受降级、提供更多资源（如更大的 GPU）、或终止当前 run。

<details>
<summary>原始反馈证据</summary>

> 对，还有就是比如我提出了一个相对strong的idea，而他因为各种原因（比如数据集找不到，环境配不好，gpu不够）之类的，给我fallback到weaker的idea之后，我感觉这个时候应该询问一下用户要不要继续跑
>
> 因为很多时候他继续跑的内容就会变得没啥意义
</details>

---

## 功能需求

### 🔵 `FEAT-001` — 论文生成后增加一致性反馈循环

- **报告者：** user1
- **描述：** 在论文生成之后，增加专门的 consistency check，检查 paper 中的声明与实际实验结果是否一致
- **建议：** 可以在 PEER_REVIEW (Stage 18) 的 prompt 中增加 claim-evidence alignment 专项检查。或者在 Stage 17 和 18 之间加一个轻量级的自动验证步骤

<details>
<summary>原始反馈</summary>

> 感觉这个可以在paper生成之后，加一些相关的consistence feedback之类的？
</details>

### 🔵 `FEAT-002` — 从 Related Works 的 GitHub 学习 Common Practice

- **报告者：** user1
- **描述：** 当前 Pipeline 的 literature 阶段只读论文，不看对应的开源代码。用户建议访问 related works 的 GitHub repo，学习 paper 中不会写的实现细节（tricks、common practice），缓解论文内容过于古老的问题
- **建议：** 在 KNOWLEDGE_EXTRACT (Stage 6) 或 EXPERIMENT_DESIGN (Stage 9) 增加 GitHub repo 分析能力。可以用 GitHub API 搜索 related works 的 repo，提取 README、主要代码结构、训练配置等信息

<details>
<summary>原始反馈</summary>

> 对就是我觉得即使不拿来用，visit related works的github也是有必要的，这样可以看到其他工作的common practice（一些不会在paper中出现的细节），应该会挺有用的。感觉可以缓解一下paper内容过于古老的问题
</details>

### 🔵 `FEAT-003` — 代码应该复用 Related Works 的框架

- **报告者：** user1
- **描述：** 当前代码都是 LLM 从零写的简单文件，用户建议从 most related works 中选一个合适的框架来用，就像真实研究中的做法
- **建议：** 可以在 BenchmarkAgent 或 CODE_GENERATION 阶段增加框架选择逻辑——从相关论文的开源实现中挑选合适的 codebase 作为起点，而不是从零生成。这是一个较大的改动，可以作为长期目标

<details>
<summary>原始反馈</summary>

> 以及他现在写的代码都比较简单，都是自己写几个文件对吧。我在想或许可以从most related works里面选一个合适的框架来用？我们平时也是这样的对吧。当然这个比较复杂，可以先不考虑
</details>

---

## 附录：按测试者分组

### 测试者：`user1`
- **学科/领域：** 计算机视觉（CV），统一图像编解码器
- **运行环境：** GPU 服务器（A5000），使用 Codex 监控
- **总计问题：** 6
- **确认 Bug：** 3（BUG-001, BUG-003, BUG-004）
- **架构改进：** 1（ARCH-002）
- **功能需求：** 3（FEAT-001, FEAT-002, FEAT-003）

| ID | 问题 | 状态 | 严重程度 |
|----|------|------|---------|
| BUG-001 | 论文硬件信息与实际不一致 | confirmed | CRITICAL |
| BUG-003 | 论文内容自相矛盾 | confirmed | HIGH |
| BUG-004 | 代码缺少数值稳定性防护 | confirmed | HIGH |
| ARCH-002 | Idea 降级不询问用户 | confirmed | HIGH |
| FEAT-001 | 一致性反馈循环 | feature_request | — |
| FEAT-002 | 从 GitHub 学习 common practice | feature_request | — |
| FEAT-003 | 复用 related works 框架 | feature_request | — |

### 测试者：`user2`
- **学科/领域：** 未知（topic 与纳米药物递送相关）
- **运行环境：** Windows
- **总计问题：** 2
- **确认 Bug：** 1（BUG-002）
- **架构改进：** 1（ARCH-001）

| ID | 问题 | 状态 | 严重程度 |
|----|------|------|---------|
| BUG-002 | Windows Docker 链式失败 | confirmed | CRITICAL |
| ARCH-001 | Stage 17 过于严格的 hard block | confirmed | HIGH |

---

## 修复执行指引

> 本文档设计为可由另一台机器上的 Claude Code agent 直接读取并执行修复。
> 建议按优先级从上到下依次修复，每修复一个 Bug 运行相关测试验证。

**修复顺序建议：**
1. BUG-002（Docker fallback）→ 解除 Windows 用户的完全阻塞
2. BUG-001（硬件一致性）→ 简单修复，prompt 注入即可
3. BUG-004（NaN guard）→ prompt 层面修复，影响面大
4. BUG-003（claim-evidence 对齐）→ 需要新增验证逻辑
5. ARCH-001（soft degradation）→ 改变 Stage 17 策略
6. ARCH-002（用户确认 Gate）→ 需要状态机和 Gate 逻辑调整
