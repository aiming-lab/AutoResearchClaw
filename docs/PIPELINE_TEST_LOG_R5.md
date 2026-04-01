# Pipeline Test Log — Round 5 (main branch)

> **分支**: `main` @ `e95527f`
> **日期**: 2026-03-18
> **目的**: 全面测试 main 分支 Pipeline 端到端流程，覆盖多领域、纯计算实验
> **环境**: Python 3.x, numpy 2.4.3, scipy 1.17.1, sklearn 1.8.0, matplotlib 3.10.8
> **LLM**: gpt-5.4 (fallback: gpt-5.1, gpt-4o) via Azure OpenAI

---

## 测试选题

| ID | 领域 | 主题 | metric_direction | 关键依赖 |
|----|------|------|-----------------|---------|
| N | 计算物理 | 随机矩阵理论：Marchenko-Pastur 分布的有限维修正分析 | minimize | numpy, scipy |
| O | 计算经济学 | 弱工具变量下 IV 估计量的 Monte Carlo 偏差-方差权衡 | minimize | numpy, scipy, sklearn |
| P | 计算流行病学 | SIR/SEIR 模型参数可辨识性：合成数据下的结构化似然分析 | maximize | numpy, scipy |
| Q | 数学/数值分析 | Krylov 子空间方法求解稀疏线性系统：预条件策略对比 | minimize | numpy, scipy |

### 选题原则
- 所有实验纯计算/模拟，无需外部数据集或 GPU
- 核心依赖仅 numpy/scipy/sklearn，sandbox 即可执行
- 覆盖 4 个不同领域：物理、经济学、流行病学、数值分析
- 避免 R4 中被 topic refinement 强行引向 ML 的问题——本轮 topic 描述更具体

### 备选 Topic（未选用）
1. **Agent/RL**: 网格世界中多智能体 emergent communication 的涌现 — 需要 gymnasium，sandbox 兼容性不确定
2. **信号处理**: 压缩感知中 RIP 条件的经验验证 — 可行但领域覆盖与 Q 重叠
3. **统计学**: Bayesian 变点检测的 MCMC 采样效率对比 — 可行，备用
4. **图论**: 随机图上 Erdos-Renyi 相变阈值的数值验证 — 可行，备用

---

## 运行状态

| Pipeline | Config | Run ID | PID | 启动时间 (UTC) | 最终阶段 | 状态 | 总耗时 |
|----------|--------|--------|-----|---------------|---------|------|--------|
| N | config_test_N.yaml | `rc-20260318-174754-fc94f2` | 2036352 | 17:47 | 28/29 (S23 fail) | ⚠️ 近完美 | ~2.5h |
| O | config_test_O.yaml | `rc-20260318-174826-01c0f3` | 2037261 | 17:48 | 28/29 (S23 fail) | ⚠️ 近完美 | ~3.0h |
| P | config_test_P.yaml | `rc-20260318-174900-d5371f` | 2037826 | 17:49 | 29/29 ✅ | ✅ 完美通过 | ~2.4h |
| Q | config_test_Q.yaml | `rc-20260318-174935-d0a717` | 2038664 | 17:49 | 28/29 (S23 fail) | ⚠️ 近完美 | ~2.5h |

---

## 观测记录

### OBS-R5-01: S2 + arXiv 429 限流（预期行为）(17:48 UTC)
- **严重度**: 🟢 预期行为
- **描述**: 4个并行 Pipeline 同时触发 S2/arXiv 429 限流
  - S2 circuit breaker: 120s cooldown (trip #1)
  - arXiv circuit breaker: 180s cooldown (trip #1)
- **关联**: R4-OBS-02 同类问题
- **影响**: 文献搜索阶段延迟增加，不阻塞

### OBS-R5-02: Pipeline Q 触发 IMP-35 Topic Refinement (17:49 UTC)
- **严重度**: 🟡 值得关注
- **描述**: Krylov 子空间方法的 topic 被评为 4/10，系统建议 refine 为 ML 相关方向
  - 原始: "Comparative Analysis of Preconditioning Strategies for Krylov Subspace Methods..."
  - 建议: "Learned preconditioner selection for Krylov solvers on sparse linear systems..."
- **评估**: IMP-35 倾向于把所有 topic 往 ML 方向引导（R4-OBS-03 同类问题）
- **影响**: 纯数值分析 topic 可能被扭曲为 ML topic，但实验代码仍应聚焦原始问题

### OBS-R5-03: 初始进度检查 (~17:55 UTC)
- N: Stage 7/SYNTHESIS ✅ 快速推进
- O: Stage 6/KNOWLEDGE_EXTRACT ✅ 正常
- P: Stage 4/LITERATURE_COLLECT — 稍慢（429 影响）
- Q: Stage 5/LITERATURE_SCREEN ✅ 正常

### OBS-R5-04: CodeSearcher query_gen.py TypeError (18:20 UTC)
- **严重度**: 🟡 中 — 不阻塞但影响代码质量
- **描述**: `researchclaw/agents/code_searcher/query_gen.py:149` 调用 `llm.chat()` 时传入不支持的 `user` 关键字参数
  ```
  TypeError: LLMClient.chat() got an unexpected keyword argument 'user'
  ```
- **影响**: CodeSearcher 无法使用 LLM 生成 GitHub 搜索 query，退化到基于规则的 query
- **关联**: R4-BUG-02 (GitHub 401) — 401 问题仍在（无 GITHUB_TOKEN），加上此 TypeError 意味着 CodeSearcher 基本失效
- **需要修复**: ✅ 是 — query_gen.py 中 `llm.chat()` 调用签名与 LLMClient 接口不匹配

### OBS-R5-05: gpt-5.4 Read Timeout 导致 fallback (18:30 UTC)
- **严重度**: 🟡 中 — 自动 fallback 工作正常
- **描述**: Pipeline N 在代码生成阶段遭遇多次 gpt-5.4 read timeout
  - 触发 fallback 到 gpt-5.1 或 gpt-4o
  - 代码生成请求因 token 量大，更容易超时
- **影响**: 代码生成速度下降，但不阻塞

### OBS-R5-06: Sandbox execution timeout 60s (18:35 UTC)
- **严重度**: 🟡 中 — 影响代码验证
- **描述**: Pipeline O 代码生成阶段的 sandbox 验证执行超时（60s）
  - 可能是验证生成的实验代码能否运行
  - 代码生成后的 AST 验证 + 试运行超时
- **影响**: 代码可能未经充分验证就进入下一阶段

### OBS-R5-07: Stage 10 Deep Quality — Copy-paste Detection (18:35 UTC)
- **严重度**: 🟡 中 — 代码质量问题
- **描述**: Pipeline O 的 models.py 中检测到多组 copy-paste 类：
  1. `FixedFullerOneBiasReducedBaseline` vs `FixedFullerFourAggressiveShrinkageBaseline` (16 vs 16 lines)
  2. `FirstStageStrengthOnlyRiskSurfaceBaseline` vs `NoLeverageGeometryRiskSurfaceAblation` (9 vs 9 lines)
  3. 多个 ablation 类仅 0-1 个非 dunder 方法
- **评估**: 这是 R4-BUG-13 的同类问题 — ablation 类之间差异不足
- **关联**: BUG-13 (copy-paste ablation)

### OBS-R5-08: 所有 Pipeline 在 Stage 10 停留超 25 分钟 (18:41 UTC)
- **严重度**: 🟢 预期行为
- **描述**: 代码生成是最重的 LLM 调用阶段，N=1 attempt, O/P=3 attempts, Q=3 attempts
- **评估**: 多次 attempt 表明 code validation loop 在工作，自动修复代码中的问题
- **耗时**: N=2441s (~41min), O=2485s (~41min), P=2796s (~47min), Q=2976s (~50min)

### OBS-R5-09: 所有已执行实验在 Stage 12 首次运行均失败 (18:55 UTC)
- **严重度**: 🔴 高 — 系统性 numpy 2.x API 不兼容
- **描述**: 3个已完成 Stage 12 的 Pipeline 均在首次实验运行失败：
  - **N**: `AttributeError: module 'numpy' has no attribute 'trapz'`
    - numpy 2.0 移除了 `np.trapz`，应使用 `np.trapezoid`
  - **O**: `numpy.linalg.LinAlgError: 1-dimensional array given. Array must be two-dimensional`
    - 代码向 linalg 函数传入了 1D 数组
  - **P**: `AttributeError: module 'numpy' has no attribute 'erfinv'`
    - `erfinv` 从未存在于 numpy 中，应使用 `scipy.special.erfinv`
- **根因**: gpt-5.4 生成的代码使用了已在 numpy 2.x 中移除或不存在的 API
- **关联**: R5-BUG-01 (见下方)

### OBS-R5-10: Stage 13 自动修复正确修复 numpy.trapz → numpy.trapezoid (18:55 UTC)
- **严重度**: 🟢 正面发现
- **描述**: Pipeline N 的 Stage 13 (ITERATIVE_REFINE) 成功检测到 `np.trapz` 错误并：
  1. 创建了 `_trapz()` 包装函数
  2. 内部使用 `np.trapezoid(y, x)` 替代
  3. 同时创建了 `_cumulative_trapezoid_1d()` 辅助函数
- **评估**: 自我修复机制在 numpy API 变更场景中工作良好

### OBS-R5-11: Pipeline Q Stage 09 YAML 解析警告 (18:40 UTC)
- **严重度**: 🟢 低 — 自动恢复
- **描述**: Pipeline Q 的 Stage 09 LLM 返回内容无法直接解析为 YAML
  - 返回了 38089 字符的响应，远超预期
  - content extraction fallback 正常工作
- **影响**: 无实际影响，pipeline 继续正常运行

### OBS-R5-12: Stage 13 自动修复成功修复所有 numpy 2.x 不兼容 (19:10 UTC)
- **严重度**: 🟢 正面发现
- **描述**: 所有 4 个 Pipeline 的 Stage 13 成功修复了 Stage 12 首次运行失败：
  - N: `np.trapz` → `np.trapezoid` (wrapper function) ✅
  - O: 1D→2D array reshape 修复 ✅
  - P: `np.erfinv` → `scipy.special.erfinv` ✅
  - Q: 修复后成功运行 ✅
- **评估**: 自我修复机制可靠，但首次成功率仍可改善

### OBS-R5-13: 所有 4 个 Pipeline 首次 Research Decision 均为 REFINE (19:23-19:49 UTC)
- **严重度**: 🟡 值得关注
- **描述**: 所有 Pipeline 在第一轮实验后都被判定需要 refine
  - 这可能意味着：(a) 实验结果不够convincing (b) 系统对首轮结果过于严格
  - N、P、Q 在第二轮后仍被 refine → 达到 max refine (2次) → 下次将 forced PROCEED
  - O 在第一轮 refine 中
- **影响**: Pipeline 总耗时增加（每次 refine 约增加 15-30 分钟实验时间）

### OBS-R5-14: Pipeline N 首先进入纸写作阶段 (~19:57 UTC)
- **严重度**: 🟢 正面进展
- **描述**: Pipeline N (Marchenko-Pastur) 完成 2 轮 refine，被 forced PROCEED 到 Stage 16
  - Stage 14 (RESULT_ANALYSIS) 耗时 553s (~9min)
  - Stage 15 decision 耗时 15s

### OBS-R5-15: Pipeline P 完美完成 29/29 stages! (20:13 UTC)
- **严重度**: 🟢🟢🟢 重大正面发现
- **描述**: Pipeline P (SIR/SEIR 流行病学) 是 R5 第一个（也是唯一一个）完美完成的 Pipeline
  - 所有 29 个 stage 成功，0 失败
  - 完整交付物：paper.tex (539行), references.bib (405行), 5 张图表, code package
  - Stage 23 citation verify 成功验证 44 条引用
  - LaTeX 编译成功（paper.aux, paper.log 生成）
  - 总耗时约 2.4 小时
- **评估**: 这是本项目自 R0 以来第一次有 Pipeline 完整通过所有 29 个 stage
  - R0: Pipeline A 29/29 但那是在较旧版本上
  - R4: 所有 4 个 Pipeline 在 Stage 20 被拒（2/10 质量分）
  - R5: Pipeline P 通过了 Stage 20（degraded 但非 rejected）

### OBS-R5-16: N 和 Q 在 Stage 23 (Citation Verify) 失败 (20:14-20:21 UTC)
- **严重度**: 🟡 中 — 不影响论文本身
- **描述**: N 和 Q 的 Stage 23 因 `references_verified.bib` 缺失而失败
  - 错误信息: `Missing or empty output: references_verified.bib`
  - Stage 23 耗时 0s — 意味着在验证前就失败了
  - Pipeline P 的 Stage 23 成功（11s），说明这不是系统性问题
- **关联**: R5-BUG-04 (见下方)

### OBS-R5-17: Pipeline O 大量 ablation failure (20:20 UTC)
- **严重度**: 🟡 中 — 代码质量问题
- **描述**: Pipeline O (IV estimators) 的 Stage 13 v2 检测到大量 copy-paste ablation 问题
  - 8+ 对 conditions 产生完全相同的输出
  - 例: `mean_bias_only_jive_evaluation_ablation` ≡ `two_stage_least_squares_wald_baseline`
  - 例: `no_instrument_density_geometry_risk_surface_ablation` ≡ `no_leverage_geometry_risk_surface_ablation`
- **关联**: R5-BUG-03, R4-BUG-13 — copy-paste ablation 问题持续存在

### OBS-R5-18: 纸面写作阶段高效 (Stage 16-22)
- **严重度**: 🟢 正面
- **描述**: 所有完成的 Pipeline 在纸面写作阶段均高效运行：
  - Stage 16 (PAPER_OUTLINE): 99-119s
  - Stage 17 (PAPER_DRAFT): 374-406s (~6-7min)
  - Stage 18 (PEER_REVIEW): 72s
  - Stage 19 (PAPER_REVISION): 242-277s (~4min)
  - Stage 20 (QUALITY_GATE): 9-12s
  - Stage 21 (KNOWLEDGE_ARCHIVE): 42-51s
  - Stage 22 (EXPORT_PUBLISH): 122-130s (~2min)
- **总计**: 纸面写作 + 导出约 15 分钟

### OBS-R5-19: Pipeline N 论文承认实验失败 (20:14 UTC)
- **严重度**: 🟡 中 — 影响论文质量
- **描述**: Pipeline N 的 paper_draft.md 中写道：
  > "the current execution failed before producing any analyzable spectral metrics"
- **分析**: 虽然 Stage 13 成功修复了 numpy 2.x 错误并重新运行了实验，但论文写作阶段可能
  没有从修复后的实验结果中获取数据，而是检测到了第一次失败的状态
- **关联**: 可能是 Stage 14 (RESULT_ANALYSIS) 没有正确读取 Stage 13 v2/v3 的结果

---

## 新发现 Bug

### R5-BUG-01: CodeSearcher query_gen.py — LLMClient.chat() 签名不匹配 ✅ 已修复
- **严重度**: 🟡 中 — 不阻塞 pipeline 但降低代码质量
- **文件**: `researchclaw/agents/code_searcher/query_gen.py:149`
- **描述**:
  - `llm.chat()` 被调用为 `llm.chat(system=..., user=..., max_tokens=...)`
  - 实际签名是 `chat(messages: list[dict], *, system=, max_tokens=)`
  - `user` 不是有效参数 → `TypeError`
  - 另外代码错误地用 `asyncio.run()` 包装同步方法
- **修复**:
  - 改为 `llm.chat([{"role": "user", "content": prompt}], system=..., max_tokens=...)`
  - 移除不必要的 `asyncio.run()` 和 `chat_sync` 分支
- **影响**: 修复后 CodeSearcher 可正常使用 LLM 生成搜索查询（仍需 GITHUB_TOKEN）

### R5-BUG-02: 代码生成使用已弃用/不存在的 numpy 2.x API（系统性）
- **严重度**: 🔴 高 — 导致所有实验首次运行失败
- **描述**: gpt-5.4 生成的代码使用了已在 numpy 2.0 中移除的 API：
  - `np.trapz` → 应使用 `np.trapezoid` (numpy 2.0 breaking change)
  - `np.erfinv` → 从未存在于 numpy，应使用 `scipy.special.erfinv`
  - `np.bool` / `np.int` 等 → 已在 numpy 1.24+ 移除
- **根因**: LLM 训练数据包含大量 numpy 1.x 代码，未适应 2.x 变化
- **自动修复**: Stage 13 (ITERATIVE_REFINE) 成功修复了这些问题 ✅
- **建议**: 在代码生成 prompt 中添加 numpy 2.x 兼容性提示，减少首次失败

### R5-BUG-03: Pipeline O copy-paste ablation 检测（已知问题复现）
- **严重度**: 🟡 中
- **描述**: Stage 10 deep quality check 检测到多组近似相同的 ablation 类
  - Fuller1 vs Fuller4: 仅超参数不同，方法体相同
  - Risk surface baseline vs ablation: 方法签名和体积完全相同
- **关联**: R4-BUG-13 (BUG-13 copy-paste ablation) — 该问题跨轮次持续存在
- **建议**: 需要在代码生成阶段强化 ablation 差异性检查

### R5-BUG-04: Stage 23 Citation Verify — references_verified.bib 缺失 ✅ FIXED
- **严重度**: 🔴 高 — 3/4 Pipeline 受影响
- **描述**: N、O 和 Q 在 Stage 23 因 `references_verified.bib` 未生成而失败
  - 错误: `Missing or empty output: references_verified.bib`
  - Stage 23 耗时 0s，说明在输出验证前就失败了
  - Pipeline P 的 Stage 23 成功（11s），同一引用验证逻辑正常工作
- **根因分析**:
  - Stage 23 在无引用时正确写入空的 `references_verified.bib`（executor.py L9082）
  - 但 contract validation（executor.py L9351）拒绝 `st_size == 0` 的文件
  - Pipeline P 有 19KB 的 references.bib → 验证后非空 → 通过
  - N/O/Q 无引用 → Stage 23 写空文件 → 被 contract validation 拒绝
- **修复**: 将空文件改为写入 BibTeX 注释 `% No references to verify\n`（executor.py L9085-9086）
  - 文件非空，通过 contract validation，同时语义上表示"无引用"

### R5-BUG-05: 论文未使用修复后的实验结果 ✅ FIXED
- **严重度**: 🔴 高 — 影响论文科学价值
- **描述**: Pipeline N/Q 的论文包含 "quality 2/10" 警告，声称实验失败
  但 Stage 13 成功修复了 numpy 错误并产生了完整的实验结果（论文表格中实际包含真实数据）
- **根因分析**: Stage 14 LLM analysis 在所有三次 refine 迭代中均给出 2/10（包括最新的非版本化 stage-14），
  而 BUG-23 guard（executor.py L7184）在 `_analysis_rating <= 2` 时强制 `has_real_metrics = False`，
  即使 `_collect_raw_experiment_metrics()` 已成功从 Stage 13 stdout 解析出真实指标
  - **注**: `_read_prior_artifact` 排序是正确的 — 非版本化目录确实是最新的（rollback 时旧目录会被重命名为 `_vN`）
- **修复**: 在 BUG-23 guard 中增加 `not _has_parsed_metrics` 条件（executor.py L7187）
  - 当 Stage 13 refinement 产生了可解析的真实指标时，不再被 analysis rating 覆盖
  - 同时保留了原始 BUG-23 防护：在确实没有真实指标时仍会触发

---

## 总结

### 整体评价

R5 是目前最成功的测试轮次：

| 指标 | R4 (feat/universal-codegen) | R5 (main) |
|------|---------------------------|-----------|
| 完美通过 (29/29) | 0/4 | **1/4 (Pipeline P)** |
| 近完美 (28/29) | 0/4 | **3/4 (N, O, Q)** |
| Stage 20 通过 | 0/4 (all rejected 2/10) | **4/4 (all degraded/pass)** |
| 崩溃/严重失败 | 1/4 (Pipeline K crash) | **0/4** |
| 平均完成阶段 | ~25/29 | **28.75/29** |
| 平均耗时 | ~3.5h | **~2.6h** |

### 关键改进
1. **Stage 20 Quality Gate 不再阻塞**: R4 中所有 Pipeline 被 2/10 拒绝，R5 全部通过
2. **自我修复能力可靠**: Stage 13 成功修复了所有 numpy 2.x API 不兼容问题
3. **跨领域能力验证**: 物理、经济学、流行病学、数值分析 4 个不同领域均可完成
4. **无崩溃**: 4/4 Pipeline 全部正常完成，无任何进程级崩溃

### 关键问题（全部已修复）
1. ✅ **R5-BUG-05**: BUG-23 guard 过度激进 → 论文声称实验失败
2. ✅ **R5-BUG-04**: Stage 23 写入空 bib 文件被 contract validation 拒绝 → 3/4 失败
3. ✅ **R5-BUG-01**: CodeSearcher query_gen.py 签名不匹配
4. ✅ **R5-BUG-02**: 代码生成使用已弃用 numpy 2.x API — 已在 7 个 prompt 中添加兼容性警告
5. ✅ **R5-BUG-03**: copy-paste ablation — 新增 <1% 近似检测 + prompt 强化
6. ✅ **R5-BUG-06**: LaTeX microtype 字体错误 — 已添加 `\usepackage{lmodern}`

### R5-BUG-06: LaTeX 编译失败 — pdfTeX font expansion 错误 ✅ FIXED
- **严重度**: 🟡 中
- **描述**: Pipeline Q 的 paper.tex 编译失败
  ```
  pdfTeX error (font expansion): auto expansion is only possible with scalable
  Fatal error occurred, no output PDF file produced!
  ```
- **根因**: `\usepackage[T1]{fontenc}` 激活了 T1 编码，但未加载可缩放字体（lmodern）
- **修复**: 在 `researchclaw/templates/conference.py` 的 NEURIPS_2024、NEURIPS_2025、GENERIC 三个模板中
  在 `fontenc` 之后添加 `\usepackage{lmodern}`

### R5-BUG-02: numpy 2.x API 不兼容 ✅ FIXED
- **修复范围**: 在以下 7 个 prompt 位置添加了 numpy 2.x 兼容性警告
  - `prompts.default.yaml` (legacy code_generation)
  - `prompts.py`: architecture_planning, generate_single_file, code_repair, iterative_improve, iterative_repair, code_exec_fix

### R5-BUG-03: copy-paste ablation ✅ IMPROVED
- **修复**: executor.py 新增 P8 近似检测（<1% relative diff → warning），补充了原有的精确匹配检测
- **注**: prompt 中已有 Rule 9 (ABLATION DIFFERENTIATION) 和 Rule 8 (METHOD RICHNESS) 的引导

### 后续排查结论
- **`_read_prior_artifact` 排序**: ✅ 确认正确 — 非版本化目录确实是最新的（rollback 重命名旧目录为 `_vN`）
- **Stage 14 quality rating 问题**: 所有 3 次 refine 迭代的 Stage 14 均给出 2/10 → 这是 LLM 分析偏保守的问题，
  但 BUG-05 的修复已绕过该问题（信任实际解析出的指标）

### 交付物检查

| Pipeline | paper.tex | references.bib | charts | code | LaTeX编译 |
|----------|-----------|---------------|--------|------|----------|
| N | ✅ | ❌ (S23 fail) | ✅ | ✅ | 未检查 |
| O | ✅ | ❌ (S23 fail) | ✅ | ✅ | 未检查 |
| P | ✅ (539行) | ✅ (405行) | ✅ (5张) | ✅ | ✅ |
| Q | ✅ | ❌ (S23 fail) | ✅ | ✅ | 未检查 |

### Pipeline 时间分布（以 Pipeline P 为例）

| 阶段 | 耗时 | 说明 |
|------|------|------|
| S1-S9 (研究+设计) | ~20min | 含 429 限流延迟 |
| S10 (代码生成) | ~47min | 最重的 LLM 阶段，3 次 attempt |
| S11 (资源规划) | ~14s | |
| S12-S13 (实验+修复) | ~15min | 首次失败 + 自动修复 + 重运行 × 2轮 refine |
| S14-S15 (分析+决策) | ~10min | 含 2 轮 refine 循环 |
| S16-S22 (论文写作+导出) | ~15min | |
| S23 (引用验证) | ~11s | |
| **总计** | **~2.4h** | |
