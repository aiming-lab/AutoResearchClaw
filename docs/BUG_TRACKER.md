# Bug Tracker & TODO

> 实验运行期间发现的 bug 和待修复事项。实验结束后统一修复。

## 已发现的 Bug

### BUG-01: Stage 2 合约缺少 queries.json 输出 (已修复)
- **状态**: ✅ 已修复 (commit `19c74a0`)
- **描述**: `contracts.py` 中 Stage 2 (PROBLEM_DECOMPOSE) 的 `output_files` 包含 `queries.json`，但实际实现只生成 `problem_tree.md`。`queries.json` 实际在 Stage 3 生成。
- **影响**: Pipeline 在 Stage 2 直接失败
- **修复**: 从 Stage 2 output_files 移除 `queries.json`，从 Stage 3 input_files 移除 `queries.json`

### BUG-02: gpt-5.4 持续 429 限流
- **状态**: ⏳ 待观察
- **描述**: 同时运行多个 pipeline 时，gpt-5.4 频繁返回 429。fallback 机制可以兜底但速度大幅下降。
- **影响**: 运行时间显著增加（Case 2 上轮从 ~2.5h 增至 ~6h）
- **建议**: 考虑增加 pipeline 间的启动间隔，或实现全局 API 调用速率协调

### BUG-03: S2/arXiv 文献搜索 429 限流
- **状态**: ✅ 已缓解 (commit `63c5a7d` circuit breaker)
- **描述**: Semantic Scholar 和 arXiv API 在并发请求时频繁 429
- **影响**: 文献收集阶段延迟，但 circuit breaker 保证最终完成

### BUG-04: Stage 10 深度质量检查 — 类方法不足
- **状态**: ✅ 已加强 (远程 commit `855c201`)
- **描述**: 生成的代码中多个类只有 1 个非 dunder 方法，质量检查报告 "algorithm classes should have at least __init__ + one core method"
- **影响**: 代码质量评分降低，但不阻塞 pipeline
- **远程修复**: 新增 Check 6 — ablation 子类必须 override 父类至少一个非 dunder 方法，否则报警告。修复写入 `validator.py` 和 `executor.py` 的 repair prompt。

### BUG-05: Stage 10 深度质量检查 — UnboundLocalError 风险
- **状态**: ✅ 已修复 (远程 commit `855c201`)
- **描述**: 生成代码中变量只在 if 分支内赋值，但在分支外使用（如 main.py:289 `mask`, main.py:300 `out` 等）
- **影响**: 生成的实验代码可能在运行时崩溃
- **远程修复**: 新增 `auto_fix_unbound_locals()` 函数（`validator.py`），在 Stage 10 代码生成后自动检测 if-only 变量并插入 `var = None` 初始化。`executor.py` 在深度检查前调用。

### BUG-05 更新: UnboundLocalError 问题在 v8r3 中大幅恶化
- **状态**: ✅ 已修复 (被 `auto_fix_unbound_locals()` 覆盖)
- **描述**: v8r3 中 Case 3 (PEFT) 生成的代码有 **47 处** UnboundLocalError 风险（data.py 27 处, methods.py 20 处, main.py 2 处），远超 v8r2 的 8 处。Case 2 也有 8 处。
- **根因**: LLM 生成的代码模式为 `if cond: x = val` 后直接 `use(x)`，缺少 else 分支或默认值初始化
- **远程修复**: 程序化自动修复已集成到 Stage 10 pipeline 中

### BUG-06: P9 Metric direction mismatch
- **状态**: ✅ 已修复
- **描述**: 配置写 `minimize` 但实验代码声明 `direction=higher`，自动纠正为 `maximize`
- **影响**: 可能影响实验结果的正确性
- **修复**: (1) Stage 9 prompt 中注入 `metric_direction` 约束; (2) Stage 12 code_generation prompt 中强制 METRIC_DEF direction 与 config 一致; (3) 取消 auto-correction，改为仅 warn 并保持 config 值

### BUG-07: Stage 23 CITATION_VERIFY 失败率高
- **状态**: ✅ 已修复
- **描述**: 上轮 Case 1 和 Case 3 都在 Stage 23 失败（28/29），仅 Case 2 通过
- **影响**: 最终 pipeline 状态标记为 failed
- **根因**: (1) `_check_citation_relevance()` 最多只处理 30 个 citation，超出的无评分; (2) 无评分的 citation 在 hard cap 排序时被当作 0.0 分全部删除
- **修复**: (1) 改为分批处理所有 citation (batch=30); (2) 无评分 citation 默认 0.7（已验证=大概率相关）

### BUG-08: CodeGen `'str' object has no attribute 'get'` (v8r3 新发现)
- **状态**: ✅ 已修复
- **严重度**: 中 — 不阻塞 pipeline（有 fallback），但连续失败 6 次
- **描述**: Case 1 在 Stage 14 (RESULT_ANALYSIS) 触发 CodeGen 时连续报 `'str' object has no attribute 'get'`。疑似 LLM 返回了纯字符串而非 dict，代码对返回值调 `.get()` 导致 AttributeError。
- **远程修复**: executor.py 中 `_check_ablation_effectiveness` 等函数已加 `isinstance` 保护
- **本地修复**: `code_agent.py` 中 `_parse_json` 结果增加 `isinstance(review, dict)` 检查

### BUG-09: FigureAgent 无法生成图表 (v8r3 新发现)
- **状态**: ✅ 已修复
- **描述**: Case 1 Stage 14 中 `FigureAgent produced no charts, falling back`。FigureAgent 可能因上游 CodeGen 失败或数据格式问题无法生成图表。
- **影响**: 论文缺少可视化图表，影响质量分数
- **根因**: `_condition_summaries` 在 metrics 不含 `/` 分隔符时为空，导致 Planner 没有数据
- **修复**: (1) 从 `metrics_summary` fallback 构建 condition_summaries; (2) 从 `structured_results` 二次 fallback; (3) 向 FigureAgent 传入 `best_run_metrics` 作为数据源兜底

### BUG-10: Degenerate refine cycle (v8r3 新发现)
- **状态**: ✅ 已修复 (远程 commit `e30443e`)
- **描述**: Case 1 出现 `P6: Degenerate refine cycle detected, injecting PROCEED hint`。Pipeline 检测到实验迭代循环没有实质进展，自动注入 PROCEED 跳出。
- **远程修复**: 根因是 LLM 在迭代 refine 时重命名/替换 condition 名称导致漂移。修复方案：在 `iterative_improve` prompt 中注入 `exp_plan.yaml` 锚定，并禁止改名条件。

## 远程额外修复（BUG_TRACKER 未记录的问题）

### RFix-01: Baselines dict→list 转换 (commit `855c201`)
- 若 LLM 输出 baselines 为 dict 而非 list，`executor.py` 现在自动转换为 `list(dict.keys())`

### RFix-02: Gymnasium 环境版本 v4→v5 (commit `855c201`)
- `benchmark_knowledge.yaml` 中 HalfCheetah-v4→v5, Hopper-v4→v5

### RFix-03: Time budget 注入到 Stage 9 (commit `855c201`)
- 实验设计 prompt 中增加 `time_budget_sec` 约束，防止生成超时的实验方案

### RFix-04: 代码模板 optimizers.py→models.py (commit `855c201`)
- 代码生成模板从 `optimizers.py` 改为 `models.py`，并禁止生成只有 import/pass 的 stub 文件

### RFix-05: RL 稳定性修复提示 (commit `e30443e`)
- `iterative_repair` prompt 中增加 gradient clipping、LR cap、reward normalization、NaN guard 等常见 RL 修复建议

## 待修复汇总

| Bug | 优先级 | 状态 |
|-----|--------|------|
| BUG-02 gpt-5.4 限流 | 低 | ⏳ 待观察 (外部限制) |

所有代码层面的 bug 已修复。

## 待办事项 (TODO)

- [x] 拉取远程更新，对比 bug 修复状态
- [x] 更新 BUG_TRACKER 标注远程已修复项
- [x] 修复 BUG-06: 在 experiment design 阶段校验 metric direction 一致性
- [x] 修复 BUG-07: 分析 Stage 23 引用验证高失败率原因
- [x] 完善 BUG-08: CodeGen 调用处增加 str 类型保护
- [x] 修复 BUG-09: FigureAgent 输入数据格式检查
- [ ] 分析本轮 (v8r3) 三个 case 的质量分数，对比上轮 (v8r2)
- [ ] 考虑增加 pipeline 间的 API 调用协调机制

## 历史质量分数对比

| 版本 | Case 1 (Graph-RAG) | Case 2 (Diffusion) | Case 3 (PEFT) | 平均 |
|------|--------------------|--------------------|---------------|------|
| v8r2 | 5.2/10 | 8.0/10 | 5.8/10 | 6.3 |
| v8r3 | 待定 | 待定 | 待定 | 待定 |

---
*最后更新: 2026-03-16*
