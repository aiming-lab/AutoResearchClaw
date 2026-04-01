# Pipeline Critical Fixes V8 — 投稿级论文质量修复

## 目标
修复所有阻止 Pipeline 产出符合 AI 顶会投稿标准论文的问题。

---

## Tier 1: 阻断性问题（必须立即修复）

### T1.1 Title 提取 Bug
- **问题**: converter 从 markdown 提取 title 时，可能错误抓取表格标题（如 "Table 1 – Aggregate primary_metric across methods."）
- **文件**: `researchclaw/templates/converter.py`
- **修复**: 加入 `_TITLE_REJECT_RE` 和 `_METRIC_DUMP_RE` 正则，`_is_bad_title()` 过滤表格/图表/指标标题，回退到真正的论文标题
- **状态**: ✅ 已修复

### T1.2 Markdown Fence 泄漏到 LaTeX
- **问题**: LLM 输出的 ` ```markdown ` fence 没有被 converter 清除，直接出现在 .tex 中
- **文件**: `researchclaw/templates/converter.py`
- **修复**: 增加智能 fence 清洗，仅移除文档类 fence（markdown/latex/text/bibtex），保留代码 fence（python/java 等）
- **状态**: ✅ 已修复

### T1.3 Section 层级错误
- **问题**: LLM 输出用 `##` (H2) 作为主要章节标题，converter 将其映射为 `\subsection` 而非 `\section`
- **文件**: `researchclaw/templates/converter.py` + `researchclaw/prompts.py`
- **修复**: converter 自动检测 body 最低 heading level 并做 level shift（H2→`\section`），prompts 明确要求用 `#` 做主标题
- **状态**: ✅ 已修复

### T1.4 BibTeX journal 字段填 arXiv 分类代码
- **问题**: `journal = {cs.CY}` 应该是 arXiv preprint 格式，不是 journal 名
- **文件**: `researchclaw/literature/models.py`
- **修复**: 检测 arXiv category 模式 → 自动转换为 `journal = {arXiv preprint arXiv:XXXX.XXXXX}` 格式
- **状态**: ✅ 已修复

### T1.5 Abstract 长度失控 + 含原始变量名
- **问题**: Abstract ~500 词（应 150-250），且包含 `bayesian_optimization/primary_metric = 0.8607` 等原始键名
- **文件**: `researchclaw/templates/converter.py`
- **修复**: `check_paper_completeness()` 增加 abstract 长度检查（>300 词警告）和原始变量名检测
- **状态**: ✅ 已修复

---

## Tier 2: 高优先级（显著提升论文质量）

### T2.1 Quality Gate 真正执行
- **问题**: Stage 20 总是返回 DONE，verdict 从未被 runner 检查
- **文件**: `researchclaw/pipeline/executor.py`
- **修复**: `_execute_quality_gate()` 当 score < threshold 时返回 `StageStatus.FAILED`，增加 pass/fail 日志
- **状态**: ✅ 已修复

### T2.2 文献筛选过于激进
- **问题**: 87 篇候选 → 仅 5 篇通过（94% 拒绝率），会议需要 20-40 篇
- **文件**: `researchclaw/pipeline/executor.py`
- **修复**: keyword pre-filter 从 ≥2 放宽到 ≥1，最低保留数从 6 提高到 15，LLM 返回太少时自动补充
- **状态**: ✅ 已修复

### T2.3 跨域论文过滤
- **问题**: 量子计算、社会学论文混入 RL 论文的参考文献
- **文件**: `researchclaw/prompts.py`
- **修复**: 已在 V7 修复中通过 `literature_screen` prompt 强化领域匹配规则（P2/P6 fixes）
- **状态**: ✅ 已修复（V7）

### T2.4 图表 DPI 不达标
- **问题**: 全部 savefig 使用 dpi=150（会议要求 ≥300）
- **文件**: `researchclaw/experiment/visualize.py`
- **修复**: 所有 `dpi=150` → `dpi=300`（5 处）
- **状态**: ✅ 已修复

### T2.5 强制必需章节验证
- **问题**: NeurIPS/ICLR 要求 Limitations 章节 — 当前不检查
- **文件**: `researchclaw/templates/converter.py` + `researchclaw/prompts.py`
- **修复**: `check_paper_completeness()` 增加 Limitations 章节检测；`writing_structure` block 增加 MARKDOWN FORMATTING 规则
- **状态**: ✅ 已修复

---

## Tier 3: 高价值架构级修复

### T3.1 RESEARCH_DECISION 质量验证
- **问题**: _parse_decision() 仅提取关键词，不验证最低标准（≥2 baselines, ≥3 seeds 等）
- **文件**: `researchclaw/pipeline/executor.py`
- **修复**: 在 decision 提取后增加质量检查，验证决策文本是否提及 baselines/seeds/metrics，警告缺失项并写入 decision_structured.json
- **状态**: ✅ 已修复

### T3.2 FigureAgent 合并到当前分支
- **问题**: FigureAgent 代码在 main 分支但不在 feat/metaclaw-integration
- **修复**: `git checkout main -- researchclaw/agents/figure_agent/ researchclaw/agents/__init__.py researchclaw/agents/base.py tests/test_figure_agent.py`
- **状态**: ✅ 已修复

### T3.3 负面结果处理
- **问题**: 当方法表现不如 baseline 时，论文仍写成 positive contribution
- **文件**: `researchclaw/pipeline/executor.py` + `researchclaw/prompts.py`
- **修复**: `_detect_result_contradictions()` 已实现 NULL/NEGATIVE 结果检测，advisories 注入 paper_draft prompt 上下文；prompts 中 `hypothesis_gen`、`paper_draft`、`paper_revision` 均已包含 negative result 处理指导
- **状态**: ✅ 已修复（已有实现）

### T3.4 Citation Verify 改为阻断性
- **问题**: CITATION_VERIFY 在 NONCRITICAL_STAGES 中，失败不阻断导出
- **文件**: `researchclaw/pipeline/stages.py`
- **修复**: 从 NONCRITICAL_STAGES 移除 CITATION_VERIFY
- **状态**: ✅ 已修复

### T3.5 论文分段写作容错
- **问题**: 3 次 LLM 调用中任一超时，对应章节丢失
- **文件**: `researchclaw/pipeline/executor.py`
- **修复**: `_write_paper_sections()` 三次 LLM 调用均增加 `retries=1`（自动重试 1 次），仍失败则用 `[PLACEHOLDER]` 标记缺失章节，确保后续流程不中断
- **状态**: ✅ 已修复

---

## 额外修复

### T-extra.1 Agent Config 集成
- **问题**: feat/metaclaw-integration 分支缺少 CodeAgentConfig / BenchmarkAgentConfig / FigureAgentConfig
- **文件**: `researchclaw/config.py`
- **修复**: 添加三个 agent config dataclass 及其解析函数，集成到 ExperimentConfig
- **状态**: ✅ 已修复

---

## 完成记录

| 时间 | 修复项 | 状态 |
|------|--------|------|
| 2026-03-15 | T1.1 Title 提取 Bug | ✅ |
| 2026-03-15 | T1.2 Markdown Fence 泄漏 | ✅ |
| 2026-03-15 | T1.3 Section 层级错误 | ✅ |
| 2026-03-15 | T1.4 BibTeX journal 字段 | ✅ |
| 2026-03-15 | T1.5 Abstract 验证 | ✅ |
| 2026-03-15 | T2.1 Quality Gate 执行 | ✅ |
| 2026-03-15 | T2.2 文献筛选放宽 | ✅ |
| 2026-03-15 | T2.4 图表 DPI 升级 | ✅ |
| 2026-03-15 | T2.5 必需章节验证 | ✅ |
| 2026-03-15 | T3.2 FigureAgent 合并 | ✅ |
| 2026-03-15 | T3.4 Citation Verify 阻断性 | ✅ |
| 2026-03-15 | T-extra.1 Agent Config | ✅ |
| 2026-03-15 | T3.1 Decision 质量验证 | ✅ |
| 2026-03-15 | T3.3 负面结果处理 | ✅ (已有) |
| 2026-03-15 | T3.5 分段写作容错 | ✅ |

**已完成**: 15/15 (100%)
