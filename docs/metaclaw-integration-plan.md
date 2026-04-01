# MetaClaw × AutoResearchClaw 集成方案

> **Status**: ✅ **Implemented & Merged to main** (v0.3.0, 2026-03-16)
>
> **目标**: 将 MetaClaw 的持续学习能力（技能注入、技能进化、PRM 评分、RL 训练）接入 AutoResearchClaw 的 23 阶段研究流水线，提升端到端论文生成质量。

---

## 一、项目概览

| 项目 | 定位 | 核心能力 |
|------|------|----------|
| **AutoResearchClaw** | 全自主研究流水线（Idea → Paper） | 23 阶段 Pipeline、文献检索、实验执行、论文写作、引用验证 |
| **MetaClaw** | Agent 持续进化平台 | 技能注入（Skill Injection）、技能进化（Skill Evolution）、PRM 奖励评分、RL 微调、空闲调度器 |

**集成核心思路**: MetaClaw 作为 AutoResearchClaw 的 **LLM 增强层**，通过多层次赋能提升每个阶段的 LLM 输出质量，并建立从研究失败中持续学习的闭环。

---

## 二、架构设计

### 2.1 集成架构总览

```
┌──────────────────────────────────────────────────────┐
│              AutoResearchClaw Pipeline                │
│  Stage 1 → 2 → ... → 23                             │
│                                                      │
│  ┌─────────────┐    ┌──────────────────────────────┐ │
│  │ LLMClient   │───▶│ MetaClaw Integration Layer   │ │
│  │ (原有)       │    │ (新增 metaclaw_bridge 模块)   │ │
│  └─────────────┘    └──────────┬───────────────────┘ │
│                                │                     │
│  ┌─────────────┐    ┌──────────▼───────────────────┐ │
│  │ Evolution   │◀──▶│ Lesson ↔ Skill 双向桥接      │ │
│  │ (原有)       │    └─────────────────────────────┘ │
│  └─────────────┘                                     │
└──────────────────────────┬───────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │     MetaClaw Proxy Server    │
            │     (FastAPI :30000)         │
            │                              │
            │  ┌────────────────────────┐  │
            │  │ SkillManager           │  │
            │  │ - 通用技能 (40+)        │  │
            │  │ - 研究专属技能 (新增)    │  │
            │  │ - 阶段映射技能检索      │  │
            │  └────────────────────────┘  │
            │                              │
            │  ┌────────────────────────┐  │
            │  │ SkillEvolver           │  │
            │  │ - 从失败中自动生成技能  │  │
            │  └────────────────────────┘  │
            │                              │
            │  ┌────────────────────────┐  │
            │  │ PRMScorer              │  │
            │  │ - 阶段输出质量评分      │  │
            │  └────────────────────────┘  │
            │                              │
            └──────────────┬──────────────┘
                           │
              ┌────────────▼────────────┐
              │   Upstream LLM API      │
              │   (OpenAI / Kimi / etc.) │
              └─────────────────────────┘
```

### 2.2 集成层次

| 层次 | 名称 | 改动范围 | 效果 |
|------|------|----------|------|
| **L1** | Proxy 透传 | 仅改配置 | AutoResearchClaw → MetaClaw Proxy → LLM，自动获得通用技能注入 |
| **L2** | 阶段感知技能 | 新增研究技能库 + 阶段映射 | 每个 Pipeline 阶段注入最相关的研究技能 |
| **L3** | Evolution 桥接 | 新增 bridge 模块 | AutoResearchClaw 失败教训 → MetaClaw 技能；双向学习闭环 |
| **L4** | PRM 质量门控 | 集成 PRMScorer | 在质量门控阶段（5/9/15/20）使用 PRM 提供客观评分 |
| **L5** | RL 持续训练 | MetaClaw RL 模式 | 从研究对话中持续微调模型（可选，需 GPU） |

---

## 三、详细任务分解

### Phase 0: 环境准备与分支管理

#### Task 0.1: 创建集成分支
```bash
cd /home/jqliu/projects/AutoResearchClaw
git checkout -b feat/metaclaw-integration
```
- 所有开发工作在此分支进行
- 定期 rebase main 保持同步

#### Task 0.2: MetaClaw 环境配置
```bash
cd /home/jqliu/projects/MetaClaw
python -m venv .venv
source .venv/bin/activate
pip install -e ".[evolve]"    # 安装核心 + 技能进化依赖
```
- 只安装 `skills_only` 模式所需依赖（不需要 GPU / RL）
- 如需 embedding 检索：`pip install -e ".[embedding]"`

#### Task 0.3: MetaClaw 基础配置
创建 `~/.metaclaw/config.yaml`:
```yaml
mode: skills_only

llm:
  provider: custom
  model_id: <与 AutoResearchClaw 相同的模型>
  api_base: <上游 LLM API 地址>
  api_key: <API Key>

proxy:
  port: 30000
  api_key: ""   # 内部调用，无需鉴权

skills:
  enabled: true
  dir: ~/.metaclaw/skills
  retrieval_mode: template
  top_k: 6
  task_specific_top_k: 10
  auto_evolve: true
```

#### Task 0.4: 验证 MetaClaw 代理可用
```bash
# 启动 MetaClaw
metaclaw start --mode skills_only

# 测试连通性
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"ping"}],"max_tokens":50}'
```

---

### Phase 1: L1 — Proxy 透传接入（最小改动）

**目标**: 零代码改动，仅通过配置让 AutoResearchClaw 经由 MetaClaw 代理调用 LLM。

#### Task 1.1: 修改 AutoResearchClaw 配置

修改 `config.researchclaw.yaml`:
```yaml
llm:
  provider: "openai-compatible"
  base_url: "http://localhost:30000"   # 指向 MetaClaw 代理
  api_key_env: ""
  api_key: ""                          # MetaClaw 无需鉴权
  primary_model: "<原模型名>"           # MetaClaw 会透传到上游
  fallback_models: []
```

#### Task 1.2: 兼容性适配

在 `researchclaw/llm/client.py` 中处理 MetaClaw 可能返回的 503 状态码（权重更新中）：

**文件**: `researchclaw/llm/client.py`
**改动**: 将 503 加入可重试状态码列表

```python
# 原有
_RETRYABLE_STATUS = {429, 500, 502, 504}

# 改为
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
```

#### Task 1.3: 端到端冒烟测试
```bash
# 1. 启动 MetaClaw
metaclaw start --mode skills_only

# 2. 运行 AutoResearchClaw 短流程
researchclaw run --topic "test topic" --config config.yaml
```

**验证点**:
- [x] AutoResearchClaw 能正常调用 LLM
- [x] MetaClaw 日志显示技能注入
- [x] 输出质量与直连 LLM 相当或更好

**预期交付**: AutoResearchClaw 透过 MetaClaw 运行，自动获得通用技能加持。

---

### Phase 2: L2 — 研究专属技能库 + 阶段映射

**目标**: 为 AutoResearchClaw 的 23 个阶段创建专属技能，并实现精准注入。

#### Task 2.1: 创建研究专属技能

在 `~/.metaclaw/skills/` 下新增以下技能（每个技能一个目录 + `SKILL.md`）：

| 技能名 | 类别 | 适用阶段 | 内容要点 |
|--------|------|----------|----------|
| `literature-search-strategy` | research | 3, 4 | 查询扩展、布尔组合、避免过宽搜索 |
| `paper-relevance-screening` | research | 5 | 相关性评分标准、排除低质量源 |
| `knowledge-card-extraction` | research | 6 | 结构化提取模板：方法/结果/局限 |
| `research-gap-identification` | research | 7 | 聚类分析、空白识别、创新角度发现 |
| `hypothesis-formulation` | research | 8 | SMART 假设、可证伪性检查 |
| `experiment-design-rigor` | research | 9 | 对照组设计、消融实验、统计功效 |
| `hardware-aware-coding` | coding | 10 | GPU/CPU 适配、内存管理、batch size 选择 |
| `experiment-debugging` | coding | 12, 13 | NaN/Inf 检测、收敛诊断、梯度检查 |
| `statistical-analysis` | data_analysis | 14 | p 值计算、效应量、置信区间 |
| `research-pivot-decision` | research | 15 | 证据权衡、PROCEED/PIVOT 决策框架 |
| `academic-writing-structure` | communication | 16, 17 | IMRaD 结构、段落逻辑、术语一致性 |
| `peer-review-methodology` | communication | 18 | 审稿视角、方法论-证据一致性检查 |
| `citation-integrity` | research | 23 | 引用验证层次、防幻觉策略 |

每个技能文件格式：
```markdown
---
name: literature-search-strategy
description: Design effective literature search queries for academic research
category: research
---
# Literature Search Strategy

## When to Use
When designing search queries for arXiv, Semantic Scholar, or other academic databases.

## Steps
1. **Decompose research topic** into 3-5 core concepts
2. **Generate synonyms** for each concept (e.g., "reinforcement learning" → "RL", "reward-based learning")
3. **Combine with Boolean operators**: (concept1 OR synonym1) AND (concept2 OR synonym2)
4. **Add temporal filters**: Prefer recent 3 years for fast-moving fields
5. **Iterative refinement**: If >200 results, narrow; if <10, broaden

## Anti-Patterns
- Avoid single-keyword queries (too broad, causes timeout)
- Avoid overly specific queries that miss relevant work
- Never rely on a single database source
```

#### Task 2.2: 阶段-技能映射模块

**新增文件**: `researchclaw/metaclaw_bridge/stage_skill_map.py`

```python
"""Maps AutoResearchClaw pipeline stages to MetaClaw skill categories."""

# 每个阶段对应的 MetaClaw 任务类型 + 推荐注入的研究专属技能
STAGE_SKILL_MAP: dict[str, dict] = {
    "topic_init": {
        "task_type": "research",
        "skills": ["literature-search-strategy"],
        "top_k": 4,
    },
    "problem_decompose": {
        "task_type": "research",
        "skills": ["research-gap-identification"],
        "top_k": 4,
    },
    "search_strategy": {
        "task_type": "research",
        "skills": ["literature-search-strategy"],
        "top_k": 6,
    },
    "literature_collect": {
        "task_type": "research",
        "skills": ["literature-search-strategy"],
        "top_k": 4,
    },
    "literature_screen": {
        "task_type": "research",
        "skills": ["paper-relevance-screening"],
        "top_k": 6,
    },
    "knowledge_extract": {
        "task_type": "research",
        "skills": ["knowledge-card-extraction"],
        "top_k": 4,
    },
    "synthesis": {
        "task_type": "research",
        "skills": ["research-gap-identification"],
        "top_k": 6,
    },
    "hypothesis_gen": {
        "task_type": "research",
        "skills": ["hypothesis-formulation"],
        "top_k": 6,
    },
    "experiment_design": {
        "task_type": "research",
        "skills": ["experiment-design-rigor"],
        "top_k": 6,
    },
    "code_generation": {
        "task_type": "coding",
        "skills": ["hardware-aware-coding"],
        "top_k": 6,
    },
    "resource_planning": {
        "task_type": "productivity",
        "skills": [],
        "top_k": 3,
    },
    "experiment_run": {
        "task_type": "automation",
        "skills": ["experiment-debugging"],
        "top_k": 4,
    },
    "iterative_refine": {
        "task_type": "coding",
        "skills": ["experiment-debugging"],
        "top_k": 6,
    },
    "result_analysis": {
        "task_type": "data_analysis",
        "skills": ["statistical-analysis"],
        "top_k": 6,
    },
    "research_decision": {
        "task_type": "research",
        "skills": ["research-pivot-decision"],
        "top_k": 4,
    },
    "paper_outline": {
        "task_type": "communication",
        "skills": ["academic-writing-structure"],
        "top_k": 4,
    },
    "paper_draft": {
        "task_type": "communication",
        "skills": ["academic-writing-structure"],
        "top_k": 6,
    },
    "peer_review": {
        "task_type": "communication",
        "skills": ["peer-review-methodology"],
        "top_k": 6,
    },
    "paper_revision": {
        "task_type": "communication",
        "skills": ["academic-writing-structure", "peer-review-methodology"],
        "top_k": 6,
    },
    "quality_gate": {
        "task_type": "research",
        "skills": ["peer-review-methodology"],
        "top_k": 4,
    },
    "knowledge_archive": {
        "task_type": "automation",
        "skills": [],
        "top_k": 2,
    },
    "export_publish": {
        "task_type": "automation",
        "skills": [],
        "top_k": 2,
    },
    "citation_verify": {
        "task_type": "research",
        "skills": ["citation-integrity"],
        "top_k": 4,
    },
}
```

#### Task 2.3: 阶段感知 HTTP Header 注入

修改 `researchclaw/llm/client.py`，在发送请求时附带阶段上下文 Header：

```python
# 在 _request() 方法中新增
headers["X-Session-Id"] = f"arc-{run_id}"
headers["X-AutoRC-Stage"] = stage_name    # 自定义 header，供 MetaClaw 日志追踪
headers["X-Turn-Type"] = "main"           # 确保触发技能注入
```

#### Task 2.4: MetaClaw 端自定义技能检索（可选增强）

如果需要更精准的阶段感知，可在 MetaClaw 的 `api_server.py` 中读取 `X-AutoRC-Stage` header，根据 `STAGE_SKILL_MAP` 调整 skill 检索策略。但这需要修改 MetaClaw 代码，可作为后续优化。

**预期交付**: 13 个研究专属技能 + 阶段映射配置，每个 Pipeline 阶段获得最相关的技能注入。

---

### Phase 3: L3 — Evolution ↔ Skill 双向桥接

**目标**: 让 AutoResearchClaw 的失败教训自动转化为 MetaClaw 技能，形成学习闭环。

#### Task 3.1: Lesson → Skill 转化器

**新增文件**: `researchclaw/metaclaw_bridge/lesson_to_skill.py`

**功能**:
1. 从 `evolution/lessons.jsonl` 读取高严重性教训（severity = "error"）
2. 按类别聚合同类失败
3. 调用 LLM 将教训批量转化为 MetaClaw 技能格式
4. 写入 `~/.metaclaw/skills/arc-xxx/SKILL.md`

**类别映射**:
```python
LESSON_CATEGORY_TO_SKILL_CATEGORY = {
    "SYSTEM": "automation",
    "EXPERIMENT": "coding",
    "WRITING": "communication",
    "ANALYSIS": "data_analysis",
    "LITERATURE": "research",
    "PIPELINE": "automation",
}
```

**转化 Prompt 模板**:
```
以下是自动化研究流水线中反复出现的失败教训。请将它们转化为可复用的技能指南。

失败教训:
{lessons_text}

请为每个关键失败模式生成一个技能，格式如下:
- name: 小写连字符命名 (如 arc-avoid-broad-queries)
- description: 一句话描述何时使用此技能
- category: {target_category}
- content: Markdown 格式的步骤指南 (5-10 行)
```

#### Task 3.2: Skill 效果回馈

**新增文件**: `researchclaw/metaclaw_bridge/skill_feedback.py`

**功能**:
1. 在每次 Pipeline 运行结束后，统计各阶段成功/失败
2. 关联当次运行中注入了哪些技能
3. 计算技能-成功率关联
4. 将低效技能标记，供 MetaClaw SkillEvolver 参考

**数据结构**:
```python
@dataclass
class SkillEffectivenessRecord:
    skill_name: str
    stage_name: str
    run_id: str
    stage_success: bool
    timestamp: str
```

存储位置: `evolution/skill_effectiveness.jsonl`

#### Task 3.3: 自动进化触发

在 `researchclaw/pipeline/executor.py` 的运行结束钩子中，添加:

```python
# Pipeline 完成后触发
async def _post_pipeline_hook(self, run_results: list[StageResult]):
    # 1. 提取教训
    lessons = extract_lessons(run_results)
    self.evolution_store.append_many(lessons)

    # 2. 将高严重性教训转化为技能 (如果 MetaClaw bridge 启用)
    if self.config.metaclaw_bridge.enabled:
        from researchclaw.metaclaw_bridge.lesson_to_skill import convert_lessons_to_skills
        new_skills = await convert_lessons_to_skills(
            lessons=[l for l in lessons if l.severity == "error"],
            llm=self.llm,
            skills_dir=self.config.metaclaw_bridge.skills_dir,
        )
        logger.info(f"Generated {len(new_skills)} new MetaClaw skills from run failures")
```

**预期交付**: 失败 → 教训 → 技能的自动闭环，每次运行都让系统变得更好。

---

### Phase 4: L4 — PRM 质量门控

**目标**: 在关键质量门控阶段使用 MetaClaw 的 PRM 评分器提供客观质量评估。

#### Task 4.1: PRM 评分器集成

**新增文件**: `researchclaw/metaclaw_bridge/prm_gate.py`

**功能**: 封装 MetaClaw PRMScorer，为 AutoResearchClaw 的质量门控提供评分。

```python
class ResearchPRMGate:
    """Uses MetaClaw's PRM scorer for objective quality assessment."""

    def __init__(self, prm_config: dict):
        self.scorer = PRMScorer(
            api_base=prm_config["api_base"],
            api_key=prm_config["api_key"],
            model=prm_config["model"],
            majority_votes=prm_config.get("votes", 3),
        )

    async def evaluate_stage_output(
        self,
        stage_name: str,
        instruction: str,
        output: str,
    ) -> float:
        """Returns -1.0 (fail), 0.0 (ambiguous), or 1.0 (pass)."""
        score = await self.scorer.score(instruction, output)
        return score
```

#### Task 4.2: 集成到质量门控阶段

在以下 4 个门控阶段添加 PRM 评分:

| 阶段 | 评分对象 | 评分指令 |
|------|----------|----------|
| Stage 5 (LITERATURE_SCREEN) | 筛选后的论文列表 | "评估文献筛选的相关性和覆盖度" |
| Stage 9 (EXPERIMENT_DESIGN) | 实验设计方案 | "评估实验设计的严谨性：对照组、消融实验、统计方法" |
| Stage 15 (RESEARCH_DECISION) | PROCEED/PIVOT 决策 | "评估决策依据的证据充分性" |
| Stage 20 (QUALITY_GATE) | 最终论文 | "评估论文的学术质量、创新性、方法论严谨性" |

**决策逻辑**:
```python
prm_score = await prm_gate.evaluate_stage_output(stage, instruction, output)

if prm_score == 1.0:
    # 通过，继续
    pass
elif prm_score == 0.0:
    # 模糊，使用原有逻辑决策
    pass
elif prm_score == -1.0:
    # 不通过，触发回退
    # Stage 5 → 回退到 Stage 4
    # Stage 9 → 回退到 Stage 8
    # Stage 15 → REFINE 或 PIVOT
    # Stage 20 → 回退到 Stage 16
```

#### Task 4.3: 配置项扩展

在 `config.researchclaw.yaml` 中新增:

```yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"
  skills_dir: "~/.metaclaw/skills"

  prm:
    enabled: true
    api_base: "https://api.openai.com/v1"   # PRM 评分用的 LLM API
    api_key_env: "PRM_API_KEY"
    model: "gpt-4o"
    votes: 3                                  # 多数投票次数
    gate_stages: [5, 9, 15, 20]              # 启用 PRM 的阶段

  lesson_to_skill:
    enabled: true
    min_severity: "error"                     # 仅转化 error 级别教训
    max_skills_per_run: 3                     # 每次运行最多生成 3 个新技能
```

**预期交付**: 关键阶段获得客观质量评分，降低低质量输出流入后续阶段的概率。

---

### Phase 5: L5 — RL 持续训练（可选）

> **注意**: 此阶段需要 GPU 和 Tinker/MinT 后端，如当前环境不具备可跳过。

**目标**: 利用研究流水线的对话数据持续微调模型。

#### Task 5.1: 切换 MetaClaw 到 MadMax 模式

```yaml
# ~/.metaclaw/config.yaml
mode: madmax

rl:
  enabled: true
  backend: tinker       # 或 mint
  model: <模型名>
  api_key: <Tinker API Key>
  batch_size: 4
  lora_rank: 32

scheduler:
  enabled: true
  sleep_start: "23:00"
  sleep_end: "07:00"
  idle_threshold_minutes: 30
```

#### Task 5.2: 会话生命周期管理

在 AutoResearchClaw 的 Pipeline runner 中管理 MetaClaw 会话:

```python
# Pipeline 开始时
headers["X-Session-Id"] = f"arc-{run_id}"

# Pipeline 结束时
headers["X-Session-Done"] = "true"  # 通知 MetaClaw 一次研究会话结束
```

这使 MetaClaw 能够在每次研究运行结束后触发技能进化和 RL 训练数据收集。

**预期交付**: 每次研究运行都成为模型改进的训练数据，长期持续提升。

---

## 四、阶段-技能映射总表

| 阶段 | 阶段名称 | MetaClaw 任务类型 | 注入技能 | top_k |
|------|----------|-------------------|----------|-------|
| 1 | TOPIC_INIT | research | literature-search-strategy | 4 |
| 2 | PROBLEM_DECOMPOSE | research | research-gap-identification | 4 |
| 3 | SEARCH_STRATEGY | research | literature-search-strategy | 6 |
| 4 | LITERATURE_COLLECT | research | literature-search-strategy | 4 |
| 5 | LITERATURE_SCREEN | research | paper-relevance-screening | 6 |
| 6 | KNOWLEDGE_EXTRACT | research | knowledge-card-extraction | 4 |
| 7 | SYNTHESIS | research | research-gap-identification | 6 |
| 8 | HYPOTHESIS_GEN | research | hypothesis-formulation | 6 |
| 9 | EXPERIMENT_DESIGN | research | experiment-design-rigor | 6 |
| 10 | CODE_GENERATION | coding | hardware-aware-coding | 6 |
| 11 | RESOURCE_PLANNING | productivity | — | 3 |
| 12 | EXPERIMENT_RUN | automation | experiment-debugging | 4 |
| 13 | ITERATIVE_REFINE | coding | experiment-debugging | 6 |
| 14 | RESULT_ANALYSIS | data_analysis | statistical-analysis | 6 |
| 15 | RESEARCH_DECISION | research | research-pivot-decision | 4 |
| 16 | PAPER_OUTLINE | communication | academic-writing-structure | 4 |
| 17 | PAPER_DRAFT | communication | academic-writing-structure | 6 |
| 18 | PEER_REVIEW | communication | peer-review-methodology | 6 |
| 19 | PAPER_REVISION | communication | academic-writing-structure, peer-review-methodology | 6 |
| 20 | QUALITY_GATE | research | peer-review-methodology | 4 |
| 21 | KNOWLEDGE_ARCHIVE | automation | — | 2 |
| 22 | EXPORT_PUBLISH | automation | — | 2 |
| 23 | CITATION_VERIFY | research | citation-integrity | 4 |

---

## 五、新增文件清单

```
AutoResearchClaw/
├── researchclaw/
│   └── metaclaw_bridge/           # 新增模块
│       ├── __init__.py
│       ├── config.py              # MetaClaw 集成配置
│       ├── stage_skill_map.py     # 阶段-技能映射
│       ├── lesson_to_skill.py     # 教训→技能转化器
│       ├── skill_feedback.py      # 技能效果追踪
│       ├── prm_gate.py            # PRM 质量门控
│       └── session.py             # MetaClaw 会话管理
├── docs/
│   └── metaclaw-integration-plan.md  # 本文档
└── tests/
    └── test_metaclaw_bridge/      # 集成测试
        ├── test_stage_skill_map.py
        ├── test_lesson_to_skill.py
        ├── test_prm_gate.py
        └── test_e2e_integration.py
```

**需修改的现有文件**:
| 文件 | 改动内容 |
|------|----------|
| `researchclaw/llm/client.py` | 添加 503 重试 + X-Session-Id/X-Turn-Type header |
| `researchclaw/config.py` | 新增 `metaclaw_bridge` 配置段 |
| `researchclaw/pipeline/executor.py` | 添加 post-pipeline hook 调用 lesson_to_skill |
| `config.researchclaw.example.yaml` | 添加 metaclaw_bridge 配置示例 |

---

## 六、实施路线图

```
Week 1: Phase 0 + Phase 1 (环境 + Proxy 透传)
  ├── Day 1-2: 环境配置、分支创建、MetaClaw 安装验证
  ├── Day 3-4: 配置修改、503 重试适配、冒烟测试
  └── Day 5:   端到端验证、记录基线指标

Week 2: Phase 2 (研究技能库)
  ├── Day 1-3: 编写 13 个研究专属技能 SKILL.md
  ├── Day 4:   实现阶段映射模块 + header 注入
  └── Day 5:   A/B 对比测试（有/无技能注入）

Week 3: Phase 3 (Evolution 桥接)
  ├── Day 1-2: 实现 lesson_to_skill 转化器
  ├── Day 3:   实现 skill_feedback 追踪
  ├── Day 4:   集成到 executor post-pipeline hook
  └── Day 5:   测试闭环：运行 → 失败 → 教训 → 技能 → 再运行改进

Week 4: Phase 4 (PRM 门控) + 收尾
  ├── Day 1-2: 实现 prm_gate + 集成到 4 个门控阶段
  ├── Day 3:   全流程端到端测试
  ├── Day 4:   性能调优、文档完善
  └── Day 5:   代码审查、合并准备
```

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| MetaClaw 代理增加延迟 | 每次 LLM 调用额外 ~50ms | 可接受；技能注入带来的质量提升远超延迟代价 |
| 技能注入增加 prompt token 消耗 | 每次调用增加 ~500-1000 tokens | 控制 top_k；关键阶段多注入，非关键阶段少注入 |
| MetaClaw 上下文截断 | 长 prompt 被静默截断 | 配置 `max_context_tokens` ≥ 32000 |
| Lesson→Skill 转化质量不稳定 | 生成无效技能 | 限制每次运行最多 3 个新技能；人工审核 |
| MetaClaw 进程崩溃 | Pipeline 中断 | 在 LLMClient 中添加 fallback：MetaClaw 不可用时直连上游 |
| 分支冲突 | 合并困难 | 改动集中在新模块，对原有代码侵入极小 |

### 关键缓解：Fallback 机制

在 `researchclaw/llm/client.py` 中实现:

```python
async def _request(self, ...):
    try:
        # 优先走 MetaClaw 代理
        return await self._http_post(self.base_url, ...)
    except (ConnectionError, Timeout):
        if self.config.metaclaw_bridge.fallback_url:
            # 降级直连上游 LLM
            logger.warning("MetaClaw proxy unavailable, falling back to direct LLM")
            return await self._http_post(self.config.metaclaw_bridge.fallback_url, ...)
        raise
```

---

## 八、成功指标

| 指标 | 基线（无 MetaClaw） | 目标（集成后） | 测量方法 |
|------|---------------------|----------------|----------|
| Pipeline 完成率 | 现有水平 | +15% | 统计 Stage 15 PROCEED 率 |
| 实验代码首次运行成功率 | 现有水平 | +20% | 统计 Stage 12 无需 Stage 13 的比例 |
| 论文 PRM 评分 | — | ≥ 0.6 平均分 | Stage 20 PRM 评分统计 |
| 引用验证通过率 | 现有水平 | +10% | Stage 23 验证通过率 |
| 技能库增长 | 40 (MetaClaw 原有) | +13 (研究专属) + 自动进化 | 技能目录文件数 |

---

## 九、API 需求清单

集成过程中可能需要以下 API（请确认可用性）:

| API | 用途 | 是否必需 |
|-----|------|----------|
| OpenAI-compatible LLM API | AutoResearchClaw + MetaClaw 共用 | **必需** |
| PRM 评分用 LLM API | Phase 4 质量门控（可与上述相同） | Phase 4 需要 |
| Tinker/MinT API | Phase 5 RL 训练 | **可选** |
| arXiv API | AutoResearchClaw 文献检索（已有） | 已配置 |
| Semantic Scholar API | AutoResearchClaw 文献检索（已有） | 已配置 |

---

## 十、快速启动命令

完成集成后的典型使用流程:

```bash
# 1. 启动 MetaClaw 代理
cd /home/jqliu/projects/MetaClaw
source .venv/bin/activate
metaclaw start --mode skills_only --port 30000

# 2. 运行增强版 AutoResearchClaw
cd /home/jqliu/projects/AutoResearchClaw
git checkout feat/metaclaw-integration
researchclaw run \
  --topic "Your research idea" \
  --config config.researchclaw.yaml

# 3. 查看 MetaClaw 技能注入日志
metaclaw status

# 4. 查看新进化出的技能
ls ~/.metaclaw/skills/arc-*/
```
