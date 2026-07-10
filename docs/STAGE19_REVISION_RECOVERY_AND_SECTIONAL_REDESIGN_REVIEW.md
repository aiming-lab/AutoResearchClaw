# Stage 19 修订恢复与章节化改造审查材料

## 审查范围

本材料包含两项必须分开裁决的工作：

1. **止血修复（已实现、未提交）**：处理 Stage 19 长度强化重试中的网络中断，并清除复用 run 目录里的旧产物。
2. **章节化改造（仅方案、尚未实现）**：把整篇论文重新生成改为逐章节修订和确定性合并。

止血修复应独立提交。章节化改造不得静默混入该提交。

## Claude 审查结论

2026-07-10 的 live-diff 复审结论：

- tactical patch：**APPROVE**，无 P0/P1；
- sectional redesign：**MODIFY**，方向正确，但应补齐下述 tokenizer、ledger、merge hash 和 authoritative release gate；
- `release_check.py` 仍未修改；
- advisory deliverables flag 暂不并入止血提交，留到 Phase C 处理。

## 真实失败证据

Run 目录：`runs/hwsec-scaffold-v2`

2026-07-10 的真实执行结果：

```text
Stage 17 PAPER_DRAFT: DONE
Stage 18 PEER_REVIEW: DONE
Stage 19 PAPER_REVISION:
  draft: 5999 words
  first revision: 3448 words
  action: 低于 80% 长度门槛，触发第二次完整重写
  retry: IncompleteRead(0 bytes read)
  final status: FAILED
```

第一次 LLM 调用返回了可读但明显缩水的修订稿。第二次调用再次请求完整论文，provider/client 重试耗尽后异常上抛，Stage 19 没有写出本轮正文。

同一 run 目录还残留更早一次执行产生的 `stage-19/paper_revised.md`。Stage metadata 正确记录了 FAILED 和零产物，deliverables manifest 也有 `not_release_ready=true`，但旧文件仍可见，形成“失败状态 + 旧正文”的混合现场。

## 根因判断

`IncompleteRead` 是直接触发因素，整篇重写才是架构根因：

1. 约 6000 词的长输出更容易被截断或发生传输中断。
2. 模型改善部分章节时，可能压缩或遗漏无关章节。
3. 当前长度门禁发现缩水后，再请求一次整篇生成，放大成本和中断概率。

止血补丁只处理本次故障，不宣称解决整篇重写的架构问题。

## 已实现的止血修复

### 修改文件

```text
researchclaw/pipeline/stage_impls/_review_publish.py
tests/test_rc_executor.py
```

`scripts/release_check.py` 未修改。

### 行为变化

#### 1. Stage 19 启动时删除本阶段旧产物

只删除当前 `stage_dir` 下由 Stage 19 拥有的三个文件：

```text
paper_revised.md
revision_notes_internal.md
revision_retry_failure.json
```

不删除 `stage-19_v*`、其他阶段、Stage 17 原稿、Stage 18 审稿意见或 deliverables。

#### 2. 首次修订调用仍然硬失败

第一次修订调用继续使用 `_chat_with_prompt(..., retries=2)`。该调用失败时异常照常上抛，不生成新的 `paper_revised.md`。

#### 3. 第二次长度强化调用按 claim scope 分流

第一次输出低于原稿 80% 时，第二次长度强化调用也使用 `retries=2`。

如果该可选调用最终抛出 `RuntimeError`：

- `pipeline_validation`：
  - 第一次短修订保存为 `revision_notes_internal.md`；
  - 写 `revision_retry_failure.json`，记录错误类型、错误信息和字数；
  - 完整原稿写为 `paper_revised.md`；
  - Stage 19 返回 DONE，由 Stage 20 执行原有严格质量检查。
- `research_release`、`exploratory` 或无效 experiment contract：
  - 异常继续上抛；
  - 不写本轮 `paper_revised.md`；
  - Stage 19 保持 fail-closed。

Claim scope 优先读取 Stage 9 不可变 experiment contract。contract 解析或校验失败时按 `research_release` 处理，禁止回退。

#### 4. 诊断文件进入 artifact 清单

发生回退时，`StageResult.artifacts` 包含：

```text
paper_revised.md
revision_notes_internal.md
revision_retry_failure.json
```

诊断文件不进入 `evidence_refs`，只有 `paper_revised.md` 是论文产物。

### 必须保持的不变量

- `research_release` 不能把失败的修订重试转换为 DONE。
- 无效 Stage 9 contract 不能启用回退。
- 首次调用失败时不能暴露旧 `paper_revised.md`。
- 回退正文必须是完整原稿，而不是缩水的第一次输出。
- Stage 20 质量阈值、degradation 语义不变。
- Stage 24 claims/provenance 门禁不变。
- `release_check.py` 不变。
- `pipeline_validation` 仍不得获得 release exit code 0。

### 测试结果

新增三条对抗测试：

1. `pipeline_validation`：第二次调用失败时保留完整原稿，写齐并登记诊断文件，两次调用均使用 `retries=2`。
2. 首次调用失败：删除旧 `paper_revised.md`，不暴露替代正文。
3. `research_release`：第二次调用失败仍硬失败，不写回退正文和诊断 JSON。

当前验证：

```text
3 targeted Stage 19 tests passed
tests/test_rc_executor.py + tests/test_release_check_v2.py: 272 passed
py_compile: passed
git diff --check: passed
```

## 长期方案：逐章节修订 + 确定性合并

### 设计原则

LLM 只负责局部语义修订。章节识别、顺序、hash、合并、证据边界和验收状态应由确定性代码拥有。

### 建议状态机

```text
paper_draft.md + reviews.md
        |
        v
确定性章节解析
        |
        v
revision plan: review issue -> section_id
        |
        v
逐章节 LLM 修订
        |
        v
逐章节确定性校验
        |
        v
按原顺序确定性合并
        |
        v
跨章节一致性审计
        |
        v
paper_revised.md + revision manifest
```

### 1. 确定性章节解析

使用 fence-aware CommonMark tokenizer 的 heading token 生成稳定记录，不使用逐行正则：

```json
{
  "section_id": "results",
  "heading": "## Results",
  "ordinal": 5,
  "original_sha256": "...",
  "body": "..."
}
```

必须做到：

- 保留 front matter 和第一个 heading 前的正文；
- 保留 heading 层级、标题文字和原始顺序；
- 正确处理 fenced code、公式、表格、figure refs 和 citation keys；
- stable ID 由代码生成，不让 LLM 命名；
- duplicate/ambiguous ID 直接失败，不能静默合并。

### 2. 有界修订计划

LLM 可以把审稿意见映射到一个或多个现有 `section_id`，但必须返回严格 JSON。Plan schema v1 额外保留 `global` 伪目标；它只能被分解为具体 section，或进入跨章节审计，不能触发整篇生成。确定性校验拒绝未知 section ID、未知字段和格式错误。

每条审稿意见必须处于以下状态之一：

```text
assigned
resolved
unresolved
not_actionable_with_reason
```

任何意见都不能从 ledger 中消失。

### 3. 逐章节修订

单次 LLM 请求只包含：

- 当前章节原文；
- 分配给该章节的审稿意见；
- grounded experiment metric whitelist；
- 允许的 citation keys 和 claim/evidence 约束；
- 必要时提供相邻章节的短摘要，而非全文。

LLM 无权新增、删除、重命名、重排或合并章节。返回值只包含修订后的 section body 和机器可读的 resolution record。

### 4. 逐章节硬校验

接受修订前，确定性代码至少检查：

- section ID 和 heading 不变；
- 未引入未知 citation key；
- 未引入 grounded whitelist 外的实验数字；
- 定量 claim 仍可进入 Stage 24 provenance closure；
- 必要 figure/table refs 未被静默删除；
- 章节长度没有异常缩水；
- Markdown fences 和结构分隔符保持平衡。
- section body 非空，且不能引入任何新 heading；
- citation 白名单来自 Stage 18 时点的规范 `references.bib`，不能从 draft 反推；
- 数字白名单按等价格式归一化，例如 `0.85`、`85%`、`8.5e-1`；
- plan、resolution record 和 manifest 均含 `schema_version`，未知字段拒绝；
- review ledger 条目总数必须与输入审稿意见数一致；
- per-section retry 上限和实际次数必须进入 manifest。

只重试失败章节。其他已通过或无需修改的章节必须保持稳定。

### 5. 确定性合并

按原始 parser 顺序合并。无需修改的章节必须 byte-for-byte 保留。LLM 不负责组装最终论文。

Manifest 应记录每节的原始/修订 hash、关联 review issue、validator 结果、重试次数和最终状态。

`paper_revised.md` 的 hash 必须等于根据 manifest 中 section hash 和原始间隔信息确定性合并后的复算结果。章节化模式下不得存在第二条正文写入路径。

### 6. 跨章节一致性审计

合并后只做有界审计：

- Abstract、Results、Conclusion 的指标一致性；
- 术语和方法名一致性；
- citation-key 一致性；
- 重复或矛盾 claim；
- 未解决审稿意见。

审计只输出 finding，并把修改重新路由到具体章节；禁止再次整篇重写。

### 建议产物

```text
stage-19/
  revision_plan.json
  sections/
    000-front-matter.original.md
    000-front-matter.revised.md
    010-abstract.original.md
    010-abstract.revised.md
    ...
  section_revision_manifest.json
  unresolved_comments.json
  consistency_audit.json
  paper_revised.md
```

### Claim-scope 语义

- `pipeline_validation`：
  - 某节 transport failure 可保留原节；
  - section 和对应 review issue 必须标记 unresolved；
  - Stage 20 必须能读取 unresolved 状态；
  - release exit code 0 仍不可能。
- `research_release`：
  - bounded retries 后仍有 required section unresolved，Stage 19 必须 FAILED；
  - DONE 不能隐藏任何回退；
  - 所有 required revisions 通过校验后才能合并。

### 三道 unresolved 门禁

1. Stage 19：`research_release` 有 required section unresolved 时直接 FAILED。
2. Stage 20：读取 `unresolved_comments.json`，非空时至少封顶为 degraded。
3. `release_check.py`：章节化模式下独立检查 unresolved 非空、manifest 缺失和 merge hash 不匹配，任一命中均 exit 非零。

Stage 20 不是 authoritative release gate，不能替代第三道检查。

## 建议实施顺序

### Phase A：确定性文档模型

- 实现 section parser 和 merger；
- 用 round-trip 测试证明 `parse -> merge` byte-identical；
- 覆盖 fence 内 heading、重复 heading、公式、表格和 front matter；
- Stage 19 启动时清理自己拥有的 `sections/`、plan、manifest 和 audit 产物，防止 resume 读取 stale 文件；
- 暂不接入 LLM。

### Phase B：修订计划和单章节执行

- 实现严格 plan schema 和 review ledger；
- 每次只修一节；
- 增加 local retry 和 section validators；
- 不修改现有 release gates。

### Phase C：一致性审计和发布集成

- 增加跨章节确定性检查；
- 把 unresolved 状态传递给 Stage 20，并在 release_check 增加 authoritative blocker；
- 增加 dropped section、fabricated metric、unknown citation、unresolved comment、stale artifact 对抗测试。

## Claude 必答问题

请审查 live diff 和本材料，不要直接修改代码。

1. 止血补丁是否存在 `research_release` P0/P1 fail-open？
2. `RuntimeError` 是否是第二次 LLM 调用的正确异常边界？是否可能吞掉非 transport 编程错误？
3. `pipeline_validation` 使用完整原稿 + 显式诊断并标 DONE 是否合理？是否应增加独立 decision/status？
4. resume 和 `stage-19_v*` 情况下，旧产物是否仍有绕过路径？
5. 当前 Stage 17 生成格式是否适合 Markdown heading parser，还是需要更完整的 document AST？
6. 接受任何 section revision 前，还缺哪些必须由确定性代码执行的不变量？
7. 全局审稿意见如何映射到章节，且不重新引入整篇生成？
8. unresolved section 应成为 release_check 直接 gate，还是传给 Stage 20 即可？请给出完整 fail-closed 路径。
9. 还缺哪些 adversarial tests 和 artifact schema 字段？
10. 分别给出裁决：
    - tactical patch：APPROVE / MODIFY / REJECT；
    - sectional redesign：ACCEPT / MODIFY / REJECT。

要求输出：

```text
## Tactical patch verdict
## P0/P1 findings
## P2 findings
## Sectional redesign verdict
## Missing invariants
## Required tests
## Recommended implementation order
## Blocking questions
```
