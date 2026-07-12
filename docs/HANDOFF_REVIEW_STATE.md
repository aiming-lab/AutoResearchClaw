# Critical Reviewer 交接备忘(Claude / Kiro 通用)

Status: living handoff note. 任何接手的 reviewer agent(Claude Cowork、Claude 4.8、
Kiro)先读本文件,再读引用的权威 spec。本文件只描述状态,不是权威契约;
契约以 docs/ 下各 spec 为准。发现本文件与 spec 或代码冲突时,以 spec/代码为准,
并更新本文件。

## 1. 项目与审查角色

- 仓库:AutoResearchClaw-v2-clean —— 25-stage 自动科研 pipeline。
- 你的角色:critical release-safety reviewer。只读审查、不改代码、不 commit,
  除非用户明确要求。每轮输出使用用户指定的固定格式(Verdict / P0 / P1 / P2 /
  专项评估 / Missing adversarial tests / Commit recommendation),P0/P1 必须给
  file:line、可复现绕过输入、最小修法。
- 审查哲学(全程一致,不得放宽):
  - fail-closed 优先;宁可失败,不可伪造。
  - 确定性代码拥有结构、hash、状态与最终裁决;LLM 只做有界局部语义。
  - 任何 artifact 不得自证(hash/counts 必须可从权威输入重算)。
  - LLM 不得自证语义支持(writer/critic 隔离、plan 约束替换)。
  - 失败路径不得残留上一轮成功产物("FAILED + 旧产物"是本项目的原罪事故)。
  - metadata / LLM summary 不是 claim-support 证据。
  - pipeline_validation 允许验证工程流程,不允许伪造证据。
  - Stage 19 sectional 语义与 release gate 只能收紧,永不放宽;
    不允许 alias、自动补 BibTeX、rejected-candidate top-up、waiver 复活。

## 2. 当前精确状态(E1 已提交,E2 就绪)

- HEAD: 236931e "literature: seal cite-key v2 registry"(= E0 已提交)。
- E1:9559fd0 "literature: enforce bounded Stage 5 screening" 已提交并通过
  二次复审。首次 APPROVE 曾因控制流探针复现 skip-stub、影子 Stage 4
  三件套和 FAILED 后 canonical partial shortlist 三条绕过而撤回;9559fd0
  已以双层禁跳、canonical 固定路径、success-only shortlist 命名和对抗测试
  关闭上述问题。
- 当前下一里程碑:E2 Stage 6 JSON evidence cards。首个 E2 commit 先强制
  Stage 6 strict 消费 screening_report.json,再实现 evidence-card schema。
  涉及:researchclaw/literature/screening.py(新)、
  researchclaw/pipeline/stage_impls/_literature.py、
  researchclaw/pipeline/contracts.py、
  tests/test_literature_screening_contract.py(新)、tests/test_rc_executor.py、
  docs/CITATION_EVIDENCE_PIPELINE_SPEC.md。
- 未跟踪目录 .agents/、experiments/、kb-* 永远不入库、不纳入审查范围。

## 3. 已落地里程碑(commit 时间线)

Stage 19 止血补丁与格式契约:
- (早期 commits)Stage 19 容错修复(claim-scope 分流回退)、
  f4b35c8 Stage 17/18 格式契约(结构校验 fail-closed)。

Phase A/B(sectional revision):
- 2965af4 Phase A 确定性 manuscript section model(parse->merge byte-identical,
  自定义 CommonMark 行协议)。
- 212eee6 B0 评论 ledger/plan 契约;f345cde B1 validator+merge+manifest;
  517c8ba B2 feature-flag 执行壳;88530c5 B3 有界 LLM planner/writer/critic。

Phase C(release gates):
- cc1270e Phase C spec;e7a2435 C-1 contract governance(共用 selector、
  运行时校验、删 synthetic waiver);d4090f2 C0 重放 schema(contract/model
  绑定、attempts_sha256、strict loaders);0202692 C1 磁盘重放 gate
  (sectional_release_audit.py + check_sectional_revision);
  66915e0 C2 非默认 dry-run 配置。

Citation evidence workstream(进行中):
- d4db0ab CITATION_EVIDENCE_PIPELINE_SPEC(v1 abstract-only + v2 HITL 全文,
  50 条对抗测试清单);236931e E0 cite-key v2 registry;E1 见上,待 commit。

真实事故背景(驱动本 workstream):
- runs/hwsec-sectional-dry-run-20260711 在 Stage 19 被正确阻断
  (draft 引用 venkatakeerthy2020ir2vec,canonical 为 ...scpecscp);
  该 run 同时暴露 Stage 5 15/15 模板 shortlist、Stage 6 全模板 cards、
  Stage 17 读全量 candidates、Stage 23 仅论文级 topical relevance。
- runs/hwsec-scaffold-v2 是更早的 Stage 19 IncompleteRead 事故现场。

## 4. 待办排程

按 CITATION_EVIDENCE_PIPELINE_SPEC §14:
- E2 Stage 6 JSON evidence cards(注意:excerpt 必须从 stage-04
  candidates.jsonl 的完整 abstract 复算,producer 显式 extraction_status,
  禁止模板伪装成功)。
- E3 citation allowlist + 顶层 citation_policy + effective policy artifact
  (绑定 run 内 config 快照,Stage 17/18/20 共享)。
- E4 Stage 16 v1 citation plan(与 E5 同发布窗口)。
- E5 Stage 17 封闭写作面(移除 candidates 目录投喂;退役 P3 全量
  preverification,不得转产过滤权威 bib;unknown key 直接 FAILED,无修复)。
- E6 Stage 18/20 policy 一致性(窄数字检测,一次 review 重生成,禁
  not_actionable 逃逸)。
- E7 Stage 23 有界验证(只验实引集合;删除 _remove_citations_from_text 的
  静默改稿行为,_review_publish.py:2960/:3235)。
- E8 Stage 24 claim-specific support + dataset-origin truth audit。
- E9 release replay(gate 类,独立 commit)。
- F0 v2 全文 HITL(PAUSED/resume 执行器支持、两轮上限、内容寻址 KB store、
  读副本单向导出)。

E9 已登记义务(来自各轮审查,勿丢):
- 确定性重放 Stage 5 prefilter/排序/top-150 admission,检出报告分区篡改
  (报告自身无法防写盘后篡改,已有探针证明)。
- 交叉校验 screening_report.claim_scope == Stage 9 contract.claim_scope。
- 扩展而非并行新建现有 check_citation_support / check_claims_provenance。
- allow_suspicious 旗标不得豁免任何新 citation/evidence 错误族。
- 新错误族永不进入 degraded exit-2 allowlist(加 allowlist 快照测试)。

小额未决(非阻塞):
- Stage 5 首批 LLM 调用可加 retries=1(降低 strict scope 单抖动整段失败)。
- B3 遗留:ProviderResponseError 与 transport 的 error_type 区分(诊断)。
- Phase C spec :61 曾有陈旧 find_experiment_contract 引用,C-1 已改代码,
  文档如仍存留请顺手清。

## 5. 不变量速查(审查时对照)

- Stage 4 三件套(candidates/references.bib/cite_key_registry)同源于单一
  collection registry,seal 自校验;bib 自 registry 绑定后 run 内不可变。
- Stage 5 是语义筛选权威边界;被拒/未筛/模板行无任何回流路径。
- allowlist = shortlist ∩ canonical bib ∩ 有效 card ∩ 可复算 excerpt;
  无 top-up。
- 数量不足:pipeline_validation 动态降目标(艺件传递到 17/18/20);
  research_release 失败(v2 才有有界回滚 Stage 3,≤2 次)。
- v1 证据仅 abstract 逐字摘录(双 hash + code-point span);
  v2 学术论文需 verified text-native PDF,其他 source kind 各有 canonical
  retained source;PDF/全文永不进 git 与 public deliverables。
- KB fulltext_store(内容寻址)是唯一真源;论文写作目录只放 KB→写作目录
  单向导出的阅读副本,可清理。
- Stage 19 与 Phase C gate 完全冻结。
- run_manifest.reviewer:writer_model/critic_model 属 Stage 15 语义,
  sectional_writer_model/sectional_critic_model 属 Stage 19,不得互相耦合。

## 6. 验证命令(本地 venv,Python ≥3.11)

```bash
.venv/bin/python -m pytest tests/test_citation_identity.py \
  tests/test_literature_screening_contract.py tests/test_rc_literature.py -q
.venv/bin/python -m pytest tests/test_manuscript_sections.py \
  tests/test_sectional_revision.py tests/test_sectional_validation.py \
  tests/test_sectional_execution.py tests/test_sectional_llm.py -q
.venv/bin/python -m pytest tests/test_rc_executor.py \
  tests/test_release_check_v2.py tests/test_rc_prompts.py -q
.venv/bin/python scripts/probe_release_gates.py
git diff --check
```

## 7. App 注意事项

Claude(Cowork 桌面,含切到 4.8):
- 沙盒 shell 是 Python 3.10:datetime.UTC 需注入 shim
  (`datetime.UTC = datetime.timezone.utc`)才能跑 sectional 测试;
  这是沙盒限制,不是代码缺陷,勿据此判 P0。
- 沙盒 bash 45s 上限:大套件用 -k 分片或后台不可靠,优先跑 targeted 套件,
  全量以用户本地结果为准。
- 用户偏好:简洁直接,少废话;中文回复;审查结论要可执行。

Kiro:
- 直接读本仓库文件即可获得全部上下文;同样遵守 §1 纪律与 §5 不变量。
- 三方对齐惯例:架构分歧先出"对齐文本",由 marcel 裁决后才写 spec,
  spec 经另一方复审 ACCEPT 后才开工代码;实现按里程碑窄 commit,
  每个 commit 由非实现方审查。

## 8. 审查指令模板(用户常用)

"请对当前未提交的 <milestone> diff 做只读 critical review,不修改文件。
范围:<files>。重点核查:<列表>。本地验证:<结果>。
输出:## Verdict / ## P0 findings / ## P1 findings / ## P2 findings /
## <专项评估> / ## Missing adversarial tests / ## Commit recommendation。
P0/P1 给 file:line、复现输入、最小修法。最后回答:能否独立 commit,
下一里程碑是否就绪。"
