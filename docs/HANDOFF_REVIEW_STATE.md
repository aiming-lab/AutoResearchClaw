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

## 2. 当前精确状态(E6 已推送,E7 已复审通过待提交)

- HEAD:eeea1cf E6 "literature: enforce effective citation policy in review and
  quality"。E4/E5 同窗口 commit 为 9f05e69,其父 commit 为 42b3dc1 E3。
  时间线新增 b3efbaa "docs: record E1 review handoff"(纯 docs)。
- E1:9559fd0 已提交。历史教训:首次 APPROVE 因控制流探针复现 skip-stub、
  影子 Stage 4 三件套、FAILED 后 canonical partial shortlist 三条绕过而
  撤回,后以双层禁跳、canonical 固定路径、success-only 命名关闭。
- E2:e9a7fbd 已提交(JSON 权威 card、25 码点 excerpt 下限双入口、
  default-deny manifest、zero-evidence 不发布、Stage 7 消费前全链重放)。
- E3:**已提交为 42b3dc1**(复审通过后落库)。内容:顶层 citation_policy 严格双层解析(_strict_int,
  拒 bool/str)、Stage 6 citation_allowlist 全量重放、run-local config
  三方绑定(pointer/history/checkpoint,CLI 每次运行写绑定、同秒冲突
  拒绝)、Stage 16 effective policy 单真源、Stage 17/18/20 三方
  load_effective 重放 + 失败先清 canonical、Stage 18 窄数字检测(含
  number-word 首 token 回退)+ 一次 regeneration、prompt bank 数字引用
  要求清零 + 回归守卫、SKIP_FORBIDDEN {4,5,6,16}。复审中修复过的 P1:
  hep.py peer_review/quality_gate 残留 30-60(双真源)、检测器两词
  count 组吞 qualifier 致 "at least fifteen unique citations" 漏检。
- E4/E5 已完成。E4 新增 strict deterministic
  `citation_plan.py`;E5 将 Stage 17 收窄为 final-plan-only prompt,
  post-HITL 生成 citation + experiment-fact closure,Stage 18 消费前重放,
  Stage 22/23 固定到 registry-bound Stage 4 bibliography,Stage 22 不再
  联网补 key 或静默删 marker。**E4/E5 必须同发布窗口**,
  中间 checkout 不得启动 pipeline。
  首轮复审结论 MODIFY:发现 Stage 22/23 仅对 full Stage 4 bib
  超集校验、未重放 allowlist/final plan,以及 experiment-fact 的
  整数/非 Results/任意 x100/影子 stage 假阴性。已修复并补对抗测试;
  二次定点复审已 APPROVE,无 P0/P1,已作为单一窄范围
  commit 落库。
- E6:Stage 18 citation-count detector 扩展为版本化闭集
  表达(`increase references to 30`/`30 references are required` 等),仍只检查
  Actionable Revisions;最多一次整份 review regeneration。Stage 20 在质量
  LLM 之前重放 allowlist+final plan,仅计 eligible planned keys,并要求
  `actual >= effective_minimum`(不强制超过 target,无硬编码 15)。
  只读复审已 APPROVE,无 P0/P1,已作为独立 E6 commit 落库;
  已有二次违规 regeneration→FAILED
  端到端测试,并补 Stage 20 已达 minimum 但混入 in-bib/non-allowlist
  key 仍 FAILED 的反凑数控制流测试。
  已提交并推送为 eeea1cf。
- E7 working tree 已获只读复审 APPROVE,无 P0/P1:Stage 23 固定读取 canonical
  `stage-22/paper_final.md`,只把最终实引 key 的 bounded bibliography 送入
  verifier;verifier results 与 relevance JSON 都要求 exact closure、有限
  0-1 分数且拒重复 key。Stage 23 不再拥有正文修复权限:
  `_remove_citations_from_text` 及 executor 导出已删除,
  `paper_final_verified.md` 始终是 Stage 22 正文逐字副本。hallucinated 在
  所有 scope FAILED;其他不完整验证仅 pipeline_validation 可 degraded,
  exploratory/research_release 均 FAILED。入口先清旧 Stage 23 自有产物,
  防 FAILED + stale verified artifacts。复审后的 P2 收尾也已落地:
  删除死代码 `annotate_paper_hallucinations`,拒绝 symlink `paper.tex`,
  no-bib 分支复用权威 citation extractor;并补 hallucinated 全 scope fatal、
  bounded-bib closure、无 canonical bib、symlink paper 对抗测试。尚未 commit。
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
  50 条对抗测试清单);236931e E0 cite-key v2 registry;9559fd0 E1;
  e9a7fbd E2;42b3dc1 E3;9f05e69 E4/E5;eeea1cf E6;E7 WIP(见 §2)。

真实事故背景(驱动本 workstream):
- runs/hwsec-sectional-dry-run-20260711 在 Stage 19 被正确阻断
  (draft 引用 venkatakeerthy2020ir2vec,canonical 为 ...scpecscp);
  该 run 同时暴露 Stage 5 15/15 模板 shortlist、Stage 6 全模板 cards、
  Stage 17 读全量 candidates、Stage 23 仅论文级 topical relevance。
- runs/hwsec-scaffold-v2 是更早的 Stage 19 IncompleteRead 事故现场。

## 4. 待办排程

按 CITATION_EVIDENCE_PIPELINE_SPEC §14:
- E2 已完成:Stage 6 JSON evidence cards(注意:excerpt 必须从 stage-04
  candidates.jsonl 的完整 abstract 复算,producer 显式 extraction_status,
  禁止模板伪装成功)。
- E3 已完成:citation allowlist + 顶层 citation_policy + effective policy artifact
  (绑定 run 内 config 快照,Stage 17/18/20 共享)。
- E4 已完成:Stage 16 v1 citation plan(与 E5 同发布窗口)。
- E5 已完成:Stage 17 封闭写作面(移除 candidates 目录投喂;退役 P3 全量
  preverification,不得转产过滤权威 bib;unknown key 直接 FAILED,无修复)。
- E6 已完成:Stage 18/20 policy 一致性(窄数字检测,一次 review 重生成,禁
  not_actionable 逃逸)。
- E7 已复审通过待提交:Stage 23 有界验证(只验实引集合;删除
  `_remove_citations_from_text` 的静默改稿 authority;严格 scope 完整性门禁)。
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
- E9 重放 active config 时必须复用 citation_policy.py 的
  resolve_active_config_snapshot 同一规则,不得自行放宽无 pointer 分支。
- card_extraction_failures.json 与 screening_partial.jsonl 均非 authority。

小额未决(非阻塞):
- Stage 5 首批 LLM 调用可加 retries=1(降低 strict scope 单抖动整段失败)。
- B3 遗留:ProviderResponseError 与 transport 的 error_type 区分(诊断)。
- Phase C spec :61 曾有陈旧 find_experiment_contract 引用,C-1 已改代码,
  文档如仍存留请顺手清。
- prompt bank 回归守卫是单向 pattern(range 在前、关键词在后),
  "Citation count 30-60" 这类关键词在前的形式不会命中;E4 顺手加反向臂
  `(citations?|references?)[^.\n]{0,80}\d+\s*-\s*\d+`。
- Stage 18 检测器仍是有意收窄的数字+引用名词+要求动词闭集;
  不做泛语义冲突推断,C3 零 unresolved 是 release 兜底。
- E4/E5 残留 P2:无单位裸整数指标(如 `an F1 of 87`)尚不解析;
  grounded 若以百分整数存储而正文写分数会 fail-closed 误报;
  public-origin 的 `our GPU measurements` 词表可能误报。均不放宽 gate,
  留 E6/E7 根据真实 run 调整。
- E6 detector 残留 P2:`cite N references` 在 Actionable Revisions 中可能把
  描述句当要求句;`should have 30 references`/`aim for 30 sources`
  尚未命中。方向分别是 fail-closed 误报与词表不完备,不改变
  Stage 20/22 确定性底线;留词表 v2 收敛。

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

## 9. 新会话启动指令(切换模型/新建 project 时用)

第一条消息建议:
"你是 AutoResearchClaw-v2-clean 的 critical release-safety reviewer。
先完整阅读 docs/HANDOFF_REVIEW_STATE.md,按其 §1 纪律、§5 不变量执行,
用 §8 模板接收审查任务。先用 git log --oneline -8 与 git status 核对
§2 状态是否仍然准确,不准确以仓库为准并更新本备忘。不要修改代码,
除非我明确要求。"

审查方法论要点(历次教训的浓缩,新模型务必内化):
- 不要只确认"校验函数存在",要确认它在实际控制流的必经路径上
  (E1 首次 APPROVE 被撤回的根因)。
- 主动构造绕过输入并在沙盒实测(探针),不要只读测试名称。
- 对"another agent 声称已修复"逐条 file:line 复核,包括自己上轮的结论。
- 承认并纠正自己的错误结论是流程的一部分,已发生过两次
  (E1 skip-stub 误判、E3 parse-layer 初判被 from_dict 内建校验推翻)。
