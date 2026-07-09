# Kiro Handoff — AutoResearchClaw Stage 10/12 Reviewer + Architect

你接手 AutoResearchClaw-v2-clean 的 critical reviewer + AI-for-Science 架构师角色。
核心职责:守住反造假 fail-closed 门禁,推进 Stage 10/12 重构。绝不为了让 run 通过而放宽任何 gate。

## 仓库与背景
- 路径:`/Users/sheva/Win工作区/AutoResearchClaw-v2-clean`(分支 `codex/reconcile-main-20260705`)
- release hardening 已闭环并 push(commits `6544065` / `5e376ac` / `74aa02b` / `ebc60c9` / `82ffab7`)。
- 真实 DeepSeek run 卡在 Stage 10/12,失败形态漂移(OpenCode timeout、DeepSeek IncompleteRead、缺 dataset_origin、ExperimentHarness API 误用、smoke rc=1、Stage 12 zero metrics)。
- 根因:Stage 10 接口无边界(自由生成完整实验工程)。目标架构见 `docs/STAGE10_12_EXPERIMENT_RUNTIME_REDESIGN.md`:契约先行 + 确定性 scaffold + 受限 plugin + 多候选统一 evaluator + sha256 封存。

## 当前未提交改动(先验证再拆两个 commit,可暂不 push)
- **Commit A(Stage 10 战术稳定化)**:`config.deepseek.yaml`、`prompts.default.yaml`、`researchclaw/experiment/{docker_sandbox,sandbox,validator}.py`、`researchclaw/pipeline/stage_impls/_code_generation.py`、`tests/{test_entry_point_validation,test_rc_validator,test_stage10_smoke_gate}.py`
- **Commit B(仅文档)**:`docs/STAGE10_12_EXPERIMENT_RUNTIME_REDESIGN.md`(以及本文件 `docs/KIRO_HANDOFF.md`)
- **不要提交**:`experiments/anomaly_representations/`、`experiments/detection_f1/`、`runs/`、`.agents/`(本地材料;不想看到就加 `.gitignore`,别提交)。
- 提交前逐行看 `config.deepseek.yaml` diff,确认没动 `fallback_to_code_agent`(须 false)/`allow_docker_fallback`(须 false)/任何门禁阈值。

## 立即执行(不调用 DeepSeek)
```
.venv/bin/python -m pytest tests/test_stage10_smoke_gate.py tests/test_entry_point_validation.py tests/test_rc_validator.py -q
.venv/bin/python -m py_compile researchclaw/pipeline/stage_impls/_code_generation.py
```
绿了才继续。

## 接下来只实现 Phase 1 + Phase 2(纯收紧,勿一口气上 scaffold/plugin)
- **Phase 1**:Stage 9 输出 `experiment_contract.yaml`(字段:`schema_version`, `claim_scope`∈{pipeline_validation,exploratory,research_release}, `dataset_origin`∈{synthetic,public,local_hardware}, `primary_metric{key,direction,minimum_valid_value}`, `smoke_budget_sec`, `run_budget_sec`, `allowed_inputs/outputs`, `evaluator{command,owner:scaffold,required_result_keys}`, `safety`, `sealing`)。契约缺字段/矛盾 → Stage 9 fail-closed。契约 Stage 9 封存后 immutable。
- **Phase 2**:Stage 10 写 `selected_candidate/` + `selected_candidate_manifest.json`;Stage 12 只读 sealed candidate 并校验 sha256;release_check 拒绝 `stage-10/candidates/*`。Phase 2 必须先于 scaffold/plugin(Phase 4)落地。

## 实现时必须写死的不变量(否则仍 fail-open,这些是历轮审查的 P1)
1. **claim_scope 后门要在 Phase 2 就接**:`research_release ∧ dataset_origin=synthetic ⇒ block` 的 release_check 拒绝不能拖到 Phase 6;Stage 9 前门 + release_check 后门同期存在。
2. **`selected_candidate/` 必须 code-only**(不含 `results.json`/任何 metric 产物);Stage 12 从零 recompute;发现数值产物即拒绝。原因:allowlist 挡得住 `stage-10/*` 路径(已实测 DENY),但挡不住"stage-10 数字被复制进 `stage-12/runs/*.json`"。
3. **sealing manifest 必须含 candidate + `scaffold_sha256` + `contract_sha256`**,且 Stage 12 拒绝 `selected_candidate/` 里任何"不在 manifest 中"的文件(完整性 default-deny,不只黑名单 smoke/runs/attempts/candidates)。
4. **`evaluator.owner=scaffold` 是必要非充分**;真正不变量是"plugin 永不接触评测 ground truth":scaffold 持有 train/test split 与 y_test,plugin 只见 X_test,metric 由 scaffold 从(plugin predictions, scaffold 持有 y_test)计算。
5. **Stage 10 candidate 记录写 `stage-10/candidates/`,绝不写 run-root `attempts/`**(后者是 release_check 允许的证据路径)。
6. **过渡期**(Phase 4 之前 scaffold 尚未拥有 results.json):保留当前 prompt 层 dataset_origin honesty 约束 + smoke gate。

## 绝不放宽
- Stage 12 zero-metric / <30s / crash-signal 三门禁;`has_real_data`/`no_real_data`(仅人工签名 waiver);`dataset_origin` 要求;claims 证据 allowlist(默认拒绝);citation 支持门禁;smoke 产物永不进 experiment_summary/claims/release_check;scaffold-owned evaluator;selected_candidate sha256 校验。
- 不许用"加大 timeout / 再加一轮 free-form repair / 让 smoke 指标下游 / 让 candidate 记录成证据 / 把 synthetic 当 public/local_hardware / 从 failed run 出 release 交付物"来解决问题。

## 验证(Phase 1+2 之后,用新 run_dir,勿复用 hwsec-v1-smoke2)
```
python -m researchclaw run -c config.deepseek.yaml -o runs/hwsec-runtime-v1 --to-stage EXPERIMENT_RUN
python scripts/release_check.py runs/hwsec-runtime-v1 --json
```
**第一轮合格标准(注意:不是 exit 0)**:
- Stage 1-12 稳定跑完;Stage 10 smoke 不污染 Stage 12 evidence;Stage 12 有真实 recompute 的 metrics。
- 若数据是 synthetic,release_check 必须 block/degraded —— 这是 PASS。synthetic 却 exit 0 = **P0 假阳性**。
- 过了再谈 Stage 13-25 和 submission_candidate。

## 输出纪律
- 每轮给:**Verdict / P0-P1 findings**(每条带 `file:line` 和"为何 fail-open")**/ 最小修法**(不放宽 gate)**/ 需补测试**。只在有 traceback 证据时才改主修点,别猜。
- 未决 blocking 问题(要先定再写 Phase 4 代码):非 ML 域(quantum/biology/physics)的 plugin 接口如何映射;无 predictions-vs-labels 型指标的 evaluator 如何 scaffold-owned;N 候选与"≥1 有效候选否则 FAILED";claim_scope=exploratory 的 release 处置;contract 一经 Stage 9 封存即 immutable(Stage 11/13 不得改)。
