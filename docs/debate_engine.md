# Debate Engine 逻辑详解

> 对应实现:`researchclaw/pipeline/debate.py`(`run_debate`)
> 当前接入阶段:Stage 8(hypothesis)。Stage 14/18 的接入见待合并的 PR #318。

debate engine 把项目原来"单模型多视角一次合成"的浅做法,升级成**多模型、多轮、评分/合成分离**的真辩论。

---

## 0. 定位与触发条件

- **opt-in**:Stage 8 未进入 tournament 分支,且 `build_panel_llms(config)` 返回非空 panel 时,才路由到这里。构建面板需要 `debate_enabled=true`；ACP 或构建失败时返回空面板。
- **接入点**:
  - Stage 8(hypothesis,`_synthesis.py`)— 传 `synthesizer=llm`
  - PR #318 提议接入 Stage 14(analysis)和 Stage 18(peer review)；当前版本尚未接入。
- **provider 无关**:引擎只调 `client.chat(messages, *, system=..., max_tokens=...)`,因此可离线 mock 测试。

入口签名:

```python
run_debate(panel, judge, roles, variables, *,
           rounds, synth_prompt, out_dir, prompts,
           author_model="", gen_max_tokens=8192, synthesizer=None)
```

---

## 1. 前置与角色绑定

1. `panel` 为空 → `ValueError`(调用方必须保证非空)。
2. `rounds = _resolve_rounds(rounds)`:rebuttal 轮数,可被环境变量 `ARC_DEBATE_ROUNDS` 覆盖,下限 0。
3. **消融钩子** `ARC_ABL_DISABLE_DEBATE=1`:把 roles 塌缩成只剩第一个角色 → 退化为单视角(消融实验用)。
4. **角色 → 模型 round-robin 绑定**:

   ```python
   role_model[name] = panel[i % len(panel)]
   ```

   panel 顺序由 `build_panel_llms` 给出:`[primary, reviewer, fallback...]` 去重。
   primary 和 fallback 使用主服务配置；模型名不同于 primary 的 reviewer 成员通过 reviewer 工厂构建,使用自己的 `reviewer_*` 服务地址、凭据和适配器。按模型名去重时保留最先出现的成员,因此 reviewer 与 primary 同名时保留 primary。
   只有一个模型时也会进入引擎,所有角色共用该模型；至少两个不同模型是形成多模型辩论的条件,不是进入引擎的门槛。
   典型配置下:innovator → claude-sonnet-4.6、pragmatist → gpt-4o、contrarian → gemini-2.5-pro。
   角色名与倾向来自 active prompt bank(ML bank = innovator / pragmatist / contrarian)。

---

## 2. 开场陈述(round 0)

对每个角色渲染其 system/user 提示词(注入 `topic` + `synthesis`),用 `_chat_with_retry` 调对应模型:

```python
text = _chat_with_retry(role_model[name], user, system, gen_max_tokens, label=...)
if not text.strip():
    continue                       # 重试后仍空/失败 → 丢弃该角色
current[name] = text
写 {name}.r0.md
```

**关键健壮性:**

- **`gen_max_tokens=8192`**:防止 reasoning 模型(如 gemini-2.5-pro)的隐藏推理 token 把可见输出饿死(配合 client 层 reasoning floor)。
- **`_chat_with_retry`(一次重试)**:空内容或抛异常先**重试一次**再放弃,堵住"一次瞬时抖动就让整个视角消失"。
- **空护栏**:两次都失败才丢弃该角色——绝不让空白进入 rebuttal/合成。

---

## 3. Rebuttal 轮

`for r in 1..rounds`(默认 `debate_rounds=1`):

```python
prev = 当前所有非空视角的快照
if len(prev) < 2: break            # 少于 2 个无从反驳
for name in role_names:
    if name not in prev: continue  # 已丢弃的角色不参与
    others = 拼接「除自己外」每个角色的上一轮发言
    sp = prompts.sub_prompt("debate_rebuttal", role, own_position, others)
    text = _chat_with_retry(...)   # 同样带重试
    if not text.strip():
        continue                   # 空 → 保留上一轮,不覆盖
    current[name] = text
    写 {name}.r{r}.md
```

要点:

- 每个角色**看到其他人上一轮的发言**再反驳(这是"辩论"而非"各说各话"的核心)。
- `prev` 是**轮开始时的快照**:同一轮内大家针对的都是上一轮立场,避免次序偏差。
- rebuttal 空产出 → **保留上一轮**(比开场更宽容,因为已有 r0 立场)。

---

## 4. 评分 + 合成(本轮重做的核心)

先 `if not current: raise RuntimeError`(全员失败才报错)。确定三个客户端:

```python
judge_client = judge or panel[0]            # 评委；是否独立取决于实际参与模型
synth_client = synthesizer or judge_client  # 合成者
split        = synth_client is not judge_client
```

把存活视角拼成 `combined`,渲染 `synth_prompt`。然后两条路:

### (A) 不拆分(`split=False`,legacy / 向后兼容)

未传 `synthesizer`、或传的就是 judge 本身时:**一次调用**让 judge 同时打分 + 排名 + 合成。保留以兼容未升级的调用方和现有测试。

### (B) 拆分(`split=True`,Stage 8 构建了单独的 reviewer 客户端时走这条)

传入与 judge 不同的 `synthesizer` 时,拆成**两步两调用**:

1. **评分(judge / gpt-4o)**:
   - system 明确"**只评估,不要改写或合并**"。
   - 给每个视角按 rigor / evidence / falsifiability 打 1–10 + best-first 排名 + 每条一句理由。
   - 结果写 `debate_scores.md`。
   - **独立性单独判断**:评委模型需要与 `author_model` 及所有成功提供视角的角色模型不同。评委、作者或参与模型的名称未知时,不能判为独立。评分和合成拆开本身不保证独立。

2. **合成(synthesizer / claude)**:
   - 把 judge 的排名**前置**到 perspectives 前,再渲染一遍 synth_prompt。只有独立性判断为真时,提示词和评审标题才称其为独立评委。
   - synth system 额外强调:取各方最强要素、**保留真实分歧**、每条主张要**具体可证伪、带可测量预测和明确阈值**。
   - 由**强模型**(primary / claude)产出终稿。

**设计要点**:之前 judge(gpt-4o)既评又合,合成被压成空泛共识;拆分后 → gpt-4o 负责评判,claude 做具体合成并锚定在 judge 排名上。实测把 "will outperform in validity" 变成 "AUROC ≥ 0.65 / ≥ 20pp cliff / 门控执行流程"。如果 gpt-4o 同时生成了一个视角,它仍是在参与评判自己的内容,记录会标记为非独立。

**PR #318 的 Stage 18 设计**:该提案不把作者模型用作评审报告的合成者,而是由 reviewer 完成评审文本。当前 Stage 18 使用 #316 的独立 reviewer 路径,尚未调用 debate 引擎。

每步都有 `try/except` 兜底:评分失败 → 无排名直接合成;合成失败 → 回退到 `combined`(原始拼接),绝不抛错中断流水线。

---

## 5. 记录

写 `debate_record.json`,字段:

```
panel_models, roles(角色→模型映射), rounds, author_model,
judge_model, synthesizer_model, split_judge_synthesis,
independent_judge, perspectives_succeeded
```

`independent_judge` 同时比较作者与成功提供内容的角色模型,不把失败/空输出角色或未使用的面板成员算作被评审内容的作者。它基于客户端配置中的模型名称,不解析供应商模型别名,也不代表评审质量或调用成功状态。tournament 对存活候选的生成模型采用同样判断。

返回 `(final_text, record)`。

---

## 三层防御(健壮性串起来看)

| 层 | 防什么 | 机制 |
|---|---|---|
| 预算 | reasoning 模型隐藏推理吃光预算 → 产空 | `gen_max_tokens=8192` + client 层 reasoning floor |
| 瞬时 | 一次 API 抖动 → 丢整个视角 | `_chat_with_retry` 一次重试 |
| 空内容 | 空白混进 rebuttal / 合成 | `text.strip()` 护栏:开场丢弃 / rebuttal 保留上一轮 / 合成回退 concat |
| 质量 | 弱合成器把终稿压成空泛 | 评分 / 合成分离,强模型合成 |
| 独立性记录 | 评委参与生成却被标成独立 | 比较作者与所有成功参与模型,重合时标为非独立 |

---

## 实证结论(2026-06-08,CAD-LLM 题,同一份 synthesis 三方对照)

- **tournament**(各出一份 → 选单一胜者):具体可证伪,但赢家通吃**丢掉 panel 广度**(contrarian 被扔)。
- **debate + 拆分合成**(本引擎):**三者最佳**——具体度 + 整合 contrarian 深度 + 保留 genuine disagreements + 门控执行流程。
- **legacy 单模型多视角**:具体且保留分歧,但**无真·跨模型多样性**。

核心结论:对**假设生成**,debate(拆分合成)优于 tournament;且 **synthesizer 模型是质量瓶颈**——gpt-4o 合成 → 空泛,claude 合成 → 具体。

---

## 配置:如何启用 debate

要让 Stage 8 走多模型 debate,`config.arc.yaml` 的 `llm:` 段:

```yaml
llm:
  # --- panel 成员:至少 2 个不同模型才形成多模型辩论；单模型也可运行 ---
  primary_model: "anthropic/claude-sonnet-4.6"   # innovator + 最终 synthesizer
  reviewer_model: "openai/gpt-4o"                # 同时作为 panel 成员和评委
  fallback_models:
    - "google/gemini-2.5-pro"                    # 第三个视角(contrarian)

  # --- 开关 ---
  debate_enabled: true        # ★ 必须 true,否则 build_panel_llms 返回空 -> 退回单模型
  debate_rounds: 1            # rebuttal 轮数(0 = 只有开场,无反驳)
  tournament_enabled: false   # 明确选择 debate；启用且 tournament_candidates >= 2 时 tournament 优先
```

### 机制要点(决定怎么填)

1. **panel 怎么来**:`build_panel_llms` 取 `primary_model + reviewer_model + fallback_models`,**按名字去重**。要真·多模型辩论,去重后需 **>= 2 个不同模型**。
   - 上例去重后 = 3 个,round-robin 绑定:innovator→claude、pragmatist→gpt-4o、contrarian→gemini。
   - 若 `reviewer_model` 空且 `fallback_models` 空 → panel 只剩 1 个模型 → debate 退化成"单模型多角色 + 评分"。
2. **tournament 优先**:Stage 8 在 `tournament_enabled=true` 且配置的 `tournament_candidates >= 2` 时进入 tournament；需要明确使用 debate 时设置 `tournament_enabled: false`。
3. **判分独立性**:上例 gpt-4o 既是 pragmatist 又是评委,因此 `independent_judge=false`。仅与 `primary_model` 不同并不足以独立；评委还需要不同于所有成功生成视角的模型。拆分路径下合成由 `primary_model` 自动担任(Stage 8 传 `synthesizer=llm`,**无需配置**)。
4. **角色倾向**来自 active prompt bank(ML bank = innovator/pragmatist/contrarian;HEP bank = theorist/phenomenologist/experimentalist),不在配置里改。

### 可选项 / 环境变量

```yaml
  debate_rounds: 2     # 更多来回反驳(每轮 = 每个存活角色再调一次,成本线性增加)
```
- `ARC_DEBATE_ROUNDS=2` — 临时覆盖轮数
- `ARC_ABL_DISABLE_DEBATE=1` — 消融:塌缩为单角色

### 成本提示

debate 成倍增加 LLM 调用:`3 视角 × (1 开场 + rounds 轮反驳) + 1 评分 + 1 合成`。`debate_rounds=1` 时约 **8 次调用**(含 reasoning 模型的推理 token),比单模型贵不少。
