<p align="center">
  <img src="../image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>شارك فكرة. احصل على ورقة بحثية. مؤتمت بالكامل & ذاتي التطور.</b></h2>



<p align="center">
  <b><i><font size="5">تحدث مع <a href="#-تكامل-openclaw">OpenClaw</a>: «ابحث عن X» → تمّ.</font></i></b>
</p>

<p align="center">
  <img src="../image/framework_v2.png" width="100%" alt="AutoResearchClaw Framework">
</p>


<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#الاختبار"><img src="https://img.shields.io/badge/Tests-1823%20passed-brightgreen?logo=pytest&logoColor=white" alt="1823 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#-تكامل-openclaw"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
  <a href="https://discord.gg/u4ksqW5P"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="../README.md">🇺🇸 English</a> ·
  <a href="README_CN.md">🇨🇳 中文</a> ·
  <a href="README_JA.md">🇯🇵 日本語</a> ·
  <a href="README_KO.md">🇰🇷 한국어</a> ·
  <a href="README_FR.md">🇫🇷 Français</a> ·
  <a href="README_DE.md">🇩🇪 Deutsch</a> ·
  <a href="README_ES.md">🇪🇸 Español</a> ·
  <a href="README_PT.md">🇧🇷 Português</a> ·
  <a href="README_RU.md">🇷🇺 Русский</a> ·
  <a href="README_AR.md">🇸🇦 العربية</a>
</p>

<p align="center">
  <a href="showcase/SHOWCASE.md">🏆 معرض الأوراق</a> · <a href="integration-guide.md">📖 دليل التكامل</a> · <a href="https://discord.gg/u4ksqW5P">💬 مجتمع Discord</a>
</p>

---

<table>
<tr>
<td width="18%">
<a href="showcase/SHOWCASE.md"><img src="showcase/thumbnails/paper_I_random_matrix-01.png" width="120" alt="ورقة نموذجية"/></a>
</td>
<td valign="middle">
<b>🏆 معرض الأوراق المُولّدة</b><br><br>
<b>8 أوراق في 8 مجالات</b> — الرياضيات، الإحصاء، الأحياء، الحوسبة، NLP، RL، الرؤية الحاسوبية، المتانة — مُولّدة بشكل مستقل تماماً بدون تدخل بشري.<br><br>
<a href="showcase/SHOWCASE.md"><img src="https://img.shields.io/badge/عرض_المعرض_الكامل_→-جميع_الأوراق_الـ8-d73a49?style=for-the-badge" alt="عرض المعرض"></a>
</td>
</tr>
</table>

---

> **🧪 نبحث عن مختبرين!** جرّب خط الأنابيب بفكرتك البحثية الخاصة — من أي مجال — و[أخبرنا برأيك](TESTER_GUIDE.md). ملاحظاتك تشكّل الإصدار القادم مباشرة. **[→ Testing Guide](TESTER_GUIDE.md)** | **[→ 中文测试指南](TESTER_GUIDE_CN.md)** | **[→ 日本語テストガイド](TESTER_GUIDE_JA.md)**

---

## 🔥 News
- **[03/22/2026]** [v0.3.2](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.2) — **دعم متعدد المنصات + استقرار كبير** — يعمل AutoResearchClaw الآن مع أي وكيل متوافق مع ACP (Claude Code، Codex CLI، Copilot CLI، Gemini CLI، Kimi CLI) ويدعم منصات المراسلة (Discord، Telegram، Lark، WeChat) عبر جسر OpenClaw. واجهة خلفية جديدة لتوليد الكود عبر CLI-agent تفوّض المرحلتين 10 و13 لوكلاء CLI خارجيين مع التحكم في الميزانية وإدارة المهلة الزمنية. يتضمن نظام مكافحة التلفيق (VerifiedRegistry + حلقة تشخيص وإصلاح التجارب)، 100+ إصلاح أخطاء، إعادة هيكلة modular executor، كشف تلقائي لـ `--resume`، تعزيز إعادة محاولات LLM، وإصلاحات المجتمع.
- **[03/18/2026]** [v0.3.1](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.1) — **OpenCode Beast Mode + Community Contributions** — New "Beast Mode" routes complex code generation to [OpenCode](https://github.com/anomalyco/opencode) with automatic complexity scoring and graceful fallback. Added Novita AI provider support, thread-safety hardening, improved LLM output parsing robustness, and 20+ bug fixes from community PRs and internal audit.
- **[03/17/2026]** [v0.3.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.0) — **MetaClaw Integration** — AutoResearchClaw now supports [MetaClaw](https://github.com/aiming-lab/MetaClaw) cross-run learning: pipeline failures → structured lessons → reusable skills, injected into all 23 stages. **+18.3%** robustness in controlled experiments. Opt-in (`metaclaw_bridge.enabled: true`), fully backward-compatible. See [Integration Guide](#-metaclaw-integration).
- **[03/16/2026]** [v0.2.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.2.0) — Three multi-agent subsystems (CodeAgent, BenchmarkAgent, FigureAgent), hardened Docker sandbox with network-policy-aware execution, 4-round paper quality audit (AI-slop detection, 7-dim review scoring, NeurIPS checklist), and 15+ bug fixes from production runs.
- **[03/15/2026]** [v0.1.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.1.0) — We release AutoResearchClaw: a fully autonomous 23-stage research pipeline that turns a single research idea into a conference-ready paper. No human intervention required.

---

## ⚡ أمر واحد. ورقة واحدة.

```bash
pip install -e . && researchclaw setup && researchclaw init && researchclaw run --topic "Your research idea here" --auto-approve
```


---

## 🤔 ما هذا؟

**أنت تفكر. AutoResearchClaw يكتب.**

أعطِ موضوعاً بحثياً — احصل على ورقة أكاديمية كاملة مع أدبيات حقيقية من OpenAlex و Semantic Scholar و arXiv، وتجارب في بيئة معزولة واعية بالعتاد (كشف تلقائي لـ GPU/MPS/CPU)، وتحليل إحصائي، ومراجعة أقران متعددة الوكلاء، و LaTeX جاهز للمؤتمرات يستهدف NeurIPS/ICML/ICLR. بدون مراقبة. بدون نسخ ولصق. بدون مراجع مُلفّقة.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>ورقة أكاديمية كاملة (مقدمة، أعمال سابقة، المنهجية، التجارب، النتائج، الخاتمة)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>LaTeX جاهز للمؤتمرات (قوالب NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>مراجع BibTeX حقيقية من OpenAlex و Semantic Scholar و arXiv — مُنقّحة تلقائياً لمطابقة الاستشهادات المضمّنة</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>تحقق من سلامة الاستشهادات على 4 طبقات + التحقق من الصلة (arXiv، CrossRef، DataCite، LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>كود مُولّد + نتائج البيئة المعزولة + مقاييس JSON منظمة</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>رسوم بيانية مُولّدة تلقائياً لمقارنة الظروف مع أشرطة الخطأ وفترات الثقة</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>مراجعة أقران متعددة الوكلاء مع فحص اتساق المنهجية والأدلة</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>دروس تعلّم ذاتي مستخلصة من كل تشغيل</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>جميع المخرجات النهائية في مجلد واحد — جاهزة للترجمة على Overleaf</td></tr>
</table>

يعمل خط الأنابيب **من البداية إلى النهاية بدون تدخل بشري**. عندما تفشل التجارب، يصلح نفسه. عندما لا تصمد الفرضيات، يغيّر المسار. عندما تكون الاستشهادات مُلفّقة، يزيلها.

🌍 **شغّله من أي مكان.** AutoResearchClaw ليس مقيّدًا بمنصة واحدة. استخدمه مستقلاً عبر CLI، أو وصّله بـ [OpenClaw](https://github.com/openclaw/openclaw)، أو ادمجه مع أي وكيل متوافق مع ACP — 🤖 Claude Code، 💻 Codex CLI، 🐙 Copilot CLI، ♊ Gemini CLI، 🌙 Kimi CLI، وغيرها. بفضل جسر الرسائل في OpenClaw، يمكنك إطلاق بحث كامل من 💬 Discord، ✈️ Telegram، 🐦 Lark (飞书)، 💚 WeChat، أو أي منصة يستخدمها فريقك بالفعل. موضوع واحد كمُدخل، ورقة بحثية كمُخرج — بغض النظر عن المكان الذي تكتب منه.

---

## 🚀 البداية السريعة

```bash
# 1. استنساخ وتثبيت
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. الإعداد (تفاعلي — يثبّت OpenCode beast mode، يتحقق من Docker/LaTeX)
researchclaw setup

# 3. التهيئة
researchclaw init          # تفاعلي: اختر مزوّد LLM، ينشئ config.arc.yaml
# أو يدوياً: cp config.researchclaw.example.yaml config.arc.yaml

# 4. التشغيل
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

المخرجات → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — LaTeX و BibTeX وكود التجارب والرسوم البيانية جاهزة للترجمة.

<details>
<summary>📝 الحد الأدنى من التهيئة المطلوبة</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models: ["gpt-4o-mini"]

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

---

## 🧠 ما الذي يميّزه

| القدرة | كيف يعمل |
|-----------|-------------|
| **🔄 حلقة PIVOT / REFINE** | المرحلة 15 تقرر بشكل مستقل: PROCEED أو REFINE (تعديل المعاملات) أو PIVOT (اتجاه جديد). المخرجات تُحفظ بإصدارات تلقائياً. |
| **🤖 نقاش متعدد الوكلاء** | توليد الفرضيات وتحليل النتائج ومراجعة الأقران تستخدم نقاشاً منظماً بوجهات نظر متعددة. |
| **🧬 التعلّم الذاتي** | دروس مستخلصة من كل تشغيل (مبررات القرارات، تحذيرات وقت التشغيل، شذوذ المقاييس) مع تناقص زمني بنصف عمر 30 يوماً. التشغيلات المستقبلية تتعلم من الأخطاء السابقة. |
| **📚 قاعدة المعرفة** | كل تشغيل يبني قاعدة معرفة منظمة عبر 6 فئات (قرارات، تجارب، اكتشافات، أدبيات، أسئلة، مراجعات). |
| **🛡️ الحارس المراقب Sentinel** | مراقب جودة في الخلفية: كشف NaN/Inf، اتساق الورقة والأدلة، تقييم صلة الاستشهادات، حماية ضد التلفيق. |

---

## 🦞 تكامل OpenClaw

<table>
<tr>

**AutoResearchClaw هو خدمة متوافقة مع [OpenClaw](https://github.com/openclaw/openclaw).** قم بتثبيته في OpenClaw وابدأ بحثاً مستقلاً برسالة واحدة — أو استخدمه بشكل مستقل عبر سطر الأوامر أو Claude Code أو أي مساعد برمجة بالذكاء الاصطناعي.

</tr>
</table>

### 🚀 الاستخدام مع OpenClaw (موصى به)

إذا كنت تستخدم [OpenClaw](https://github.com/openclaw/openclaw) بالفعل كمساعد ذكاء اصطناعي:

```
1️⃣  شارك رابط مستودع GitHub مع OpenClaw
2️⃣  OpenClaw يقرأ تلقائياً RESEARCHCLAW_AGENTS.md → يفهم خط الأنابيب
3️⃣  قل: "ابحث عن [موضوعك]"
4️⃣  تم — OpenClaw يستنسخ، يثبّت، يهيّئ، يشغّل، ويعيد النتائج
```

**هذا كل شيء.** يتعامل OpenClaw مع `git clone`، `pip install`، إعداد التهيئة، وتنفيذ خط الأنابيب تلقائياً. أنت فقط تتحدث.

<details>
<summary>💡 ماذا يحدث خلف الكواليس</summary>

1. يقرأ OpenClaw ملف `RESEARCHCLAW_AGENTS.md` → يتعلم دور منسّق البحث
2. يقرأ OpenClaw ملف `README.md` → يفهم التثبيت وبنية خط الأنابيب
3. يقرأ OpenClaw ملف `config.researchclaw.example.yaml` → `config.yaml`
4. يسأل عن مفتاح API لنموذج اللغة (أو يستخدم متغير البيئة)
5. يشغّل `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. يعيد الورقة و LaTeX والتجارب والاستشهادات

</details>

### 🔌 جسر OpenClaw (متقدم)

للتكامل الأعمق، يتضمن AutoResearchClaw **نظام محوّلات جسر** مع 6 إمكانيات اختيارية:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ عمليات تشغيل بحث مجدولة
  use_message: true           # 💬 إشعارات التقدم (Discord/Slack/Telegram)
  use_memory: true            # 🧠 استمرارية المعرفة عبر الجلسات
  use_sessions_spawn: true    # 🔀 إطلاق جلسات فرعية متوازية للمراحل المتزامنة
  use_web_fetch: true         # 🌐 بحث ويب مباشر أثناء مراجعة الأدبيات
  use_browser: false          # 🖥️ جمع الأوراق عبر المتصفح
```

كل علامة تفعّل بروتوكول محوّل مُحدد النوع. عندما يوفر OpenClaw هذه الإمكانيات، تستهلكها المحوّلات بدون تغييرات في الكود. راجع [`integration-guide.md`](integration-guide.md) للتفاصيل الكاملة.

### ACP (Agent Client Protocol)

يمكن لـ AutoResearchClaw استخدام **أي وكيل برمجة متوافق مع ACP** كواجهة خلفية لنموذج اللغة — بدون الحاجة لمفاتيح API. يتواصل الوكيل عبر [acpx](https://github.com/openclaw/acpx)، ويحافظ على جلسة واحدة مستمرة عبر جميع مراحل خط الأنابيب الـ 23.

| الوكيل | الأمر | ملاحظات |
|-------|---------|-------|
| Claude Code | `claude` | Anthropic |
| Codex CLI | `codex` | OpenAI |
| Copilot CLI | `gh` | GitHub |
| Gemini CLI | `gemini` | Google |
| OpenCode | `opencode` | SST |
| Kimi CLI | `kimi` | Moonshot |

```yaml
# config.yaml — مثال ACP
llm:
  provider: "acp"
  acp:
    agent: "claude"   # أي أمر CLI لوكيل متوافق مع ACP
    cwd: "."          # دليل العمل للوكيل
  # لا حاجة لـ base_url أو api_key — الوكيل يدير مصادقته بنفسه.
```

```bash
# فقط شغّل — الوكيل يستخدم بيانات اعتماده الخاصة
researchclaw run --config config.yaml --topic "Your research idea" --auto-approve
```

### 🛠️ طرق أخرى للتشغيل

| الطريقة | الكيفية |
|--------|-----|
| **سطر أوامر مستقل** | `researchclaw setup` → `researchclaw init` → `researchclaw run --topic "..." --auto-approve` |
| **واجهة Python البرمجية** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | يقرأ `RESEARCHCLAW_CLAUDE.md` — فقط قل *"شغّل بحثاً عن [موضوع]"* |
| **Copilot CLI** | `researchclaw run --topic "..."` مع `llm.acp.agent: "gh"` |
| **OpenCode** | يقرأ `.claude/skills/` — نفس واجهة اللغة الطبيعية |
| **أي واجهة ذكاء اصطناعي** | قدّم `RESEARCHCLAW_AGENTS.md` كسياق → الوكيل يبدأ تلقائياً |

---

## 🔬 خط الأنابيب: 23 مرحلة، 8 أطوار

```
Phase A: تحديد نطاق البحث          Phase E: تنفيذ التجارب
  1. TOPIC_INIT                      12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE               13. ITERATIVE_REFINE  ← إصلاح ذاتي

Phase B: اكتشاف الأدبيات          Phase F: التحليل والقرار
  3. SEARCH_STRATEGY                 14. RESULT_ANALYSIS    ← متعدد الوكلاء
  4. LITERATURE_COLLECT  ← API حقيقي  15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [بوابة]
  6. KNOWLEDGE_EXTRACT               Phase G: كتابة الورقة
                                     16. PAPER_OUTLINE
Phase C: توليف المعرفة              17. PAPER_DRAFT
  7. SYNTHESIS                       18. PEER_REVIEW        ← فحص الأدلة
  8. HYPOTHESIS_GEN    ← نقاش        19. PAPER_REVISION

Phase D: تصميم التجارب            Phase H: الإنهاء
  9. EXPERIMENT_DESIGN   [بوابة]      20. QUALITY_GATE      [بوابة]
 10. CODE_GENERATION                 21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING               22. EXPORT_PUBLISH     ← LaTeX
                                     23. CITATION_VERIFY    ← فحص الصلة
```

> **مراحل البوابات** (5، 9، 20) تتوقف للحصول على موافقة بشرية أو موافقة تلقائية مع `--auto-approve`. عند الرفض، يعود خط الأنابيب للخلف.

> **حلقات القرار**: يمكن للمرحلة 15 تفعيل REFINE (→ المرحلة 13) أو PIVOT (→ المرحلة 8)، مع إصدار تلقائي للمخرجات.

<details>
<summary>📋 ماذا يفعل كل طور</summary>

| الطور | ما يحدث |
|-------|-------------|
| **A: تحديد النطاق** | يفكك نموذج اللغة الموضوع إلى شجرة مشاكل منظمة مع أسئلة بحثية |
| **A+: العتاد** | كشف تلقائي لـ GPU (NVIDIA CUDA / Apple MPS / CPU فقط)، تحذير إذا كان العتاد المحلي محدوداً، تكييف توليد الكود وفقاً لذلك |
| **B: الأدبيات** | بحث متعدد المصادر (OpenAlex → Semantic Scholar → arXiv) عن أوراق حقيقية، فرز حسب الصلة، استخلاص بطاقات معرفية |
| **C: التوليف** | تجميع النتائج، تحديد فجوات البحث، توليد فرضيات قابلة للاختبار عبر نقاش متعدد الوكلاء |
| **D: التصميم** | تصميم خطة التجارب، توليد كود Python قابل للتشغيل واعٍ بالعتاد (مستوى GPU → اختيار الحزم)، تقدير احتياجات الموارد |
| **E: التنفيذ** | تشغيل التجارب في بيئة معزولة، كشف NaN/Inf وأخطاء وقت التشغيل، إصلاح ذاتي للكود عبر إصلاح مُستهدف بنموذج اللغة |
| **F: التحليل** | تحليل متعدد الوكلاء للنتائج؛ قرار مستقل PROCEED / REFINE / PIVOT مع المبررات |
| **G: الكتابة** | مخطط → صياغة قسم بقسم (5,000-6,500 كلمة) → مراجعات أقران (مع اتساق المنهجية والأدلة) → مراجعة مع حماية الطول |
| **H: الإنهاء** | بوابة جودة، أرشفة المعرفة، تصدير LaTeX مع قالب المؤتمر، التحقق من سلامة الاستشهادات + الصلة |

</details>

---

## ✨ الميزات الرئيسية

| الميزة | الوصف |
|---------|------------|
| **📚 أدبيات متعددة المصادر** | أوراق حقيقية من OpenAlex و Semantic Scholar و arXiv — توسيع الاستعلام، إزالة التكرار، قاطع دائرة مع تدهور أنيق |
| **🔍 تحقق من الاستشهادات على 4 طبقات** | فحص arXiv ID → CrossRef/DataCite DOI → مطابقة عنوان Semantic Scholar → تقييم صلة LLM. المراجع المُلفّقة تُزال تلقائياً. |
| **🖥️ تنفيذ واعٍ بالعتاد** | كشف تلقائي لـ GPU (NVIDIA CUDA / Apple MPS / CPU فقط) مع تكييف توليد الكود والاستيرادات ونطاق التجارب |
| **🦾 OpenCode Beast Mode** | التجارب المعقدة تُوجّه تلقائياً إلى [OpenCode](https://github.com/anomalyco/opencode) — يولّد مشاريع متعددة الملفات مع بنى مخصصة وحلقات تدريب ودراسات استئصال. التثبيت عبر `researchclaw setup`. |
| **🧪 تجارب في بيئة معزولة** | كود مُتحقق بـ AST، إطار غير قابل للتعديل، فشل سريع عند NaN/Inf، إصلاح ذاتي، تحسين تكراري (حتى 10 جولات)، التقاط نتائج جزئية |
| **📝 كتابة بمستوى المؤتمرات** | قوالب NeurIPS/ICML/ICLR، صياغة قسم بقسم (5,000-6,500 كلمة)، حماية ضد التلفيق، حماية طول المراجعة، فرض مضاد لإخلاءات المسؤولية |
| **📐 تبديل القوالب** | `neurips_2025`، `iclr_2026`، `icml_2026` — Markdown → LaTeX مع رياضيات وجداول وأشكال ومراجع تبادلية و `\cite{}` |
| **🚦 بوابات الجودة** | 3 بوابات بمشاركة بشرية (المراحل 5، 9، 20) مع إمكانية التراجع. تخطّ باستخدام `--auto-approve`. |

---

## 🧠 تكامل MetaClaw

**AutoResearchClaw + [MetaClaw](https://github.com/aiming-lab/MetaClaw) = خط أنابيب يتعلم من كل تشغيل.**

يضيف MetaClaw **نقل المعرفة عبر التشغيلات** إلى AutoResearchClaw. عند التفعيل، يلتقط خط الأنابيب تلقائياً الدروس من الإخفاقات والتحذيرات، ويحوّلها إلى مهارات قابلة لإعادة الاستخدام، ويحقنها في جميع مراحل خط الأنابيب الـ 23 في التشغيلات اللاحقة — بحيث لا تتكرر نفس الأخطاء أبداً.

### كيف يعمل

```
Run N ينفّذ → الإخفاقات/التحذيرات تُلتقط كـ Lessons
                      ↓
          MetaClaw Lesson → تحويل إلى Skill
                      ↓
          ملفات arc-* Skill تُخزّن في ~/.metaclaw/skills/
                      ↓
Run N+1 → build_overlay() يحقن المهارات في كل أمر LLM
                      ↓
          LLM يتجنب المزالق المعروفة → جودة أعلى، محاولات أقل
```

### الإعداد السريع

```bash
# 1. تثبيت MetaClaw (إذا لم يكن مُثبّتاً)
pip install metaclaw

# 2. التفعيل في التهيئة
```

```yaml
# config.arc.yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"        # وكيل MetaClaw (اختياري)
  skills_dir: "~/.metaclaw/skills"          # أين تُخزّن المهارات
  fallback_url: "https://api.openai.com/v1" # بديل LLM مباشر
  fallback_api_key: ""                      # مفتاح API لعنوان البديل
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # تحويل التحذيرات + الأخطاء
    max_skills_per_run: 3
```

```bash
# 3. شغّل كالمعتاد — MetaClaw يعمل بشفافية
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve
```

بعد كل تشغيل، تحقق من `~/.metaclaw/skills/arc-*/SKILL.md` لمشاهدة المهارات التي تعلّمها خط أنابيبك.

### نتائج التجارب

في تجارب A/B مُحكمة (نفس الموضوع، نفس LLM، نفس التهيئة):

| المقياس | خط الأساس | مع MetaClaw | التحسين |
|---------|----------|---------------|----------|
| معدل إعادة المحاولة لكل مرحلة | 10.5% | 7.9% | **-24.8%** |
| عدد دورات REFINE | 2.0 | 1.2 | **-40.0%** |
| إكمال مراحل خط الأنابيب | 18/19 | 19/19 | **+5.3%** |
| درجة المتانة الإجمالية (مركّبة) | 0.714 | 0.845 | **+18.3%** |

> درجة المتانة المركّبة هي متوسط مرجّح لمعدل إكمال المراحل (40%) وتقليل المحاولات (30%) وكفاءة دورات REFINE (30%).

### التوافق العكسي

- **الافتراضي: مُعطّل.** إذا كان `metaclaw_bridge` غائباً أو `enabled: false`، يعمل خط الأنابيب تماماً كما كان.
- **بدون تبعيات جديدة.** MetaClaw اختياري — خط الأنابيب الأساسي يعمل بدونه.
- **جميع الاختبارات الـ 1,823 الحالية تنجح** مع وجود كود التكامل.

---

## ⚙️ مرجع التهيئة

<details>
<summary>انقر لتوسيع مرجع التهيئة الكامل</summary>

```yaml
# === المشروع ===
project:
  name: "my-research"              # معرّف المشروع
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === البحث ===
research:
  topic: "..."                     # موضوع البحث (مطلوب)
  domains: ["ml", "nlp"]           # مجالات البحث للبحث في الأدبيات
  daily_paper_count: 8             # عدد الأوراق المستهدف لكل استعلام بحث
  quality_threshold: 4.0           # الحد الأدنى لدرجة جودة الأوراق

# === وقت التشغيل ===
runtime:
  timezone: "America/New_York"     # للطوابع الزمنية
  max_parallel_tasks: 3            # حد التجارب المتزامنة
  approval_timeout_hours: 12       # مهلة مرحلة البوابة
  retry_limit: 2                   # عدد إعادة المحاولة عند فشل المرحلة

# === نموذج اللغة ===
llm:
  provider: "openai-compatible"    # openai | openrouter | deepseek | minimax | acp | openai-compatible
  base_url: "https://..."          # نقطة نهاية API (مطلوب لـ openai-compatible)
  api_key_env: "OPENAI_API_KEY"    # متغير بيئة لمفتاح API (مطلوب لـ openai-compatible)
  api_key: ""                      # أو ضع المفتاح هنا مباشرة
  primary_model: "gpt-4o"          # النموذج الأساسي
  fallback_models: ["gpt-4o-mini"] # سلسلة النماذج الاحتياطية
  s2_api_key: ""                   # مفتاح Semantic Scholar API (اختياري، حدود معدل أعلى)
  acp:                             # يُستخدم فقط عند provider: "acp"
    agent: "claude"                # أمر CLI لوكيل ACP (claude، codex، gemini، إلخ)
    cwd: "."                       # دليل العمل للوكيل

# === التجارب ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 300             # أقصى وقت تنفيذ لكل تشغيل (الافتراضي: 300 ثانية)
  max_iterations: 10               # أقصى عدد تكرارات التحسين
  metric_key: "val_loss"           # اسم المقياس الأساسي
  metric_direction: "minimize"     # minimize | maximize
  sandbox:
    python_path: ".venv/bin/python"
    gpu_required: false
    allowed_imports: [math, random, json, csv, numpy, torch, sklearn]
    max_memory_mb: 4096
  docker:
    image: "researchclaw/experiment:latest"
    network_policy: "setup_only"   # none | setup_only | pip_only | full
    gpu_enabled: true
    memory_limit_mb: 8192
    auto_install_deps: true        # كشف تلقائي للاستيراد → requirements.txt
  ssh_remote:
    host: ""                       # اسم مضيف خادم GPU
    gpu_ids: []                    # معرّفات GPU المتاحة
    remote_workdir: "/tmp/researchclaw_experiments"
  opencode:                          # OpenCode Beast Mode (يُثبّت تلقائياً عبر `researchclaw setup`)
    enabled: true                    # المفتاح الرئيسي (الافتراضي: true)
    auto: true                       # تشغيل تلقائي بدون تأكيد (الافتراضي: true)
    complexity_threshold: 0.2        # 0.0-1.0 — أعلى = فقط للتجارب المعقدة
    model: ""                        # تجاوز النموذج (فارغ = يستخدم llm.primary_model)
    timeout_sec: 600                 # أقصى ثوانٍ لتوليد OpenCode
    max_retries: 1                   # عدد المحاولات عند الفشل
    workspace_cleanup: true          # حذف مساحة العمل المؤقتة بعد الجمع

# === التصدير ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === الأوامر النصية ===
prompts:
  custom_file: ""                  # مسار ملف YAML للأوامر المخصصة (فارغ = الافتراضي)

# === الأمان ===
security:
  hitl_required_stages: [5, 9, 20] # المراحل التي تتطلب موافقة بشرية
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === قاعدة المعرفة ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === الإشعارات ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === جسر MetaClaw (اختياري) ===
metaclaw_bridge:
  enabled: false                   # اضبط على true لتفعيل التعلم عبر التشغيلات
  proxy_url: "http://localhost:30000"  # عنوان وكيل MetaClaw
  skills_dir: "~/.metaclaw/skills" # أين تُخزّن مهارات arc-*
  fallback_url: ""                 # بديل LLM مباشر عند عدم توفر الوكيل
  fallback_api_key: ""             # مفتاح API لنقطة نهاية البديل
  lesson_to_skill:
    enabled: true                  # تحويل الدروس إلى مهارات تلقائياً
    min_severity: "warning"        # أدنى شدة للتحويل
    max_skills_per_run: 3          # أقصى مهارات جديدة لكل تشغيل

# === جسر OpenClaw ===
openclaw_bridge:
  use_cron: false                  # عمليات تشغيل بحث مجدولة
  use_message: false               # إشعارات التقدم
  use_memory: false                # استمرارية المعرفة عبر الجلسات
  use_sessions_spawn: false        # إطلاق جلسات فرعية متوازية
  use_web_fetch: false             # بحث ويب مباشر
  use_browser: false               # جمع الأوراق عبر المتصفح
```

</details>

---

## 🙏 شكر وتقدير

مستوحى من:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — رائد البحث الآلي
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — أتمتة البحث من البداية إلى النهاية
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — نظام بحث مؤتمت بالكامل

---

## 📄 الرخصة

MIT — راجع [LICENSE](../LICENSE) للتفاصيل.

---

## 📌 الاستشهاد

إذا وجدت AutoResearchClaw مفيداً، يرجى الاستشهاد:

```bibtex
@misc{liu2026autoresearchclaw,
  author       = {Liu, Jiaqi and Xia, Peng and Han, Siwei and Qiu, Shi and Zhang, Letian and Chen, Guiming  and Tu, Haoqin and Yang, Xinyu and and Zhou, Jiawei and Zhu, Hongtu and Li, Yun and Zhou, Yuyin and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  title        = {AutoResearchClaw: Fully Autonomous Research from Idea to Paper},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/aiming-lab/AutoResearchClaw},
}
```

<p align="center">
  <sub>بُني بـ 🦞 بواسطة فريق AutoResearchClaw</sub>
</p>
