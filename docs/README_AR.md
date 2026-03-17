<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>شارك فكرة. احصل على ورقة بحثية. مؤتمت بالكامل.</b></h2>



<p align="center">
  <i>تحدث مع <a href="#openclaw-integration">OpenClaw</a>: «ابحث عن X» → تمّ.</i>
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/Tests-1284%20passed-brightgreen?logo=pytest&logoColor=white" alt="1284 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-integration"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="integration-guide.md">📖 دليل التكامل</a>
</p>

---

## ⚡ سطر واحد للبدء

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```

---

## 🤔 ما هذا؟

لديك فكرة. تريد ورقة بحثية. **هذا كل شيء.**

يأخذ AutoResearchClaw موضوعاً بحثياً وينتج بشكل مستقل ورقة أكاديمية كاملة — مع أدبيات حقيقية من arXiv و Semantic Scholar (متعدد المصادر، أولوية arXiv لتجنب تحديد المعدل)، وتجارب في بيئة معزولة واعية بالعتاد (كشف تلقائي لـ GPU/MPS/CPU)، وتحليل إحصائي، ومراجعة أقران، و LaTeX جاهز للمؤتمرات (يستهدف 5,000-6,500 كلمة لـ NeurIPS/ICML/ICLR). بدون مراقبة. بدون نسخ ولصق بين الأدوات.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>ورقة أكاديمية كاملة (مقدمة، أعمال سابقة، المنهجية، التجارب، النتائج، الخاتمة)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>LaTeX جاهز للمؤتمرات (قوالب NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>مراجع BibTeX حقيقية من Semantic Scholar و arXiv — مُنقّحة تلقائياً لمطابقة الاستشهادات المضمّنة</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>تحقق من سلامة الاستشهادات على 4 طبقات + التحقق من الصلة (arXiv، CrossRef، DataCite، LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>كود مُولّد + نتائج البيئة المعزولة + مقاييس JSON منظمة</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>رسوم بيانية مُولّدة تلقائياً لمقارنة الظروف مع أشرطة الخطأ وفترات الثقة</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>مراجعة أقران متعددة الوكلاء مع فحص اتساق المنهجية والأدلة</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>دروس تعلّم ذاتي مستخلصة من كل تشغيل</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>جميع المخرجات النهائية في مجلد واحد — جاهزة للترجمة على Overleaf</td></tr>
</table>

يعمل خط الأنابيب **من البداية إلى النهاية بدون تدخل بشري** (ما لم تقم بتهيئة مراحل البوابات للمراجعة اليدوية). عندما تفشل التجارب، يصلح نفسه. عندما لا تصمد الفرضيات، يغيّر المسار.

### 🎯 جرّبه

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 ما الذي يميّزه

### 🔄 حلقة قرار PIVOT / REFINE

لا يعمل خط الأنابيب بشكل خطي فقط. المرحلة 15 (RESEARCH_DECISION) تقيّم نتائج التجارب مقابل الفرضيات وتتخذ قراراً مستقلاً:

- **PROCEED** — النتائج تدعم الفرضيات، الاستمرار في كتابة الورقة
- **REFINE** — النتائج واعدة لكنها تحتاج تحسيناً، العودة لتحسين الكود/المعاملات
- **PIVOT** — تم اكتشاف مشكلة جوهرية، إعادة البدء من توليد الفرضيات باتجاه جديد

كل دورة PIVOT/REFINE **تحفظ نسخاً من المخرجات السابقة** (`stage-08_v1/`، `stage-08_v2/`، ...) بحيث لا يُفقد أي عمل ويمكن تتبع تطور القرارات بالكامل.

### 🤖 نقاش متعدد الوكلاء

تستخدم المراحل الحرجة بروتوكول نقاش منظم مع وجهات نظر متعددة من نماذج اللغة:

- **توليد الفرضيات** — وكلاء متنوعون يقترحون ويتحدّون الأفكار
- **تحليل النتائج** — متفائل ومتشكك وواقعي يحللون المخرجات
- **مراجعة الأقران** — فحص اتساق المنهجية والأدلة (هل تدّعي الورقة 50 تجربة بينما الكود نفّذ 5 فقط؟)

### 🧬 التطور: التعلّم الذاتي عبر التشغيلات

كل تشغيل لخط الأنابيب يستخلص دروساً دقيقة — ليس فقط "فشلت" بل *لماذا*:

- مبررات القرار من اختيارات PIVOT/REFINE
- تحذيرات وقت التشغيل من stderr للتجارب (مثل `RuntimeWarning: division by zero`)
- شذوذ المقاييس (NaN، Inf، سرعات تقارب متطابقة)

هذه الدروس تُحفظ في مخزن JSONL مع **تثقيل بتناقص زمني بنصف عمر 30 يوماً** وتُحقن كطبقات إضافية في الأوامر النصية للتشغيلات المستقبلية. خط الأنابيب يتعلّم حرفياً من أخطائه.

### 📚 قاعدة المعرفة

كل تشغيل يبني قاعدة معرفة منظمة (مخزّنة في `docs/kb/`) مع 6 فئات:

- **decisions/** — تصميم التجارب، بوابات الجودة، قرارات البحث، تخطيط الموارد، استراتيجيات البحث، أرشيفات المعرفة
- **experiments/** — سجلات توليد الكود، تشغيلات التجارب، التحسينات التكرارية
- **findings/** — التحقق من الاستشهادات، تحليل النتائج، تقارير التوليف
- **literature/** — استخلاص المعرفة، جمع الأدبيات، نتائج الفرز
- **questions/** — توليد الفرضيات، تفكيك المشكلة، تهيئة الموضوع
- **reviews/** — تقارير التصدير/النشر، مسودات الأوراق، المخططات، المراجعات، مراجعات الأقران

### 🛡️ الحارس المراقب (Sentinel Watchdog)

مراقب جودة في الخلفية يكتشف المشاكل التي قد يفوتها خط الأنابيب الرئيسي:

- **كشف أخطاء وقت التشغيل** — NaN/Inf في المقاييس، تحذيرات stderr تُغذّى مرة أخرى لنموذج اللغة للإصلاح المُستهدف
- **اتساق الورقة والأدلة** — كود التجارب الفعلي ونتائج التشغيل وسجلات التحسين تُحقن في مراجعة الأقران
- **تقييم صلة الاستشهادات** — إلى ما هو أبعد من التحقق من الوجود، يقيّم نموذج اللغة الصلة الموضوعية لكل مرجع
- **فرض التقارب** — يكتشف التجارب ذات التكرار الثابت ويطالب بمعايير إيقاف مبكر مناسبة
- **التحقق من الاستئصال** — يكتشف ظروف الاستئصال المكررة/المتطابقة ويُبلّغ عن المقارنات المعطلة
- **حماية ضد التلفيق** — يمنع كتابة الورقة تماماً عندما لا تنتج التجارب أي مقاييس

---

## 🦞 تكامل OpenClaw

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw هو خدمة متوافقة مع [OpenClaw](https://github.com/openclaw/openclaw).** قم بتثبيته في OpenClaw وابدأ بحثاً مستقلاً برسالة واحدة — أو استخدمه بشكل مستقل عبر سطر الأوامر أو Claude Code أو أي مساعد برمجة بالذكاء الاصطناعي.

</td>
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

### 🛠️ طرق أخرى للتشغيل

| الطريقة | الكيفية |
|--------|-----|
| **سطر أوامر مستقل** | `researchclaw run --topic "..." --auto-approve` |
| **واجهة Python البرمجية** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | يقرأ `RESEARCHCLAW_CLAUDE.md` — فقط قل *"شغّل بحثاً عن [موضوع]"* |
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
| **B: الأدبيات** | بحث متعدد المصادر (أولوية arXiv، ثم Semantic Scholar) عن أوراق حقيقية، فرز حسب الصلة، استخلاص بطاقات معرفية |
| **C: التوليف** | تجميع النتائج، تحديد فجوات البحث، توليد فرضيات قابلة للاختبار عبر نقاش متعدد الوكلاء |
| **D: التصميم** | تصميم خطة التجارب، توليد كود Python قابل للتشغيل واعٍ بالعتاد (مستوى GPU → اختيار الحزم)، تقدير احتياجات الموارد |
| **E: التنفيذ** | تشغيل التجارب في بيئة معزولة، كشف NaN/Inf وأخطاء وقت التشغيل، إصلاح ذاتي للكود عبر إصلاح مُستهدف بنموذج اللغة |
| **F: التحليل** | تحليل متعدد الوكلاء للنتائج؛ قرار مستقل PROCEED / REFINE / PIVOT مع المبررات |
| **G: الكتابة** | مخطط → صياغة قسم بقسم (5,000-6,500 كلمة) → مراجعات أقران (مع اتساق المنهجية والأدلة) → مراجعة مع حماية الطول |
| **H: الإنهاء** | بوابة جودة، أرشفة المعرفة، تصدير LaTeX مع قالب المؤتمر، التحقق من سلامة الاستشهادات + الصلة |

</details>

---

## 🚀 البداية السريعة

### المتطلبات الأساسية

- 🐍 Python 3.11+
- 🔑 نقطة نهاية API متوافقة مع OpenAI لنموذج لغة (GPT-4o، GPT-5.x، أو أي مزود متوافق)

### التثبيت

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### التهيئة

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 الحد الأدنى من التهيئة المطلوبة</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"     # Any OpenAI-compatible endpoint
  api_key_env: "OPENAI_API_KEY"              # Env var name containing your key
  primary_model: "gpt-4o"                    # Any model your endpoint supports
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                             # Optional: Semantic Scholar API key for higher rate limits

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

### التشغيل

```bash
# تعيين مفتاح API
export OPENAI_API_KEY="sk-..."

# 🚀 تشغيل خط الأنابيب الكامل
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 تحديد موضوع مباشرة
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ التحقق من التهيئة
researchclaw validate --config config.arc.yaml

# ⏩ الاستئناف من مرحلة محددة
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

المخرجات → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` مع دليل فرعي لكل مرحلة.

جميع المخرجات النهائية للمستخدم تُجمع تلقائياً في مجلد **`deliverables/`** واحد:

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # الورقة النهائية (Markdown)
├── paper.tex                  # LaTeX جاهز للمؤتمر
├── references.bib             # قائمة مراجع BibTeX موثّقة (مُنقّحة تلقائياً)
├── neurips_2025.sty           # ملف نمط المؤتمر (مختار تلقائياً)
├── code/                      # كود التجارب + requirements.txt
├── verification_report.json   # تقرير سلامة الاستشهادات
├── charts/                    # رسوم بيانية للنتائج (مقارنة الظروف، أشرطة الخطأ)
└── manifest.json              # فهرس المخرجات مع البيانات الوصفية
```

مجلد `deliverables/` **جاهز للترجمة** — يتضمن ملفات `.sty` و `.bst` الخاصة بالمؤتمر بحيث يمكنك ترجمة `paper.tex` مباشرة باستخدام `pdflatex` + `bibtex` أو رفعه إلى Overleaf بدون تحميل أي شيء إضافي.

---

## ✨ الميزات الرئيسية

### 📚 بحث أدبيات متعدد المصادر

تستعلم المرحلة 4 من **واجهات API أكاديمية حقيقية** — وليس أوراقاً مُتخيّلة من نموذج اللغة. تستخدم استراتيجية **أولوية arXiv** لتجنب تحديد معدل Semantic Scholar.

- **arXiv API** (أساسي) — مسودات مع معرّفات arXiv وبيانات وصفية حقيقية، بدون حدود للمعدل
- **Semantic Scholar API** (ثانوي) — أوراق حقيقية مع عناوين وملخصات وأماكن نشر وعدد استشهادات ومعرّفات DOI
- **توسيع الاستعلام** — يولّد تلقائياً استعلامات أوسع (مسح، معيار مرجعي، متغيرات المقارنة) لتغطية شاملة (30-60 مرجعاً)
- **إزالة التكرار التلقائية** — DOI → معرّف arXiv → مطابقة عناوين ضبابية
- **توليد BibTeX** — إدخالات صالحة `@article{cite_key, ...}` مع بيانات وصفية حقيقية
- **قاطع دائرة ثلاثي الحالات** — CLOSED → OPEN → HALF_OPEN مع تعافي بتأخير أسّي (لا يُعطّل نهائياً أبداً)
- **تدهور أنيق** — فشل S2 لا يحجب نتائج arXiv؛ يعود لنتائج مُعزّزة بنموذج اللغة إذا فشلت جميع الواجهات

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 التحقق من الاستشهادات (المرحلة 23)

بعد كتابة الورقة، تقوم المرحلة 23 بـ**التحقق الفعلي من كل مرجع** من حيث السلامة والصلة:

| الطبقة | الطريقة | ما تتحقق منه |
|-------|--------|----------------|
| L1 | arXiv API `id_list` | الأوراق ذات معرّفات arXiv — تتحقق من وجود المعرّف فعلاً |
| L2 | CrossRef `/works/{doi}` + احتياطي DataCite | الأوراق ذات DOI — تتحقق من أن DOI يعمل والعنوان يتطابق (DataCite تتعامل مع DOI الخاص بـ arXiv `10.48550`) |
| L3 | Semantic Scholar + بحث عنوان arXiv | جميع المتبقية — مطابقة عناوين ضبابية (≥0.80 تشابه) |
| L4 | تقييم صلة بنموذج اللغة | جميع المراجع الموثّقة — تقييم الصلة الموضوعية بالبحث |

كل مرجع → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**تنظيف تلقائي**: الاستشهادات المُلفّقة تُزال بصمت من نص الورقة (بدون علامات `[HALLUCINATED]`). إدخالات قائمة المراجع غير المُستشهد بها تُحذف. ملف `references.bib` النهائي يحتوي فقط على مراجع موثّقة ومُستشهد بها.

### 🖥️ تنفيذ واعٍ بالعتاد

تكتشف المرحلة 1 تلقائياً قدرات GPU المحلية وتكيّف خط الأنابيب بالكامل:

| المستوى | الكشف | السلوك |
|------|-----------|----------|
| **عالي** | GPU من NVIDIA مع ≥8 جيجابايت VRAM | توليد كود PyTorch/GPU كامل، تثبيت torch تلقائياً إذا كان مفقوداً |
| **محدود** | NVIDIA <8 جيجابايت أو Apple MPS | تجارب خفيفة (<1 مليون معامل، ≤20 حقبة)، تحذير للمستخدم |
| **CPU فقط** | لم يُكتشف GPU | NumPy/sklearn فقط، بدون استيراد torch، تحذير مع توصية بـ GPU عن بعد |

ملف تعريف العتاد يُحفظ في `stage-01/hardware_profile.json` ويؤثر على توليد الكود واستيرادات البيئة المعزولة وقيود الأوامر النصية.

### 🧪 تنفيذ التجارب في بيئة معزولة

- **التحقق من الكود** — تحليل AST، قائمة بيضاء للاستيرادات، بدون I/O للملفات خارج البيئة المعزولة
- **حارس ميزانية الحوسبة** — ميزانية زمنية (قابلة للتهيئة، الافتراضي 600 ثانية) تُحقن في أمر توليد الكود؛ يجب على نموذج اللغة تصميم تجارب تناسب مهلة البيئة المعزولة
- **إطار التجارب** — ملف `experiment_harness.py` غير قابل للتعديل يُحقن في البيئة المعزولة مع حارس وقت `should_stop()`، رفض NaN/Inf في `report_metric()`، وكتابة النتائج في `finalize()` (مستوحى من نمط التقييم غير القابل للتعديل في karpathy/autoresearch)
- **مخرجات منظمة** — التجارب تنتج `results.json` مع مقاييس مُحددة النوع (وليس فقط تحليل stdout)
- **تحليل ذكي للمقاييس** — يُصفّي سطور السجل من المقاييس باستخدام كشف الكلمات المفتاحية (`is_metric_name()`)
- **فشل سريع عند NaN/التباعد** — قيم NaN/Inf تُصفّى من المقاييس؛ خسارة متباعدة (>100) تُكتشف وتُبلّغ
- **فرض التقارب** — الكود المُولّد يجب أن يتضمن معايير إيقاف مبكر، وليس عدد تكرارات ثابت
- **كشف أخطاء وقت التشغيل** — مقاييس NaN/Inf وتحذيرات stderr (القسمة على صفر، الفيضان) تُكتشف تلقائياً
- **إصلاح ذاتي** — مشاكل وقت التشغيل تُغذّى مرة أخرى لنموذج اللغة مع تشخيص مُستهدف لإصلاحات السبب الجذري (وليس try/except كحل مؤقت)
- **تحسين تكراري** — المرحلة 13 تحلل النتائج وتعيد التشغيل مع كود/معاملات محسّنة (حتى 10 تكرارات، مع أوامر واعية بالمهلة)
- **التقاط نتائج جزئية** — التجارب المنتهية المهلة مع مقاييس ملتقطة تحصل على حالة "partial" بدلاً من "failed"، مما يحفظ البيانات القابلة للاستخدام
- **محاذاة الموضوع والتجربة** — فحص ما بعد التوليد بنموذج اللغة يضمن أن كود التجربة يختبر فعلاً موضوع البحث المُعلن

### 📝 كتابة ورقة بمستوى المؤتمرات

يستهدف خط أنابيب الكتابة معايير NeurIPS/ICML/ICLR (9+ صفحات، 5,000-6,500 كلمة):

- **فرض سلامة البيانات** — كتابة الورقة تُحظر تماماً عندما لا تنتج التجارب مقاييس (يمنع نموذج اللغة من تلفيق النتائج)؛ تعليمات مضادة للتلفيق تُحقن في كل من أوامر المسودة والمراجعة
- **أوامر بمستوى المؤتمرات** — الأوامر النظامية تتضمن مبادئ أساسية من تحليلات الأوراق المقبولة: الجِدّة، السرد، خطوط أساس قوية، استئصالات، صدق، قابلية التكرار؛ أسباب الرفض الشائعة مُبلّغة
- **إرشادات العنوان والتأطير** — إشارة الجِدّة، اختبار القابلية للانتشار، بنية ملخص من 5 جمل، كشف العناوين العامة مع إعادة التوليد
- **صياغة قسم بقسم** — 3 استدعاءات متسلسلة لنموذج اللغة (مقدمة+أعمال سابقة → منهجية+تجارب → نتائج+خاتمة) لتجنب اقتطاع المخرجات
- **أهداف عدد كلمات لكل قسم** — الملخص (150-250)، المقدمة (800-1000)، الأعمال السابقة (600-800)، المنهجية (1000-1500)، التجارب (800-1200)، النتائج (600-800)، المناقشة (400-600)
- **حماية طول المراجعة** — إذا كانت الورقة المُراجَعة أقصر من المسودة، تُعاد المحاولة تلقائياً مع تطبيق أقوى؛ يعود للمسودة+التعليقات إذا لزم الأمر
- **فرض مضاد لإخلاءات المسؤولية** — يحد من "بسبب القيود الحسابية" إلى حدوث واحد على الأكثر؛ أوامر المراجعة تزيل التحفظات المتكررة بنشاط
- **صرامة إحصائية** — فترات الثقة وقيم p وأحجام التأثير مطلوبة في جداول النتائج؛ الاستئصالات المعطلة تُبلّغ وتُستبعد من الادعاءات
- **مراجعة أقران بمعايير المؤتمرات** — المراجعون يقيّمون 1-10 وفق معايير NeurIPS/ICML (الجِدّة، خطوط الأساس، الاستئصالات، الادعاءات مقابل الأدلة، القيود)

### 📐 تبديل قوالب المؤتمرات

```yaml
export:
  target_conference: "neurips_2025"   # or "iclr_2026" or "icml_2026"
```

| المؤتمر | حزمة النمط | الأعمدة |
|------------|--------------|---------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

محوّل Markdown → LaTeX يتعامل مع: عناوين الأقسام (مع إزالة تكرار الترقيم التلقائي)، الرياضيات المضمّنة/المعروضة، الخط العريض/المائل، القوائم، الجداول (مع `\caption`/`\label`)، الأشكال (`\includegraphics`)، كتل الكود (آمنة لـ Unicode)، المراجع التبادلية، ومراجع `\cite{}`.

### 🚦 بوابات الجودة

| البوابة | المرحلة | عند الرفض → العودة إلى |
|------|-------|---------------------------|
| فرز الأدبيات | 5 | إعادة جمع الأدبيات (المرحلة 4) |
| تصميم التجارب | 9 | إعادة توليد الفرضيات (المرحلة 8) |
| بوابة الجودة | 20 | إعادة كتابة الورقة من المخطط (المرحلة 16) |

استخدم `--auto-approve` لتخطي جميع البوابات، أو هيّئ مراحل محددة في `security.hitl_required_stages`.

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
  provider: "openai-compatible"    # نوع المزوّد
  base_url: "https://..."          # نقطة نهاية API (مطلوب)
  api_key_env: "OPENAI_API_KEY"    # متغير بيئة لمفتاح API (مطلوب)
  api_key: ""                      # أو ضع المفتاح هنا مباشرة
  primary_model: "gpt-4o"          # النموذج الأساسي
  fallback_models: ["gpt-4o-mini"] # سلسلة النماذج الاحتياطية
  s2_api_key: ""                   # مفتاح Semantic Scholar API (اختياري، حدود معدل أعلى)

# === التجارب ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # أقصى وقت تنفيذ لكل تشغيل (الافتراضي: 600 ثانية)
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

<p align="center">
  <sub>بُني بـ 🦞 بواسطة فريق AutoResearchClaw</sub>
</p>
