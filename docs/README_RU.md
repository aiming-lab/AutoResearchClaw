<p align="center">
  <img src="../image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Напишите идею. Получите статью. Автономный, Совместный & Самоэволюционирующий.</b></h2>

<p align="center">
  <b><i><font size="5">Просто напишите <a href="#-интеграция-с-openclaw">OpenClaw</a>: «Исследуй X» → готово.</font></i></b>
</p>

<p align="center">
  <img src="../image/framework_v2.png" width="100%" alt="AutoResearchClaw Framework">
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#тестирование"><img src="https://img.shields.io/badge/Tests-2699%20passed-brightgreen?logo=pytest&logoColor=white" alt="2699 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#-интеграция-с-openclaw"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="showcase/SHOWCASE.md">🏆 Галерея статей</a> · <a href="HITL_GUIDE.md">🧑‍✈️ Руководство Co-Pilot</a> · <a href="integration-guide.md">📖 Руководство по интеграции</a> · <a href="https://discord.gg/u4ksqW5P">💬 Сообщество в Discord</a>
</p>

---

<table>
<tr>
<td width="18%">
<a href="showcase/SHOWCASE.md"><img src="showcase/thumbnails/paper_I_random_matrix-01.png" width="120" alt="Пример статьи"/></a>
</td>
<td valign="middle">
<b>🏆 Галерея сгенерированных статей</b><br><br>
<b>8 статей в 8 областях</b> — математика, статистика, биология, информатика, NLP, RL, компьютерное зрение, робастность — сгенерированы полностью автономно или с направляющим участием Human-in-the-Loop Co-Pilot.<br><br>
<a href="showcase/SHOWCASE.md"><img src="https://img.shields.io/badge/Посмотреть_галерею_→-Все_8_статей-d73a49?style=for-the-badge" alt="Посмотреть галерею"></a>
</td>
</tr>
</table>

---

> **🧪 Мы ищем тестировщиков!** Попробуйте запустить пайплайн со своей исследовательской идеей из любой области и [расскажите нам о результатах](TESTER_GUIDE.md). Ваш фидбек напрямую влияет на развитие проекта. **[→ Руководство по тестированию](TESTER_GUIDE.md)** | **[→ 中文测试指南](TESTER_GUIDE_CN.md)** | **[→ 日本語テストガイド](TESTER_GUIDE_JA.md)**

---

## 🔥 Новости
- **[01.04.2026]** **v0.4.0** — **Система Human-in-the-Loop Co-Pilot** — AutoResearchClaw больше не является чисто автономным. Новая HITL-система добавляет 6 режимов вмешательства (`full-auto`, `gate-only`, `checkpoint`, `step-by-step`, `co-pilot`, `custom`), настраиваемые политики для каждого этапа и глубокое взаимодействие человека и ИИ. Включает: Мастерскую идей для совместного формирования гипотез, Навигатор бейзлайнов для обзора дизайна экспериментов, Совместное написание статьи с Paper Co-Writer, SmartPause (динамическое вмешательство по уровню уверенности), обучение на интервенциях (ALHF), проверку утверждений на антигаллюцинацию, контроль бюджета затрат, ветвление пайплайна для параллельного исследования гипотез и CLI-команды (`attach`/`status`/`approve`/`reject`/`guide`). **[→ Полное руководство HITL](HITL_GUIDE.md)**
- **[30.03.2026]** **Гибкая загрузка навыков** — AutoResearchClaw теперь поддерживает загрузку открытых и пользовательских навыков из любой дисциплины для расширения исследовательских возможностей. 19 предустановленных навыков включены в качестве готовых примеров — научное письмо, дизайн экспериментов, химия, биология и др., включая навык агентной эволюции [A-Evolve](https://github.com/A-EVO-Lab/a-evolve), предоставленный сообществом. Загружайте свои навыки через `researchclaw skills install` или поместите `SKILL.md` в `.claude/skills/`. См. [Библиотека навыков](#-библиотека-навыков).
- **[22.03.2026]** [v0.3.2](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.2) — **Кроссплатформенная поддержка + крупное обновление стабильности** — AutoResearchClaw теперь работает с любым ACP-совместимым агентом (Claude Code, Codex CLI, Copilot CLI, Gemini CLI, Kimi CLI) и поддерживает мессенджеры (Discord, Telegram, Lark, WeChat) через мост OpenClaw. Новый CLI-agent бэкенд генерации кода делегирует Stage 10 и 13 внешним CLI-агентам с контролем бюджета и управлением таймаутами. Включает систему защиты от фабрикации (VerifiedRegistry + цикл диагностики и ремонта экспериментов), 100+ исправлений багов, модульный рефакторинг executor, автоопределение `--resume`, усиление повторов LLM и исправления от сообщества.

<details>
<summary>Предыдущие версии</summary>

- **[18.03.2026]** [v0.3.1](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.1) — **OpenCode Beast Mode + Контрибьюты сообщества** — Новый режим "Beast Mode" перенаправляет сложную генерацию кода в [OpenCode](https://github.com/anomalyco/opencode) с автоматической оценкой сложности и безопасным фоллбэком. Добавлена поддержка провайдера Novita AI, улучшена потокобезопасность, повышена надежность парсинга ответов LLM, а также исправлено более 20 багов благодаря PR от сообщества и внутреннему аудиту.
- **[17.03.2026]** [v0.3.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.0) — **Интеграция с MetaClaw** — AutoResearchClaw теперь поддерживает кросс-сессионное обучение через [MetaClaw](https://github.com/aiming-lab/MetaClaw): ошибки пайплайна → структурированные уроки → переиспользуемые навыки, которые внедряются во все 23 этапа. Робастность в контролируемых экспериментах выросла на **+18.3%**. Фича опциональна (`metaclaw_bridge.enabled: true`) и полностью обратно совместима. См. [Руководство по интеграции](#-интеграция-с-metaclaw).
- **[16.03.2026]** [v0.2.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.2.0) — Три мультиагентные подсистемы (CodeAgent, BenchmarkAgent, FigureAgent), защищенная Docker-песочница с поддержкой сетевых политик, 4-этапный аудит качества статьи (поиск ИИ-галлюцинаций, оценка по 7 критериям, чек-лист NeurIPS) и более 15 исправлений багов с продакшена.
- **[15.03.2026]** [v0.1.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.1.0) — Релиз AutoResearchClaw: полностью автономный исследовательский пайплайн из 23 этапов, который превращает одну идею в готовую для конференции статью. Без вмешательства человека.

</details>

---

## ⚡ Одна команда. Одна статья.

```bash
# Полностью автономный режим — без участия человека
pip install -e . && researchclaw setup && researchclaw init && researchclaw run --topic "Ваша исследовательская идея" --auto-approve

# Режим Co-Pilot — совместная работа с ИИ в ключевых точках принятия решений
researchclaw run --topic "Ваша исследовательская идея" --mode co-pilot
```

---

## 🤔 Что это такое?

**Вы придумываете. AutoResearchClaw пишет. Вы направляете ключевые решения.**

Задайте тему исследования — и получите полноценную академическую статью с реальным обзором литературы из OpenAlex, Semantic Scholar и arXiv, экспериментами в песочнице с учетом вашего железа (автоопределение GPU/MPS/CPU), статистическим анализом, мультиагентным рецензированием и готовым LaTeX-кодом для конференций NeurIPS/ICML/ICLR. Запускайте полностью автономно или используйте **режим Co-Pilot**, чтобы направлять ИИ в критических точках — выбирайте направления исследований, проверяйте дизайн экспериментов и совместно пишите статью. Никаких выдуманных ссылок.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Полная академическая статья (Введение, Обзор литературы, Метод, Эксперименты, Результаты, Заключение)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>Готовый LaTeX-код (шаблоны NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Реальные BibTeX-ссылки из OpenAlex, Semantic Scholar и arXiv — автоматически отфильтрованные под цитаты в тексте</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4-уровневая проверка целостности и релевантности цитирования (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Сгенерированный код + результаты из песочницы + структурированные JSON-метрики</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Автоматически сгенерированные графики сравнения с планками погрешностей и доверительными интервалами</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Мультиагентное рецензирование с проверкой согласованности методологии и результатов</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Уроки для самообучения, извлеченные из каждого запуска</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>Все итоговые материалы в одной папке — готовы к загрузке в Overleaf</td></tr>
</table>

Пайплайн работает **от начала до конца** — полностью автономно или с совместным участием человека (human-in-the-loop). Если эксперименты падают — он чинит код. Если гипотезы не подтверждаются — он меняет направление. Если цитаты оказываются фейковыми — он их удаляет. Если вы хотите направить процесс — он останавливается и слушает.

🌍 **Запускайте где угодно.** AutoResearchClaw не привязан к одной платформе. Используйте его автономно через CLI, подключите к [OpenClaw](https://github.com/openclaw/openclaw) или интегрируйте с любым ACP-совместимым агентом — 🤖 Claude Code, 💻 Codex CLI, 🐙 Copilot CLI, ♊ Gemini CLI, 🌙 Kimi CLI. Благодаря мосту сообщений OpenClaw вы можете запустить полное исследование из 💬 Discord, ✈️ Telegram, 🐦 Lark (飞书), 💚 WeChat или любой другой платформы, которую использует ваша команда. Один топик на входе, одна статья на выходе — откуда бы вы ни писали.

---

## 🚀 Быстрый старт

```bash
# 1. Клонируйте и установите
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Настройка (интерактивная — устанавливает OpenCode beast mode, проверяет Docker/LaTeX)
researchclaw setup

# 3. Конфигурация
researchclaw init          # Интерактивный режим: выбор провайдера LLM, создание config.arc.yaml
# Или вручную: cp config.researchclaw.example.yaml config.arc.yaml

# 4. Запуск
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Ваша исследовательская идея" --auto-approve
```

Результаты → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — готовые к компиляции LaTeX, BibTeX, код экспериментов, графики.

<details>
<summary>📝 Минимальная конфигурация</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Ваша тема исследования"

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

## 🧠 В чем отличие от других

| Фича | Как это работает |
|-----------|-------------|
| **🧑‍✈️ Режим Co-Pilot** | 6 режимов вмешательства — от полностью автономного до пошагового. Направляйте ИИ в критических решениях (гипотезы, бейзлайны, написание статьи) или дайте ему работать самостоятельно. SmartPause автоматически определяет, когда помощь человека будет полезна. |
| **🔄 Цикл PIVOT / REFINE** | На 15-м этапе система автономно решает: ПРОДОЛЖИТЬ, УЛУЧШИТЬ (подобрать параметры) или СМЕНИТЬ КУРС (PIVOT). Артефакты версионируются автоматически. |
| **🤖 Мультиагентные дебаты** | Генерация гипотез, анализ результатов и рецензирование проходят в формате структурированных дебатов с разных точек зрения. |
| **🧬 Самообучение** | Из каждого запуска извлекаются уроки (обоснование решений, ошибки в коде, аномалии в метриках) с периодом полураспада в 30 дней. Будущие запуски учатся на прошлых ошибках. |
| **📚 База знаний** | Каждый запуск пополняет структурированную базу знаний по 6 категориями (решения, эксперименты, находки, литература, вопросы, рецензии). |
| **🛡️ Сторожевой модуль Sentinel** | Фоновый мониторинг качества: обнаружение NaN/Inf, проверка соответствия текста статьи реальным данным, оценка релевантности цитат, защита от фабрикации фактов. |
| **🔍 Проверка утверждений** | Инлайн-фактчекинг: извлекает утверждения из текста, сгенерированного ИИ, и сверяет их с собранной литературой. Отмечает необоснованные цитаты и выдуманные числа. |
| **🌿 Ветвление исследований** | Ответвление пайплайна для одновременного исследования нескольких направлений, сравнение результатов бок о бок и слияние лучшего пути. |

---

## 🦞 Интеграция с OpenClaw

<table>
<tr>

**AutoResearchClaw полностью совместим с [OpenClaw](https://github.com/openclaw/openclaw).** Установите его в OpenClaw и запускайте автономные исследования одним сообщением — или используйте отдельно через CLI, Claude Code или любой другой ИИ-ассистент.

</tr>
</table>

### 🚀 Использование с OpenClaw (Рекомендуется)

Если вы уже используете [OpenClaw](https://github.com/openclaw/openclaw) как своего ИИ-ассистента:

```
1️⃣  Отправьте URL репозитория в OpenClaw
2️⃣  OpenClaw автоматически прочитает RESEARCHCLAW_AGENTS.md → поймет структуру пайплайна
3️⃣  Напишите: "Проведи исследование на тему [ваша тема]"
4️⃣  Готово — OpenClaw сам склонирует, установит, настроит, запустит и вернет результаты
```

**Вот и всё.** OpenClaw берет на себя `git clone`, `pip install`, настройку конфигов и запуск пайплайна. Вы просто общаетесь в чате.

<details>
<summary>💡 Что происходит под капотом</summary>

1. OpenClaw читает `RESEARCHCLAW_AGENTS.md` → принимает на себя роль исследовательского оркестратора
2. OpenClaw читает `README.md` → понимает процесс установки и структуру пайплайна
3. OpenClaw копирует `config.researchclaw.example.yaml` → `config.yaml`
4. Запрашивает ваш API-ключ (или использует переменную окружения)
5. Выполняет `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. Возвращает готовую статью, LaTeX, код экспериментов и список литературы

</details>

### 🔌 Мост OpenClaw (Продвинутый уровень)

Для более глубокой интеграции в AutoResearchClaw встроена **система адаптеров** с 6 опциональными возможностями:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Запуск исследований по расписанию
  use_message: true           # 💬 Уведомления о прогрессе (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Сохранение знаний между сессиями
  use_sessions_spawn: true    # 🔀 Запуск параллельных подсессий для независимых этапов
  use_web_fetch: true         # 🌐 Поиск в интернете в реальном времени при обзоре литературы
  use_browser: false          # 🖥️ Сбор статей через браузер
```

Каждый флаг активирует типизированный протокол адаптера. Если OpenClaw поддерживает эти функции, адаптеры используют их без изменения кода. Подробности см. в [`integration-guide.md`](integration-guide.md).

### ACP (Agent Client Protocol)

AutoResearchClaw может использовать **любого ACP-совместимого агента** в качестве LLM-бэкенда — API-ключи не требуются. Агент общается через [acpx](https://github.com/openclaw/acpx), поддерживая единую сессию на протяжении всех 23 этапов.

| Агент | Команда | Примечания |
|-------|---------|-------|
| Claude Code | `claude` | Anthropic |
| Codex CLI | `codex` | OpenAI |
| Copilot CLI | `gh` | GitHub |
| Gemini CLI | `gemini` | Google |
| OpenCode | `opencode` | SST |
| Kimi CLI | `kimi` | Moonshot |

```yaml
# config.yaml — пример ACP
llm:
  provider: "acp"
  acp:
    agent: "claude"   # Любая команда CLI ACP-совместимого агента
    cwd: "."          # Рабочая директория для агента
  # base_url и api_key не нужны — агент сам управляет авторизацией.
```

```bash
# Просто запускайте — агент использует свои собственные учетные данные
researchclaw run --config config.yaml --topic "Ваша идея" --auto-approve
```

### 🛠️ Другие способы запуска

| Способ | Как запустить |
|--------|-----|
| **CLI** | `researchclaw run --topic "..." --auto-approve` (автономный) или `--mode co-pilot` (совместный) |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Читает `RESEARCHCLAW_CLAUDE.md` — просто напишите *"Run research on [topic]"* |
| **Copilot CLI** | `researchclaw run --topic "..."` с `llm.acp.agent: "gh"` |
| **OpenCode** | Читает `.claude/skills/` — такой же интерфейс на естественном языке |
| **Любой AI CLI** | Скормите `RESEARCHCLAW_AGENTS.md` в контекст → агент сам поймет, что делать |

---

## 🔬 Пайплайн: 23 этапа, 8 фаз

```
Фаза A: Определение области          Фаза E: Выполнение экспериментов
  1. TOPIC_INIT                         12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE                  13. ITERATIVE_REFINE  ← самовосстановление

Фаза B: Поиск литературы             Фаза F: Анализ и принятие решений
  3. SEARCH_STRATEGY                    14. RESULT_ANALYSIS    ← мультиагентный анализ
  4. LITERATURE_COLLECT  ← API          15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [гейт]
  6. KNOWLEDGE_EXTRACT                  Фаза G: Написание статьи
                                        16. PAPER_OUTLINE
Фаза C: Синтез знаний                   17. PAPER_DRAFT
  7. SYNTHESIS                          18. PEER_REVIEW        ← проверка доказательств
  8. HYPOTHESIS_GEN    ← дебаты         19. PAPER_REVISION

Фаза D: Дизайн экспериментов         Фаза H: Финализация
  9. EXPERIMENT_DESIGN   [гейт]         20. QUALITY_GATE      [гейт]
 10. CODE_GENERATION                    21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING                  22. EXPORT_PUBLISH     ← LaTeX
                                        23. CITATION_VERIFY    ← проверка релевантности
```

> **Гейты (Контрольные точки)** (5, 9, 20) ставят пайплайн на паузу для апрува человеком (или пропускаются флагом `--auto-approve`). При отклонении пайплайн откатывается назад.

> **Режим Co-Pilot** (`--mode co-pilot`): Глубокое взаимодействие человека и ИИ на этапах 7-8 (Мастерская идей), этапе 9 (Навигатор бейзлайнов) и этапах 16-17 (Совместное написание статьи). Остальные этапы выполняются автоматически с мониторингом SmartPause.

> **Циклы принятия решений**: На 15-м этапе система может уйти на доработку (REFINE → Этап 13) или сменить курс (PIVOT → Этап 8), автоматически сохраняя версии артефактов.

<details>
<summary>📋 Что происходит на каждой фазе</summary>

| Фаза | Описание |
|-------|-------------|
| **A: Определение области** | LLM разбивает тему на структурированное дерево проблем с исследовательскими вопросами. |
| **A+: Железо** | Автоопределение GPU (NVIDIA CUDA / Apple MPS / CPU), предупреждения о нехватке ресурсов, адаптация генерации кода под доступное железо. |
| **B: Литература** | Поиск по нескольким базам (OpenAlex → Semantic Scholar → arXiv) реальных статей, фильтрация по релевантности, извлечение карточек знаний. |
| **C: Синтез** | Кластеризация находок, поиск пробелов в исследованиях, генерация проверяемых гипотез через мультиагентные дебаты. |
| **D: Дизайн** | Проектирование плана экспериментов, генерация Python-кода с учетом железа (выбор пакетов под GPU), оценка требуемых ресурсов. |
| **E: Выполнение** | Запуск экспериментов в песочнице, отлов NaN/Inf и багов в рантайме, самовосстановление кода через LLM. |
| **F: Анализ** | Мультиагентный анализ результатов; автономное решение ПРОДОЛЖИТЬ / УЛУЧШИТЬ / СМЕНИТЬ КУРС с подробным обоснованием. |
| **G: Написание** | План → написание по разделам (5,000-6,500 слов) → рецензирование (с проверкой соответствия методологии и результатов) → редактура с контролем объема. |
| **H: Финализация** | Контроль качества, архивация знаний, экспорт в LaTeX по шаблонам конференций, проверка целостности и релевантности цитат. |

</details>

---

## ✨ Ключевые фичи

| Фича | Описание |
|---------|------------|
| **📚 Мультиисточниковая литература** | Реальные статьи из OpenAlex, Semantic Scholar и arXiv — расширение запросов, дедупликация, защита от падений API с постепенной деградацией. |
| **🔍 4-уровневая проверка цитат** | Проверка arXiv ID → CrossRef/DataCite DOI → совпадение заголовков в Semantic Scholar → оценка релевантности через LLM. Выдуманные ссылки удаляются автоматически. |
| **🖥️ Адаптация под железо** | Автоопределение GPU (NVIDIA CUDA / Apple MPS / CPU) и адаптация генерации кода, импортов и масштаба экспериментов. |
| **🦾 OpenCode Beast Mode** | Сложные эксперименты автоматически перенаправляются в [OpenCode](https://github.com/anomalyco/opencode) — генерация многофайловых проектов с кастомными архитектурами, циклами обучения и ablation studies. Устанавливается через `researchclaw setup`. |
| **🧪 Эксперименты в песочнице** | Валидация кода через AST, неизменяемая обвязка, быстрый отказ при NaN/Inf, самовосстановление, итеративное улучшение (до 10 раундов), сохранение частичных результатов. |
| **📝 Написание уровня конференций** | Шаблоны NeurIPS/ICML/ICLR, написание по разделам (5,000-6,500 слов), защита от выдуманных фактов, контроль объема при редактуре, удаление типичных ИИ-оговорок. |
| **📐 Переключение шаблонов** | `neurips_2025`, `iclr_2026`, `icml_2026` — Markdown → LaTeX с формулами, таблицами, графиками, перекрестными ссылками и `\cite{}`. |
| **🛡️ Анти-фабрикация** | VerifiedRegistry обеспечивает использование проверенных экспериментальных данных в статьях. Автоматическая диагностика и восстановление неудачных экспериментов перед написанием. Непроверенные числа очищаются. |
| **🚦 Гейты качества** | 3 точки контроля человеком (Этапы 5, 9, 20) с возможностью отката. Можно пропустить флагом `--auto-approve`. |
| **🧑‍✈️ HITL Co-Pilot** | 6 режимов вмешательства с настраиваемыми политиками для каждого этапа. Мастерская идей, Навигатор бейзлайнов, Paper Co-Writer для глубокого взаимодействия. SmartPause, контроль бюджета, политики эскалации и обучение на интервенциях для безопасности в продакшене. Адаптеры CLI/WebSocket/MCP. |
| **💰 Контроль бюджета** | Мониторинг затрат с настраиваемыми порогами оповещений (50%/80%/100%). Пайплайн автоматически приостанавливается при превышении бюджета. |
| **🔐 Воспроизводимость** | SHA256-контрольные суммы для всех артефактов этапов. Неизменяемые манифесты для верификации. Многоуровневый откат с версионированными снимками. |

---

## 🧑‍✈️ Human-in-the-Loop Co-Pilot

**AutoResearchClaw v0.4.0 представляет полноценную систему Human-in-the-Loop (HITL)**, которая превращает пайплайн из чисто автономного в совместный исследовательский движок человека и ИИ. Выберите свой уровень вовлеченности:

### Режимы вмешательства

| Режим | Команда | Что делает |
|------|---------|-------------|
| **Full Auto** | `--auto-approve` | Исходное поведение — без вмешательства человека |
| **Gate Only** | `--mode gate-only` | Пауза на 3 гейтах (5, 9, 20) для одобрения |
| **Checkpoint** | `--mode checkpoint` | Пауза на границе каждой фазы (8 контрольных точек) |
| **Co-Pilot** | `--mode co-pilot` | Глубокое взаимодействие на критических этапах, автоматический режим на остальных |
| **Step-by-Step** | `--mode step-by-step` | Пауза после каждого этапа — для изучения пайплайна |
| **Express** | `--mode express` | Быстрый обзор — только 3 самых критических гейта |

### Рабочий процесс Co-Pilot

```
You: researchclaw run --topic "Квантовый шум как регуляризация нейронных сетей" --mode co-pilot

Пайплайн выполняет этапы 1-7 автоматически...

  ┌─────────────────────────────────────────────────────────────┐
  │  HITL | Stage 08: HYPOTHESIS_GEN                            │
  │  Пост-этапная проверка                                      │
  │                                                             │
  │  Упомянуто гипотез: 3                                       │
  │  Оценка новизны: 0.72 (умеренная)                           │
  │                                                             │
  │  [a] Одобрить  [r] Отклонить  [e] Редактировать  [c] Чат    │
  │  [i] Дать указание  [v] Посмотреть вывод  [q] Прервать      │
  └─────────────────────────────────────────────────────────────┘

You: c  (начать совместный чат)
You: Гипотеза 3 интересна, но нужны бейзлайны Dropout/Label Smoothing
AI:  Обновлено — добавлены Dropout, Label Smoothing, MixUp, CutMix в качестве бейзлайнов...
You: approve

Пайплайн продолжает с вашей улучшенной гипотезой...
```

### CLI-команды

```bash
# Запуск в режиме HITL
researchclaw run --topic "..." --mode co-pilot

# Подключение к приостановленному пайплайну (из другого терминала)
researchclaw attach artifacts/rc-2026-xxx

# Проверка статуса пайплайна и HITL
researchclaw status artifacts/rc-2026-xxx

# Одобрение/отклонение из другого терминала или скрипта
researchclaw approve artifacts/rc-2026-xxx --message "LGTM"
researchclaw reject artifacts/rc-2026-xxx --reason "Отсутствует ключевой бейзлайн"

# Предоставление указаний для этапа (даже до его выполнения)
researchclaw guide artifacts/rc-2026-xxx --stage 9 --message "Использовать ResNet-50 как основной бейзлайн"
```

### Ключевые возможности

| Возможность | Описание |
|---------|------------|
| **Мастерская идей** | Совместное обсуждение, оценка и доработка гипотез (Этапы 7-8) |
| **Навигатор бейзлайнов** | ИИ предлагает бейзлайны + человек добавляет/удаляет + чек-лист воспроизводимости (Этап 9) |
| **Paper Co-Writer** | Написание по разделам с редактированием человеком и полировкой ИИ (Этапы 16-19) |
| **SmartPause** | Динамическая пауза по уровню уверенности — автоматически определяет, когда помощь человека будет полезна |
| **Проверка утверждений** | Инлайн-фактчекинг по собранной литературе — отмечает необоснованные утверждения |
| **Контроль бюджета** | Мониторинг затрат с порогами оповещений 50%/80%/100% |
| **Обучение на интервенциях** | ALHF — учится на ваших паттернах ревью для оптимизации будущих пауз |
| **Ветвление исследований** | Ответвление пайплайна для исследования нескольких гипотез, сравнение, слияние лучшего |
| **Политика эскалации** | Многоуровневые уведомления (терминал → Slack → email → авто-остановка) при отсутствии оператора |
| **3 адаптера** | CLI (терминал), WebSocket (веб-панель), MCP (внешние агенты) |

### Конфигурация

```yaml
# config.arc.yaml
hitl:
  enabled: true
  mode: co-pilot                     # full-auto | gate-only | checkpoint | co-pilot | custom
  cost_budget_usd: 50.0              # Пауза при превышении бюджета (0 = без лимита)

  notifications:
    on_pause: true
    on_quality_drop: true
    channels: ["terminal"]            # terminal | slack | webhook

  timeouts:
    default_human_timeout_sec: 86400  # Ожидание до 24ч
    auto_proceed_on_timeout: false

  collaboration:
    max_chat_turns: 50
    save_chat_history: true

  # Кастомные политики для этапов (опционально, для режима 'custom')
  stage_policies:
    8: { require_approval: true, enable_collaboration: true }
    9: { require_approval: true, allow_edit_output: true }
```

### Обратная совместимость

- **По умолчанию: ВЫКЛЮЧЕНО.** Без `hitl.enabled: true` или `--mode` пайплайн работает как раньше.
- **`--auto-approve` по-прежнему работает.** Он перекрывает режим HITL.
- **Все 2 699 тестов проходят успешно** с кодом HITL.

---

## 🧠 Интеграция с MetaClaw

**AutoResearchClaw + [MetaClaw](https://github.com/aiming-lab/MetaClaw) = Пайплайн, который учится на каждом запуске.**

MetaClaw добавляет **перенос знаний между запусками**. Если эта функция включена, пайплайн автоматически извлекает уроки из ошибок и предупреждений, превращает их в переиспользуемые навыки и внедряет во все 23 этапа при следующих запусках — чтобы больше никогда не повторять одни и те же ошибки.

### Как это работает

```
Запуск N выполняется → ошибки/предупреждения сохраняются как Уроки (Lessons)
                      ↓
          MetaClaw конвертирует Урок → Навык (Skill)
                      ↓
          Файлы навыков arc-* сохраняются в ~/.metaclaw/skills/
                      ↓
Запуск N+1 → build_overlay() внедряет навыки в каждый промпт LLM
                      ↓
          LLM избегает известных ошибок → выше качество, меньше ретраев
```

### Быстрая настройка

```bash
# 1. Установите MetaClaw (если еще не установлен)
pip install metaclaw

# 2. Включите в конфиге
```

```yaml
# config.arc.yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"        # Прокси MetaClaw (опционально)
  skills_dir: "~/.metaclaw/skills"          # Папка для хранения навыков
  fallback_url: "https://api.openai.com/v1" # Прямой фоллбэк к LLM
  fallback_api_key: ""                      # API-ключ для фоллбэка
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # Конвертировать предупреждения и ошибки
    max_skills_per_run: 3
```

```bash
# 3. Запускайте как обычно — MetaClaw работает прозрачно
researchclaw run --config config.arc.yaml --topic "Ваша идея" --auto-approve
```

После каждого запуска заглядывайте в `~/.metaclaw/skills/arc-*/SKILL.md`, чтобы посмотреть, чему научился ваш пайплайн.

### Результаты экспериментов

В контролируемых A/B тестах (одна тема, одна LLM, один конфиг):

| Метрика | База | С MetaClaw | Улучшение |
|--------|----------|---------------|-------------|
| Частота ретраев на этапах | 10.5% | 7.9% | **-24.8%** |
| Количество циклов доработки (Refine) | 2.0 | 1.2 | **-40.0%** |
| Успешное завершение пайплайна | 18/19 | 19/19 | **+5.3%** |
| Общий индекс робастности (композитный) | 0.714 | 0.845 | **+18.3%** |

> Композитный индекс робастности — это взвешенное среднее из процента завершения (40%), снижения ретраев (30%) и эффективности циклов доработки (30%).

### Обратная совместимость

- **По умолчанию: ВЫКЛЮЧЕНО.** Если блока `metaclaw_bridge` нет или `enabled: false`, пайплайн работает как раньше.
- **Никаких новых зависимостей.** MetaClaw опционален — ядро работает и без него.
- **Все 2 699 тестов проходят успешно** даже с кодом интеграции.

---

## 🧩 Библиотека навыков

AutoResearchClaw теперь поддерживает загрузку **открытых и пользовательских навыков** для расширения исследовательских возможностей. Мы также поставляем **19 предустановленных навыков** (научное письмо, поиск литературы, химия, биология и др.) в качестве готовых примеров, обеспечивая высокую гибкость из коробки. Отключите любой навык, добавив `enabled: false` в его метаданные.

**Примеры встроенных навыков:**

| Категория | Навык | Описание |
|----------|-------|-------------|
| **Написание** | `scientific-writing` | Структура IMRAD, форматирование цитат, стандарты отчетности |
| **Домен** | `chemistry-rdkit` | Молекулярный анализ, SMILES, фингерпринты, открытие лекарств |
| **Эксперимент** | `literature-search` | Систематический обзор, методология PRISMA |

> Смотрите все 19 навыков командой `researchclaw skills list`.

### Загрузка своих навыков

```bash
# Вариант 1: Установить навык (сохраняется между проектами)
researchclaw skills install /path/to/my-skill/

# Вариант 2: Поместить SKILL.md в проект
mkdir -p .claude/skills/my-custom-skill
# Затем создайте SKILL.md с YAML-метаданными (name, description, trigger-keywords, applicable-stages)

# Вариант 3: Настроить общие директории навыков в config.arc.yaml
# skills:
#   custom_dirs:
#     - /path/to/team-shared-skills
```

### Использование навыков

Навыки загружаются и внедряются в промпты LLM автоматически — ручная активация не требуется. Используйте CLI для просмотра:

```bash
researchclaw skills list               # Показать все загруженные навыки с источниками
researchclaw skills validate ./my-skill # Проверить формат SKILL.md
```

Навыки от сообщества: [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) (150+ научных навыков по множеству дисциплин).

---

## ⚙️ Справочник по конфигурации

<details>
<summary>Нажмите, чтобы развернуть полный конфиг</summary>

```yaml
# === Проект ===
project:
  name: "my-research"              # Идентификатор проекта
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Исследование ===
research:
  topic: "..."                     # Тема исследования (обязательно)
  domains: ["ml", "nlp"]           # Домены для поиска литературы
  daily_paper_count: 8             # Целевое количество статей на один запрос
  quality_threshold: 4.0           # Минимальный порог качества для статей

# === Рантайм ===
runtime:
  timezone: "Europe/Moscow"        # Для таймстемпов
  max_parallel_tasks: 3            # Лимит параллельных экспериментов
  approval_timeout_hours: 12       # Таймаут ожидания на гейтах
  retry_limit: 2                   # Количество ретраев при падении этапа

# === LLM ===
llm:
  provider: "openai-compatible"    # openai | openrouter | deepseek | minimax | acp | openai-compatible
  base_url: "https://..."          # API endpoint (обязательно для openai-compatible)
  api_key_env: "OPENAI_API_KEY"    # Переменная окружения с ключом (обязательно для openai-compatible)
  api_key: ""                      # Или можно захардкодить ключ здесь
  primary_model: "gpt-4o"          # Основная модель
  fallback_models: ["gpt-4o-mini"] # Цепочка фоллбэков
  s2_api_key: ""                   # API-ключ Semantic Scholar (опционально, дает лимиты выше)
  acp:                             # Используется только если provider: "acp"
    agent: "claude"                # Команда CLI ACP-агента (claude, codex, gemini и т.д.)
    cwd: "."                       # Рабочая директория агента

# === Эксперименты ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 300             # Макс. время на один запуск (по умолчанию: 300с)
  max_iterations: 10               # Макс. количество итераций оптимизации
  metric_key: "val_loss"           # Название главной метрики
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
    auto_install_deps: true        # Автоопределение импортов → requirements.txt
  ssh_remote:
    host: ""                       # Хостнейм GPU-сервера
    gpu_ids: []                    # Доступные ID видеокарт
    remote_workdir: "/tmp/researchclaw_experiments"
  opencode:                          # OpenCode Beast Mode (устанавливается через `researchclaw setup`)
    enabled: true                    # Главный рубильник (по умолчанию: true)
    auto: true                       # Автозапуск без подтверждения (по умолчанию: true)
    complexity_threshold: 0.2        # 0.0-1.0 — чем выше, тем реже триггерится (только на сложных задачах)
    model: ""                        # Переопределение модели (пусто = использовать llm.primary_model)
    timeout_sec: 600                 # Макс. время на генерацию в OpenCode
    max_retries: 1                   # Количество ретраев при падении
    workspace_cleanup: true          # Удалять временный воркспейс после сбора результатов
  code_agent:                        # CodeAgent v2 — многофазная генерация кода
    enabled: true                    # Использовать CodeAgent вместо устаревшей однопромптной генерации
    architecture_planning: true      # Генерировать подробный план реализации перед кодированием
    sequential_generation: true      # Генерировать файлы по одному согласно DAG зависимостей
    hard_validation: true            # AST-валидация (блокирует идентичные ablation, захардкоженные метрики)
    hard_validation_max_repairs: 2   # Макс. попыток исправления при провале валидации
    exec_fix_max_iterations: 3       # Попыток исправления при выполнении
    exec_fix_timeout_sec: 60         # Таймаут на одну попытку исправления
  benchmark_agent:                   # BenchmarkAgent — автоматический подбор датасетов и бейзлайнов
    enabled: true                    # Включить 4-агентный пайплайн (Surveyor→Selector→Acquirer→Validator)
    enable_hf_search: true           # Поиск по HuggingFace Datasets
    enable_web_search: true          # Поиск бенчмарков в Google Scholar
    tier_limit: 2                    # Фильтрация датасетов по уровню (1=малые/кэшированные, 2=средние, 3=большие)
    min_benchmarks: 1                # Минимум необходимых датасетов
    min_baselines: 2                 # Минимум бейзлайнов
  figure_agent:                      # FigureAgent — генерация академических графиков
    enabled: true                    # Включить 5-агентный пайплайн (Planner→CodeGen→Renderer→Critic→Integrator)
    min_figures: 3                   # Минимум генерируемых графиков
    max_figures: 8                   # Максимум графиков
    max_iterations: 3                # Итераций улучшения через Critic
    dpi: 300                         # Разрешение вывода
    strict_mode: false               # Провал пайплайна при ошибке генерации графиков
  repair:                            # Антифабрикация — ремонт экспериментов
    enabled: true                    # Автодиагностика и ремонт упавших экспериментов
    max_cycles: 3                    # Количество циклов ремонта
    min_completion_rate: 0.5         # >=50% условий должны завершиться для продолжения
    min_conditions: 2                # Минимум 2 условия для валидного эксперимента
    use_opencode: true               # Направлять ремонт через OpenCode Beast Mode

# === Веб-поиск (Опционально) ===
web_search:
  enabled: true                      # Включить веб-расширенный поиск литературы
  tavily_api_key_env: "TAVILY_API_KEY"  # Переменная окружения для Tavily API-ключа (опционально)
  enable_scholar: true               # Поиск в Google Scholar
  enable_pdf_extraction: true        # Извлечение текста из PDF
  max_web_results: 10                # Макс. веб-результатов на запрос

# === Экспорт ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Промпты ===
prompts:
  custom_file: ""                  # Путь к кастомному YAML с промптами (пусто = дефолтные)

# === HITL Co-Pilot (НОВОЕ в v0.4.0) ===
hitl:
  enabled: false                     # Установите true для включения HITL
  mode: co-pilot                     # full-auto | gate-only | checkpoint | step-by-step | co-pilot | custom
  cost_budget_usd: 0.0              # Лимит затрат в USD (0 = без лимита)
  notifications:
    on_pause: true                   # Уведомлять при паузе пайплайна
    on_quality_drop: true            # Уведомлять при проблемах с качеством
    channels: ["terminal"]           # terminal | slack | webhook
  timeouts:
    default_human_timeout_sec: 86400 # Ожидание до 24ч
    auto_proceed_on_timeout: false   # Если true, авто-одобрение по таймауту
  collaboration:
    max_chat_turns: 50               # Макс. реплик за сессию взаимодействия
    save_chat_history: true          # Сохранять логи чата
  stage_policies: {}                 # Переопределения для этапов (для режима 'custom')

# === Безопасность ===
security:
  hitl_required_stages: [5, 9, 20] # Этапы, требующие апрува человеком (Human-in-the-loop)
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === База знаний ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === Уведомления ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === Мост MetaClaw (Опционально) ===
metaclaw_bridge:
  enabled: false                   # Включить кросс-сессионное обучение
  proxy_url: "http://localhost:30000"  # URL прокси MetaClaw
  skills_dir: "~/.metaclaw/skills" # Папка для хранения навыков arc-*
  fallback_url: ""                 # Прямой фоллбэк к LLM, если прокси лежит
  fallback_api_key: ""             # API-ключ для фоллбэка
  lesson_to_skill:
    enabled: true                  # Автоматически конвертировать уроки в навыки
    min_severity: "warning"        # Минимальная серьезность для конвертации
    max_skills_per_run: 3          # Макс. количество новых навыков за один запуск
  prm:                             # Process Reward Model — гейт качества (опционально)
    enabled: false                 # Использовать LLM-as-judge для оценки результатов этапов
    model: "gpt-5.4"              # Модель-судья PRM
    votes: 3                       # Количество голосов (мажоритарное голосование)
    gate_stages: [5, 9, 15, 20]   # Этапы для применения PRM-гейтов

# === Мост OpenClaw ===
openclaw_bridge:
  use_cron: false                  # Запуск исследований по расписанию
  use_message: false               # Уведомления о прогрессе
  use_memory: false                # Сохранение знаний между сессиями
  use_sessions_spawn: false        # Запуск параллельных подсессий
  use_web_fetch: false             # Поиск в интернете в реальном времени
  use_browser: false               # Сбор статей через браузер
```

</details>

---

## 🙏 Благодарности

Вдохновлено проектами:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Пионер автоматизированных исследований
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — Сквозная автоматизация исследований
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Полностью автоматизированная исследовательская система

---

## ⚖️ Этика и ответственное использование

AutoResearchClaw — это инструмент для помощи в исследованиях, а не замена человеку-исследователю. Мы просим всех пользователей соблюдать следующие принципы:

**Академическая честность.** Статьи, сгенерированные AutoResearchClaw, следует рассматривать как черновики, требующие тщательной проверки, верификации и доработки человеком перед подачей. Авторы, указанные в статье, несут полную ответственность за её содержание, утверждения и корректность. Использование текста, сгенерированного ИИ, без надлежащего человеческого контроля или раскрытия информации может нарушать политику академической честности вашего учреждения или целевой конференции.

**Прозрачность и раскрытие информации.** Мы настоятельно рекомендуем раскрывать использование AutoResearchClaw (или любой помощи ИИ) в рукописях, следуя политике целевой конференции (например, NeurIPS, ICML, ICLR и другие крупные конференции теперь требуют раскрытия помощи ИИ в написании). Co-Pilot с участием человека существует именно для того, чтобы обеспечить значимый человеческий контроль над исследовательскими решениями.

**Цитирование и атрибуция.** AutoResearchClaw проверяет цитаты через 4-уровневый конвейер, но ни одна автоматизированная система не совершенна. Пользователи должны вручную проверить перед подачей, что все цитаты являются реальными, релевантными и правильно оформленными. Поддельные или неправильно атрибутированные цитаты подрывают научное доверие.

**Риск злоупотребления.** Как и любой мощный инструмент, AutoResearchClaw может быть использован не по назначению для массового производства низкокачественных или вводящих в заблуждение исследований. Мы не поддерживаем использование данной системы для создания фабрик статей, мошеннических подач или контента, направленного на манипулирование рецензированием. В случае обнаружения систематического злоупотребления мы оставляем за собой право обновить лицензию или условия использования.

**Двойное назначение.** Автономные исследовательские системы поднимают более широкие вопросы о будущем научного труда, нормах авторства и процессах рецензирования. Мы приветствуем дискуссии сообщества по этим темам и стремимся к ответственной разработке этой технологии.

Используя AutoResearchClaw, вы соглашаетесь использовать его в соответствии с этими принципами, а также этическими нормами вашего учреждения и исследовательского сообщества.

---

## 📄 Лицензия

MIT — подробности см. в [LICENSE](../LICENSE).

---

## 📌 Цитирование

Если AutoResearchClaw оказался вам полезен, пожалуйста, процитируйте:

```bibtex
@misc{liu2026autoresearchclaw,
  author       = {Liu, Jiaqi and Xia, Peng and Han, Siwei and Qiu, Shi and Zhang, Letian and Chen, Guiming and Tu, Haoqin and Yang, Xinyu and Zhou, Jiawei and Zhu, Hongtu and Li, Yun and Zhang, Jiaheng and Zhou, Yuyin and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  title        = {AutoResearchClaw: Fully Autonomous Research from Idea to Paper},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/aiming-lab/AutoResearchClaw},
}
```

<p align="center">
  <sub>Создано с 🦞 командой AutoResearchClaw</sub>
</p>