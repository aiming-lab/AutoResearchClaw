<p align="center">
  <img src="../image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Comparte una idea. Obten un articulo. Totalmente autonomo & autoevolutivo.</b></h2>



<p align="center">
  <b><i><font size="5">Chatea con <a href="#-integracion-con-openclaw">OpenClaw</a>: "Investiga X" → hecho.</font></i></b>
</p>

<p align="center">
  <img src="../image/framework_v2.png" width="100%" alt="AutoResearchClaw Framework">
</p>


<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/Tests-1823%20passed-brightgreen?logo=pytest&logoColor=white" alt="1823 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#-integracion-con-openclaw"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="showcase/SHOWCASE.md">🏆 Galeria de articulos</a> · <a href="integration-guide.md">📖 Guia de integracion</a> · <a href="https://discord.gg/u4ksqW5P">💬 Comunidad Discord</a>
</p>

---

<table>
<tr>
<td width="18%">
<a href="showcase/SHOWCASE.md"><img src="showcase/thumbnails/paper_I_random_matrix-01.png" width="120" alt="Sample Paper"/></a>
</td>
<td valign="middle">
<b>🏆 Galeria de articulos generados</b><br><br>
<b>8 articulos en 8 dominios</b> — matematicas, estadistica, biologia, computacion, NLP, RL, vision, robustez — generados de forma completamente autonoma sin intervencion humana.<br><br>
<a href="showcase/SHOWCASE.md"><img src="https://img.shields.io/badge/View_Full_Showcase_→-All_8_Papers-d73a49?style=for-the-badge" alt="View Showcase"></a>
</td>
</tr>
</table>

---

> **🧪 Buscamos testers!** Prueba el pipeline con tu propia idea de investigacion — de cualquier campo — y [cuentanos que piensas](TESTER_GUIDE.md). Tu feedback da forma directamente a la proxima version. **[→ Testing Guide](TESTER_GUIDE.md)** | **[→ 中文测试指南](TESTER_GUIDE_CN.md)** | **[→ 日本語テストガイド](TESTER_GUIDE_JA.md)**

---

## 🔥 News
- **[03/22/2026]** [v0.3.2](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.2) — **Soporte multiplataforma + estabilidad mayor** — AutoResearchClaw ahora funciona con cualquier agente compatible con ACP (Claude Code, Codex CLI, Copilot CLI, Gemini CLI, Kimi CLI) y soporta plataformas de mensajeria (Discord, Telegram, Lark, WeChat) via el puente OpenClaw. Nuevo backend de generacion de codigo CLI-agent que delega las Stages 10 y 13 a agentes CLI externos con control de presupuesto y gestion de timeouts. Incluye sistema anti-fabricacion (VerifiedRegistry + bucle de diagnostico y reparacion), 100+ correcciones de bugs, refactorizacion modular del executor, auto-deteccion de `--resume`, endurecimiento de reintentos LLM y correcciones de la comunidad.
- **[03/18/2026]** [v0.3.1](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.1) — **OpenCode Beast Mode + Community Contributions** — New "Beast Mode" routes complex code generation to [OpenCode](https://github.com/anomalyco/opencode) with automatic complexity scoring and graceful fallback. Added Novita AI provider support, thread-safety hardening, improved LLM output parsing robustness, and 20+ bug fixes from community PRs and internal audit.
- **[03/17/2026]** [v0.3.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.0) — **MetaClaw Integration** — AutoResearchClaw now supports [MetaClaw](https://github.com/aiming-lab/MetaClaw) cross-run learning: pipeline failures → structured lessons → reusable skills, injected into all 23 stages. **+18.3%** robustness in controlled experiments. Opt-in (`metaclaw_bridge.enabled: true`), fully backward-compatible. See [Integration Guide](#-integracion-metaclaw).
- **[03/16/2026]** [v0.2.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.2.0) — Three multi-agent subsystems (CodeAgent, BenchmarkAgent, FigureAgent), hardened Docker sandbox with network-policy-aware execution, 4-round paper quality audit (AI-slop detection, 7-dim review scoring, NeurIPS checklist), and 15+ bug fixes from production runs.
- **[03/15/2026]** [v0.1.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.1.0) — We release AutoResearchClaw: a fully autonomous 23-stage research pipeline that turns a single research idea into a conference-ready paper. No human intervention required.

---

## ⚡ Un comando. Un articulo.

```bash
pip install -e . && researchclaw setup && researchclaw init && researchclaw run --topic "Your research idea here" --auto-approve
```


---

## 🤔 Que es esto?

**Tu lo piensas. AutoResearchClaw lo escribe.**

Proporciona un tema de investigacion — recibe un articulo academico completo con literatura real de OpenAlex, Semantic Scholar y arXiv, experimentos en sandbox adaptados al hardware (deteccion automatica GPU/MPS/CPU), analisis estadistico, revision multi-agentes, y LaTeX listo para conferencia orientado a NeurIPS/ICML/ICLR. Sin supervision. Sin copiar y pegar. Sin referencias alucinadas.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Articulo academico completo (Introduccion, Trabajo relacionado, Metodo, Experimentos, Resultados, Conclusion)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>LaTeX listo para conferencia (plantillas NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Referencias BibTeX reales de OpenAlex, Semantic Scholar y arXiv — auto-depuradas para coincidir con las citas en linea</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>Verificacion de integridad + relevancia de citas en 4 capas (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Codigo generado + resultados en sandbox + metricas JSON estructuradas</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Graficos de comparacion de condiciones auto-generados con barras de error e intervalos de confianza</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Revision por pares multi-agente con verificacion de consistencia metodologia-evidencia</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Lecciones de auto-aprendizaje extraidas de cada ejecucion</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>Todos los entregables finales en una sola carpeta — listos para compilar en Overleaf</td></tr>
</table>

El pipeline se ejecuta **de principio a fin sin intervencion humana**. Cuando los experimentos fallan, se auto-repara. Cuando las hipotesis no se sostienen, pivotea. Cuando las citas son falsas, las elimina.

🌍 **Ejecutalo en cualquier lugar.** AutoResearchClaw no esta atado a una sola plataforma. Usalo de forma independiente por CLI, conectalo a [OpenClaw](https://github.com/openclaw/openclaw), o integralo con cualquier agente compatible con ACP — 🤖 Claude Code, 💻 Codex CLI, 🐙 Copilot CLI, ♊ Gemini CLI, 🌙 Kimi CLI, y mas. Gracias al puente de mensajeria de OpenClaw, puedes iniciar una investigacion completa desde 💬 Discord, ✈️ Telegram, 🐦 Lark (飞书), 💚 WeChat, o cualquier plataforma que tu equipo ya utilice. Un tema de entrada, un paper de salida — sin importar donde lo escribas.

---

## 🚀 Inicio rapido

```bash
# 1. Clonar e instalar
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Setup (interactivo — instala OpenCode beast mode, verifica Docker/LaTeX)
researchclaw setup

# 3. Configurar
researchclaw init          # Interactivo: elegir proveedor LLM, crea config.arc.yaml
# O manualmente: cp config.researchclaw.example.yaml config.arc.yaml

# 4. Ejecutar
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

Salida → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — LaTeX listo para compilar, BibTeX, codigo experimental, graficos.

<details>
<summary>📝 Configuracion minima requerida</summary>

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

## 🧠 Que lo hace diferente

| Capacidad | Como funciona |
|-----------|--------------|
| **🔄 Bucle PIVOT / REFINE** | La etapa 15 decide de forma autonoma: PROCEED, REFINE (ajustar parametros) o PIVOT (nueva direccion). Artefactos auto-versionados. |
| **🤖 Debate multi-agente** | La generacion de hipotesis, el analisis de resultados y la revision por pares utilizan cada uno debate estructurado multi-perspectiva. |
| **🧬 Auto-aprendizaje** | Lecciones extraidas por ejecucion (justificacion de decisiones, advertencias de ejecucion, anomalias de metricas) con decaimiento temporal de 30 dias. Las ejecuciones futuras aprenden de errores pasados. |
| **📚 Base de conocimiento** | Cada ejecucion construye una KB estructurada en 6 categorias (decisiones, experimentos, hallazgos, literatura, preguntas, revisiones). |
| **🛡️ Vigilante Sentinel** | Monitor de calidad en segundo plano: deteccion NaN/Inf, consistencia articulo-evidencia, puntuacion de relevancia de citas, guardia anti-fabricacion. |

---

## 🦞 Integracion con OpenClaw

<table>
<tr>

**AutoResearchClaw es un servicio compatible con [OpenClaw](https://github.com/openclaw/openclaw).** Instalalo en OpenClaw y lanza investigacion autonoma con un solo mensaje — o usalo de forma independiente via CLI, Claude Code o cualquier asistente de programacion con IA.

</tr>
</table>

### 🚀 Uso con OpenClaw (Recomendado)

Si ya usas [OpenClaw](https://github.com/openclaw/openclaw) como tu asistente de IA:

```
1️⃣  Comparte la URL del repositorio de GitHub con OpenClaw
2️⃣  OpenClaw lee automaticamente RESEARCHCLAW_AGENTS.md → comprende el pipeline
3️⃣  Di: "Research [tu tema]"
4️⃣  Listo — OpenClaw clona, instala, configura, ejecuta y devuelve los resultados
```

**Eso es todo.** OpenClaw se encarga de `git clone`, `pip install`, configuracion y ejecucion del pipeline automaticamente. Tu solo chateas.

<details>
<summary>💡 Que sucede internamente</summary>

1. OpenClaw lee `RESEARCHCLAW_AGENTS.md` → aprende el rol de orquestador de investigacion
2. OpenClaw lee `README.md` → comprende la instalacion y la estructura del pipeline
3. OpenClaw copia `config.researchclaw.example.yaml` → `config.yaml`
4. Solicita tu clave API del LLM (o usa tu variable de entorno)
5. Ejecuta `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. Devuelve el articulo, LaTeX, experimentos y citas

</details>

### 🔌 Bridge de OpenClaw (Avanzado)

Para una integracion mas profunda, AutoResearchClaw incluye un **sistema de adaptadores bridge** con 6 capacidades opcionales:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Ejecuciones de investigacion programadas
  use_message: true           # 💬 Notificaciones de progreso (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Persistencia de conocimiento entre sesiones
  use_sessions_spawn: true    # 🔀 Generar sub-sesiones paralelas para etapas concurrentes
  use_web_fetch: true         # 🌐 Busqueda web en vivo durante la revision de literatura
  use_browser: false          # 🖥️ Recopilacion de articulos basada en navegador
```

Cada flag activa un protocolo de adaptador tipado. Cuando OpenClaw proporciona estas capacidades, los adaptadores las consumen sin cambios en el codigo. Consulta [`integration-guide.md`](integration-guide.md) para mas detalles.

### ACP (Agent Client Protocol)

AutoResearchClaw puede usar **cualquier agente de programacion compatible con ACP** como backend LLM — sin necesidad de claves API. El agente se comunica via [acpx](https://github.com/openclaw/acpx), manteniendo una sola sesion persistente a traves de las 23 etapas del pipeline.

| Agente | Comando | Notas |
|--------|---------|-------|
| Claude Code | `claude` | Anthropic |
| Codex CLI | `codex` | OpenAI |
| Copilot CLI | `gh` | GitHub |
| Gemini CLI | `gemini` | Google |
| OpenCode | `opencode` | SST |
| Kimi CLI | `kimi` | Moonshot |

```yaml
# config.yaml — ejemplo ACP
llm:
  provider: "acp"
  acp:
    agent: "claude"   # Cualquier comando CLI de agente compatible con ACP
    cwd: "."          # Directorio de trabajo para el agente
  # No se necesita base_url ni api_key — el agente gestiona su propia autenticacion.
```

```bash
# Solo ejecuta — el agente usa sus propias credenciales
researchclaw run --config config.yaml --topic "Your research idea" --auto-approve
```

### 🛠️ Otras formas de ejecucion

| Metodo | Como |
|--------|------|
| **CLI independiente** | `researchclaw setup` → `researchclaw init` → `researchclaw run --topic "..." --auto-approve` |
| **API de Python** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Lee `RESEARCHCLAW_CLAUDE.md` — solo di *"Run research on [tema]"* |
| **Copilot CLI** | `researchclaw run --topic "..."` con `llm.acp.agent: "gh"` |
| **OpenCode** | Lee `.claude/skills/` — la misma interfaz en lenguaje natural |
| **Cualquier CLI de IA** | Proporciona `RESEARCHCLAW_AGENTS.md` como contexto → el agente se auto-configura |

---

## 🔬 Pipeline: 23 etapas, 8 fases

```
Fase A: Alcance de investigacion     Fase E: Ejecucion de experimentos
  1. TOPIC_INIT                        12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE                 13. ITERATIVE_REFINE  ← auto-reparacion

Fase B: Descubrimiento de literatura Fase F: Analisis y decision
  3. SEARCH_STRATEGY                   14. RESULT_ANALYSIS    ← multi-agente
  4. LITERATURE_COLLECT  ← API real    15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [compuerta]
  6. KNOWLEDGE_EXTRACT                 Fase G: Redaccion del articulo
                                       16. PAPER_OUTLINE
Fase C: Sintesis de conocimiento       17. PAPER_DRAFT
  7. SYNTHESIS                         18. PEER_REVIEW        ← verif. evidencia
  8. HYPOTHESIS_GEN    ← debate        19. PAPER_REVISION

Fase D: Diseno experimental          Fase H: Finalizacion
  9. EXPERIMENT_DESIGN   [compuerta]   20. QUALITY_GATE      [compuerta]
 10. CODE_GENERATION                   21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING                 22. EXPORT_PUBLISH     ← LaTeX
                                       23. CITATION_VERIFY    ← verif. relevancia
```

> Las **etapas con compuerta** (5, 9, 20) se pausan para aprobacion humana o se auto-aprueban con `--auto-approve`. Al rechazar, el pipeline retrocede.

> **Bucles de decision**: La etapa 15 puede activar REFINE (→ Etapa 13) o PIVOT (→ Etapa 8), con versionado automatico de artefactos.

<details>
<summary>📋 Que hace cada fase</summary>

| Fase | Que sucede |
|------|-----------|
| **A: Alcance** | El LLM descompone el tema en un arbol de problemas estructurado con preguntas de investigacion |
| **A+: Hardware** | Deteccion automatica de GPU (NVIDIA CUDA / Apple MPS / solo CPU), advierte si el hardware local es limitado, adapta la generacion de codigo en consecuencia |
| **B: Literatura** | Busqueda multi-fuente (OpenAlex → Semantic Scholar → arXiv) de articulos reales, filtrado por relevancia, extraccion de fichas de conocimiento |
| **C: Sintesis** | Agrupa hallazgos, identifica brechas de investigacion, genera hipotesis comprobables mediante debate multi-agente |
| **D: Diseno** | Disena plan experimental, genera Python ejecutable adaptado al hardware (nivel de GPU → seleccion de paquetes), estima necesidades de recursos |
| **E: Ejecucion** | Ejecuta experimentos en sandbox, detecta NaN/Inf y errores en tiempo de ejecucion, auto-repara codigo mediante reparacion LLM dirigida |
| **F: Analisis** | Analisis multi-agente de resultados; decision autonoma PROCEED / REFINE / PIVOT con justificacion |
| **G: Redaccion** | Esquema → redaccion seccion por seccion (5,000-6,500 palabras) → revision por pares (con consistencia metodologia-evidencia) → revision con guardia de longitud |
| **H: Finalizacion** | Compuerta de calidad, archivado de conocimiento, exportacion LaTeX con plantilla de conferencia, verificacion de integridad + relevancia de citas |

</details>

---

## ✨ Caracteristicas principales

| Caracteristica | Descripcion |
|----------------|------------|
| **📚 Literatura multi-fuente** | Articulos reales de OpenAlex, Semantic Scholar y arXiv — expansion de consultas, deduplicacion, circuit breaker con degradacion gradual |
| **🔍 Verificacion de citas en 4 capas** | Verificacion de arXiv ID → DOI CrossRef/DataCite → coincidencia de titulo Semantic Scholar → puntuacion de relevancia LLM. Referencias alucinadas auto-eliminadas. |
| **🖥️ Ejecucion adaptada al hardware** | Deteccion automatica de GPU (NVIDIA CUDA / Apple MPS / solo CPU) y adaptacion de la generacion de codigo, imports y escala experimental |
| **🦾 OpenCode Beast Mode** | Los experimentos complejos se enrutan automaticamente a [OpenCode](https://github.com/anomalyco/opencode) — genera proyectos multi-archivo con arquitecturas personalizadas, bucles de entrenamiento y estudios de ablacion. Instalacion via `researchclaw setup`. |
| **🧪 Experimentos en sandbox** | Codigo validado por AST, harness inmutable, fallo rapido NaN/Inf, reparacion auto-curativa, refinamiento iterativo (hasta 10 rondas), captura de resultados parciales |
| **📝 Redaccion de calidad conferencia** | Plantillas NeurIPS/ICML/ICLR, redaccion seccion por seccion (5,000-6,500 palabras), guardia anti-fabricacion, guardia de longitud en revision, enforcement anti-disclaimer |
| **📐 Cambio de plantilla** | `neurips_2025`, `iclr_2026`, `icml_2026` — Markdown → LaTeX con formulas, tablas, figuras, referencias cruzadas, `\cite{}` |
| **🚦 Compuertas de calidad** | 3 compuertas con intervencion humana posible (etapas 5, 9, 20) con retroceso. Omitir con `--auto-approve`. |

---

## 🧠 Integracion MetaClaw

**AutoResearchClaw + [MetaClaw](https://github.com/aiming-lab/MetaClaw) = Un pipeline que aprende de cada ejecucion.**

MetaClaw agrega **transferencia de conocimiento entre ejecuciones** a AutoResearchClaw. Cuando esta habilitado, el pipeline captura automaticamente lecciones de fallos y advertencias, las convierte en habilidades reutilizables, e inyecta esas habilidades en las 23 etapas del pipeline en ejecuciones posteriores — para que los mismos errores nunca se repitan.

### Como funciona

```
Ejecucion N se ejecuta → fallos/advertencias capturados como Lecciones
                      ↓
          MetaClaw Leccion → conversion a Habilidad
                      ↓
          Archivos de habilidades arc-* almacenados en ~/.metaclaw/skills/
                      ↓
Ejecucion N+1 → build_overlay() inyecta habilidades en cada prompt LLM
                      ↓
          El LLM evita trampas conocidas → mayor calidad, menos reintentos
```

### Configuracion rapida

```bash
# 1. Instalar MetaClaw (si no esta instalado)
pip install metaclaw

# 2. Habilitar en tu configuracion
```

```yaml
# config.arc.yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"        # Proxy MetaClaw (opcional)
  skills_dir: "~/.metaclaw/skills"          # Donde se almacenan las habilidades
  fallback_url: "https://api.openai.com/v1" # Fallback directo al LLM
  fallback_api_key: ""                      # Clave API para la URL de fallback
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # Convertir advertencias + errores
    max_skills_per_run: 3
```

```bash
# 3. Ejecuta como siempre — MetaClaw funciona de forma transparente
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve
```

Despues de cada ejecucion, revisa `~/.metaclaw/skills/arc-*/SKILL.md` para ver las habilidades que tu pipeline ha aprendido.

### Resultados experimentales

En experimentos controlados A/B (mismo tema, mismo LLM, misma configuracion):

| Metrica | Linea base | Con MetaClaw | Mejora |
|---------|------------|--------------|--------|
| Tasa de reintento de etapas | 10.5% | 7.9% | **-24.8%** |
| Conteo de ciclos REFINE | 2.0 | 1.2 | **-40.0%** |
| Completacion de etapas del pipeline | 18/19 | 19/19 | **+5.3%** |
| Puntuacion de robustez global (compuesta) | 0.714 | 0.845 | **+18.3%** |

> La puntuacion de robustez compuesta es un promedio ponderado de la tasa de completacion de etapas (40%), reduccion de reintentos (30%) y eficiencia de ciclos REFINE (30%).

### Retrocompatibilidad

- **Por defecto: DESACTIVADO.** Si `metaclaw_bridge` esta ausente o `enabled: false`, el pipeline se comporta exactamente como antes.
- **Sin nuevas dependencias.** MetaClaw es opcional — el pipeline base funciona sin el.
- **Los 1,823 tests existentes pasan** con el codigo de integracion presente.

---

## ⚙️ Referencia de configuracion

<details>
<summary>Haz clic para expandir la referencia completa de configuracion</summary>

```yaml
# === Proyecto ===
project:
  name: "my-research"              # Identificador del proyecto
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Investigacion ===
research:
  topic: "..."                     # Tema de investigacion (requerido)
  domains: ["ml", "nlp"]           # Dominios de investigacion para busqueda de literatura
  daily_paper_count: 8             # Articulos objetivo por consulta de busqueda
  quality_threshold: 4.0           # Puntuacion minima de calidad para articulos

# === Tiempo de ejecucion ===
runtime:
  timezone: "America/New_York"     # Para marcas de tiempo
  max_parallel_tasks: 3            # Limite de experimentos concurrentes
  approval_timeout_hours: 12       # Timeout de etapas con compuerta
  retry_limit: 2                   # Numero de reintentos por fallo de etapa

# === LLM ===
llm:
  provider: "openai-compatible"    # openai | openrouter | deepseek | minimax | acp | openai-compatible
  base_url: "https://..."          # Endpoint de API (requerido para openai-compatible)
  api_key_env: "OPENAI_API_KEY"    # Variable de entorno para la clave API (requerido para openai-compatible)
  api_key: ""                      # O codifica la clave aqui directamente
  primary_model: "gpt-4o"          # Modelo principal
  fallback_models: ["gpt-4o-mini"] # Cadena de fallback
  s2_api_key: ""                   # Clave API de Semantic Scholar (opcional, mayores limites de tasa)
  acp:                             # Solo se usa cuando provider: "acp"
    agent: "claude"                # Comando CLI del agente ACP (claude, codex, gemini, etc.)
    cwd: "."                       # Directorio de trabajo para el agente

# === Experimento ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 300             # Tiempo maximo de ejecucion por corrida (por defecto: 300s)
  max_iterations: 10               # Maximo de iteraciones de optimizacion
  metric_key: "val_loss"           # Nombre de la metrica principal
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
    auto_install_deps: true        # Deteccion automatica de imports → requirements.txt
  ssh_remote:
    host: ""                       # Nombre de host del servidor GPU
    gpu_ids: []                    # IDs de GPU disponibles
    remote_workdir: "/tmp/researchclaw_experiments"
  opencode:                          # OpenCode Beast Mode (auto-instalado via `researchclaw setup`)
    enabled: true                    # Interruptor principal (por defecto: true)
    auto: true                       # Auto-activacion sin confirmacion (por defecto: true)
    complexity_threshold: 0.2        # 0.0-1.0 — mas alto = solo se activa para experimentos complejos
    model: ""                        # Modelo a forzar (vacio = usa llm.primary_model)
    timeout_sec: 600                 # Segundos maximos para generacion OpenCode
    max_retries: 1                   # Numero de reintentos por fallo
    workspace_cleanup: true          # Eliminar workspace temporal despues de recoleccion

# === Exportacion ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # Ruta a YAML de prompts personalizados (vacio = valores por defecto)

# === Seguridad ===
security:
  hitl_required_stages: [5, 9, 20] # Etapas que requieren aprobacion humana
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === Base de conocimiento ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === Notificaciones ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === Puente MetaClaw (Opcional) ===
metaclaw_bridge:
  enabled: false                   # Establecer en true para habilitar aprendizaje entre ejecuciones
  proxy_url: "http://localhost:30000"  # URL del proxy MetaClaw
  skills_dir: "~/.metaclaw/skills" # Donde se almacenan las habilidades arc-*
  fallback_url: ""                 # Fallback directo al LLM cuando el proxy esta caido
  fallback_api_key: ""             # Clave API para el endpoint de fallback
  lesson_to_skill:
    enabled: true                  # Convertir lecciones en habilidades automaticamente
    min_severity: "warning"        # Severidad minima para conversion
    max_skills_per_run: 3          # Max de nuevas habilidades por ejecucion del pipeline

# === Bridge de OpenClaw ===
openclaw_bridge:
  use_cron: false                  # Ejecuciones de investigacion programadas
  use_message: false               # Notificaciones de progreso
  use_memory: false                # Persistencia de conocimiento entre sesiones
  use_sessions_spawn: false        # Generar sub-sesiones paralelas
  use_web_fetch: false             # Busqueda web en vivo
  use_browser: false               # Recopilacion de articulos basada en navegador
```

</details>

---

## 🙏 Agradecimientos

Inspirado por:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Pionero en investigacion automatizada
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — Automatizacion de investigacion de principio a fin
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Sistema de investigacion completamente automatizado

---

## 📄 Licencia

MIT — consulta [LICENSE](../LICENSE) para mas detalles.

---

## 📌 Citacion

Si encuentras AutoResearchClaw util, por favor cita:

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
  <sub>Construido con 🦞 por el equipo de AutoResearchClaw</sub>
</p>
