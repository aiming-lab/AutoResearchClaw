<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Comparte una idea. Obtén un artículo. Totalmente autónomo.</b></h2>



<p align="center">
  <i>Chatea con <a href="#openclaw-integration">OpenClaw</a>: "Investiga X" → hecho.</i>
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
  <a href="integration-guide.md">📖 Guía de integración</a>
</p>

---

## ⚡ En una línea

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```

---

## 🤔 ¿Qué es esto?

Tienes una idea. Quieres un artículo. **Eso es todo.**

AutoResearchClaw toma un tema de investigación y produce de forma autónoma un artículo académico completo — con literatura real de arXiv y Semantic Scholar (multi-fuente, arXiv primero para evitar limitaciones de tasa), experimentos en sandbox adaptados al hardware (detección automática de GPU/MPS/CPU), análisis estadístico, revisión por pares y LaTeX listo para conferencia (orientado a 5,000-6,500 palabras para NeurIPS/ICML/ICLR). Sin supervisión. Sin copiar y pegar entre herramientas.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Artículo académico completo (Introducción, Trabajo relacionado, Método, Experimentos, Resultados, Conclusión)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>LaTeX listo para conferencia (plantillas NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Referencias BibTeX reales de Semantic Scholar y arXiv — auto-depuradas para coincidir con las citas en línea</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>Verificación de integridad + relevancia de citas en 4 capas (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Código generado + resultados en sandbox + métricas JSON estructuradas</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Gráficos de comparación de condiciones auto-generados con barras de error e intervalos de confianza</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Revisión por pares multi-agente con verificación de consistencia metodología-evidencia</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Lecciones de auto-aprendizaje extraídas de cada ejecución</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>Todos los entregables finales en una sola carpeta — listos para compilar en Overleaf</td></tr>
</table>

El pipeline se ejecuta **de principio a fin sin intervención humana** (a menos que configures etapas con compuerta para revisión manual). Cuando los experimentos fallan, se auto-repara. Cuando las hipótesis no se sostienen, pivotea.

### 🎯 Pruébalo

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 ¿Qué lo hace diferente?

### 🔄 Bucle de decisión PIVOT / REFINE

El pipeline no se ejecuta de forma lineal. La etapa 15 (RESEARCH_DECISION) evalúa los resultados experimentales frente a las hipótesis y toma una decisión autónoma:

- **PROCEED** — los resultados respaldan las hipótesis, continuar con la redacción del artículo
- **REFINE** — los resultados son prometedores pero necesitan mejora, volver a refinar código/parámetros
- **PIVOT** — se detectó un problema fundamental, reiniciar desde la generación de hipótesis con una nueva dirección

Cada ciclo PIVOT/REFINE **versiona los artefactos anteriores** (`stage-08_v1/`, `stage-08_v2/`, ...) para que no se pierda trabajo y la evolución de decisiones sea completamente rastreable.

### 🤖 Debate multi-agente

Las etapas críticas utilizan un protocolo de debate estructurado con múltiples perspectivas LLM:

- **Generación de hipótesis** — agentes diversos proponen y cuestionan ideas
- **Análisis de resultados** — un optimista, un escéptico y un pragmático analizan los resultados
- **Revisión por pares** — verificación de consistencia metodología-evidencia (¿el artículo afirma 50 ensayos cuando el código ejecutó 5?)

### 🧬 Evolución: auto-aprendizaje entre ejecuciones

Cada ejecución del pipeline extrae lecciones detalladas — no solo "falló" sino *por qué*:

- Justificación de decisiones de las elecciones PIVOT/REFINE
- Advertencias en tiempo de ejecución del stderr de los experimentos (p. ej., `RuntimeWarning: division by zero`)
- Anomalías en métricas (NaN, Inf, velocidades de convergencia idénticas)

Estas lecciones persisten en un almacén JSONL con **ponderación por decaimiento temporal con vida media de 30 días** y se inyectan como capas de prompt en ejecuciones futuras. El pipeline literalmente aprende de sus errores.

### 📚 Base de conocimiento

Cada ejecución construye una base de conocimiento estructurada (almacenada en `docs/kb/`) con 6 categorías:

- **decisions/** — diseño experimental, compuertas de calidad, decisiones de investigación, planificación de recursos, estrategias de búsqueda, archivos de conocimiento
- **experiments/** — registros de generación de código, ejecuciones de experimentos, refinamientos iterativos
- **findings/** — verificación de citas, análisis de resultados, informes de síntesis
- **literature/** — extracción de conocimiento, recopilación de literatura, resultados de filtrado
- **questions/** — generación de hipótesis, descomposición de problemas, inicialización de temas
- **reviews/** — informes de exportación/publicación, borradores de artículos, esquemas, revisiones, revisiones por pares

### 🛡️ Vigilante Sentinel

Un monitor de calidad en segundo plano que detecta problemas que el pipeline principal podría pasar por alto:

- **Detección de errores en tiempo de ejecución** — NaN/Inf en métricas, advertencias de stderr retroalimentadas al LLM para reparación dirigida
- **Consistencia artículo-evidencia** — código experimental real, resultados de ejecución y registros de refinamiento inyectados en la revisión por pares
- **Puntuación de relevancia de citas** — más allá de la verificación de existencia, el LLM evalúa la relevancia temática de cada referencia
- **Cumplimiento de convergencia** — detecta experimentos de iteración fija y exige criterios de parada temprana adecuados
- **Validación de ablación** — detecta condiciones de ablación duplicadas/idénticas y señala comparaciones defectuosas
- **Guardia anti-fabricación** — bloquea completamente la redacción del artículo cuando los experimentos no producen métricas

---

## 🦞 Integración con OpenClaw

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw es un servicio compatible con [OpenClaw](https://github.com/openclaw/openclaw).** Instálalo en OpenClaw y lanza investigación autónoma con un solo mensaje — o úsalo de forma independiente vía CLI, Claude Code o cualquier asistente de programación con IA.

</td>
</tr>
</table>

### 🚀 Uso con OpenClaw (Recomendado)

Si ya usas [OpenClaw](https://github.com/openclaw/openclaw) como tu asistente de IA:

```
1️⃣  Comparte la URL del repositorio de GitHub con OpenClaw
2️⃣  OpenClaw lee automáticamente RESEARCHCLAW_AGENTS.md → comprende el pipeline
3️⃣  Di: "Investiga [tu tema]"
4️⃣  Listo — OpenClaw clona, instala, configura, ejecuta y devuelve los resultados
```

**Eso es todo.** OpenClaw se encarga de `git clone`, `pip install`, configuración y ejecución del pipeline automáticamente. Tú solo chateas.

<details>
<summary>💡 Qué sucede internamente</summary>

1. OpenClaw lee `RESEARCHCLAW_AGENTS.md` → aprende el rol de orquestador de investigación
2. OpenClaw lee `README.md` → comprende la instalación y la estructura del pipeline
3. OpenClaw copia `config.researchclaw.example.yaml` → `config.yaml`
4. Solicita tu clave API del LLM (o usa tu variable de entorno)
5. Ejecuta `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. Devuelve el artículo, LaTeX, experimentos y citas

</details>

### 🔌 Bridge de OpenClaw (Avanzado)

Para una integración más profunda, AutoResearchClaw incluye un **sistema de adaptadores bridge** con 6 capacidades opcionales:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Ejecuciones de investigación programadas
  use_message: true           # 💬 Notificaciones de progreso (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Persistencia de conocimiento entre sesiones
  use_sessions_spawn: true    # 🔀 Generar sub-sesiones paralelas para etapas concurrentes
  use_web_fetch: true         # 🌐 Búsqueda web en vivo durante la revisión de literatura
  use_browser: false          # 🖥️ Recopilación de artículos basada en navegador
```

Cada flag activa un protocolo de adaptador tipado. Cuando OpenClaw proporciona estas capacidades, los adaptadores las consumen sin cambios en el código. Consulta [`integration-guide.md`](integration-guide.md) para más detalles.

### 🛠️ Otras formas de ejecución

| Método | Cómo |
|--------|------|
| **CLI independiente** | `researchclaw run --topic "..." --auto-approve` |
| **API de Python** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Lee `RESEARCHCLAW_CLAUDE.md` — solo di *"Ejecuta investigación sobre [tema]"* |
| **OpenCode** | Lee `.claude/skills/` — la misma interfaz en lenguaje natural |
| **Cualquier CLI de IA** | Proporciona `RESEARCHCLAW_AGENTS.md` como contexto → el agente se auto-configura |

---

## 🔬 Pipeline: 23 etapas, 8 fases

```
Fase A: Alcance de investigación     Fase E: Ejecución de experimentos
  1. TOPIC_INIT                        12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE                 13. ITERATIVE_REFINE  ← auto-reparación

Fase B: Descubrimiento de literatura Fase F: Análisis y decisión
  3. SEARCH_STRATEGY                   14. RESULT_ANALYSIS    ← multi-agente
  4. LITERATURE_COLLECT  ← API real    15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [compuerta]
  6. KNOWLEDGE_EXTRACT                 Fase G: Redacción del artículo
                                       16. PAPER_OUTLINE
Fase C: Síntesis de conocimiento       17. PAPER_DRAFT
  7. SYNTHESIS                         18. PEER_REVIEW        ← verif. evidencia
  8. HYPOTHESIS_GEN    ← debate        19. PAPER_REVISION

Fase D: Diseño experimental          Fase H: Finalización
  9. EXPERIMENT_DESIGN   [compuerta]   20. QUALITY_GATE      [compuerta]
 10. CODE_GENERATION                   21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING                 22. EXPORT_PUBLISH     ← LaTeX
                                       23. CITATION_VERIFY    ← verif. relevancia
```

> Las **etapas con compuerta** (5, 9, 20) se pausan para aprobación humana o se auto-aprueban con `--auto-approve`. Al rechazar, el pipeline retrocede.

> **Bucles de decisión**: La etapa 15 puede activar REFINE (→ Etapa 13) o PIVOT (→ Etapa 8), con versionado automático de artefactos.

<details>
<summary>📋 Qué hace cada fase</summary>

| Fase | Qué sucede |
|------|-----------|
| **A: Alcance** | El LLM descompone el tema en un árbol de problemas estructurado con preguntas de investigación |
| **A+: Hardware** | Detección automática de GPU (NVIDIA CUDA / Apple MPS / solo CPU), advierte si el hardware local es limitado, adapta la generación de código en consecuencia |
| **B: Literatura** | Búsqueda multi-fuente (arXiv primero, luego Semantic Scholar) de artículos reales, filtrado por relevancia, extracción de fichas de conocimiento |
| **C: Síntesis** | Agrupa hallazgos, identifica brechas de investigación, genera hipótesis comprobables mediante debate multi-agente |
| **D: Diseño** | Diseña plan experimental, genera Python ejecutable adaptado al hardware (nivel de GPU → selección de paquetes), estima necesidades de recursos |
| **E: Ejecución** | Ejecuta experimentos en sandbox, detecta NaN/Inf y errores en tiempo de ejecución, auto-repara código mediante reparación LLM dirigida |
| **F: Análisis** | Análisis multi-agente de resultados; decisión autónoma PROCEED / REFINE / PIVOT con justificación |
| **G: Redacción** | Esquema → redacción sección por sección (5,000-6,500 palabras) → revisión por pares (con consistencia metodología-evidencia) → revisión con guardia de longitud |
| **H: Finalización** | Compuerta de calidad, archivado de conocimiento, exportación LaTeX con plantilla de conferencia, verificación de integridad + relevancia de citas |

</details>

---

## 🚀 Inicio rápido

### Requisitos previos

- 🐍 Python 3.11+
- 🔑 Un endpoint de API LLM compatible con OpenAI (GPT-4o, GPT-5.x o cualquier proveedor compatible)

### Instalación

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuración

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 Configuración mínima requerida</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"     # Cualquier endpoint compatible con OpenAI
  api_key_env: "OPENAI_API_KEY"              # Nombre de la variable de entorno con tu clave
  primary_model: "gpt-4o"                    # Cualquier modelo soportado por tu endpoint
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                             # Opcional: clave API de Semantic Scholar para mayores límites de tasa

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

### Ejecución

```bash
# Configura tu clave API
export OPENAI_API_KEY="sk-..."

# 🚀 Ejecuta el pipeline completo
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 Especifica un tema en línea
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ Valida la configuración
researchclaw validate --config config.arc.yaml

# ⏩ Reanuda desde una etapa específica
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

Salida → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` con un subdirectorio por etapa.

Todos los entregables para el usuario se recopilan automáticamente en una única carpeta **`deliverables/`**:

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # Artículo final (Markdown)
├── paper.tex                  # LaTeX listo para conferencia
├── references.bib             # Bibliografía BibTeX verificada (auto-depurada)
├── neurips_2025.sty           # Archivo de estilo de conferencia (auto-seleccionado)
├── code/                      # Código experimental + requirements.txt
├── verification_report.json   # Informe de integridad de citas
├── charts/                    # Visualizaciones de resultados (comparación de condiciones, barras de error)
└── manifest.json              # Índice de entregables con metadatos
```

La carpeta `deliverables/` está **lista para compilar** — incluye los archivos `.sty` y `.bst` de la conferencia para que puedas compilar `paper.tex` directamente con `pdflatex` + `bibtex` o subirlo a Overleaf sin descargar nada adicional.

---

## ✨ Características principales

### 📚 Búsqueda de literatura multi-fuente

La etapa 4 consulta **APIs académicas reales** — no artículos alucinados por el LLM. Usa una estrategia **arXiv primero** para evitar limitaciones de tasa de Semantic Scholar.

- **arXiv API** (primaria) — preprints con IDs reales de arXiv y metadatos, sin límites de tasa
- **Semantic Scholar API** (secundaria) — artículos reales con títulos, resúmenes, venues, conteos de citas, DOIs
- **Expansión de consultas** — genera automáticamente consultas más amplias (variantes de survey, benchmark, comparación) para cobertura completa (30-60 referencias)
- **Deduplicación automática** — DOI → arXiv ID → coincidencia difusa de títulos
- **Generación de BibTeX** — entradas `@article{cite_key, ...}` válidas con metadatos reales
- **Circuit breaker de tres estados** — CLOSED → OPEN → HALF_OPEN con backoff exponencial de recuperación (nunca se desactiva permanentemente)
- **Degradación gradual** — un fallo de S2 no bloquea los resultados de arXiv; recurre a resultados aumentados por LLM si todas las APIs fallan

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 Verificación de citas (Etapa 23)

Después de redactar el artículo, la etapa 23 **verifica cada referencia** tanto en integridad como en relevancia:

| Capa | Método | Qué verifica |
|------|--------|--------------|
| L1 | arXiv API `id_list` | Artículos con IDs de arXiv — verifica que el ID realmente existe |
| L2 | CrossRef `/works/{doi}` + fallback DataCite | Artículos con DOIs — verifica que el DOI resuelve y el título coincide (DataCite maneja DOIs `10.48550` de arXiv) |
| L3 | Semantic Scholar + búsqueda por título en arXiv | Resto — coincidencia difusa de títulos (≥0.80 de similitud) |
| L4 | Puntuación de relevancia por LLM | Todas las refs verificadas — evalúa relevancia temática para la investigación |

Cada referencia → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**Limpieza automática**: Las citas alucinadas se eliminan silenciosamente del texto del artículo (sin etiquetas `[HALLUCINATED]`). Las entradas bibliográficas no citadas se podan. El `references.bib` final contiene solo referencias verificadas y citadas.

### 🖥️ Ejecución adaptada al hardware

La etapa 1 detecta automáticamente las capacidades de GPU locales y adapta todo el pipeline:

| Nivel | Detección | Comportamiento |
|-------|-----------|----------------|
| **Alto** | GPU NVIDIA con ≥8 GB VRAM | Generación de código PyTorch/GPU completa, auto-instala torch si falta |
| **Limitado** | NVIDIA <8 GB o Apple MPS | Experimentos ligeros (<1M parámetros, ≤20 epochs), advertencia al usuario |
| **Solo CPU** | No se detecta GPU | Solo NumPy/sklearn, sin imports de torch, advertencia con recomendación de GPU remota |

El perfil de hardware se guarda en `stage-01/hardware_profile.json` e influye en la generación de código, los imports del sandbox y las restricciones de los prompts.

### 🧪 Ejecución de experimentos en sandbox

- **Validación de código** — análisis AST, lista blanca de imports, sin E/S de archivos fuera del sandbox
- **Guardia de presupuesto de cómputo** — presupuesto de tiempo (configurable, 600s por defecto) inyectado en el prompt de generación de código; el LLM debe diseñar experimentos que se ajusten al timeout del sandbox
- **Harness de experimentos** — `experiment_harness.py` inmutable inyectado en el sandbox con guardia de tiempo `should_stop()`, rechazo de NaN/Inf en `report_metric()` y escritura de resultados `finalize()` (inspirado en el patrón de eval inmutable de karpathy/autoresearch)
- **Salida estructurada** — los experimentos producen `results.json` con métricas tipadas (no solo análisis de stdout)
- **Análisis inteligente de métricas** — filtra líneas de log de las métricas usando detección de palabras clave (`is_metric_name()`)
- **Fallo rápido por NaN/divergencia** — valores NaN/Inf filtrados de las métricas; pérdida divergente (>100) detectada y señalada
- **Cumplimiento de convergencia** — el código generado debe incluir criterios de parada temprana, no conteos de iteración fijos
- **Detección de errores en tiempo de ejecución** — métricas NaN/Inf y advertencias de stderr (división por cero, desbordamiento) detectadas automáticamente
- **Reparación auto-curativa** — los problemas en tiempo de ejecución se retroalimentan al LLM con diagnóstico dirigido para correcciones de causa raíz (no parches try/except)
- **Refinamiento iterativo** — la etapa 13 analiza resultados y re-ejecuta con código/parámetros mejorados (hasta 10 iteraciones, con prompts conscientes del timeout)
- **Captura de resultados parciales** — los experimentos que agotan el tiempo con métricas capturadas obtienen estado "partial" en lugar de "failed", preservando datos utilizables
- **Alineación tema-experimento** — verificación post-generación basada en LLM que asegura que el código experimental realmente prueba el tema de investigación planteado

### 📝 Redacción de artículos con calidad de conferencia

El pipeline de redacción apunta a estándares NeurIPS/ICML/ICLR (9+ páginas, 5,000-6,500 palabras):

- **Cumplimiento de integridad de datos** — la redacción del artículo se bloquea completamente cuando los experimentos no producen métricas (previene que el LLM fabrique resultados); instrucciones anti-fabricación inyectadas tanto en borradores como en prompts de revisión
- **Prompts de calidad de conferencia** — los prompts del sistema incluyen principios clave del análisis de artículos aceptados: novedad, narrativa, baselines sólidos, ablaciones, honestidad, reproducibilidad; se señalan razones comunes de rechazo
- **Directrices de título y enfoque** — señalización de novedad, test de memorabilidad, estructura de abstract de 5 oraciones, detección de títulos genéricos con re-generación
- **Redacción sección por sección** — 3 llamadas secuenciales al LLM (Intro+Trabajo relacionado → Método+Experimentos → Resultados+Conclusión) para evitar truncamiento de salida
- **Objetivos de conteo de palabras por sección** — Abstract (150-250), Introducción (800-1000), Trabajo relacionado (600-800), Método (1000-1500), Experimentos (800-1200), Resultados (600-800), Discusión (400-600)
- **Guardia de longitud en revisión** — si el artículo revisado es más corto que el borrador, reintenta automáticamente con mayor enforcement; recurre a borrador+anotaciones si es necesario
- **Enforcement anti-disclaimer** — limita "due to computational constraints" a máximo 1 ocurrencia; los prompts de revisión eliminan activamente el hedging repetitivo
- **Rigor estadístico** — intervalos de confianza, valores p y tamaños del efecto requeridos en tablas de resultados; ablaciones defectuosas señaladas y excluidas de las afirmaciones
- **Revisión por pares con rúbrica de conferencia** — los revisores puntúan 1-10 siguiendo la rúbrica NeurIPS/ICML (novedad, baselines, ablaciones, afirmaciones vs evidencia, limitaciones)

### 📐 Cambio de plantilla de conferencia

```yaml
export:
  target_conference: "neurips_2025"   # o "iclr_2026" o "icml_2026"
```

| Conferencia | Paquete de estilo | Columnas |
|-------------|------------------|----------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

El convertidor Markdown → LaTeX maneja: encabezados de sección (con deduplicación de auto-numeración), matemáticas inline/display, negrita/cursiva, listas, tablas (con `\caption`/`\label`), figuras (`\includegraphics`), bloques de código (seguros para Unicode), referencias cruzadas y referencias `\cite{}`.

### 🚦 Compuertas de calidad

| Compuerta | Etapa | Al rechazar → Retrocede a |
|-----------|-------|--------------------------|
| Filtrado de literatura | 5 | Re-recopilar literatura (Etapa 4) |
| Diseño experimental | 9 | Re-generar hipótesis (Etapa 8) |
| Compuerta de calidad | 20 | Re-escribir artículo desde el esquema (Etapa 16) |

Usa `--auto-approve` para omitir todas las compuertas, o configura etapas específicas en `security.hitl_required_stages`.

---

## ⚙️ Referencia de configuración

<details>
<summary>Haz clic para expandir la referencia completa de configuración</summary>

```yaml
# === Proyecto ===
project:
  name: "my-research"              # Identificador del proyecto
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Investigación ===
research:
  topic: "..."                     # Tema de investigación (requerido)
  domains: ["ml", "nlp"]           # Dominios de investigación para búsqueda de literatura
  daily_paper_count: 8             # Artículos objetivo por consulta de búsqueda
  quality_threshold: 4.0           # Puntuación mínima de calidad para artículos

# === Tiempo de ejecución ===
runtime:
  timezone: "America/New_York"     # Para marcas de tiempo
  max_parallel_tasks: 3            # Límite de experimentos concurrentes
  approval_timeout_hours: 12       # Timeout de etapas con compuerta
  retry_limit: 2                   # Número de reintentos por fallo de etapa

# === LLM ===
llm:
  provider: "openai-compatible"    # Tipo de proveedor
  base_url: "https://..."          # Endpoint de API (requerido)
  api_key_env: "OPENAI_API_KEY"    # Variable de entorno para la clave API (requerido)
  api_key: ""                      # O codifica la clave aquí directamente
  primary_model: "gpt-4o"          # Modelo principal
  fallback_models: ["gpt-4o-mini"] # Cadena de fallback
  s2_api_key: ""                   # Clave API de Semantic Scholar (opcional, mayores límites de tasa)

# === Experimento ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # Tiempo máximo de ejecución por corrida (por defecto: 600s)
  max_iterations: 10               # Máximo de iteraciones de optimización
  metric_key: "val_loss"           # Nombre de la métrica principal
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
    auto_install_deps: true        # Detección automática de imports → requirements.txt
  ssh_remote:
    host: ""                       # Nombre de host del servidor GPU
    gpu_ids: []                    # IDs de GPU disponibles
    remote_workdir: "/tmp/researchclaw_experiments"

# === Exportación ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # Ruta a YAML de prompts personalizados (vacío = valores por defecto)

# === Seguridad ===
security:
  hitl_required_stages: [5, 9, 20] # Etapas que requieren aprobación humana
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

# === Bridge de OpenClaw ===
openclaw_bridge:
  use_cron: false                  # Ejecuciones de investigación programadas
  use_message: false               # Notificaciones de progreso
  use_memory: false                # Persistencia de conocimiento entre sesiones
  use_sessions_spawn: false        # Generar sub-sesiones paralelas
  use_web_fetch: false             # Búsqueda web en vivo
  use_browser: false               # Recopilación de artículos basada en navegador
```

</details>

---

## 🙏 Agradecimientos

Inspirado por:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Pionero en investigación automatizada
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — Automatización de investigación de principio a fin
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Sistema de investigación completamente automatizado

---

## 📄 Licencia

MIT — consulta [LICENSE](../LICENSE) para más detalles.

<p align="center">
  <sub>Construido con 🦞 por el equipo de AutoResearchClaw</sub>
</p>
