<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Idee besprechen. Paper erhalten. Vollautomatisch.</b></h2>



<p align="center">
  <i>Einfach mit <a href="#openclaw-integration">OpenClaw</a> chatten: „Erforsche X" → erledigt.</i>
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
  <a href="integration-guide.md">📖 Integrationsanleitung</a>
</p>

---

## ⚡ Einzeiler

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```

---

## 🤔 Was ist das?

Du hast eine Idee. Du willst ein Paper. **Das war's.**

AutoResearchClaw nimmt ein Forschungsthema und erstellt autonom ein vollständiges wissenschaftliches Paper — mit echter Literatur von OpenAlex, Semantic Scholar und arXiv (Multi-Source), hardwarebewussten Sandbox-Experimenten (automatische GPU/MPS/CPU-Erkennung), statistischer Analyse, Peer-Review und konferenzfertigem LaTeX (Ziel: 5.000–6.500 Wörter für NeurIPS/ICML/ICLR). Kein Babysitting. Kein Hin-und-her-Kopieren zwischen Tools.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Vollständiges wissenschaftliches Paper (Einleitung, Verwandte Arbeiten, Methode, Experimente, Ergebnisse, Fazit)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>Konferenzfertiges LaTeX (NeurIPS / ICLR / ICML Templates)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Echte BibTeX-Referenzen von Semantic Scholar und arXiv — automatisch bereinigt, um Inline-Zitationen zu entsprechen</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4-Schicht-Zitationsintegritäts- und Relevanzprüfung (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Generierter Code + Sandbox-Ergebnisse + strukturierte JSON-Metriken</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Automatisch generierte Vergleichsdiagramme mit Fehlerbalken und Konfidenzintervallen</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Multi-Agenten-Peer-Review mit Methodik-Evidenz-Konsistenzprüfungen</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Selbstlernende Erkenntnisse aus jedem Durchlauf</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>Alle finalen Ergebnisse in einem Ordner — kompilierbereit für Overleaf</td></tr>
</table>

Die Pipeline läuft **vollständig ohne menschliches Eingreifen** (sofern keine Gate-Stufen für manuelle Überprüfung konfiguriert sind). Wenn Experimente fehlschlagen, repariert sie sich selbst. Wenn Hypothesen nicht bestätigt werden, schwenkt sie um.

### 🎯 Ausprobieren

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 Was macht es anders

### 🔄 PIVOT / REFINE Entscheidungsschleife

Die Pipeline läuft nicht einfach linear ab. Stufe 15 (RESEARCH_DECISION) bewertet Experimentergebnisse im Vergleich zu Hypothesen und trifft eine autonome Entscheidung:

- **PROCEED** — Ergebnisse stützen die Hypothesen, weiter zur Papiererstellung
- **REFINE** — Ergebnisse sind vielversprechend, aber verbesserungswürdig, Rücksprung zur Code-/Parameterverfeinerung
- **PIVOT** — Grundlegendes Problem erkannt, Neustart ab Hypothesengenerierung mit neuer Richtung

Jeder PIVOT/REFINE-Zyklus **versioniert vorherige Artefakte** (`stage-08_v1/`, `stage-08_v2/`, ...), sodass keine Arbeit verloren geht und die Entscheidungsentwicklung vollständig nachvollziehbar ist.

### 🤖 Multi-Agenten-Debatte

Kritische Stufen verwenden ein strukturiertes Debattenprotokoll mit mehreren LLM-Perspektiven:

- **Hypothesengenerierung** — verschiedene Agenten schlagen Ideen vor und hinterfragen sie
- **Ergebnisanalyse** — Optimist, Skeptiker und Pragmatiker analysieren Resultate
- **Peer-Review** — Methodik-Evidenz-Konsistenzprüfung (behauptet das Paper 50 Versuche, wenn der Code nur 5 ausgeführt hat?)

### 🧬 Evolution: Lernen über Durchläufe hinweg

Jeder Pipeline-Durchlauf extrahiert detaillierte Erkenntnisse — nicht nur „es ist fehlgeschlagen", sondern *warum*:

- Entscheidungsbegründungen aus PIVOT/REFINE-Entscheidungen
- Laufzeitwarnungen aus Experiment-stderr (z. B. `RuntimeWarning: division by zero`)
- Metrikanaomalien (NaN, Inf, identische Konvergenzgeschwindigkeiten)

Diese Erkenntnisse werden in einem JSONL-Speicher mit **30-Tage-Halbwertszeit-Zeitabklinggewichtung** persistiert und als Prompt-Overlays in zukünftige Durchläufe injiziert. Die Pipeline lernt buchstäblich aus ihren Fehlern.

### 📚 Wissensdatenbank

Jeder Durchlauf erstellt eine strukturierte Wissensdatenbank (gespeichert in `docs/kb/`) mit 6 Kategorien:

- **decisions/** — Experimentdesign, Qualitäts-Gates, Forschungsentscheidungen, Ressourcenplanung, Suchstrategien, Wissensarchive
- **experiments/** — Codegenerierungsprotokolle, Experimentdurchläufe, iterative Verfeinerungen
- **findings/** — Zitationsverifikation, Ergebnisanalyse, Syntheseberichte
- **literature/** — Wissensextraktion, Literatursammlung, Screening-Ergebnisse
- **questions/** — Hypothesengenerierung, Problemdekomposition, Themeninitialisierung
- **reviews/** — Export-/Publikationsberichte, Paperentwürfe, Gliederungen, Revisionen, Peer-Reviews

### 🛡️ Sentinel Watchdog

Ein Hintergrund-Qualitätsmonitor, der Probleme erkennt, die die Hauptpipeline möglicherweise übersieht:

- **Laufzeit-Fehlererkennung** — NaN/Inf in Metriken, stderr-Warnungen werden an das LLM zur gezielten Reparatur weitergeleitet
- **Paper-Evidenz-Konsistenz** — tatsächlicher Experimentcode, Laufergebnisse und Verfeinerungsprotokolle werden in das Peer-Review eingespeist
- **Zitationsrelevanz-Bewertung** — über die reine Existenzprüfung hinaus bewertet das LLM die thematische Relevanz jeder Referenz
- **Konvergenzdurchsetzung** — erkennt Experimente mit fester Iterationszahl und fordert ordnungsgemäßes Early Stopping
- **Ablationsvalidierung** — erkennt doppelte/identische Ablationsbedingungen und markiert fehlerhafte Vergleiche
- **Anti-Fabrikationsschutz** — blockiert die Papiererstellung, wenn Experimente keine Metriken liefern

---

## 🦞 OpenClaw-Integration

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw ist ein [OpenClaw](https://github.com/openclaw/openclaw)-kompatibler Dienst.** Installiere es in OpenClaw und starte autonome Forschung mit einer einzigen Nachricht — oder verwende es eigenständig über CLI, Claude Code oder jeden anderen KI-Coding-Assistenten.

</td>
</tr>
</table>

### 🚀 Verwendung mit OpenClaw (empfohlen)

Wenn du bereits [OpenClaw](https://github.com/openclaw/openclaw) als KI-Assistenten nutzt:

```
1️⃣  Teile die GitHub-Repo-URL mit OpenClaw
2️⃣  OpenClaw liest automatisch RESEARCHCLAW_AGENTS.md → versteht die Pipeline
3️⃣  Sage: "Research [dein Thema]"
4️⃣  Fertig — OpenClaw klont, installiert, konfiguriert, führt aus und liefert Ergebnisse
```

**Das war's.** OpenClaw übernimmt `git clone`, `pip install`, Konfiguration und Pipeline-Ausführung automatisch. Du chattest einfach.

<details>
<summary>💡 Was unter der Haube passiert</summary>

1. OpenClaw liest `RESEARCHCLAW_AGENTS.md` → lernt die Forschungs-Orchestrator-Rolle
2. OpenClaw liest `README.md` → versteht Installation und Pipeline-Struktur
3. OpenClaw kopiert `config.researchclaw.example.yaml` → `config.yaml`
4. Fragt nach deinem LLM-API-Schlüssel (oder verwendet deine Umgebungsvariable)
5. Führt `pip install -e .` + `researchclaw run --topic "..." --auto-approve` aus
6. Liefert Paper, LaTeX, Experimente und Zitationen zurück

</details>

### 🔌 OpenClaw Bridge (Fortgeschritten)

Für tiefere Integration enthält AutoResearchClaw ein **Bridge-Adapter-System** mit 6 optionalen Fähigkeiten:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Geplante Forschungsdurchläufe
  use_message: true           # 💬 Fortschrittsbenachrichtigungen (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Sitzungsübergreifende Wissenspersistenz
  use_sessions_spawn: true    # 🔀 Parallele Sub-Sessions für gleichzeitige Stufen
  use_web_fetch: true         # 🌐 Live-Websuche während der Literaturrecherche
  use_browser: false          # 🖥️ Browserbasierte Paper-Sammlung
```

Jedes Flag aktiviert ein typisiertes Adapter-Protokoll. Wenn OpenClaw diese Fähigkeiten bereitstellt, nutzen die Adapter sie ohne Codeänderungen. Siehe [`integration-guide.md`](integration-guide.md) für vollständige Details.

### 🛠️ Weitere Ausführungsmöglichkeiten

| Methode | Anleitung |
|---------|-----------|
| **Standalone CLI** | `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Liest `RESEARCHCLAW_CLAUDE.md` — sage einfach *„Forsche zu [Thema]"* |
| **OpenCode** | Liest `.claude/skills/` — gleiche natürliche Sprachschnittstelle |
| **Jeder KI-CLI** | Übergib `RESEARCHCLAW_AGENTS.md` als Kontext → Agent bootstrappt automatisch |

---

## 🔬 Pipeline: 23 Stufen, 8 Phasen

```
Phase A: Forschungsplanung            Phase E: Experimentausführung
  1. TOPIC_INIT                          12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE                   13. ITERATIVE_REFINE  ← Selbstheilung

Phase B: Literaturrecherche            Phase F: Analyse & Entscheidung
  3. SEARCH_STRATEGY                     14. RESULT_ANALYSIS    ← Multi-Agent
  4. LITERATURE_COLLECT  ← echte API     15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [Gate]
  6. KNOWLEDGE_EXTRACT                   Phase G: Papiererstellung
                                         16. PAPER_OUTLINE
Phase C: Wissenssynthese                 17. PAPER_DRAFT
  7. SYNTHESIS                           18. PEER_REVIEW        ← Evidenzprüfung
  8. HYPOTHESIS_GEN    ← Debatte         19. PAPER_REVISION

Phase D: Experimentdesign             Phase H: Finalisierung
  9. EXPERIMENT_DESIGN   [Gate]          20. QUALITY_GATE      [Gate]
 10. CODE_GENERATION                     21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING                   22. EXPORT_PUBLISH     ← LaTeX
                                         23. CITATION_VERIFY    ← Relevanzprüfung
```

> **Gate-Stufen** (5, 9, 20) pausieren für menschliche Genehmigung oder werden mit `--auto-approve` automatisch genehmigt. Bei Ablehnung wird die Pipeline zurückgesetzt.

> **Entscheidungsschleifen**: Stufe 15 kann REFINE (→ Stufe 13) oder PIVOT (→ Stufe 8) auslösen, mit automatischer Artefakt-Versionierung.

<details>
<summary>📋 Was jede Phase bewirkt</summary>

| Phase | Beschreibung |
|-------|-------------|
| **A: Planung** | LLM zerlegt das Thema in einen strukturierten Problembaum mit Forschungsfragen |
| **A+: Hardware** | Automatische GPU-Erkennung (NVIDIA CUDA / Apple MPS / nur CPU), Warnung bei eingeschränkter Hardware, Codegenerierung wird entsprechend angepasst |
| **B: Literatur** | Multi-Source-Suche (OpenAlex → Semantic Scholar → arXiv) nach echten Papern, Relevanzscreening, Extraktion von Wissenskarten |
| **C: Synthese** | Clustering der Ergebnisse, Identifizierung von Forschungslücken, Generierung testbarer Hypothesen via Multi-Agenten-Debatte |
| **D: Design** | Experimentplan entwerfen, hardwarebewussten ausführbaren Python-Code generieren (GPU-Stufe → Paketauswahl), Ressourcenbedarf schätzen |
| **E: Ausführung** | Experimente in Sandbox ausführen, NaN/Inf und Laufzeitfehler erkennen, Code via gezielter LLM-Reparatur selbst heilen |
| **F: Analyse** | Multi-Agenten-Analyse der Ergebnisse; autonome PROCEED / REFINE / PIVOT Entscheidung mit Begründung |
| **G: Schreiben** | Gliederung → abschnittsweises Verfassen (5.000–6.500 Wörter) → Peer-Review (mit Methodik-Evidenz-Konsistenz) → Revision mit Längenprüfung |
| **H: Finalisierung** | Qualitäts-Gate, Wissensarchivierung, LaTeX-Export mit Konferenztemplate, Zitationsintegritäts- und Relevanzprüfung |

</details>

---

## 🚀 Schnellstart

### Voraussetzungen

- 🐍 Python 3.11+
- 🔑 Ein OpenAI-kompatibler LLM-API-Endpunkt (GPT-4o, GPT-5.x oder jeder kompatible Anbieter)

### Installation

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Konfiguration

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 Minimale erforderliche Konfiguration</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"     # Jeder OpenAI-kompatible Endpunkt
  api_key_env: "OPENAI_API_KEY"              # Name der Umgebungsvariable mit deinem Schlüssel
  primary_model: "gpt-4o"                    # Jedes Modell, das dein Endpunkt unterstützt
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                             # Optional: Semantic Scholar API-Schlüssel für höhere Rate-Limits

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

### Ausführung

```bash
# API-Schlüssel setzen
export OPENAI_API_KEY="sk-..."

# 🚀 Vollständige Pipeline ausführen
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 Thema inline angeben
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ Konfiguration validieren
researchclaw validate --config config.arc.yaml

# ⏩ Ab einer bestimmten Stufe fortsetzen
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

Ausgabe → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` mit einem Unterverzeichnis pro Stufe.

Alle benutzerseitigen Ergebnisse werden automatisch in einem einzigen **`deliverables/`**-Ordner gesammelt:

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # Finales Paper (Markdown)
├── paper.tex                  # Konferenzfertiges LaTeX
├── references.bib             # Verifizierte BibTeX-Bibliographie (automatisch bereinigt)
├── neurips_2025.sty           # Konferenz-Stildatei (automatisch ausgewählt)
├── code/                      # Experimentcode + requirements.txt
├── verification_report.json   # Zitationsintegritätsbericht
├── charts/                    # Ergebnisvisualisierungen (Bedingungsvergleich, Fehlerbalken)
└── manifest.json              # Ergebnisindex mit Metadaten
```

Der `deliverables/`-Ordner ist **kompilierbereit** — er enthält die Konferenz-`.sty`- und `.bst`-Dateien, sodass `paper.tex` direkt mit `pdflatex` + `bibtex` kompiliert oder ohne weitere Downloads auf Overleaf hochgeladen werden kann.

---

## ✨ Hauptfunktionen

### 📚 Multi-Source-Literatursuche

Stufe 4 durchsucht **echte akademische APIs** — keine LLM-halluzinierten Paper. Verwendet eine **OpenAlex-first**-Strategie (OpenAlex → Semantic Scholar → arXiv) zur optimalen Abdeckung bei minimalem Rate-Limiting.

- **arXiv API** (primär) — Preprints mit echten arXiv-IDs und Metadaten, keine Rate-Limits
- **Semantic Scholar API** (sekundär) — echte Paper mit Titeln, Abstracts, Venues, Zitationszahlen, DOIs
- **Abfrageerweiterung** — generiert automatisch breitere Abfragen (Survey-, Benchmark-, Vergleichsvarianten) für umfassende Abdeckung (30–60 Referenzen)
- **Automatische Deduplizierung** — DOI → arXiv-ID → unscharfer Titelabgleich
- **BibTeX-Generierung** — gültige `@article{cite_key, ...}`-Einträge mit echten Metadaten
- **Dreistufiger Circuit Breaker** — CLOSED → OPEN → HALF_OPEN Wiederherstellung mit exponentiellem Backoff-Cooldown (nie dauerhaft deaktiviert)
- **Graceful Degradation** — S2-Ausfall blockiert arXiv-Ergebnisse nicht; Fallback auf LLM-augmentierte Ergebnisse, wenn alle APIs ausfallen

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 Zitationsverifikation (Stufe 23)

Nachdem das Paper geschrieben wurde, **überprüft Stufe 23 jede Referenz** auf Integrität und Relevanz:

| Schicht | Methode | Was geprüft wird |
|---------|---------|-------------------|
| L1 | arXiv API `id_list` | Paper mit arXiv-IDs — prüft, ob die ID tatsächlich existiert |
| L2 | CrossRef `/works/{doi}` + DataCite-Fallback | Paper mit DOIs — prüft, ob der DOI auflösbar ist und der Titel übereinstimmt (DataCite behandelt arXiv `10.48550`-DOIs) |
| L3 | Semantic Scholar + arXiv-Titelsuche | Alle übrigen — unscharfer Titelabgleich (≥0,80 Ähnlichkeit) |
| L4 | LLM-Relevanzbewertung | Alle verifizierten Refs — bewertet thematische Relevanz zur Forschung |

Jede Referenz → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**Automatische Bereinigung**: Halluzinierte Zitationen werden stillschweigend aus dem Papertext entfernt (keine `[HALLUCINATED]`-Tags). Nicht zitierte Bibliographieeinträge werden bereinigt. Die finale `references.bib` enthält nur verifizierte, zitierte Referenzen.

### 🖥️ Hardwarebewusste Ausführung

Stufe 1 erkennt automatisch lokale GPU-Fähigkeiten und passt die gesamte Pipeline an:

| Stufe | Erkennung | Verhalten |
|-------|-----------|-----------|
| **Hoch** | NVIDIA GPU mit ≥8 GB VRAM | Volle PyTorch/GPU-Codegenerierung, automatische torch-Installation falls fehlend |
| **Eingeschränkt** | NVIDIA <8 GB oder Apple MPS | Leichtgewichtige Experimente (<1M Parameter, ≤20 Epochen), Benutzerwarnung |
| **Nur CPU** | Keine GPU erkannt | Nur NumPy/sklearn, keine torch-Imports, Benutzerwarnung mit Empfehlung für Remote-GPU |

Das Hardwareprofil wird in `stage-01/hardware_profile.json` gespeichert und beeinflusst Codegenerierung, Sandbox-Imports und Prompt-Einschränkungen.

### 🧪 Sandbox-Experimentausführung

- **Code-Validierung** — AST-Parsing, Import-Whitelist, keine Datei-I/O außerhalb der Sandbox
- **Rechenbudget-Schutz** — Zeitbudget (konfigurierbar, Standard 600s) wird in den Codegenerierungs-Prompt injiziert; LLM muss Experimente entwerfen, die innerhalb des Sandbox-Timeouts ablaufen
- **Experiment-Harness** — unveränderliches `experiment_harness.py` wird in die Sandbox injiziert mit `should_stop()` Zeitschutz, `report_metric()` NaN/Inf-Ablehnung und `finalize()` Ergebnisschreibung (inspiriert von karpathy/autoresearch's unveränderlichem Eval-Muster)
- **Strukturierte Ausgabe** — Experimente erzeugen `results.json` mit typisierten Metriken (nicht nur stdout-Parsing)
- **Intelligentes Metrik-Parsing** — filtert Logzeilen aus Metriken mithilfe von Schlüsselworterkennung (`is_metric_name()`)
- **NaN/Divergenz-Schnellabbruch** — NaN/Inf-Werte werden aus Metriken gefiltert; divergierender Loss (>100) wird erkannt und gemeldet
- **Konvergenzdurchsetzung** — generierter Code muss Early-Stopping-Kriterien enthalten, keine festen Iterationszahlen
- **Laufzeit-Fehlererkennung** — NaN/Inf-Metriken und stderr-Warnungen (Division durch Null, Overflow) werden automatisch erkannt
- **Selbstheilende Reparatur** — Laufzeitprobleme werden mit gezielter Diagnose an das LLM zurückgegeben für ursachengerechte Behebung (kein oberflächliches try/except)
- **Iterative Verfeinerung** — Stufe 13 analysiert Ergebnisse und führt mit verbessertem Code/Parametern erneut aus (bis zu 10 Iterationen, mit Timeout-bewussten Prompts)
- **Teilergebnis-Erfassung** — Experimente mit Timeout, die bereits Metriken erfasst haben, erhalten den Status „partial" statt „failed", um verwertbare Daten zu erhalten
- **Themen-Experiment-Alignment** — LLM-basierte Post-Generierungsprüfung stellt sicher, dass der Experimentcode tatsächlich das angegebene Forschungsthema testet

### 📝 Konferenzqualität beim Schreiben

Die Schreib-Pipeline zielt auf NeurIPS/ICML/ICLR-Standards ab (9+ Seiten, 5.000–6.500 Wörter):

- **Datenintegritätsdurchsetzung** — Papiererstellung wird blockiert, wenn Experimente keine Metriken liefern (verhindert, dass das LLM Ergebnisse fabriziert); Anti-Fabrikationsanweisungen werden sowohl in Entwurfs- als auch Revisions-Prompts injiziert
- **Konferenzqualitäts-Prompts** — System-Prompts enthalten Schlüsselprinzipien aus analysierten akzeptierten Papern: Neuheit, Narrativ, starke Baselines, Ablationen, Ehrlichkeit, Reproduzierbarkeit; häufige Ablehnungsgründe werden markiert
- **Titel- und Rahmungsrichtlinien** — Neuheitssignalisierung, Einprägsamkeitstest, 5-Satz-Abstract-Struktur, Erkennung generischer Titel mit Neugenerierung
- **Abschnittsweises Verfassen** — 3 sequenzielle LLM-Aufrufe (Einleitung+Verwandte Arbeiten → Methode+Experimente → Ergebnisse+Fazit) zur Vermeidung von Ausgabetrunkierung
- **Wortanzahl-Ziele pro Abschnitt** — Abstract (150–250), Einleitung (800–1000), Verwandte Arbeiten (600–800), Methode (1000–1500), Experimente (800–1200), Ergebnisse (600–800), Diskussion (400–600)
- **Revisionslängen-Schutz** — wenn das revidierte Paper kürzer als der Entwurf ist, wird automatisch mit stärkerer Durchsetzung wiederholt; bei Bedarf Fallback auf Entwurf+Annotationen
- **Anti-Disclaimer-Durchsetzung** — begrenzt „due to computational constraints" auf höchstens 1 Vorkommen; Revisions-Prompts entfernen aktiv wiederholte Absicherungsformulierungen
- **Statistische Strenge** — Konfidenzintervalle, p-Werte und Effektstärken in Ergebnistabellen erforderlich; fehlerhafte Ablationen werden markiert und von Behauptungen ausgeschlossen
- **Peer-Review mit Konferenz-Rubrik** — Reviewer bewerten 1–10 nach NeurIPS/ICML-Rubrik (Neuheit, Baselines, Ablationen, Behauptungen vs. Evidenz, Limitierungen)

### 📐 Konferenztemplate-Umschaltung

```yaml
export:
  target_conference: "neurips_2025"   # oder "iclr_2026" oder "icml_2026"
```

| Konferenz | Stilpaket | Spalten |
|-----------|-----------|---------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

Der Markdown → LaTeX Konverter verarbeitet: Abschnittsüberschriften (mit automatischer Nummerierungsdeduplizierung), Inline-/Display-Mathematik, Fett-/Kursivschrift, Listen, Tabellen (mit `\caption`/`\label`), Abbildungen (`\includegraphics`), Codeblöcke (Unicode-sicher), Querverweise und `\cite{}`-Referenzen.

### 🚦 Qualitäts-Gates

| Gate | Stufe | Bei Ablehnung → Zurück zu |
|------|-------|---------------------------|
| Literatur-Screening | 5 | Literatur erneut sammeln (Stufe 4) |
| Experimentdesign | 9 | Hypothesen erneut generieren (Stufe 8) |
| Qualitäts-Gate | 20 | Paper ab Gliederung neu schreiben (Stufe 16) |

Verwende `--auto-approve`, um alle Gates zu überspringen, oder konfiguriere bestimmte Stufen in `security.hitl_required_stages`.

---

## ⚙️ Konfigurationsreferenz

<details>
<summary>Klicken zum Aufklappen der vollständigen Konfigurationsreferenz</summary>

```yaml
# === Projekt ===
project:
  name: "my-research"              # Projektbezeichner
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Forschung ===
research:
  topic: "..."                     # Forschungsthema (erforderlich)
  domains: ["ml", "nlp"]           # Forschungsdomänen für Literatursuche
  daily_paper_count: 8             # Ziel-Paperzahl pro Suchabfrage
  quality_threshold: 4.0           # Mindestqualitätswert für Paper

# === Laufzeit ===
runtime:
  timezone: "America/New_York"     # Für Zeitstempel
  max_parallel_tasks: 3            # Limit gleichzeitiger Experimente
  approval_timeout_hours: 12       # Gate-Stufen-Timeout
  retry_limit: 2                   # Wiederholungsanzahl bei Stufenfehler

# === LLM ===
llm:
  provider: "openai-compatible"    # Anbietertyp
  base_url: "https://..."          # API-Endpunkt (erforderlich)
  api_key_env: "OPENAI_API_KEY"    # Umgebungsvariable für API-Schlüssel (erforderlich)
  api_key: ""                      # Oder Schlüssel direkt eintragen
  primary_model: "gpt-4o"          # Primäres Modell
  fallback_models: ["gpt-4o-mini"] # Fallback-Kette
  s2_api_key: ""                   # Semantic Scholar API-Schlüssel (optional, höhere Rate-Limits)

# === Experiment ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # Max. Ausführungszeit pro Durchlauf (Standard: 600s)
  max_iterations: 10               # Max. Optimierungsiterationen
  metric_key: "val_loss"           # Primärer Metrikname
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
    auto_install_deps: true        # Automatische Import-Erkennung → requirements.txt
  ssh_remote:
    host: ""                       # GPU-Server-Hostname
    gpu_ids: []                    # Verfügbare GPU-IDs
    remote_workdir: "/tmp/researchclaw_experiments"

# === Export ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # Pfad zur benutzerdefinierten Prompts-YAML (leer = Standardwerte)

# === Sicherheit ===
security:
  hitl_required_stages: [5, 9, 20] # Stufen, die menschliche Genehmigung erfordern
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === Wissensdatenbank ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === Benachrichtigungen ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === OpenClaw Bridge ===
openclaw_bridge:
  use_cron: false                  # Geplante Forschungsdurchläufe
  use_message: false               # Fortschrittsbenachrichtigungen
  use_memory: false                # Sitzungsübergreifende Wissenspersistenz
  use_sessions_spawn: false        # Parallele Sub-Sessions starten
  use_web_fetch: false             # Live-Websuche
  use_browser: false               # Browserbasierte Paper-Sammlung
```

</details>

---

## 🙏 Danksagungen

Inspiriert von:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Pionier der automatisierten Forschung
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — End-to-End-Forschungsautomatisierung
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Fully Automated Research System

---

## 📄 Lizenz

MIT — siehe [LICENSE](../LICENSE) für Details.

<p align="center">
  <sub>Gebaut mit 🦞 vom AutoResearchClaw-Team</sub>
</p>
