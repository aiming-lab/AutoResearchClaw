<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Converse uma ideia. Receba um artigo. Totalmente autônomo.</b></h2>



<p align="center">
  <i>Converse com o <a href="#integração-openclaw">OpenClaw</a>: "Pesquise X" → pronto.</i>
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#testes"><img src="https://img.shields.io/badge/Tests-1284%20passed-brightgreen?logo=pytest&logoColor=white" alt="1284 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#integração-openclaw"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="integration-guide.md">📖 Guia de Integração</a>
</p>

---

## ⚡ Em Uma Linha

```bash
pip install -e . && researchclaw run --topic "Sua ideia de pesquisa aqui" --auto-approve
```

---

## 🤔 O Que É Isto?

Você tem uma ideia. Você quer um artigo. **É só isso.**

O AutoResearchClaw recebe um tópico de pesquisa e produz autonomamente um artigo acadêmico completo — com literatura real do OpenAlex, Semantic Scholar e arXiv (multi-fonte), experimentos em sandbox com detecção automática de hardware (GPU/MPS/CPU), análise estatística, revisão por pares e LaTeX pronto para conferência (mirando 5.000-6.500 palavras para NeurIPS/ICML/ICLR). Sem babá. Sem copiar e colar entre ferramentas.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Artigo acadêmico completo (Introdução, Trabalhos Relacionados, Método, Experimentos, Resultados, Conclusão)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>LaTeX pronto para conferência (templates NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Referências BibTeX reais do Semantic Scholar e arXiv — auto-podadas para corresponder às citações inline</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>Verificação de integridade + relevância de citações em 4 camadas (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Código gerado + resultados do sandbox + métricas JSON estruturadas</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Gráficos de comparação de condições gerados automaticamente com barras de erro e intervalos de confiança</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Revisão por pares multi-agente com verificações de consistência metodologia-evidência</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Lições de autoaprendizagem extraídas de cada execução</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>Todas as saídas finais em uma pasta — pronto para compilar no Overleaf</td></tr>
</table>

O pipeline roda **de ponta a ponta sem intervenção humana** (a menos que você configure estágios de gate para revisão manual). Quando experimentos falham, ele se auto-repara. Quando hipóteses não se sustentam, ele pivota.

### 🎯 Experimente

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 O Que o Torna Diferente

### 🔄 Loop de Decisão PIVOT / REFINE

O pipeline não executa apenas linearmente. O Estágio 15 (RESEARCH_DECISION) avalia os resultados dos experimentos contra as hipóteses e toma uma decisão autônoma:

- **PROCEED** — os resultados suportam as hipóteses, continua para a escrita do artigo
- **REFINE** — os resultados são promissores mas precisam de melhorias, volta para refinar código/parâmetros
- **PIVOT** — problema fundamental detectado, reinicia a partir da geração de hipóteses com nova direção

Cada ciclo PIVOT/REFINE **versiona os artefatos anteriores** (`stage-08_v1/`, `stage-08_v2/`, ...) para que nenhum trabalho seja perdido e a evolução das decisões seja totalmente rastreável.

### 🤖 Debate Multi-Agente

Estágios críticos usam um protocolo de debate estruturado com múltiplas perspectivas de LLM:

- **Geração de hipóteses** — agentes diversos propõem e desafiam ideias
- **Análise de resultados** — otimista, cético e pragmático analisam os resultados
- **Revisão por pares** — verificação de consistência metodologia-evidência (o artigo afirma 50 trials quando o código executou 5?)

### 🧬 Evolução: Autoaprendizagem Entre Execuções

Cada execução do pipeline extrai lições detalhadas — não apenas "falhou", mas *por quê*:

- Justificativa das decisões de PIVOT/REFINE
- Avisos de runtime do stderr dos experimentos (ex.: `RuntimeWarning: division by zero`)
- Anomalias em métricas (NaN, Inf, velocidades de convergência idênticas)

Essas lições persistem em um armazenamento JSONL com **ponderação por decaimento temporal de meia-vida de 30 dias** e são injetadas como overlays de prompt em execuções futuras. O pipeline literalmente aprende com seus erros.

### 📚 Base de Conhecimento

Cada execução constrói uma base de conhecimento estruturada (armazenada em `docs/kb/`) com 6 categorias:

- **decisions/** — design de experimentos, quality gates, decisões de pesquisa, planejamento de recursos, estratégias de busca, arquivos de conhecimento
- **experiments/** — logs de geração de código, execuções de experimentos, refinamentos iterativos
- **findings/** — verificação de citações, análise de resultados, relatórios de síntese
- **literature/** — extração de conhecimento, coleta de literatura, resultados de triagem
- **questions/** — geração de hipóteses, decomposição de problemas, inicialização de tópicos
- **reviews/** — relatórios de exportação/publicação, rascunhos de artigos, outlines, revisões, revisão por pares

### 🛡️ Sentinel Watchdog

Um monitor de qualidade em segundo plano que captura problemas que o pipeline principal pode não detectar:

- **Detecção de bugs em runtime** — NaN/Inf em métricas, avisos do stderr enviados de volta ao LLM para reparo direcionado
- **Consistência artigo-evidência** — código real dos experimentos, resultados de execução e logs de refinamento injetados na revisão por pares
- **Pontuação de relevância de citações** — além da verificação de existência, o LLM avalia a relevância temática de cada referência
- **Imposição de convergência** — detecta experimentos de iteração fixa e exige early stopping adequado
- **Validação de ablação** — detecta condições de ablação duplicadas/idênticas e sinaliza comparações quebradas
- **Guarda anti-fabricação** — bloqueia fortemente a escrita do artigo quando experimentos não produzem métricas

---

## 🦞 Integração OpenClaw

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw é um serviço compatível com [OpenClaw](https://github.com/openclaw/openclaw).** Instale-o no OpenClaw e inicie pesquisa autônoma com uma única mensagem — ou use-o de forma independente via CLI, Claude Code ou qualquer assistente de codificação IA.

</td>
</tr>
</table>

### 🚀 Usar com OpenClaw (Recomendado)

Se você já usa o [OpenClaw](https://github.com/openclaw/openclaw) como seu assistente de IA:

```
1️⃣  Compartilhe a URL do repositório GitHub com o OpenClaw
2️⃣  O OpenClaw lê automaticamente RESEARCHCLAW_AGENTS.md → entende o pipeline
3️⃣  Diga: "Pesquise [seu tópico]"
4️⃣  Pronto — o OpenClaw clona, instala, configura, executa e retorna os resultados
```

**É isso.** O OpenClaw gerencia `git clone`, `pip install`, configuração e execução do pipeline automaticamente. Você apenas conversa.

<details>
<summary>💡 O que acontece por baixo dos panos</summary>

1. O OpenClaw lê `RESEARCHCLAW_AGENTS.md` → aprende o papel de orquestrador de pesquisa
2. O OpenClaw lê `README.md` → entende a instalação e estrutura do pipeline
3. O OpenClaw copia `config.researchclaw.example.yaml` → `config.yaml`
4. Solicita sua chave de API do LLM (ou usa sua variável de ambiente)
5. Executa `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. Retorna o artigo, LaTeX, experimentos e citações

</details>

### 🔌 Bridge OpenClaw (Avançado)

Para integração mais profunda, o AutoResearchClaw inclui um **sistema de adaptadores bridge** com 6 capacidades opcionais:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Execuções de pesquisa agendadas
  use_message: true           # 💬 Notificações de progresso (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Persistência de conhecimento entre sessões
  use_sessions_spawn: true    # 🔀 Criar sub-sessões paralelas para estágios concorrentes
  use_web_fetch: true         # 🌐 Busca web ao vivo durante revisão de literatura
  use_browser: false          # 🖥️ Coleta de artigos baseada em navegador
```

Cada flag ativa um protocolo de adaptador tipado. Quando o OpenClaw fornece essas capacidades, os adaptadores as consomem sem alterações no código. Consulte [`integration-guide.md`](integration-guide.md) para detalhes completos.

### 🛠️ Outras Formas de Executar

| Método | Como |
|--------|------|
| **CLI Independente** | `researchclaw run --topic "..." --auto-approve` |
| **API Python** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Lê `RESEARCHCLAW_CLAUDE.md` — basta dizer *"Execute pesquisa sobre [tópico]"* |
| **OpenCode** | Lê `.claude/skills/` — mesma interface em linguagem natural |
| **Qualquer CLI de IA** | Forneça `RESEARCHCLAW_AGENTS.md` como contexto → o agente faz bootstrap automaticamente |

---

## 🔬 Pipeline: 23 Estágios, 8 Fases

```
Fase A: Escopo da Pesquisa           Fase E: Execução de Experimentos
  1. TOPIC_INIT                        12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE                 13. ITERATIVE_REFINE  ← auto-reparo

Fase B: Descoberta de Literatura     Fase F: Análise & Decisão
  3. SEARCH_STRATEGY                   14. RESULT_ANALYSIS    ← multi-agente
  4. LITERATURE_COLLECT  ← API real    15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [gate]
  6. KNOWLEDGE_EXTRACT                 Fase G: Escrita do Artigo
                                       16. PAPER_OUTLINE
Fase C: Síntese de Conhecimento       17. PAPER_DRAFT
  7. SYNTHESIS                         18. PEER_REVIEW        ← verif. evidência
  8. HYPOTHESIS_GEN    ← debate        19. PAPER_REVISION

Fase D: Design de Experimentos      Fase H: Finalização
  9. EXPERIMENT_DESIGN   [gate]        20. QUALITY_GATE      [gate]
 10. CODE_GENERATION                   21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING                 22. EXPORT_PUBLISH     ← LaTeX
                                       23. CITATION_VERIFY    ← verif. relevância
```

> **Estágios gate** (5, 9, 20) pausam para aprovação humana ou aprovam automaticamente com `--auto-approve`. Em caso de rejeição, o pipeline faz rollback.

> **Loops de decisão**: O Estágio 15 pode acionar REFINE (→ Estágio 13) ou PIVOT (→ Estágio 8), com versionamento automático de artefatos.

<details>
<summary>📋 O Que Cada Fase Faz</summary>

| Fase | O Que Acontece |
|------|----------------|
| **A: Escopo** | O LLM decompõe o tópico em uma árvore de problemas estruturada com questões de pesquisa |
| **A+: Hardware** | Detecta automaticamente GPU (NVIDIA CUDA / Apple MPS / apenas CPU), avisa se o hardware local é limitado, adapta a geração de código adequadamente |
| **B: Literatura** | Busca multi-fonte (OpenAlex → Semantic Scholar → arXiv) por artigos reais, triagem por relevância, extração de fichas de conhecimento |
| **C: Síntese** | Agrupa descobertas, identifica lacunas de pesquisa, gera hipóteses testáveis via debate multi-agente |
| **D: Design** | Projeta plano de experimento, gera Python executável com consciência de hardware (tier de GPU → seleção de pacotes), estima necessidades de recursos |
| **E: Execução** | Executa experimentos em sandbox, detecta NaN/Inf e bugs de runtime, auto-repara código via reparo direcionado por LLM |
| **F: Análise** | Análise multi-agente dos resultados; decisão autônoma PROCEED / REFINE / PIVOT com justificativa |
| **G: Escrita** | Outline → redação seção por seção (5.000-6.500 palavras) → revisão por pares (com consistência metodologia-evidência) → revisão com guarda de tamanho |
| **H: Finalização** | Quality gate, arquivamento de conhecimento, exportação LaTeX com template de conferência, verificação de integridade + relevância de citações |

</details>

---

## 🚀 Início Rápido

### Pré-requisitos

- 🐍 Python 3.11+
- 🔑 Um endpoint de API LLM compatível com OpenAI (GPT-4o, GPT-5.x ou qualquer provedor compatível)

### Instalação

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuração

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 Configuração mínima necessária</summary>

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

### Execução

```bash
# Defina sua chave de API
export OPENAI_API_KEY="sk-..."

# 🚀 Execute o pipeline completo
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 Especifique um tópico inline
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ Valide a configuração
researchclaw validate --config config.arc.yaml

# ⏩ Retome a partir de um estágio específico
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

Saída → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` com um subdiretório por estágio.

Todos os entregáveis voltados ao usuário são automaticamente coletados em uma única pasta **`deliverables/`**:

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # Artigo final (Markdown)
├── paper.tex                  # LaTeX pronto para conferência
├── references.bib             # Bibliografia BibTeX verificada (auto-podada)
├── neurips_2025.sty           # Arquivo de estilo da conferência (auto-selecionado)
├── code/                      # Código do experimento + requirements.txt
├── verification_report.json   # Relatório de integridade de citações
├── charts/                    # Visualizações de resultados (comparação de condições, barras de erro)
└── manifest.json              # Índice de entregáveis com metadados
```

A pasta `deliverables/` está **pronta para compilação** — inclui os arquivos `.sty` e `.bst` da conferência para que você possa compilar `paper.tex` diretamente com `pdflatex` + `bibtex` ou fazer upload para o Overleaf sem precisar baixar nada extra.

---

## ✨ Funcionalidades Principais

### 📚 Busca de Literatura Multi-Fonte

O Estágio 4 consulta **APIs acadêmicas reais** — não artigos alucinados por LLM. Usa uma estratégia **OpenAlex-first** (OpenAlex → Semantic Scholar → arXiv) para cobertura ideal com mínimo de limitação de taxa.

- **arXiv API** (primário) — preprints com IDs arXiv reais e metadados, sem limites de taxa
- **Semantic Scholar API** (secundário) — artigos reais com títulos, resumos, venues, contagens de citação, DOIs
- **Expansão de consultas** — gera automaticamente consultas mais amplas (variantes de survey, benchmark, comparação) para cobertura abrangente (30-60 referências)
- **Deduplicação automática** — DOI → arXiv ID → correspondência fuzzy de títulos
- **Geração de BibTeX** — entradas válidas `@article{cite_key, ...}` com metadados reais
- **Circuit breaker de três estados** — CLOSED → OPEN → HALF_OPEN com recuperação e backoff exponencial (nunca desabilitado permanentemente)
- **Degradação graciosa** — falha do S2 não bloqueia resultados do arXiv; faz fallback para resultados aumentados por LLM se todas as APIs falharem

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — citado {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 Verificação de Citações (Estágio 23)

Após a escrita do artigo, o Estágio 23 **verifica cada referência** quanto à integridade e relevância:

| Camada | Método | O Que Verifica |
|--------|--------|----------------|
| L1 | arXiv API `id_list` | Artigos com IDs arXiv — verifica se o ID realmente existe |
| L2 | CrossRef `/works/{doi}` + fallback DataCite | Artigos com DOIs — verifica se o DOI resolve e o título corresponde (DataCite lida com DOIs arXiv `10.48550`) |
| L3 | Semantic Scholar + busca por título no arXiv | Todos os restantes — correspondência fuzzy de títulos (≥0,80 de similaridade) |
| L4 | Pontuação de relevância por LLM | Todas as refs verificadas — avalia relevância temática para a pesquisa |

Cada referência → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**Auto-limpeza**: Citações alucinadas são silenciosamente removidas do texto do artigo (sem tags `[HALLUCINATED]`). Entradas de bibliografia não citadas são podadas. O `references.bib` final contém apenas referências verificadas e citadas.

### 🖥️ Execução com Consciência de Hardware

O Estágio 1 detecta automaticamente as capacidades de GPU local e adapta todo o pipeline:

| Tier | Detecção | Comportamento |
|------|----------|---------------|
| **Alto** | GPU NVIDIA com ≥8 GB VRAM | Geração completa de código PyTorch/GPU, instala torch automaticamente se ausente |
| **Limitado** | NVIDIA <8 GB ou Apple MPS | Experimentos leves (<1M parâmetros, ≤20 épocas), aviso ao usuário |
| **Apenas CPU** | Nenhuma GPU detectada | Apenas NumPy/sklearn, sem imports de torch, aviso ao usuário com recomendação de GPU remota |

O perfil de hardware é salvo em `stage-01/hardware_profile.json` e influencia a geração de código, imports do sandbox e restrições de prompt.

### 🧪 Execução de Experimentos em Sandbox

- **Validação de código** — parsing AST, whitelist de imports, sem I/O de arquivo fora do sandbox
- **Guarda de orçamento computacional** — orçamento de tempo (configurável, padrão 600s) injetado no prompt de geração de código; o LLM deve projetar experimentos que caibam dentro do timeout do sandbox
- **Harness de experimento** — `experiment_harness.py` imutável injetado no sandbox com guarda de tempo `should_stop()`, rejeição de NaN/Inf em `report_metric()`, e escrita de resultados via `finalize()` (inspirado no padrão de eval imutável do karpathy/autoresearch)
- **Saída estruturada** — experimentos produzem `results.json` com métricas tipadas (não apenas parsing de stdout)
- **Parsing inteligente de métricas** — filtra linhas de log das métricas usando detecção de palavras-chave (`is_metric_name()`)
- **Fast-fail de NaN/divergência** — valores NaN/Inf filtrados das métricas; loss divergente (>100) detectado e sinalizado
- **Imposição de convergência** — código gerado deve incluir critérios de early stopping, não contagens de iteração fixas
- **Detecção de bugs em runtime** — métricas NaN/Inf e avisos do stderr (divisão por zero, overflow) detectados automaticamente
- **Reparo auto-reparável** — problemas de runtime enviados de volta ao LLM com diagnóstico direcionado para correções de causa raiz (não band-aid try/except)
- **Refinamento iterativo** — Estágio 13 analisa resultados e re-executa com código/parâmetros melhorados (até 10 iterações, com prompts conscientes de timeout)
- **Captura de resultados parciais** — experimentos com timeout que capturaram métricas recebem status "partial" em vez de "failed", preservando dados utilizáveis
- **Alinhamento tópico-experimento** — verificação pós-geração baseada em LLM garante que o código do experimento realmente testa o tópico de pesquisa declarado

### 📝 Escrita de Artigos com Qualidade de Conferência

O pipeline de escrita mira os padrões NeurIPS/ICML/ICLR (9+ páginas, 5.000-6.500 palavras):

- **Imposição de integridade de dados** — a escrita do artigo é fortemente bloqueada quando experimentos não produzem métricas (impede o LLM de fabricar resultados); instruções anti-fabricação injetadas tanto nos prompts de rascunho quanto de revisão
- **Prompts com qualidade de conferência** — prompts de sistema incluem princípios-chave de análises de artigos aceitos: novidade, narrativa, baselines fortes, ablações, honestidade, reprodutibilidade; razões comuns de rejeição sinalizadas
- **Diretrizes de título e enquadramento** — sinalização de novidade, teste de memorabilidade, estrutura de abstract em 5 frases, detecção de títulos genéricos com re-geração
- **Redação seção por seção** — 3 chamadas sequenciais ao LLM (Intro+Trabalhos Relacionados → Método+Experimentos → Resultados+Conclusão) para evitar truncamento de saída
- **Metas de contagem de palavras por seção** — Resumo (150-250), Introdução (800-1000), Trabalhos Relacionados (600-800), Método (1000-1500), Experimentos (800-1200), Resultados (600-800), Discussão (400-600)
- **Guarda de tamanho na revisão** — se o artigo revisado for mais curto que o rascunho, automaticamente retenta com imposição mais forte; faz fallback para rascunho+anotações se necessário
- **Imposição anti-disclaimer** — limita "due to computational constraints" a no máximo 1 ocorrência; prompts de revisão removem ativamente hedging repetido
- **Rigor estatístico** — intervalos de confiança, p-values e tamanhos de efeito exigidos em tabelas de resultados; ablações quebradas sinalizadas e excluídas das afirmações
- **Revisão por pares com rubrica de conferência** — revisores pontuam de 1-10 seguindo a rubrica NeurIPS/ICML (novidade, baselines, ablações, afirmações vs evidência, limitações)

### 📐 Troca de Template de Conferência

```yaml
export:
  target_conference: "neurips_2025"   # or "iclr_2026" or "icml_2026"
```

| Conferência | Pacote de Estilo | Colunas |
|-------------|-----------------|---------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

O conversor Markdown → LaTeX lida com: títulos de seção (com dedup de auto-numeração), matemática inline/display, negrito/itálico, listas, tabelas (com `\caption`/`\label`), figuras (`\includegraphics`), blocos de código (Unicode-safe), referências cruzadas e referências `\cite{}`.

### 🚦 Quality Gates

| Gate | Estágio | Em Caso de Rejeição → Volta Para |
|------|---------|----------------------------------|
| Triagem de Literatura | 5 | Re-coletar literatura (Estágio 4) |
| Design de Experimento | 9 | Re-gerar hipóteses (Estágio 8) |
| Quality Gate | 20 | Re-escrever artigo a partir do outline (Estágio 16) |

Use `--auto-approve` para pular todos os gates, ou configure estágios específicos em `security.hitl_required_stages`.

---

## ⚙️ Referência de Configuração

<details>
<summary>Clique para expandir a referência completa de configuração</summary>

```yaml
# === Projeto ===
project:
  name: "my-research"              # Identificador do projeto
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Pesquisa ===
research:
  topic: "..."                     # Tópico de pesquisa (obrigatório)
  domains: ["ml", "nlp"]           # Domínios de pesquisa para busca de literatura
  daily_paper_count: 8             # Artigos alvo por consulta de busca
  quality_threshold: 4.0           # Pontuação mínima de qualidade para artigos

# === Runtime ===
runtime:
  timezone: "America/New_York"     # Para timestamps
  max_parallel_tasks: 3            # Limite de experimentos concorrentes
  approval_timeout_hours: 12       # Timeout de estágios gate
  retry_limit: 2                   # Contagem de retentativas em falha de estágio

# === LLM ===
llm:
  provider: "openai-compatible"    # Tipo de provedor
  base_url: "https://..."          # Endpoint da API (obrigatório)
  api_key_env: "OPENAI_API_KEY"    # Variável de ambiente para chave da API (obrigatório)
  api_key: ""                      # Ou insira a chave diretamente aqui
  primary_model: "gpt-4o"          # Modelo primário
  fallback_models: ["gpt-4o-mini"] # Cadeia de fallback
  s2_api_key: ""                   # Chave API do Semantic Scholar (opcional, limites de taxa maiores)

# === Experimento ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # Tempo máximo de execução por run (padrão: 600s)
  max_iterations: 10               # Máximo de iterações de otimização
  metric_key: "val_loss"           # Nome da métrica primária
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
    auto_install_deps: true        # Detecção automática de imports → requirements.txt
  ssh_remote:
    host: ""                       # Hostname do servidor GPU
    gpu_ids: []                    # IDs de GPU disponíveis
    remote_workdir: "/tmp/researchclaw_experiments"

# === Exportação ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # Caminho para YAML de prompts customizados (vazio = padrões)

# === Segurança ===
security:
  hitl_required_stages: [5, 9, 20] # Estágios que requerem aprovação humana
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === Base de Conhecimento ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === Notificações ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === Bridge OpenClaw ===
openclaw_bridge:
  use_cron: false                  # Execuções de pesquisa agendadas
  use_message: false               # Notificações de progresso
  use_memory: false                # Persistência de conhecimento entre sessões
  use_sessions_spawn: false        # Criar sub-sessões paralelas
  use_web_fetch: false             # Busca web ao vivo
  use_browser: false               # Coleta de artigos baseada em navegador
```

</details>

---

## 🙏 Agradecimentos

Inspirado por:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Pioneiro em pesquisa automatizada
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — Automação de pesquisa de ponta a ponta
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Fully Automated Research System

---

## 📄 Licença

MIT — veja [LICENSE](../LICENSE) para detalhes.

<p align="center">
  <sub>Construído com 🦞 pela equipe AutoResearchClaw</sub>
</p>
