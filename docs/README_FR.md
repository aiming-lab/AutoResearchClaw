<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Discutez une idée. Obtenez un article. Entièrement autonome.</b></h2>



<p align="center">
  <i>Discutez avec <a href="#intégration-openclaw">OpenClaw</a> : « Recherche X » → terminé.</i>
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/Tests-1284%20passed-brightgreen?logo=pytest&logoColor=white" alt="1284 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#intégration-openclaw"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="integration-guide.md">📖 Guide d'intégration</a>
</p>

---

## ⚡ En une ligne

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```

---

## 🤔 De quoi s'agit-il ?

Vous avez une idée. Vous voulez un article. **C'est tout.**

AutoResearchClaw prend un sujet de recherche et produit de manière autonome un article académique complet — avec de la vraie littérature provenant d'arXiv et de Semantic Scholar (multi-sources, arXiv en priorité pour éviter la limitation de débit), des expériences en sandbox adaptées au matériel (détection automatique GPU/MPS/CPU), une analyse statistique, une relecture par les pairs, et du LaTeX prêt pour les conférences (ciblant 5 000-6 500 mots pour NeurIPS/ICML/ICLR). Aucune supervision. Aucun copier-coller entre outils.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Article académique complet (Introduction, Travaux connexes, Méthode, Expériences, Résultats, Conclusion)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>LaTeX prêt pour les conférences (templates NeurIPS / ICLR / ICML)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Références BibTeX réelles provenant de Semantic Scholar et arXiv — auto-élaguées pour correspondre aux citations dans le texte</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>Vérification d'intégrité et de pertinence des citations sur 4 couches (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Code généré + résultats sandbox + métriques JSON structurées</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Graphiques de comparaison de conditions auto-générés avec barres d'erreur et intervalles de confiance</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Relecture multi-agents avec vérification de cohérence méthodologie-preuves</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Leçons d'auto-apprentissage extraites de chaque exécution</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>Tous les livrables finaux dans un seul dossier — prêt à compiler pour Overleaf</td></tr>
</table>

Le pipeline s'exécute **de bout en bout sans intervention humaine** (sauf si vous configurez des étapes de validation pour une revue manuelle). Quand les expériences échouent, il s'auto-répare. Quand les hypothèses ne tiennent pas, il pivote.

### 🎯 Essayez

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 Ce qui le distingue

### 🔄 Boucle de décision PIVOT / REFINE

Le pipeline ne se contente pas de s'exécuter linéairement. L'étape 15 (RESEARCH_DECISION) évalue les résultats expérimentaux par rapport aux hypothèses et prend une décision autonome :

- **PROCEED** — les résultats confirment les hypothèses, on continue vers la rédaction
- **REFINE** — les résultats sont prometteurs mais nécessitent des améliorations, retour en arrière pour affiner le code/les paramètres
- **PIVOT** — problème fondamental détecté, redémarrage depuis la génération d'hypothèses avec une nouvelle direction

Chaque cycle PIVOT/REFINE **versionne les artefacts précédents** (`stage-08_v1/`, `stage-08_v2/`, ...) afin qu'aucun travail ne soit perdu et que l'évolution des décisions soit entièrement traçable.

### 🤖 Débat multi-agents

Les étapes critiques utilisent un protocole de débat structuré avec plusieurs perspectives LLM :

- **Génération d'hypothèses** — des agents diversifiés proposent et contestent des idées
- **Analyse des résultats** — un optimiste, un sceptique et un pragmatique analysent les résultats
- **Relecture par les pairs** — vérification de la cohérence méthodologie-preuves (l'article affirme-t-il 50 essais alors que le code n'en a exécuté que 5 ?)

### 🧬 Évolution : auto-apprentissage inter-exécutions

Chaque exécution du pipeline extrait des leçons granulaires — pas seulement « ça a échoué » mais *pourquoi* :

- Justification des décisions PIVOT/REFINE
- Avertissements d'exécution depuis stderr (ex. `RuntimeWarning: division by zero`)
- Anomalies métriques (NaN, Inf, vitesses de convergence identiques)

Ces leçons sont conservées dans un magasin JSONL avec **pondération par décroissance temporelle à demi-vie de 30 jours** et sont injectées comme surcouches de prompts dans les exécutions futures. Le pipeline apprend littéralement de ses erreurs.

### 📚 Base de connaissances

Chaque exécution construit une base de connaissances structurée (stockée dans `docs/kb/`) avec 6 catégories :

- **decisions/** — conception d'expériences, portes qualité, décisions de recherche, planification des ressources, stratégies de recherche, archives de connaissances
- **experiments/** — journaux de génération de code, exécutions d'expériences, affinements itératifs
- **findings/** — vérification de citations, analyse de résultats, rapports de synthèse
- **literature/** — extraction de connaissances, collecte de littérature, résultats de filtrage
- **questions/** — génération d'hypothèses, décomposition de problèmes, initialisation du sujet
- **reviews/** — rapports d'export/publication, brouillons d'articles, plans, révisions, relectures par les pairs

### 🛡️ Sentinel Watchdog

Un moniteur de qualité en arrière-plan qui détecte les problèmes que le pipeline principal pourrait manquer :

- **Détection de bugs à l'exécution** — NaN/Inf dans les métriques, avertissements stderr renvoyés au LLM pour réparation ciblée
- **Cohérence article-preuves** — le code d'expérience réel, les résultats d'exécution et les journaux d'affinement sont injectés dans la relecture
- **Score de pertinence des citations** — au-delà de la vérification d'existence, le LLM évalue la pertinence thématique de chaque référence
- **Application de la convergence** — détecte les expériences à itérations fixes et exige un arrêt anticipé approprié
- **Validation des ablations** — détecte les conditions d'ablation dupliquées/identiques et signale les comparaisons invalides
- **Protection anti-fabrication** — bloque strictement la rédaction quand les expériences ne produisent aucune métrique

---

## 🦞 Intégration OpenClaw

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw est un service compatible [OpenClaw](https://github.com/openclaw/openclaw).** Installez-le dans OpenClaw et lancez une recherche autonome avec un seul message — ou utilisez-le de manière autonome via CLI, Claude Code, ou tout assistant de codage IA.

</td>
</tr>
</table>

### 🚀 Utilisation avec OpenClaw (recommandé)

Si vous utilisez déjà [OpenClaw](https://github.com/openclaw/openclaw) comme assistant IA :

```
1️⃣  Partagez l'URL du dépôt GitHub avec OpenClaw
2️⃣  OpenClaw lit automatiquement RESEARCHCLAW_AGENTS.md → comprend le pipeline
3️⃣  Dites : "Research [votre sujet]"
4️⃣  C'est fait — OpenClaw clone, installe, configure, exécute et renvoie les résultats
```

**C'est tout.** OpenClaw gère `git clone`, `pip install`, la configuration et l'exécution du pipeline automatiquement. Vous n'avez qu'à discuter.

<details>
<summary>💡 Ce qui se passe en coulisses</summary>

1. OpenClaw lit `RESEARCHCLAW_AGENTS.md` → apprend le rôle d'orchestrateur de recherche
2. OpenClaw lit `README.md` → comprend l'installation et la structure du pipeline
3. OpenClaw copie `config.researchclaw.example.yaml` → `config.yaml`
4. Demande votre clé API LLM (ou utilise votre variable d'environnement)
5. Exécute `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. Renvoie l'article, le LaTeX, les expériences et les citations

</details>

### 🔌 Pont OpenClaw (avancé)

Pour une intégration plus poussée, AutoResearchClaw inclut un **système d'adaptateurs pont** avec 6 fonctionnalités optionnelles :

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Exécutions de recherche planifiées
  use_message: true           # 💬 Notifications de progression (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Persistance des connaissances inter-sessions
  use_sessions_spawn: true    # 🔀 Lancement de sous-sessions parallèles pour les étapes concurrentes
  use_web_fetch: true         # 🌐 Recherche web en direct pendant la revue de littérature
  use_browser: false          # 🖥️ Collecte d'articles via navigateur
```

Chaque option active un protocole d'adaptateur typé. Quand OpenClaw fournit ces fonctionnalités, les adaptateurs les consomment sans modification de code. Voir [`integration-guide.md`](integration-guide.md) pour tous les détails.

### 🛠️ Autres méthodes d'exécution

| Méthode | Comment |
|---------|---------|
| **CLI autonome** | `researchclaw run --topic "..." --auto-approve` |
| **API Python** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Lit `RESEARCHCLAW_CLAUDE.md` — dites simplement *"Run research on [sujet]"* |
| **OpenCode** | Lit `.claude/skills/` — même interface en langage naturel |
| **Tout CLI IA** | Fournissez `RESEARCHCLAW_AGENTS.md` comme contexte → l'agent s'auto-initialise |

---

## 🔬 Pipeline : 23 étapes, 8 phases

```
Phase A : Cadrage de la recherche     Phase E : Exécution des expériences
  1. TOPIC_INIT                         12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE                  13. ITERATIVE_REFINE  ← auto-réparation

Phase B : Découverte de littérature   Phase F : Analyse et décision
  3. SEARCH_STRATEGY                    14. RESULT_ANALYSIS    ← multi-agents
  4. LITERATURE_COLLECT  ← API réelle   15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [porte]
  6. KNOWLEDGE_EXTRACT                  Phase G : Rédaction de l'article
                                        16. PAPER_OUTLINE
Phase C : Synthèse des connaissances    17. PAPER_DRAFT
  7. SYNTHESIS                          18. PEER_REVIEW        ← vérif. preuves
  8. HYPOTHESIS_GEN    ← débat          19. PAPER_REVISION

Phase D : Conception expérimentale    Phase H : Finalisation
  9. EXPERIMENT_DESIGN   [porte]        20. QUALITY_GATE      [porte]
 10. CODE_GENERATION                    21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING                  22. EXPORT_PUBLISH     ← LaTeX
                                        23. CITATION_VERIFY    ← vérif. pertinence
```

> **Étapes de validation** (5, 9, 20) : pause pour approbation humaine ou approbation automatique avec `--auto-approve`. En cas de rejet, le pipeline revient en arrière.

> **Boucles de décision** : l'étape 15 peut déclencher REFINE (→ étape 13) ou PIVOT (→ étape 8), avec versionnement automatique des artefacts.

<details>
<summary>📋 Ce que fait chaque phase</summary>

| Phase | Ce qui se passe |
|-------|-----------------|
| **A : Cadrage** | Le LLM décompose le sujet en un arbre de problèmes structuré avec des questions de recherche |
| **A+ : Matériel** | Détection automatique du GPU (NVIDIA CUDA / Apple MPS / CPU uniquement), avertissement si le matériel local est limité, adaptation de la génération de code en conséquence |
| **B : Littérature** | Recherche multi-sources (arXiv en priorité, puis Semantic Scholar) de vrais articles, filtrage par pertinence, extraction de fiches de connaissances |
| **C : Synthèse** | Regroupement des résultats, identification des lacunes de recherche, génération d'hypothèses testables via débat multi-agents |
| **D : Conception** | Conception du plan expérimental, génération de Python exécutable adapté au matériel (niveau GPU → sélection de packages), estimation des besoins en ressources |
| **E : Exécution** | Exécution des expériences en sandbox, détection de NaN/Inf et bugs d'exécution, auto-réparation du code via réparation ciblée par LLM |
| **F : Analyse** | Analyse multi-agents des résultats ; décision autonome PROCEED / REFINE / PIVOT avec justification |
| **G : Rédaction** | Plan → rédaction section par section (5 000-6 500 mots) → relecture (avec vérification de cohérence méthodologie-preuves) → révision avec contrôle de longueur |
| **H : Finalisation** | Porte qualité, archivage des connaissances, export LaTeX avec template de conférence, vérification d'intégrité et de pertinence des citations |

</details>

---

## 🚀 Démarrage rapide

### Prérequis

- 🐍 Python 3.11+
- 🔑 Un point d'accès API LLM compatible OpenAI (GPT-4o, GPT-5.x, ou tout fournisseur compatible)

### Installation

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuration

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 Configuration minimale requise</summary>

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

### Exécution

```bash
# Définissez votre clé API
export OPENAI_API_KEY="sk-..."

# 🚀 Exécuter le pipeline complet
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 Spécifier un sujet en ligne de commande
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ Valider la configuration
researchclaw validate --config config.arc.yaml

# ⏩ Reprendre à partir d'une étape spécifique
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

Sortie → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` avec un sous-répertoire par étape.

Tous les livrables destinés à l'utilisateur sont automatiquement rassemblés dans un seul dossier **`deliverables/`** :

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # Article final (Markdown)
├── paper.tex                  # LaTeX prêt pour la conférence
├── references.bib             # Bibliographie BibTeX vérifiée (auto-élaguée)
├── neurips_2025.sty           # Fichier de style conférence (auto-sélectionné)
├── code/                      # Code d'expérience + requirements.txt
├── verification_report.json   # Rapport d'intégrité des citations
├── charts/                    # Visualisations des résultats (comparaison de conditions, barres d'erreur)
└── manifest.json              # Index des livrables avec métadonnées
```

Le dossier `deliverables/` est **prêt à compiler** — il inclut les fichiers `.sty` et `.bst` de la conférence pour que vous puissiez compiler `paper.tex` directement avec `pdflatex` + `bibtex` ou le télécharger sur Overleaf sans rien télécharger de plus.

---

## ✨ Fonctionnalités clés

### 📚 Recherche de littérature multi-sources

L'étape 4 interroge de **vraies API académiques** — pas des articles hallucinés par un LLM. Utilise une stratégie **arXiv en priorité** pour éviter la limitation de débit de Semantic Scholar.

- **arXiv API** (primaire) — prépublications avec de vrais identifiants arXiv et métadonnées, sans limite de débit
- **Semantic Scholar API** (secondaire) — vrais articles avec titres, résumés, lieux de publication, nombre de citations, DOIs
- **Expansion de requêtes** — génère automatiquement des requêtes plus larges (variantes survey, benchmark, comparaison) pour une couverture complète (30-60 références)
- **Déduplication automatique** — DOI → identifiant arXiv → correspondance floue par titre
- **Génération BibTeX** — entrées valides `@article{cite_key, ...}` avec de vraies métadonnées
- **Disjoncteur à trois états** — CLOSED → OPEN → HALF_OPEN avec récupération par backoff exponentiel (jamais désactivé de manière permanente)
- **Dégradation gracieuse** — une défaillance de S2 ne bloque pas les résultats arXiv ; repli sur des résultats augmentés par LLM si toutes les API échouent

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 Vérification des citations (étape 23)

Après la rédaction de l'article, l'étape 23 **vérifie chaque référence** en termes d'intégrité et de pertinence :

| Couche | Méthode | Ce qu'elle vérifie |
|--------|---------|-------------------|
| L1 | arXiv API `id_list` | Articles avec identifiants arXiv — vérifie que l'identifiant existe réellement |
| L2 | CrossRef `/works/{doi}` + fallback DataCite | Articles avec DOIs — vérifie que le DOI résout et que le titre correspond (DataCite gère les DOIs arXiv `10.48550`) |
| L3 | Semantic Scholar + recherche par titre arXiv | Tout le reste — correspondance floue par titre (similarité ≥ 0,80) |
| L4 | Score de pertinence LLM | Toutes les réf. vérifiées — évalue la pertinence thématique par rapport à la recherche |

Chaque référence → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**Nettoyage automatique** : les citations hallucinées sont silencieusement supprimées du texte de l'article (pas de balises `[HALLUCINATED]`). Les entrées bibliographiques non citées sont élaguées. Le fichier final `references.bib` ne contient que des références vérifiées et citées.

### 🖥️ Exécution adaptée au matériel

L'étape 1 détecte automatiquement les capacités GPU locales et adapte l'ensemble du pipeline :

| Niveau | Détection | Comportement |
|--------|-----------|--------------|
| **Élevé** | GPU NVIDIA avec ≥ 8 Go VRAM | Génération de code PyTorch/GPU complète, installation automatique de torch si absent |
| **Limité** | NVIDIA < 8 Go ou Apple MPS | Expériences légères (< 1M paramètres, ≤ 20 époques), avertissement utilisateur |
| **CPU uniquement** | Aucun GPU détecté | NumPy/sklearn uniquement, pas d'imports torch, avertissement avec recommandation de GPU distant |

Le profil matériel est sauvegardé dans `stage-01/hardware_profile.json` et influence la génération de code, les imports sandbox et les contraintes de prompts.

### 🧪 Exécution d'expériences en sandbox

- **Validation du code** — analyse AST, liste blanche d'imports, pas d'E/S fichier hors de la sandbox
- **Garde de budget de calcul** — budget temps (configurable, 600s par défaut) injecté dans le prompt de génération de code ; le LLM doit concevoir des expériences qui respectent le timeout de la sandbox
- **Harnais d'expérimentation** — `experiment_harness.py` immuable injecté dans la sandbox avec garde temporelle `should_stop()`, rejet NaN/Inf par `report_metric()`, et écriture de résultats `finalize()` (inspiré du pattern d'évaluation immuable de karpathy/autoresearch)
- **Sortie structurée** — les expériences produisent un `results.json` avec des métriques typées (pas seulement de l'analyse stdout)
- **Analyse intelligente des métriques** — filtre les lignes de log des métriques via détection de mots-clés (`is_metric_name()`)
- **Échec rapide NaN/divergence** — les valeurs NaN/Inf sont filtrées des métriques ; une perte divergente (> 100) est détectée et signalée
- **Application de la convergence** — le code généré doit inclure des critères d'arrêt anticipé, pas un nombre fixe d'itérations
- **Détection de bugs à l'exécution** — les métriques NaN/Inf et les avertissements stderr (division par zéro, débordement) sont détectés automatiquement
- **Réparation auto-guérison** — les problèmes d'exécution sont renvoyés au LLM avec un diagnostic ciblé pour des corrections à la racine (pas de pansement try/except)
- **Affinement itératif** — l'étape 13 analyse les résultats et relance avec du code/paramètres améliorés (jusqu'à 10 itérations, avec prompts tenant compte du timeout)
- **Capture de résultats partiels** — les expériences en timeout avec des métriques capturées obtiennent le statut « partial » au lieu de « failed », préservant les données utilisables
- **Alignement sujet-expérience** — vérification post-génération par LLM que le code d'expérience teste réellement le sujet de recherche annoncé

### 📝 Rédaction d'articles de qualité conférence

Le pipeline de rédaction cible les standards NeurIPS/ICML/ICLR (9+ pages, 5 000-6 500 mots) :

- **Application de l'intégrité des données** — la rédaction est bloquée quand les expériences ne produisent aucune métrique (empêche le LLM de fabriquer des résultats) ; instructions anti-fabrication injectées dans les prompts de brouillon et de révision
- **Prompts de qualité conférence** — les prompts système incluent des principes clés issus de l'analyse d'articles acceptés : nouveauté, narratif, baselines solides, ablations, honnêteté, reproductibilité ; raisons courantes de rejet signalées
- **Directives de titre et de cadrage** — signalement de nouveauté, test de mémorabilité, structure d'abstract en 5 phrases, détection de titre générique avec re-génération
- **Rédaction section par section** — 3 appels LLM séquentiels (Intro+Travaux connexes → Méthode+Expériences → Résultats+Conclusion) pour éviter la troncature de sortie
- **Objectifs de nombre de mots par section** — Résumé (150-250), Introduction (800-1000), Travaux connexes (600-800), Méthode (1000-1500), Expériences (800-1200), Résultats (600-800), Discussion (400-600)
- **Garde de longueur en révision** — si l'article révisé est plus court que le brouillon, relance automatique avec un renforcement plus strict ; repli sur brouillon+annotations si nécessaire
- **Application anti-clause de non-responsabilité** — limite « due to computational constraints » à au plus 1 occurrence ; les prompts de révision suppriment activement les réserves répétitives
- **Rigueur statistique** — intervalles de confiance, valeurs p et tailles d'effet requis dans les tableaux de résultats ; ablations invalides signalées et exclues des affirmations
- **Relecture avec grille de conférence** — les relecteurs notent de 1 à 10 selon la grille NeurIPS/ICML (nouveauté, baselines, ablations, affirmations vs preuves, limites)

### 📐 Changement de template de conférence

```yaml
export:
  target_conference: "neurips_2025"   # or "iclr_2026" or "icml_2026"
```

| Conférence | Package de style | Colonnes |
|------------|-----------------|----------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

Le convertisseur Markdown → LaTeX gère : les titres de sections (avec dédoublonnage de la numérotation automatique), les formules mathématiques en ligne/display, le gras/italique, les listes, les tableaux (avec `\caption`/`\label`), les figures (`\includegraphics`), les blocs de code (compatibles Unicode), les références croisées et les références `\cite{}`.

### 🚦 Portes qualité

| Porte | Étape | En cas de rejet → retour à |
|-------|-------|---------------------------|
| Filtrage de littérature | 5 | Re-collecter la littérature (étape 4) |
| Conception expérimentale | 9 | Re-générer les hypothèses (étape 8) |
| Porte qualité | 20 | Re-rédiger l'article depuis le plan (étape 16) |

Utilisez `--auto-approve` pour passer toutes les portes, ou configurez des étapes spécifiques dans `security.hitl_required_stages`.

---

## ⚙️ Référence de configuration

<details>
<summary>Cliquez pour afficher la référence complète de configuration</summary>

```yaml
# === Projet ===
project:
  name: "my-research"              # Identifiant du projet
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Recherche ===
research:
  topic: "..."                     # Sujet de recherche (requis)
  domains: ["ml", "nlp"]           # Domaines de recherche pour la revue de littérature
  daily_paper_count: 8             # Nombre cible d'articles par requête de recherche
  quality_threshold: 4.0           # Score qualité minimum pour les articles

# === Exécution ===
runtime:
  timezone: "America/New_York"     # Pour les horodatages
  max_parallel_tasks: 3            # Limite d'expériences concurrentes
  approval_timeout_hours: 12       # Timeout des étapes de validation
  retry_limit: 2                   # Nombre de tentatives en cas d'échec d'étape

# === LLM ===
llm:
  provider: "openai-compatible"    # Type de fournisseur
  base_url: "https://..."          # Point d'accès API (requis)
  api_key_env: "OPENAI_API_KEY"    # Variable d'env pour la clé API (requis)
  api_key: ""                      # Ou clé en dur ici
  primary_model: "gpt-4o"          # Modèle principal
  fallback_models: ["gpt-4o-mini"] # Chaîne de repli
  s2_api_key: ""                   # Clé API Semantic Scholar (optionnel, limites de débit plus élevées)

# === Expérience ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # Temps d'exécution max par lancement (défaut : 600s)
  max_iterations: 10               # Itérations d'optimisation max
  metric_key: "val_loss"           # Nom de la métrique principale
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
    auto_install_deps: true        # Détection auto des imports → requirements.txt
  ssh_remote:
    host: ""                       # Nom d'hôte du serveur GPU
    gpu_ids: []                    # Identifiants GPU disponibles
    remote_workdir: "/tmp/researchclaw_experiments"

# === Export ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # Chemin vers un YAML de prompts personnalisés (vide = défauts)

# === Sécurité ===
security:
  hitl_required_stages: [5, 9, 20] # Étapes nécessitant une approbation humaine
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === Base de connaissances ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === Notifications ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === Pont OpenClaw ===
openclaw_bridge:
  use_cron: false                  # Exécutions de recherche planifiées
  use_message: false               # Notifications de progression
  use_memory: false                # Persistance des connaissances inter-sessions
  use_sessions_spawn: false        # Lancement de sous-sessions parallèles
  use_web_fetch: false             # Recherche web en direct
  use_browser: false               # Collecte d'articles via navigateur
```

</details>

---

## 🙏 Remerciements

Inspiré par :

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Pionnier de la recherche automatisée
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — Automatisation de la recherche de bout en bout
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Système de recherche entièrement automatisé

---

## 📄 Licence

MIT — voir [LICENSE](../LICENSE) pour les détails.

<p align="center">
  <sub>Construit avec 🦞 par l'équipe AutoResearchClaw</sub>
</p>
