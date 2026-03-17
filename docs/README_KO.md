<p align="center">
  <img src="../image/logo.png" width="800" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>아이디어를 말하다. 논문을 받다. 완전 자동.</b></h2>



<p align="center">
  <i><a href="#openclaw-integration">OpenClaw</a>에 채팅하세요: "X 연구해줘" → 완료.</i>
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
  <a href="integration-guide.md">📖 통합 가이드</a>
</p>

---

## ⚡ 한 줄 실행

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```

---

## 🤔 이것은 무엇인가요?

아이디어가 있고, 논문이 필요합니다. **그게 전부입니다.**

AutoResearchClaw는 연구 주제를 받아 완전한 학술 논문을 자율적으로 작성합니다 — arXiv와 Semantic Scholar에서 실제 문헌을 검색하고(다중 소스, 속도 제한 회피를 위한 arXiv 우선), 하드웨어 인식 샌드박스 실험(GPU/MPS/CPU 자동 감지), 통계 분석, 피어 리뷰, 그리고 학회 수준의 LaTeX(NeurIPS/ICML/ICLR 대상 5,000-6,500단어)를 수행합니다. 수동 관리가 필요 없습니다. 도구 간 복사-붙여넣기도 필요 없습니다.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>완성된 학술 논문 (서론, 관련 연구, 방법론, 실험, 결과, 결론)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>학회 제출용 LaTeX (NeurIPS / ICLR / ICML 템플릿)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Semantic Scholar 및 arXiv에서 가져온 실제 BibTeX 참고문헌 — 인라인 인용과 일치하도록 자동 정리</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4계층 인용 무결성 + 관련성 검증 (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>생성된 코드 + 샌드박스 결과 + 구조화된 JSON 메트릭</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>오차 막대와 신뢰 구간이 포함된 자동 생성 조건 비교 차트</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>방법론-증거 일관성 검사를 포함한 멀티 에이전트 피어 리뷰</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>각 실행에서 추출된 자기 학습 교훈</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>모든 최종 산출물을 하나의 폴더에 — Overleaf에 바로 컴파일 가능</td></tr>
</table>

파이프라인은 **사람의 개입 없이 처음부터 끝까지 실행**됩니다 (수동 검토를 위한 게이트 단계를 설정하지 않는 한). 실험이 실패하면 자가 복구합니다. 가설이 성립하지 않으면 방향을 전환합니다.

### 🎯 사용해 보세요

```bash
researchclaw run --topic "Agent-based Reinforcement Learning for Automated Scientific Discovery" --auto-approve
```

---

## 🧠 차별화 요소

### 🔄 PIVOT / REFINE 의사결정 루프

파이프라인은 단순히 선형으로 실행되지 않습니다. 15단계(RESEARCH_DECISION)에서 실험 결과를 가설과 비교 평가하고 자율적으로 결정을 내립니다:

- **PROCEED** — 결과가 가설을 지지하므로 논문 작성으로 진행
- **REFINE** — 결과가 유망하나 개선이 필요하므로 코드/매개변수를 개선하기 위해 루프 복귀
- **PIVOT** — 근본적인 문제가 발견되어 새로운 방향으로 가설 생성부터 재시작

각 PIVOT/REFINE 주기는 **이전 산출물을 버전 관리**합니다 (`stage-08_v1/`, `stage-08_v2/`, ...) 따라서 작업이 손실되지 않으며 의사결정 과정이 완전히 추적 가능합니다.

### 🤖 멀티 에이전트 토론

핵심 단계에서는 다수의 LLM 관점을 활용한 구조화된 토론 프로토콜을 사용합니다:

- **가설 생성** — 다양한 에이전트가 아이디어를 제안하고 반론을 제기
- **결과 분석** — 낙관론자, 회의론자, 실용주의자가 결과를 분석
- **피어 리뷰** — 방법론-증거 일관성 검사 (논문이 50회 시행을 주장하는데 코드는 5회만 실행했는지 확인)

### 🧬 진화: 실행 간 자기 학습

모든 파이프라인 실행은 세밀한 교훈을 추출합니다 — 단순히 "실패했다"가 아니라 *왜* 실패했는지:

- PIVOT/REFINE 선택의 의사결정 근거
- 실험 stderr의 런타임 경고 (예: `RuntimeWarning: division by zero`)
- 메트릭 이상 (NaN, Inf, 동일한 수렴 속도)

이 교훈들은 **30일 반감기 시간 감쇠 가중치**를 적용하여 JSONL 저장소에 보존되며, 향후 실행에 프롬프트 오버레이로 주입됩니다. 파이프라인은 말 그대로 실수로부터 학습합니다.

### 📚 지식 기반

모든 실행은 6개 카테고리로 구성된 구조화된 지식 기반을 구축합니다 (`docs/kb/`에 저장):

- **decisions/** — 실험 설계, 품질 게이트, 연구 결정, 자원 계획, 검색 전략, 지식 아카이브
- **experiments/** — 코드 생성 로그, 실험 실행, 반복적 개선
- **findings/** — 인용 검증, 결과 분석, 종합 보고서
- **literature/** — 지식 추출, 문헌 수집, 선별 결과
- **questions/** — 가설 생성, 문제 분해, 주제 초기화
- **reviews/** — 내보내기/출판 보고서, 논문 초안, 개요, 수정, 피어 리뷰

### 🛡️ 센티넬 감시견

메인 파이프라인이 놓칠 수 있는 문제를 포착하는 백그라운드 품질 모니터:

- **런타임 버그 감지** — 메트릭의 NaN/Inf, stderr 경고를 LLM에 피드백하여 표적화된 수리
- **논문-증거 일관성** — 실제 실험 코드, 실행 결과, 개선 로그를 피어 리뷰에 주입
- **인용 관련성 점수** — 존재 여부 검증 외에도 LLM이 각 참고문헌의 주제 관련성을 평가
- **수렴 기준 적용** — 고정 반복 실험을 감지하고 적절한 조기 종료를 요구
- **절제 실험 검증** — 중복/동일한 절제 조건을 감지하고 잘못된 비교를 표시
- **날조 방지 가드** — 실험이 메트릭을 생성하지 않으면 논문 작성을 강제 차단

---

## 🦞 OpenClaw 통합

<table>
<tr>
<td width="60">🦞</td>
<td>

**AutoResearchClaw는 [OpenClaw](https://github.com/openclaw/openclaw) 호환 서비스입니다.** OpenClaw에 설치하고 단일 메시지로 자율 연구를 시작하거나, CLI, Claude Code 또는 기타 AI 코딩 어시스턴트를 통해 독립적으로 사용하세요.

</td>
</tr>
</table>

### 🚀 OpenClaw와 함께 사용 (권장)

[OpenClaw](https://github.com/openclaw/openclaw)을 이미 AI 어시스턴트로 사용하고 있다면:

```
1️⃣  GitHub 저장소 URL을 OpenClaw에 공유
2️⃣  OpenClaw이 자동으로 RESEARCHCLAW_AGENTS.md를 읽고 → 파이프라인을 이해
3️⃣  "Research [주제]"라고 말하기
4️⃣  완료 — OpenClaw이 클론, 설치, 설정, 실행, 결과 반환까지 자동 처리
```

**그게 전부입니다.** OpenClaw이 `git clone`, `pip install`, 설정 구성, 파이프라인 실행을 자동으로 처리합니다. 채팅만 하면 됩니다.

<details>
<summary>💡 내부 동작 과정</summary>

1. OpenClaw이 `RESEARCHCLAW_AGENTS.md`를 읽고 → 연구 오케스트레이터 역할을 학습
2. OpenClaw이 `README.md`를 읽고 → 설치 및 파이프라인 구조를 이해
3. OpenClaw이 `config.researchclaw.example.yaml`을 → `config.yaml`로 복사
4. LLM API 키를 요청 (또는 환경 변수를 사용)
5. `pip install -e .` + `researchclaw run --topic "..." --auto-approve` 실행
6. 논문, LaTeX, 실험, 인용을 반환

</details>

### 🔌 OpenClaw 브릿지 (고급)

더 깊은 통합을 위해 AutoResearchClaw는 6가지 선택적 기능을 갖춘 **브릿지 어댑터 시스템**을 포함합니다:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ 예약된 연구 실행
  use_message: true           # 💬 진행 상황 알림 (Discord/Slack/Telegram)
  use_memory: true            # 🧠 세션 간 지식 영속성
  use_sessions_spawn: true    # 🔀 동시 단계를 위한 병렬 서브세션 생성
  use_web_fetch: true         # 🌐 문헌 검토 중 실시간 웹 검색
  use_browser: false          # 🖥️ 브라우저 기반 논문 수집
```

각 플래그는 타입이 지정된 어댑터 프로토콜을 활성화합니다. OpenClaw이 이러한 기능을 제공하면 어댑터가 코드 변경 없이 이를 소비합니다. 전체 세부 사항은 [`integration-guide.md`](integration-guide.md)를 참조하세요.

### 🛠️ 기타 실행 방법

| 방법 | 사용법 |
|------|--------|
| **독립형 CLI** | `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | `RESEARCHCLAW_CLAUDE.md`를 읽음 — *"Run research on [주제]"*라고 말하기 |
| **OpenCode** | `.claude/skills/`를 읽음 — 동일한 자연어 인터페이스 |
| **기타 AI CLI** | `RESEARCHCLAW_AGENTS.md`를 컨텍스트로 제공 → 에이전트가 자동 부트스트랩 |

---

## 🔬 파이프라인: 23단계, 8단계

```
단계 A: 연구 범위 설정            단계 E: 실험 실행
  1. TOPIC_INIT                      12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE               13. ITERATIVE_REFINE  ← 자가 복구

단계 B: 문헌 탐색                단계 F: 분석 및 의사결정
  3. SEARCH_STRATEGY                 14. RESULT_ANALYSIS    ← 멀티 에이전트
  4. LITERATURE_COLLECT  ← 실제 API  15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [게이트]
  6. KNOWLEDGE_EXTRACT               단계 G: 논문 작성
                                     16. PAPER_OUTLINE
단계 C: 지식 종합                   17. PAPER_DRAFT
  7. SYNTHESIS                       18. PEER_REVIEW        ← 증거 확인
  8. HYPOTHESIS_GEN    ← 토론        19. PAPER_REVISION

단계 D: 실험 설계               단계 H: 최종화
  9. EXPERIMENT_DESIGN   [게이트]      20. QUALITY_GATE      [게이트]
 10. CODE_GENERATION                 21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING               22. EXPORT_PUBLISH     ← LaTeX
                                     23. CITATION_VERIFY    ← 관련성 확인
```

> **게이트 단계** (5, 9, 20)는 사람의 승인을 기다리거나 `--auto-approve`로 자동 승인합니다. 거부 시 파이프라인이 롤백됩니다.

> **의사결정 루프**: 15단계에서 REFINE (→ 13단계) 또는 PIVOT (→ 8단계)을 트리거할 수 있으며, 산출물 버전 관리가 자동으로 이루어집니다.

<details>
<summary>📋 각 단계별 상세 설명</summary>

| 단계 | 수행 내용 |
|------|----------|
| **A: 범위 설정** | LLM이 주제를 연구 질문이 포함된 구조화된 문제 트리로 분해 |
| **A+: 하드웨어** | GPU 자동 감지 (NVIDIA CUDA / Apple MPS / CPU 전용), 로컬 하드웨어가 제한적인 경우 경고, 이에 맞게 코드 생성 적응 |
| **B: 문헌** | 다중 소스 검색 (arXiv 우선, 이후 Semantic Scholar)으로 실제 논문 수집, 관련성별 선별, 지식 카드 추출 |
| **C: 종합** | 연구 결과 클러스터링, 연구 갭 식별, 멀티 에이전트 토론을 통한 검증 가능한 가설 생성 |
| **D: 설계** | 실험 계획 설계, 하드웨어 인식 실행 가능 Python 생성 (GPU 등급 → 패키지 선택), 리소스 요구 사항 추정 |
| **E: 실행** | 샌드박스에서 실험 실행, NaN/Inf 및 런타임 버그 감지, LLM을 통한 표적화된 코드 자가 복구 |
| **F: 분석** | 결과에 대한 멀티 에이전트 분석; 근거가 포함된 자율 PROCEED / REFINE / PIVOT 결정 |
| **G: 작성** | 개요 → 섹션별 작성 (5,000-6,500단어) → 피어 리뷰 (방법론-증거 일관성 포함) → 길이 제한 적용 수정 |
| **H: 최종화** | 품질 게이트, 지식 아카이빙, 학회 템플릿 포함 LaTeX 내보내기, 인용 무결성 + 관련성 검증 |

</details>

---

## 🚀 빠른 시작

### 사전 요구 사항

- 🐍 Python 3.11+
- 🔑 OpenAI 호환 LLM API 엔드포인트 (GPT-4o, GPT-5.x 또는 기타 호환 제공자)

### 설치

```bash
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 설정

```bash
cp config.researchclaw.example.yaml config.arc.yaml
```

<details>
<summary>📝 최소 필수 설정</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"     # OpenAI 호환 엔드포인트
  api_key_env: "OPENAI_API_KEY"              # API 키가 포함된 환경 변수 이름
  primary_model: "gpt-4o"                    # 엔드포인트가 지원하는 모든 모델
  fallback_models: ["gpt-4o-mini"]
  s2_api_key: ""                             # 선택 사항: 더 높은 속도 제한을 위한 Semantic Scholar API 키

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

### 실행

```bash
# API 키 설정
export OPENAI_API_KEY="sk-..."

# 🚀 전체 파이프라인 실행
researchclaw run --config config.arc.yaml --auto-approve

# 🎯 주제를 인라인으로 지정
researchclaw run --config config.arc.yaml --topic "Transformer attention for time series" --auto-approve

# ✅ 설정 검증
researchclaw validate --config config.arc.yaml

# ⏩ 특정 단계부터 재개
researchclaw run --config config.arc.yaml --from-stage PAPER_OUTLINE --auto-approve
```

출력 → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` 각 단계별 하위 디렉토리 포함.

모든 사용자 대상 결과물은 자동으로 하나의 **`deliverables/`** 폴더에 수집됩니다:

```
artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/
├── paper_final.md             # 최종 논문 (Markdown)
├── paper.tex                  # 학회 제출용 LaTeX
├── references.bib             # 검증된 BibTeX 참고문헌 (자동 정리)
├── neurips_2025.sty           # 학회 스타일 파일 (자동 선택)
├── code/                      # 실험 코드 + requirements.txt
├── verification_report.json   # 인용 무결성 보고서
├── charts/                    # 결과 시각화 (조건 비교, 오차 막대)
└── manifest.json              # 메타데이터가 포함된 결과물 인덱스
```

`deliverables/` 폴더는 **바로 컴파일 가능**합니다 — 학회 `.sty` 및 `.bst` 파일이 포함되어 있어 `pdflatex` + `bibtex`로 `paper.tex`를 바로 컴파일하거나 추가 다운로드 없이 Overleaf에 업로드할 수 있습니다.

---

## ✨ 주요 기능

### 📚 다중 소스 문헌 검색

4단계에서 **실제 학술 API**를 쿼리합니다 — LLM이 생성한 가짜 논문이 아닙니다. Semantic Scholar 속도 제한을 피하기 위해 **arXiv 우선** 전략을 사용합니다.

- **arXiv API** (주요) — 실제 arXiv ID와 메타데이터가 포함된 프리프린트, 속도 제한 없음
- **Semantic Scholar API** (보조) — 제목, 초록, 학회, 인용 수, DOI가 포함된 실제 논문
- **쿼리 확장** — 포괄적인 커버리지(30-60개 참고문헌)를 위해 더 넓은 쿼리를 자동 생성 (서베이, 벤치마크, 비교 변형)
- **자동 중복 제거** — DOI → arXiv ID → 퍼지 제목 매칭
- **BibTeX 생성** — 실제 메타데이터가 포함된 유효한 `@article{cite_key, ...}` 항목
- **3상태 서킷 브레이커** — CLOSED → OPEN → HALF_OPEN 복구, 지수 백오프 쿨다운 (영구 비활성화 없음)
- **단계적 성능 저하** — S2 실패 시 arXiv 결과를 차단하지 않음; 모든 API 실패 시 LLM 보강 결과로 폴백

```python
from researchclaw.literature import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

### 🔍 인용 검증 (23단계)

논문 작성 후, 23단계에서 모든 참고문헌의 무결성과 관련성을 **팩트체크**합니다:

| 계층 | 방법 | 검증 내용 |
|------|------|----------|
| L1 | arXiv API `id_list` | arXiv ID가 있는 논문 — ID의 실제 존재 여부 확인 |
| L2 | CrossRef `/works/{doi}` + DataCite 폴백 | DOI가 있는 논문 — DOI 해석 및 제목 일치 확인 (DataCite는 arXiv `10.48550` DOI 처리) |
| L3 | Semantic Scholar + arXiv 제목 검색 | 나머지 모든 논문 — 퍼지 제목 매칭 (유사도 ≥0.80) |
| L4 | LLM 관련성 점수 | 모든 검증된 참고문헌 — 연구 주제와의 관련성 평가 |

각 참고문헌 → **VERIFIED** ✅ · **SUSPICIOUS** ⚠️ · **HALLUCINATED** ❌ · **SKIPPED** ⏭️ · **LOW_RELEVANCE** 📉

**자동 정리**: 환각된 인용은 논문 텍스트에서 조용히 제거됩니다 (`[HALLUCINATED]` 태그 없음). 인용되지 않은 참고문헌 항목은 정리됩니다. 최종 `references.bib`에는 검증되고 인용된 참고문헌만 포함됩니다.

### 🖥️ 하드웨어 인식 실행

1단계에서 로컬 GPU 기능을 자동 감지하고 전체 파이프라인을 적응시킵니다:

| 등급 | 감지 | 동작 |
|------|------|------|
| **고성능** | NVIDIA GPU, 8 GB VRAM 이상 | 전체 PyTorch/GPU 코드 생성, torch 미설치 시 자동 설치 |
| **제한적** | NVIDIA 8 GB 미만 또는 Apple MPS | 경량 실험 (1M 파라미터 미만, 20 에포크 이하), 사용자 경고 |
| **CPU 전용** | GPU 미감지 | NumPy/sklearn만 사용, torch import 없음, 원격 GPU 추천 사용자 경고 |

하드웨어 프로필은 `stage-01/hardware_profile.json`에 저장되며 코드 생성, 샌드박스 임포트, 프롬프트 제약에 영향을 줍니다.

### 🧪 샌드박스 실험 실행

- **코드 검증** — AST 파싱, 임포트 화이트리스트, 샌드박스 외부 파일 I/O 금지
- **컴퓨팅 예산 보호** — 시간 예산 (설정 가능, 기본값 600초)을 코드 생성 프롬프트에 주입; LLM은 샌드박스 타임아웃 내에서 완료되는 실험을 설계해야 함
- **실험 하네스** — 불변 `experiment_harness.py`가 샌드박스에 주입되며 `should_stop()` 시간 가드, `report_metric()` NaN/Inf 거부, `finalize()` 결과 기록 기능 포함 (karpathy/autoresearch의 불변 평가 패턴에서 영감)
- **구조화된 출력** — 실험이 타입이 지정된 메트릭이 포함된 `results.json`을 생성 (단순 stdout 파싱이 아님)
- **스마트 메트릭 파싱** — 키워드 감지(`is_metric_name()`)를 사용하여 로그 라인을 메트릭에서 필터링
- **NaN/발산 즉시 실패** — 메트릭에서 NaN/Inf 값 필터링; 발산하는 손실 (>100) 감지 및 표시
- **수렴 기준 적용** — 생성된 코드에 고정 반복 횟수가 아닌 조기 종료 기준이 반드시 포함되어야 함
- **런타임 버그 감지** — NaN/Inf 메트릭 및 stderr 경고 (0으로 나눔, 오버플로우)가 자동 감지
- **자가 복구 수리** — 런타임 문제를 LLM에 피드백하여 근본 원인을 수정하는 표적화된 진단 (임시방편 try/except가 아님)
- **반복적 개선** — 13단계에서 결과를 분석하고 개선된 코드/매개변수로 재실행 (최대 10회 반복, 타임아웃 인식 프롬프트)
- **부분 결과 캡처** — 캡처된 메트릭이 있는 타임아웃 실험은 "failed" 대신 "partial" 상태를 받아 사용 가능한 데이터를 보존
- **주제-실험 정렬** — LLM 기반 사후 생성 검사로 실험 코드가 실제로 명시된 연구 주제를 테스트하는지 확인

### 📝 학회 수준 논문 작성

작성 파이프라인은 NeurIPS/ICML/ICLR 기준 (9+ 페이지, 5,000-6,500단어)을 목표로 합니다:

- **데이터 무결성 적용** — 실험이 메트릭을 생성하지 않으면 논문 작성이 강제 차단됨 (LLM의 결과 날조 방지); 날조 방지 지침이 초안 및 수정 프롬프트에 모두 주입
- **학회 수준 프롬프트** — 시스템 프롬프트에 채택된 논문 분석의 핵심 원칙 포함: 참신성, 서사, 강력한 기준선, 절제, 정직성, 재현성; 일반적인 거절 사유 표시
- **제목 및 프레이밍 가이드라인** — 참신성 신호, 기억에 남는 제목 테스트, 5문장 초록 구조, 일반적인 제목 감지 및 재생성
- **섹션별 작성** — 3회 순차 LLM 호출 (서론+관련 연구 → 방법론+실험 → 결과+결론)로 출력 잘림 방지
- **섹션별 목표 단어 수** — 초록 (150-250), 서론 (800-1000), 관련 연구 (600-800), 방법론 (1000-1500), 실험 (800-1200), 결과 (600-800), 토론 (400-600)
- **수정 길이 제한** — 수정된 논문이 초안보다 짧으면 더 강한 규칙으로 자동 재시도; 필요시 초안+주석으로 폴백
- **면책 조항 방지 적용** — "계산 제약으로 인해"를 최대 1회로 제한; 수정 프롬프트에서 반복되는 단서를 적극 제거
- **통계적 엄밀성** — 결과 표에 신뢰 구간, p-값, 효과 크기 필요; 결함 있는 절제 실험은 표시 및 주장에서 제외
- **학회 평가 기준 피어 리뷰** — 리뷰어가 NeurIPS/ICML 기준에 따라 1-10점 평가 (참신성, 기준선, 절제, 주장 대 증거, 한계점)

### 📐 학회 템플릿 전환

```yaml
export:
  target_conference: "neurips_2025"   # 또는 "iclr_2026" 또는 "icml_2026"
```

| 학회 | 스타일 패키지 | 컬럼 |
|------|-------------|------|
| NeurIPS 2025 | `neurips_2025` | 1 |
| ICLR 2026 | `iclr2026_conference` | 1 |
| ICML 2026 | `icml2026` | 2 |
| NeurIPS 2024 | `neurips_2024` | 1 |
| ICLR 2025 | `iclr2025_conference` | 1 |
| ICML 2025 | `icml2025` | 2 |

Markdown → LaTeX 변환기는 다음을 처리합니다: 섹션 제목 (자동 번호 매김 중복 제거), 인라인/디스플레이 수학, 굵게/기울임, 목록, 표 (`\caption`/`\label` 포함), 그림 (`\includegraphics`), 코드 블록 (유니코드 안전), 교차 참조, `\cite{}` 참고문헌.

### 🚦 품질 게이트

| 게이트 | 단계 | 거부 시 → 롤백 대상 |
|--------|------|---------------------|
| 문헌 선별 | 5 | 문헌 재수집 (4단계) |
| 실험 설계 | 9 | 가설 재생성 (8단계) |
| 품질 게이트 | 20 | 개요부터 논문 재작성 (16단계) |

`--auto-approve`를 사용하여 모든 게이트를 건너뛰거나, `security.hitl_required_stages`에서 특정 단계를 설정하세요.

---

## ⚙️ 설정 참고서

<details>
<summary>전체 설정 참고서 펼치기</summary>

```yaml
# === 프로젝트 ===
project:
  name: "my-research"              # 프로젝트 식별자
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === 연구 ===
research:
  topic: "..."                     # 연구 주제 (필수)
  domains: ["ml", "nlp"]           # 문헌 검색용 연구 분야
  daily_paper_count: 8             # 검색 쿼리당 목표 논문 수
  quality_threshold: 4.0           # 논문 최소 품질 점수

# === 런타임 ===
runtime:
  timezone: "America/New_York"     # 타임스탬프용
  max_parallel_tasks: 3            # 동시 실험 제한
  approval_timeout_hours: 12       # 게이트 단계 타임아웃
  retry_limit: 2                   # 단계 실패 시 재시도 횟수

# === LLM ===
llm:
  provider: "openai-compatible"    # 제공자 유형
  base_url: "https://..."          # API 엔드포인트 (필수)
  api_key_env: "OPENAI_API_KEY"    # API 키용 환경 변수 (필수)
  api_key: ""                      # 또는 키를 직접 입력
  primary_model: "gpt-4o"          # 기본 모델
  fallback_models: ["gpt-4o-mini"] # 폴백 체인
  s2_api_key: ""                   # Semantic Scholar API 키 (선택, 더 높은 속도 제한)

# === 실험 ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 600             # 실행당 최대 실행 시간 (기본값: 600초)
  max_iterations: 10               # 최대 최적화 반복 횟수
  metric_key: "val_loss"           # 기본 메트릭 이름
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
    auto_install_deps: true        # import 자동 감지 → requirements.txt
  ssh_remote:
    host: ""                       # GPU 서버 호스트명
    gpu_ids: []                    # 사용 가능한 GPU ID
    remote_workdir: "/tmp/researchclaw_experiments"

# === 내보내기 ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === 프롬프트 ===
prompts:
  custom_file: ""                  # 사용자 정의 프롬프트 YAML 경로 (비어 있으면 기본값)

# === 보안 ===
security:
  hitl_required_stages: [5, 9, 20] # 사람의 승인이 필요한 단계
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === 지식 기반 ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === 알림 ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === OpenClaw 브릿지 ===
openclaw_bridge:
  use_cron: false                  # 예약된 연구 실행
  use_message: false               # 진행 상황 알림
  use_memory: false                # 세션 간 지식 영속성
  use_sessions_spawn: false        # 병렬 서브세션 생성
  use_web_fetch: false             # 실시간 웹 검색
  use_browser: false               # 브라우저 기반 논문 수집
```

</details>

---

## 🙏 감사의 말

다음 프로젝트에서 영감을 받았습니다:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — 자동화 연구의 선구자
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — 엔드투엔드 연구 자동화
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — 완전 자동 연구 시스템

---

## 📄 라이선스

MIT — 자세한 내용은 [LICENSE](../LICENSE)를 참조하세요.

<p align="center">
  <sub>Built with 🦞 by the AutoResearchClaw team</sub>
</p>
