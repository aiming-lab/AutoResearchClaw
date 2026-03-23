<p align="center">
  <img src="../image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>아이디어를 말하다. 논문을 받다. 완전 자동 & 자기 진화.</b></h2>



<p align="center">
  <b><i><font size="5"><a href="#openclaw-통합">OpenClaw</a>에 채팅하세요: "X 연구해줘" → 완료.</font></i></b>
</p>

<p align="center">
  <img src="../image/framework_v2.png" width="100%" alt="AutoResearchClaw Framework">
</p>


<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#테스트"><img src="https://img.shields.io/badge/Tests-1823%20passed-brightgreen?logo=pytest&logoColor=white" alt="1823 Tests Passed"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-통합"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
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
  <a href="showcase/SHOWCASE.md">🏆 논문 쇼케이스</a> · <a href="integration-guide.md">📖 통합 가이드</a> · <a href="https://discord.gg/u4ksqW5P">💬 Discord 커뮤니티</a>
</p>

---

<table>
<tr>
<td width="18%">
<a href="showcase/SHOWCASE.md"><img src="showcase/thumbnails/paper_I_random_matrix-01.png" width="120" alt="Sample Paper"/></a>
</td>
<td valign="middle">
<b>🏆 생성된 논문 쇼케이스</b><br><br>
<b>8개 분야에 걸친 8편의 논문</b> — 수학, 통계, 생물학, 컴퓨팅, NLP, RL, 비전, 견고성 — 인간 개입 없이 완전 자율 생성.<br><br>
<a href="showcase/SHOWCASE.md"><img src="https://img.shields.io/badge/View_Full_Showcase_→-All_8_Papers-d73a49?style=for-the-badge" alt="View Showcase"></a>
</td>
</tr>
</table>

---

> **🧪 테스터를 모집합니다!** 여러분의 연구 아이디어로 — 어떤 분야든 — 파이프라인을 시험해 보시고 [의견을 들려주세요](TESTER_GUIDE.md). 여러분의 피드백이 다음 버전에 직접 반영됩니다. **[→ Testing Guide](TESTER_GUIDE.md)** | **[→ 中文测试指南](TESTER_GUIDE_CN.md)** | **[→ 日本語テストガイド](TESTER_GUIDE_JA.md)**

---

## 🔥 News
- **[03/22/2026]** [v0.3.2](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.2) — **크로스 플랫폼 지원 + 주요 안정성 개선** — ACP 호환 AI 에이전트 백엔드(Claude Code, Codex CLI, Copilot CLI, Gemini CLI, Kimi CLI) 지원 및 OpenClaw 브릿지를 통한 메시징 플랫폼(Discord, Telegram, Lark, WeChat) 지원 추가. 새로운 CLI-agent 코드 생성 백엔드가 Stage 10 및 13을 외부 CLI 에이전트에 위임하며, 예산 제어 및 타임아웃 관리를 지원. 반데이터 조작 시스템(VerifiedRegistry + 실험 진단 및 복구 루프), 100건 이상의 버그 수정, 모듈러 executor 리팩토링, `--resume` 자동 감지, LLM 재시도 강화, 커뮤니티 보고 수정 포함.
- **[03/18/2026]** [v0.3.1](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.1) — **OpenCode Beast Mode + Community Contributions** — New "Beast Mode" routes complex code generation to [OpenCode](https://github.com/anomalyco/opencode) with automatic complexity scoring and graceful fallback. Added Novita AI provider support, thread-safety hardening, improved LLM output parsing robustness, and 20+ bug fixes from community PRs and internal audit.
- **[03/17/2026]** [v0.3.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.3.0) — **MetaClaw Integration** — AutoResearchClaw now supports [MetaClaw](https://github.com/aiming-lab/MetaClaw) cross-run learning: pipeline failures → structured lessons → reusable skills, injected into all 23 stages. **+18.3%** robustness in controlled experiments. Opt-in (`metaclaw_bridge.enabled: true`), fully backward-compatible. See [Integration Guide](#-metaclaw-integration).
- **[03/16/2026]** [v0.2.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.2.0) — Three multi-agent subsystems (CodeAgent, BenchmarkAgent, FigureAgent), hardened Docker sandbox with network-policy-aware execution, 4-round paper quality audit (AI-slop detection, 7-dim review scoring, NeurIPS checklist), and 15+ bug fixes from production runs.
- **[03/15/2026]** [v0.1.0](https://github.com/aiming-lab/AutoResearchClaw/releases/tag/v0.1.0) — We release AutoResearchClaw: a fully autonomous 23-stage research pipeline that turns a single research idea into a conference-ready paper. No human intervention required.

---

## ⚡ 하나의 명령. 하나의 논문.

```bash
pip install -e . && researchclaw setup && researchclaw init && researchclaw run --topic "Your research idea here" --auto-approve
```


---

## 🤔 이것은 무엇인가요?

**당신이 생각하면, AutoResearchClaw가 씁니다.**

연구 주제를 입력하면 — OpenAlex, Semantic Scholar, arXiv의 실제 문헌, 하드웨어 인식 샌드박스 실험 (GPU/MPS/CPU 자동 감지), 통계 분석, 멀티 에이전트 피어 리뷰, NeurIPS/ICML/ICLR 대상 학회 수준 LaTeX를 포함한 완전한 학술 논문을 받을 수 있습니다. 관리가 필요 없습니다. 복사-붙여넣기도 필요 없습니다. 환각된 참고문헌도 없습니다.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>완성된 학술 논문 (서론, 관련 연구, 방법론, 실험, 결과, 결론)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>학회 제출용 LaTeX (NeurIPS / ICLR / ICML 템플릿)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>OpenAlex, Semantic Scholar, arXiv에서 가져온 실제 BibTeX 참고문헌 — 인라인 인용과 일치하도록 자동 정리</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4계층 인용 무결성 + 관련성 검증 (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>생성된 코드 + 샌드박스 결과 + 구조화된 JSON 메트릭</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>오차 막대와 신뢰 구간이 포함된 자동 생성 조건 비교 차트</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>방법론-증거 일관성 검사를 포함한 멀티 에이전트 피어 리뷰</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>각 실행에서 추출된 자기 학습 교훈</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>모든 최종 산출물을 하나의 폴더에 — Overleaf에 바로 컴파일 가능</td></tr>
</table>

파이프라인은 **사람의 개입 없이 처음부터 끝까지 실행**됩니다. 실험이 실패하면 자가 복구합니다. 가설이 성립하지 않으면 방향을 전환합니다. 인용이 가짜면 삭제합니다.

🌍 **어디서든 실행 가능.** AutoResearchClaw는 특정 플랫폼에 종속되지 않습니다. CLI로 독립 실행하거나, [OpenClaw](https://github.com/openclaw/openclaw)에 연결하거나, ACP 호환 AI 에이전트 —— 🤖 Claude Code, 💻 Codex CLI, 🐙 Copilot CLI, ♊ Gemini CLI, 🌙 Kimi CLI 등 —— 와 연동할 수 있습니다. OpenClaw의 메시지 브릿지 덕분에 💬 Discord, ✈️ Telegram, 🐦 Lark(飞书), 💚 WeChat 등 팀이 이미 사용 중인 플랫폼에서 연구를 시작할 수 있습니다. 주제 하나 입력하면 논문 하나 완성 — 어디서 입력하든 상관없습니다.

---

## 🚀 빠른 시작

```bash
# 1. 클론 & 설치
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. 설정 (대화형 — OpenCode Beast Mode 설치, Docker/LaTeX 확인)
researchclaw setup

# 3. 구성
researchclaw init          # 대화형: LLM 제공자 선택, config.arc.yaml 생성
# 또는 수동: cp config.researchclaw.example.yaml config.arc.yaml

# 4. 실행
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

출력 → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — 컴파일 가능한 LaTeX, BibTeX, 실험 코드, 차트.

<details>
<summary>📝 최소 필수 설정</summary>

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

## 🧠 차별화 요소

| 기능 | 작동 방식 |
|------|----------|
| **🔄 PIVOT / REFINE 루프** | 15단계에서 자율적으로 결정: PROCEED, REFINE (매개변수 조정), 또는 PIVOT (새 방향). 산출물 자동 버전 관리. |
| **🤖 멀티 에이전트 토론** | 가설 생성, 결과 분석, 피어 리뷰 각각에서 구조화된 다관점 토론을 수행. |
| **🧬 자기 학습** | 각 실행에서 교훈 추출 (의사결정 근거, 런타임 경고, 메트릭 이상), 30일 시간 감쇠. 향후 실행이 과거의 실수에서 학습. |
| **📚 지식 기반** | 각 실행에서 6개 카테고리 (결정, 실험, 발견, 문헌, 질문, 리뷰)에 걸친 구조화된 지식 기반 구축. |
| **🛡️ 센티넬 감시견** | 백그라운드 품질 모니터: NaN/Inf 감지, 논문-증거 일관성, 인용 관련성 점수, 날조 방지 가드. |

---

## 🦞 OpenClaw 통합

<table>
<tr>

**AutoResearchClaw는 [OpenClaw](https://github.com/openclaw/openclaw) 호환 서비스입니다.** OpenClaw에 설치하고 단일 메시지로 자율 연구를 시작하거나 — CLI, Claude Code 또는 기타 AI 코딩 어시스턴트를 통해 독립적으로 사용하세요.

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

### ACP (Agent Client Protocol)

AutoResearchClaw는 **모든 ACP 호환 코딩 에이전트**를 LLM 백엔드로 사용할 수 있습니다 — API 키가 필요 없습니다. 에이전트는 [acpx](https://github.com/openclaw/acpx)를 통해 통신하며, 전체 23개 파이프라인 단계에 걸쳐 단일 영구 세션을 유지합니다.

| 에이전트 | 명령어 | 비고 |
|---------|--------|------|
| Claude Code | `claude` | Anthropic |
| Codex CLI | `codex` | OpenAI |
| Copilot CLI | `gh` | GitHub |
| Gemini CLI | `gemini` | Google |
| OpenCode | `opencode` | SST |
| Kimi CLI | `kimi` | Moonshot |

```yaml
# config.yaml — ACP 예시
llm:
  provider: "acp"
  acp:
    agent: "claude"   # 모든 ACP 호환 에이전트 CLI 명령어
    cwd: "."          # 에이전트의 작업 디렉토리
  # base_url이나 api_key 불필요 — 에이전트가 자체 인증을 처리합니다.
```

```bash
# 바로 실행 — 에이전트가 자체 자격 증명 사용
researchclaw run --config config.yaml --topic "Your research idea" --auto-approve
```

### 🛠️ 기타 실행 방법

| 방법 | 사용법 |
|------|--------|
| **독립형 CLI** | `researchclaw setup` → `researchclaw init` → `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | `RESEARCHCLAW_CLAUDE.md`를 읽음 — *"Run research on [주제]"*라고 말하기 |
| **Copilot CLI** | `researchclaw run --topic "..."` 에 `llm.acp.agent: "gh"` 사용 |
| **OpenCode** | `.claude/skills/`를 읽음 — 동일한 자연어 인터페이스 |
| **기타 AI CLI** | `RESEARCHCLAW_AGENTS.md`를 컨텍스트로 제공 → 에이전트가 자동 부트스트랩 |

---

## 🔬 파이프라인: 23단계, 8페이즈

```
페이즈 A: 연구 범위 설정            페이즈 E: 실험 실행
  1. TOPIC_INIT                      12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE               13. ITERATIVE_REFINE  ← 자가 복구

페이즈 B: 문헌 탐색                페이즈 F: 분석 및 의사결정
  3. SEARCH_STRATEGY                 14. RESULT_ANALYSIS    ← 멀티 에이전트
  4. LITERATURE_COLLECT  ← 실제 API  15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [게이트]
  6. KNOWLEDGE_EXTRACT               페이즈 G: 논문 작성
                                     16. PAPER_OUTLINE
페이즈 C: 지식 종합                   17. PAPER_DRAFT
  7. SYNTHESIS                       18. PEER_REVIEW        ← 증거 확인
  8. HYPOTHESIS_GEN    ← 토론        19. PAPER_REVISION

페이즈 D: 실험 설계               페이즈 H: 최종화
  9. EXPERIMENT_DESIGN   [게이트]      20. QUALITY_GATE      [게이트]
 10. CODE_GENERATION                 21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING               22. EXPORT_PUBLISH     ← LaTeX
                                     23. CITATION_VERIFY    ← 관련성 확인
```

> **게이트 단계** (5, 9, 20)는 사람의 승인을 기다리거나 `--auto-approve`로 자동 승인합니다. 거부 시 파이프라인이 롤백됩니다.

> **의사결정 루프**: 15단계에서 REFINE (→ 13단계) 또는 PIVOT (→ 8단계)을 트리거할 수 있으며, 산출물 버전 관리가 자동으로 이루어집니다.

<details>
<summary>📋 각 페이즈별 상세 설명</summary>

| 페이즈 | 수행 내용 |
|--------|----------|
| **A: 범위 설정** | LLM이 주제를 연구 질문이 포함된 구조화된 문제 트리로 분해 |
| **A+: 하드웨어** | GPU 자동 감지 (NVIDIA CUDA / Apple MPS / CPU 전용), 로컬 하드웨어가 제한적인 경우 경고, 이에 맞게 코드 생성 적응 |
| **B: 문헌** | 다중 소스 검색 (OpenAlex → Semantic Scholar → arXiv)으로 실제 논문 수집, 관련성별 선별, 지식 카드 추출 |
| **C: 종합** | 연구 결과 클러스터링, 연구 갭 식별, 멀티 에이전트 토론을 통한 검증 가능한 가설 생성 |
| **D: 설계** | 실험 계획 설계, 하드웨어 인식 실행 가능 Python 생성 (GPU 등급 → 패키지 선택), 리소스 요구 사항 추정 |
| **E: 실행** | 샌드박스에서 실험 실행, NaN/Inf 및 런타임 버그 감지, LLM을 통한 표적화된 코드 자가 복구 |
| **F: 분석** | 결과에 대한 멀티 에이전트 분석; 근거가 포함된 자율 PROCEED / REFINE / PIVOT 결정 |
| **G: 작성** | 개요 → 섹션별 작성 (5,000-6,500단어) → 피어 리뷰 (방법론-증거 일관성 포함) → 길이 제한 적용 수정 |
| **H: 최종화** | 품질 게이트, 지식 아카이빙, 학회 템플릿 포함 LaTeX 내보내기, 인용 무결성 + 관련성 검증 |

</details>

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **📚 다중 소스 문헌** | OpenAlex, Semantic Scholar, arXiv에서 실제 논문 — 쿼리 확장, 중복 제거, 3상태 서킷 브레이커와 단계적 성능 저하 |
| **🔍 4계층 인용 검증** | arXiv ID 확인 → CrossRef/DataCite DOI → Semantic Scholar 제목 매칭 → LLM 관련성 점수. 환각된 참고문헌 자동 삭제. |
| **🖥️ 하드웨어 인식 실행** | GPU (NVIDIA CUDA / Apple MPS / CPU 전용) 자동 감지, 이에 맞게 코드 생성, import, 실험 규모 적응 |
| **🦾 OpenCode Beast Mode** | 복잡한 실험을 [OpenCode](https://github.com/anomalyco/opencode)로 자동 라우팅 — 커스텀 아키텍처, 학습 루프, 절제 연구가 포함된 다중 파일 프로젝트 생성. `researchclaw setup`으로 설치. |
| **🧪 샌드박스 실험** | AST 검증 코드, 불변 하네스, NaN/Inf 즉시 실패, 자가 복구, 반복적 개선 (최대 10라운드), 부분 결과 캡처 |
| **📝 학회 수준 작성** | NeurIPS/ICML/ICLR 템플릿, 섹션별 작성 (5,000-6,500단어), 날조 방지 가드, 수정 길이 제한, 면책 조항 방지 적용 |
| **📐 템플릿 전환** | `neurips_2025`, `iclr_2026`, `icml_2026` — Markdown → LaTeX (수학, 표, 그림, 교차 참조, `\cite{}` 포함) |
| **🚦 품질 게이트** | 3개의 Human-in-the-loop 게이트 (단계 5, 9, 20), 롤백 지원. `--auto-approve`로 건너뛰기. |

---

## 🧠 MetaClaw 통합

**AutoResearchClaw + [MetaClaw](https://github.com/aiming-lab/MetaClaw) = 모든 실행에서 학습하는 파이프라인.**

MetaClaw는 AutoResearchClaw에 **교차 실행 지식 전이**를 추가합니다. 활성화되면 파이프라인이 실패와 경고에서 자동으로 교훈을 추출하고, 이를 재사용 가능한 스킬로 변환하여 후속 실행의 전체 23단계에 주입합니다 — 같은 실수를 다시 반복하지 않습니다.

### 작동 방식

```
Run N executes → failures/warnings captured as Lessons
                      ↓
          MetaClaw Lesson → Skill conversion
                      ↓
          arc-* Skill files stored in ~/.metaclaw/skills/
                      ↓
Run N+1 → build_overlay() injects skills into every LLM prompt
                      ↓
          LLM avoids known pitfalls → higher quality, fewer retries
```

### 빠른 설정

```bash
# 1. MetaClaw 설치 (미설치 시)
pip install metaclaw

# 2. 설정에서 활성화
```

```yaml
# config.arc.yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000"        # MetaClaw 프록시 (선택)
  skills_dir: "~/.metaclaw/skills"          # 스킬 저장 위치
  fallback_url: "https://api.openai.com/v1" # 직접 LLM 폴백
  fallback_api_key: ""                      # 폴백 URL의 API 키
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # warning + error 변환
    max_skills_per_run: 3
```

```bash
# 3. 평소대로 실행 — MetaClaw가 투명하게 작동
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve
```

각 실행 후 `~/.metaclaw/skills/arc-*/SKILL.md`를 확인하여 파이프라인이 학습한 스킬을 확인하세요.

### 실험 결과

대조 A/B 실험 (동일 주제, 동일 LLM, 동일 설정):

| 메트릭 | 기준선 | MetaClaw 사용 시 | 개선 |
|--------|--------|-----------------|------|
| 단계 재시도율 | 10.5% | 7.9% | **-24.8%** |
| Refine 사이클 수 | 2.0 | 1.2 | **-40.0%** |
| 파이프라인 단계 완료율 | 18/19 | 19/19 | **+5.3%** |
| 전체 견고성 점수 (종합) | 0.714 | 0.845 | **+18.3%** |

> 종합 견고성 점수는 단계 완료율 (40%), 재시도 감소 (30%), Refine 사이클 효율성 (30%)의 가중 평균입니다.

### 하위 호환성

- **기본값: 꺼짐.** `metaclaw_bridge`가 없거나 `enabled: false`이면 파이프라인은 이전과 정확히 동일하게 동작합니다.
- **새로운 종속성 없음.** MetaClaw는 선택 사항입니다 — 핵심 파이프라인은 MetaClaw 없이도 동작합니다.
- **기존 1,823개 테스트 모두 통과** (통합 코드 포함).

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
  provider: "openai-compatible"    # openai | openrouter | deepseek | minimax | acp | openai-compatible
  base_url: "https://..."          # API 엔드포인트 (openai-compatible 필수)
  api_key_env: "OPENAI_API_KEY"    # API 키용 환경 변수 (openai-compatible 필수)
  api_key: ""                      # 또는 키를 직접 입력
  primary_model: "gpt-4o"          # 기본 모델
  fallback_models: ["gpt-4o-mini"] # 폴백 체인
  s2_api_key: ""                   # Semantic Scholar API 키 (선택, 더 높은 속도 제한)
  acp:                             # provider: "acp" 인 경우에만 사용
    agent: "claude"                # ACP 에이전트 CLI 명령어 (claude, codex, gemini 등)
    cwd: "."                       # 에이전트의 작업 디렉토리

# === 실험 ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | docker | ssh_remote
  time_budget_sec: 300             # 실행당 최대 실행 시간 (기본값: 300초)
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
  opencode:                          # OpenCode Beast Mode (`researchclaw setup`으로 자동 설치)
    enabled: true                    # 마스터 스위치 (기본값: true)
    auto: true                       # 확인 없이 자동 트리거 (기본값: true)
    complexity_threshold: 0.2        # 0.0-1.0 — 높을수록 복잡한 실험에서만 트리거
    model: ""                        # 모델 오버라이드 (비어있으면 llm.primary_model 사용)
    timeout_sec: 600                 # OpenCode 생성 최대 초
    max_retries: 1                   # 실패 시 재시도 횟수
    workspace_cleanup: true          # 수집 후 임시 작업 공간 제거

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

# === MetaClaw Bridge (선택) ===
metaclaw_bridge:
  enabled: false                   # true로 설정하여 교차 실행 학습 활성화
  proxy_url: "http://localhost:30000"  # MetaClaw 프록시 URL
  skills_dir: "~/.metaclaw/skills" # arc-* 스킬 저장 위치
  fallback_url: ""                 # 프록시 장애 시 직접 LLM 폴백
  fallback_api_key: ""             # 폴백 엔드포인트의 API 키
  lesson_to_skill:
    enabled: true                  # 교훈을 스킬로 자동 변환
    min_severity: "warning"        # 변환할 최소 심각도
    max_skills_per_run: 3          # 파이프라인 실행당 최대 새 스킬 수

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

---

## 📌 인용

AutoResearchClaw가 유용했다면, 아래를 인용해 주세요:

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
  <sub>Built with 🦞 by the AutoResearchClaw team</sub>
</p>
