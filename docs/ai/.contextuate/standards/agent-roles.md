# Agent Roles (Team Roster)

> **Purpose:** Authoritative registry of all specialist agents. Reference this when deciding which agent to delegate to.

All agents have dedicated definition files in `docs/ai/agents/` containing their full responsibilities and context.

---

## Specialist Agents

| Agent | Model | Domain | When to Delegate |
|-------|-------|--------|------------------|
| [**PYTHIA**](../../agents/pythia.md) | Opus | Planning/Research | Pre-implementation research, ideation, specification (use for complex/unfamiliar work requiring deep planning) |
| [**ARCHON**](../../agents/archon.md) | Opus | Orchestration | Complex multi-agent tasks requiring coordination (sub-orchestration) |
| [**AEGIS**](../../agents/aegis.md) | Sonnet | Quality/Review | Code review, best practices, quality assurance |
| [**ATLAS**](../../agents/atlas.md) | Sonnet | Navigation | Codebase exploration, file search, pattern discovery |
| [**CANVAS**](../../agents/canvas.md) | Sonnet | Frontend/UX | UI components, state management, theming, design systems |
| [**CHRONICLE**](../../agents/chronicle.md) | Sonnet | Documentation | Comments, markdown, changelogs |
| [**CHRONOS**](../../agents/chronos.md) | Sonnet | Data/State | Database administration, caching, state management, performance |
| [**CIPHER**](../../agents/cipher.md) | Sonnet | Data Transformation | Data utilities, formatting, transformations |
| [**CRUCIBLE**](../../agents/crucible.md) | Sonnet | Testing | Test writing, execution, coverage |
| [**ECHO**](../../agents/echo.md) | Sonnet | Frontend JS | JavaScript, UI interactions, client-side |
| [**FORGE**](../../agents/forge.md) | Sonnet | Infrastructure | Scaffolding, deployment, DevOps, tooling |
| [**LEDGER**](../../agents/ledger.md) | Sonnet | Task Management | Multi-step tasks, session continuity, progress tracking |
| [**MERIDIAN**](../../agents/meridian.md) | Sonnet | Schema/Migrations | Database schema changes, migrations |
| [**NEXUS**](../../agents/nexus.md) | Sonnet | Backend/Services | Service classes, external APIs, business logic |
| [**THOTH**](../../agents/thoth.md) | Opus | Database/Queries | Complex database queries, schema design, data operations |
| [**SCRIBE**](../../agents/scribe.md) | Sonnet | Documentation | API docs, technical writing, user guides |
| [**SENTINEL**](../../agents/sentinel.md) | Opus | Security | Validation, permissions, sanitization, security analysis |
| [**UNITY**](../../agents/unity.md) | Sonnet | Version Control | Git merges, conflict resolution, release management |
| [**VOX**](../../agents/vox.md) | Sonnet | Media/Communications | WebRTC, streaming, audio/video processing |
| [**WEAVER**](../../agents/weaver.md) | Sonnet | Controllers/Views | Page actions, view rendering, permissions |

---

## Quick Reference by Task Type

### Planning & Research
- **PYTHIA** - Use BEFORE implementation for complex/unfamiliar work

### Code Implementation
- **NEXUS** - Backend services, APIs, business logic
- **ECHO** - Frontend JavaScript, client-side
- **CANVAS** - UI components, design systems
- **WEAVER** - Controllers, views, page actions
- **CIPHER** - Data transformations, utilities

### Data & Database
- **THOTH** - Complex queries, schema design
- **CHRONOS** - Database admin, caching, performance
- **MERIDIAN** - Schema migrations

### Quality & Security
- **AEGIS** - Code review, best practices
- **CRUCIBLE** - Testing, coverage
- **SENTINEL** - Security, validation, permissions

### Infrastructure & Ops
- **FORGE** - Scaffolding, DevOps, deployment
- **UNITY** - Git, version control, releases

### Documentation & Communication
- **CHRONICLE** - Comments, changelogs
- **SCRIBE** - API docs, technical writing
- **VOX** - Media, WebRTC, streaming

### Coordination
- **ARCHON** - Multi-agent orchestration
- **LEDGER** - Task tracking, progress management
- **ATLAS** - Codebase navigation, exploration

---

## Delegation Guidelines

1. **Check this roster** before delegating to ensure you pick the right specialist
2. **Provide context** - Include relevant files, patterns, and constraints
3. **Be specific** - Clear objectives get better results
4. **Consider dependencies** - Some tasks need sequential agents (e.g., THOTH before NEXUS for data-heavy APIs)

For full orchestration protocols, see [ARCHON](../../agents/archon.md).
