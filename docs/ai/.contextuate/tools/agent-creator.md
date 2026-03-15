# Agent Creator Tool

> **Type:** AI Tool Guide
> **Purpose:** Create new AI agent definitions following Contextuate standards

---

## When to Use This Tool

Use this tool when:
- User requests a new agent for a specific domain
- Project needs specialized AI expertise documented
- Onboarding AI to a new area of the codebase

---

## Input

**Required:**
- Domain/expertise area for the agent
- Primary responsibilities

**Optional:**
- Existing documentation to reference
- Specific file patterns the agent covers
- Related agents for delegation

---

## Process

### Step 1: Determine Agent Scope

Answer these questions:
- What domain/expertise does this agent cover?
- What files/patterns are in scope?
- Is there existing documentation to reference?
- Does a quickref need to be created first?

### Step 2: Create Supporting Documentation (if needed)

Before creating the agent, ensure documentation exists:

| Need | Create |
|------|--------|
| Comprehensive docs | `docs/{topic}.md` |
| API/method reference | `docs/ai/quickrefs/{name}.quickref.md` |

Use the [Quickref Generator](quickref.md) if needed.

### Step 3: Create Agent File

**Location:** `docs/ai/agents/{domain}-expert.md`

**Naming conventions:**
- Use lowercase, hyphen-separated
- Be specific: `api-auth-expert` not just `api-expert`
- Pattern: `{domain}-expert.md`

### Step 4: Fill Template

Use the template below, replacing all `{placeholders}`.

### Step 5: Quality Check

Verify:
- [ ] Inherits from base configuration
- [ ] Required context lists only domain-specific docs
- [ ] Core competencies are specific and actionable
- [ ] Includes practical code examples
- [ ] Anti-patterns show real mistakes to avoid
- [ ] Decision framework helps with common choices

---

## Output Template

```markdown
# {Name} Expert Agent

> **Inherits:** [../.contextuate/agents/base.md](../.contextuate/agents/base.md)
> **Role:** {One-line description of expertise}
> **Domain:** `{file patterns covered}`

---

## Agent Identity

You are an expert in {domain description}. Your role is to:

1. **{Primary responsibility}**
2. **{Secondary responsibility}**
3. **{Tertiary responsibility}**

---

## Required Context

In addition to base agent context, you MUST read:

1. **[{Primary Doc}]({path})** - {Why needed}
2. **[{Secondary Doc}]({path})** - {Why needed} *(if applicable)*

---

## Core Competencies

### {Competency Area 1}
- {Specific skill}
- {Specific skill}

### {Competency Area 2}
- {Specific skill}
- {Specific skill}

---

## Decision Framework

### {Key Decision Type}

{Decision tree or guidelines}

---

## Common Patterns

### {Pattern Name}

\`\`\`{language}
{code example}
\`\`\`

---

## Anti-Patterns

\`\`\`{language}
// BAD: {description}
{bad pattern}

// GOOD: {description}
{good pattern}
\`\`\`

---

## Quick Reference

| Task | Approach |
|------|----------|
| {common task} | {solution/method} |
| {common task} | {solution/method} |
```

---

## Example

**Request:** "Create an agent for database operations"

### Step 1: Assess Scope
- Domain: Database queries and schema management
- Files: `*.sql`, `migrations/`, database service files
- Existing docs: Check `docs/` for database documentation

### Step 2: Create Supporting Docs (if needed)
```
docs/
└── database.md              # If comprehensive docs needed
docs/ai/quickrefs/
└── database.quickref.md     # If large API needs summary
```

### Step 3: Create Agent File

File: `docs/ai/agents/database-expert.md`

```markdown
# Database Expert Agent

> **Inherits:** [../.contextuate/agents/base.md](../.contextuate/agents/base.md)
> **Role:** Expert in database operations, queries, and schema design
> **Domain:** `*.sql`, `migrations/`, `**/db/**`

## Agent Identity

You are an expert in database operations. Your role is to:

1. **Write efficient queries** following project conventions
2. **Design schemas** that are normalized and performant
3. **Create migrations** that are safe and reversible

## Required Context

1. **[../../database.md](../../database.md)** - Schema and conventions
2. **[../quickrefs/database.quickref.md](../quickrefs/database.quickref.md)** - API reference
```

---

## Agent Relationships

### Inheritance Hierarchy
```
base.md (framework - immutable)
    └── {user-agents}.md (project-specific - in docs/ai/agents/)
```

### Delegation Pattern
Agents should delegate when task requires expertise outside their domain:

```markdown
## Delegation

| Situation | Delegate To |
|-----------|-------------|
| Need API design help | api-expert |
| Need testing help | testing-expert |
```

---

## When to Create Quickrefs

Create a quickref (`docs/ai/quickrefs/{name}.quickref.md`) when:
- Source documentation exceeds ~300 lines
- Methods/APIs are frequently looked up
- Agent needs awareness without full context load

---

## Reporting

After creating an agent, report:

1. Agent file created
2. Supporting docs created (if any)
3. Recommended next steps

```
Agent Created
=============
File: docs/ai/agents/database-expert.md

Supporting docs:
  - Created: docs/ai/quickrefs/database.quickref.md

Next steps:
  - Review and customize the agent file
  - Add project-specific patterns and anti-patterns
```
