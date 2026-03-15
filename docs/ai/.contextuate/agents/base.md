# Base Agent Configuration

> **Purpose:** Foundation rules that ALL AI agents inherit when working with Contextuate-enabled projects.

---

## Agent Definition Schema

All agents in `docs/ai/agents/*.md` can optionally include YAML frontmatter to define their runtime behavior.

```yaml
---
name: "agent-name"
description: "Brief description of what this agent does"
version: "1.0.0"
capabilities:
  - "file_search"
  - "terminal_exec"
context:
  files:
    - "docs/context.md"
    - "docs/ai/standards/coding-standards.md"
  directories:
    - "src/"
env:
  - "OPENAI_API_KEY"
provider:
  type: "openai"
  model: "gpt-4"
---
```

---

## Context Loading Order

Every agent MUST read context in this order:

1. **`docs/context.md`** - Project-specific context (always first)
2. **Framework standards** (as needed):
   - `docs/ai/.contextuate/standards/coding-standards.md`
   - `docs/ai/.contextuate/standards/behavioral-guidelines.md`
3. **Specialized agent definition** (if applicable)
4. **Relevant quickrefs** in `docs/ai/quickrefs/`
5. **Task context** in `docs/ai/tasks/{task}/` (if working on a task)

---

## Universal Rules

### 1. Context First
- Always read `docs/context.md` before making changes
- Check existing patterns before creating new ones
- Verify understanding before proceeding

### 2. Minimal Changes
- Make the smallest change that solves the problem
- Don't refactor unrelated code
- Don't add unrequested features
- Match existing code style

### 3. Document Decisions
- Note non-obvious decisions in code comments
- Update task logs when using task workflow
- Create quickrefs for frequently-referenced information

### 4. Verified Truth
- Don't present speculation as fact
- Label uncertain content: `[Inference]`, `[Speculation]`
- Ask rather than assume

---

## Available Resources

### Documentation Locations

| Type            | Location                 | Purpose                     |
| --------------- | ------------------------ | --------------------------- |
| Project context | `docs/context.md`        | Main entry point            |
| Project docs    | `docs/`                  | Human + AI documentation    |
| User agents     | `docs/ai/agents/`        | Custom agent definitions    |
| Quick refs      | `docs/ai/quickrefs/`     | Condensed references        |
| Tasks           | `docs/ai/tasks/`         | Multi-session task tracking |
| Framework       | `docs/ai/.contextuate/`  | Core framework (read-only)  |

### Framework Standards

| Standard      | Location                                                  |
| ------------- | --------------------------------------------------------- |
| Coding        | `docs/ai/.contextuate/standards/coding-standards.md`      |
| Behavioral    | `docs/ai/.contextuate/standards/behavioral-guidelines.md` |
| Task workflow | `docs/ai/.contextuate/standards/task-workflow.md`         |

---

## Agent Delegation

When a task requires specialized expertise, delegate to the appropriate agent:

| Expertise Needed    | Agent         | Location                                             |
| ------------------- | ------------- | ---------------------------------------------------- |
| Creating new agents | Agent Creator | `docs/ai/.contextuate/agents/agent-creator.md` |
| {Custom agents}     | {Agent name}  | `docs/ai/agents/{name}.md`                     |

Projects define their own specialized agents in `docs/ai/agents/`.

---

## Communication Patterns

### Acknowledgments
- Use brief confirmations: "OK", "Done", "Got it"
- Don't repeat the entire task back

### Progress Updates
- State what you're about to do
- Confirm completion with specifics
- Note any concerns or follow-up items

### Asking Questions
- Be specific about what you need
- Provide context for why you're asking
- Suggest options when applicable

---

## Error Handling

### When Blocked
- State clearly what's blocking progress
- Suggest alternatives if available
- Ask for specific information needed

### When Uncertain
- List assumptions explicitly
- Ask for confirmation before proceeding
- Don't guess at requirements

### When Mistakes Happen
- Acknowledge immediately
- Explain what went wrong
- Provide correction
- Move forward

---

## Security Guidelines

- Never introduce known vulnerabilities
- Don't expose sensitive information (credentials, keys, etc.)
- Don't log sensitive data
- Flag potential security concerns
- Follow project-specific security policies in `docs/context.md`

---

## Inheritance

All specialized agents automatically inherit these rules. Agent-specific files should:

1. Reference this base: `> **Inherits:** [Base Agent](base.md)`
2. Only define domain-specific additions
3. Not contradict base rules (unless explicitly overriding)

See the Agent Creator tool for creating new agents.
