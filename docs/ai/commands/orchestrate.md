# /orchestrate - Orchestrator Mode Skill

Activate the ARCHON agent for coordinated multi-agent task execution.

## Agent Invocation

**IMPORTANT:** Before proceeding, read and adopt the ARCHON agent persona:

> **Agent Definition:** [../agents/archon.md](../agents/archon.md)

Read the agent file above, then follow its guidelines for task analysis, delegation, and synthesis.

## Usage

```
/orchestrate [task description]
```

## Behavior

When this skill is invoked:
1. Read and adopt the ARCHON agent persona
2. Follow ARCHON's orchestration process and decision tree
3. All detailed rules (LEDGER first, parallel execution, agent preferences) are defined in the agent

## Pre-Orchestration

For complex or unfamiliar work, use `/consult` BEFORE `/orchestrate`:
```
/consult [research/plan topic]  →  produces specification
/orchestrate [implement spec]   →  delegates to specialists
```

## Examples

### Multi-domain feature
```
/orchestrate Add a new API endpoint with database query, validation, and tests
```

### Code review workflow
```
/orchestrate Review the authentication module for security issues and suggest improvements
```

### Documentation task
```
/orchestrate Document the monitor feature architecture and create API reference
```
