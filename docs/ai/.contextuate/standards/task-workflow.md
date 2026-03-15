# Contextuate Task Workflow Standard

> **Purpose:** Standardized structure for multi-session AI tasks that persist across conversations.

---

## When to Use Task Workflow

Use this workflow when:
- Task spans multiple AI sessions
- Multiple phases of work required
- Need to track progress over time
- Collaborating with multiple AI agents/platforms
- **Complex Orchestration**: For multi-agent teams, see [Agentic Workflow](agent-workflow.md)
- Complex project requiring documentation

Don't use for:
- Simple, single-session tasks
- Quick questions or lookups
- Minor code changes

---

## Agentic Orchestration

For complex tasks requiring specialized roles (Archon, Chronos, etc.), this Task Workflow serves as the **Session Governance** layer.

*   **Relationship**: The `task.md` file becomes the "Shared State" (or blackboard) for the Agentic Workflow.
*   **Orchestrator**: The **Archon** agent manages the `task.md` file, while sub-agents focus on their specific files.
*   **Reference**: See **[Agentic Workflow Standard](agent-workflow.md)** for the full dispatch protocol.

---

## Directory Structure

```
docs/ai/tasks/{task-name}/
├── 00-project-scope.md      # Task definition (required)
├── 01-{phase-name}.md       # Phase documentation (as needed)
├── 02-{phase-name}.md
├── ...
├── files/                   # Input files, specs, references
│   └── {relevant-files}
└── logs/                    # Progress tracking
    └── {number}-{date}-{summary}.md
```

### Location
- All tasks live in `docs/ai/tasks/`
- This folder is gitignored (user-specific)
- Each task gets its own subfolder

---

## File Templates

### 00-project-scope.md (Required)

```markdown
# {Task Name}

> **Status:** {Planning | In Progress | Blocked | Complete}
> **Created:** {YYYY-MM-DD}
> **Updated:** {YYYY-MM-DD}

## Objective

{One paragraph describing what this task accomplishes}

## Requirements

- [ ] {Requirement 1}
- [ ] {Requirement 2}
- [ ] {Requirement 3}

## Success Criteria

- {Measurable outcome 1}
- {Measurable outcome 2}

## Phases

| Phase | Description    | Status                    |
| ----- | -------------- | ------------------------- |
| 1     | {Phase 1 name} | {Pending/Active/Complete} |
| 2     | {Phase 2 name} | {Pending/Active/Complete} |

## Constraints

- {Technical constraint}
- {Timeline constraint}
- {Resource constraint}

## Open Questions

- [ ] {Question needing resolution}
```

### Phase Files (01-{name}.md, 02-{name}.md, ...)

```markdown
# Phase {N}: {Phase Name}

> **Status:** {Pending | Active | Complete}
> **Started:** {YYYY-MM-DD}
> **Completed:** {YYYY-MM-DD or blank}

## Objectives

- {Phase objective 1}
- {Phase objective 2}

## Approach

{Description of how this phase will be executed}

## Deliverables

- [ ] {Deliverable 1}
- [ ] {Deliverable 2}

## Notes

{Working notes, decisions made, issues encountered}

## Outcome

{Summary of what was accomplished - filled in when complete}
```

### Log Files (logs/{number}-{date}-{summary}.md)

```markdown
# Log: {Summary}

> **Date:** {YYYY-MM-DD}
> **Session:** {number}
> **Platform:** {Claude Code | Cursor | Copilot | etc.}

## Work Completed

- {Action taken 1}
- {Action taken 2}

## Decisions Made

- {Decision and rationale}

## Blockers/Issues

- {Issue encountered}

## Next Steps

- {What should be done next}
```

---

## Naming Conventions

### Task Folders
- Use lowercase
- Use hyphens for spaces
- Be descriptive but concise
- Examples: `user-auth-refactor`, `api-v2-migration`, `performance-audit`

### Phase Files
- Prefix with two-digit number: `01-`, `02-`, etc.
- Use descriptive names: `01-analysis.md`, `02-design.md`

### Log Files
- Format: `{number}-{YYYY-MM-DD}-{summary}.md`
- Example: `01-2024-01-15-initial-analysis.md`

---

## Workflow Process

### Starting a Task

1. Create task folder: `docs/ai/tasks/{task-name}/`
2. Create `00-project-scope.md` with objectives and requirements
3. Create `files/` folder if you have input materials
4. Create first log entry

### During Work

1. Update phase files as work progresses
2. Create log entries for each significant session
3. Check off requirements as completed
4. Update status in scope file

### Completing a Task

1. Mark all requirements complete
2. Update scope status to "Complete"
3. Write final log entry with summary
4. Archive or delete if no longer needed

---

## Cross-Platform Considerations

### Platform Detection
Log entries should note which AI platform performed the work. This helps when:
- Debugging issues
- Understanding context limitations
- Coordinating between platforms

### Context Limits
Different AI platforms have different context windows. Structure files so:
- Scope file is self-contained and readable alone
- Phase files focus on single phases
- Log files are chronological and scannable

### Handoff
When switching platforms mid-task:
1. Create a log entry summarizing current state
2. Note any platform-specific context
3. List explicit next steps
4. Reference relevant files

---

## Best Practices

### Keep It Updated
- Update scope status regularly
- Create log entries for significant sessions
- Check off requirements as completed

### Be Specific
- Use concrete, measurable requirements
- Document decisions and rationale
- Include file paths and code references

### Stay Focused
- One task per folder
- Don't mix unrelated work
- Split large tasks into subtasks if needed

### Clean Up
- Archive completed tasks periodically
- Remove obsolete files from `files/`
- Delete tasks that were abandoned
