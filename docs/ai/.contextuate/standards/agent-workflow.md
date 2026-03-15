# Agentic Workflow & Orchestration Guide

## Overview
This document outlines the standard operating procedure for orchestrating tasks within a multi-agent architecture. It describes how the "Orchestrator" (Archon) should spawn, context-load, and manage specific domain experts from the main thread.

## Core Roles
Refer to `agent-roles.md` for the definitive list of Agents and their Responsibilities.
*   **Archon**: The Main Thread / Orchestrator.
*   **Specialists**: Forge, Chronos, Vox, Ledger, Nexus, Canvas, Scribe.

## The Dispatch Protocol

### 1. Task Inception (Main Thread)
All work begins in the **Archon** (Main) context.
*   **Input**: A high-level User Request.
*   **Action**: Archon analyzes the request against project specifications and architecture.
*   **Output**: A Decomposed Task List in `task.md`.

### 2. Spawning an Agent
To "spawn" an agent means to start a new context window (or chat session) specifically scoped for one of the Expert Roles.

#### Step A: Select the Role
Identify which agent owns the domain of the sub-task.
*   *Example*: "Update Database Schema" -> **Chronos**.

#### Step B: Defined Context (Contextualizing)
You must strictly limit the context provided to the sub-agent to prevent hallucination and token waste. Provide only the distinct files relevant to that agent's domain (e.g., schema files for Chronos, UI components for Canvas).

#### Step C: The Dispatch Prompt
When invoking the sub-agent, use a structured prompt:

```markdown
Role: [Agent Name] (e.g., Chronos)
Objective: [Specific Sub-Task] (e.g., Add 'column_name' to 'table_name')
Context:
- I have provided [Context Files].
- Follow the style guide in [Specifications].
Constraints:
- Do not modify API logic, only database schema.
- Output strictly [Language/Format].
```

### 3. Execution & Context Isolation
The sub-agent works in its isolated thread.
*   **Files**: It should only touch files within its domain.
*   **Tools**: It uses standard tools (`write_to_file`, `run_command`).
*   **Output**: It returns a specific artifact (Code, SQL, Config) or confirmation of a change.

### 4. Reintegration (Main Thread)
Once the sub-agent reports completion:
1.  **Archon** reviews the changes (diffs).
2.  **Archon** runs integration tests.
3.  **Archon** marks the item as `[x]` in `task.md`.

---

## Inter-Agent Dependencies (Archon-Mediated)

Complex tasks often require Agents to rely on each other. To maintain context isolation and preventing "Context Bleed," **Sub-Agents must never communicate directly.** 

Instead, they direct requests to **Archon**, who manages the dependency.

### The Protocol
When a Sub-Agent encounters a dependency, it returns a structured **Dependency Request** instead of a final result.

#### Format
```yaml
Status: PAUSED
Dependency:
  Target: [Role Name] (e.g., Nexus)
  Type: [BLOCKING | ASYNC]
  Request: "[Clear instruction for the other agent]"
  Context: "[Specific file paths or snippets needed]"
```

### 1. Blocking Dependency (Synchronous)
Used when Agent A *cannot proceed* without the output of Agent B.

*   **Scenario**: `Canvas` (Frontend) needs an API endpoint before it can fetch data.
*   **Flow**:
    1.  **Canvas** reaches the point of needing the API.
    2.  **Canvas** returns a `BLOCKING` dependency request for **Nexus**.
    3.  **Archon**:
        *   Pauses Canvas's thread.
        *   Spawns **Nexus** with the request.
        *   Waits for Nexus to complete.
    4.  **Nexus**: Implements endpoint, returns "Endpoint Created".
    5.  **Archon**:
        *   Resumes **Canvas** thread.
        *   Injects Nexus's output into Canvas's context.
    6.  **Canvas**: Finishes the component code.

### 2. Fire-and-Forget (Asynchronous)
Used when Agent A needs something done but *does not need the result* to finish its own current task.

*   **Scenario**: `Nexus` (Backend) implements a new feature and knows it needs documentation.
*   **Flow**:
    1.  **Nexus** finishes the code.
    2.  **Nexus** returns an `ASYNC` dependency request for a **Scribe**.
    3.  **Archon**:
        *   Mark's Nexus's task as done.
        *   Adds the new task ("Update Docs") to `task.md` for later execution.

---

## Conflict Avoidance & File Locking

When multiple agents run in parallel, there is a risk of race conditions on shared files. To prevent this, use an **Intent-First Locking Protocol**.

### The Protocol
Before an Agent generates any code or edits any files, it must declare its **Intent**.

#### Step 1: Analysis & Intent Declaration
The Agent reads its context and determines which files need modification. It returns:
```yaml
Status: PLANNING
Intent:
  - Modify: src/path/to/file.js
  - Create: src/path/to/new-file.js
```

#### Step 2: Archon Validation
The Archon checks these files against its **Active Lock Table** (a list of files currently being edited by other running agents).

*   **Scenario A (Clear)**: No other agent is using these files.
    *   Archon Action: **Locks** the files.
    *   Archon Prompt to Agent: "Plan Approved. Proceed with execution."
*   **Scenario B (Conflict)**: Another agent is editing the file.
    *   Archon Action: **Queues** the agent.
    *   Archon Prompt to Agent: "File is currently locked by Agent [Name]. Standby."

### Alternative: Git Worktree Isolation
For highly parallel tasks where locking is too restrictive, use **Git Worktrees**.
1.  **Spawn**: Archon creates a disposable Git Worktree (branch) for the Agent.
2.  **Execution**: The Agent runs entirely within the worktree.
3.  **Completion**: Agent commits changes and signals ready to merge.
4.  **Merge**: Archon calls **Unity** (Merge Specialist) to merge the branch into main.

---

## Session Governance (Unique Task Files)

To keep complex workflows organized and debuggable, **Archon never overwrites a global status file** for specific runs.

### Task Isolation
For every major User Request, Archon creates a dedicated **Session Directory**:

**Path**: `docs/ai/tasks/[YYYY-MM-DD]-[kebab-case-feature-name]/`

### The Session `task.md`
Inside this directory, Archon creates a `task.md`. This is the **Single Source of Truth** for that specific orchestration session.

*   **Visibility**: Only Archon writes to this file. Sub-agents do not see it.
*   **Purpose**:
    *   Tracks the breakdown of the specific request.
    *   Logs the status of spawned sub-agents.
    *   Records "Blocking Dependencies".

#### Example Structure
```markdown
# Session: Implement WhatsApp Support
> Date: 2024-01-01
> ID: wa-integration-001

## Master Checklist
- [x] (Archon) Schema Design
- [>] (Chronos) Create Tables [LOCKED: schema.sql]
- [ ] (Nexus) API Endpoints [WAITING: Tables]
- [ ] (Canvas) UI Components
```
