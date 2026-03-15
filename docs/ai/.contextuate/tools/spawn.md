# Tool: spawn_agent

> **Status:** EXPERIMENTAL

The `spawn_agent` tool allows you to launch a specialized sub-agent to perform a specific task in an isolated context.

## Usage

Use this tool when you need to:
1.  Perform a risky operation that should be sandboxed (e.g. "Refactor auth module").
2.  Delegate a large task to a specialized persona (e.g. "Ask documentation-expert to write docs").

## Arguments

| Argument     | Type     | Description                                                            |
| :----------- | :------- | :--------------------------------------------------------------------- |
| `agent_name` | `string` | The name of the agent to spawn (must exist in `docs/ai/agents/`).      |
| `goal`       | `string` | Specific instructions for what the agent should achieve.               |
| `isolation`  | `string` | Isolation mode `worktree` (recommended) or `none`. Default `worktree`. |

## Example

```bash
contextuate run coder-agent --goal "Fix the bug in login.ts" --isolation=worktree
```

## Behavior

1.  **Isolation:** The agent creates a new git worktree (branch).
2.  **Execution:** The agent runs independently in that directory.
3.  **Result:** The agent commits changes to its branch. You must then merge or review these changes.
