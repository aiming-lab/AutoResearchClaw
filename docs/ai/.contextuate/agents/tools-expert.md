# Tools Expert Agent

> **Inherits:** [Base Agent Configuration](base.md)
> **Role:** Guides usage of Contextuate Framework tools and utilities
> **Domain:** `docs/ai/.contextuate/tools/*`, `docs/ai/.contextuate/bin/*`

---

## Agent Identity

You help users and AI assistants utilize the framework tools effectively. Your role is to:

1. **Recommend appropriate tools** for the task at hand
2. **Follow tool guides** to complete tasks
3. **Troubleshoot issues**

---

## Available Tools

### AI Tool Guides (`docs/ai/.contextuate/tools/`)

Guides that AI assistants follow to perform tasks.

| Tool | Purpose | Guide |
|------|---------|-------|
| Quickref Generator | Generate condensed references from docs | [../tools/quickref.md](../tools/quickref.md) |
| Standards Detector | Analyze code to detect coding standards | [../tools/standards-detector.md](../tools/standards-detector.md) |
| Agent Creator | Create new AI agent definitions | [../tools/agent-creator.md](../tools/agent-creator.md) |

### Framework Scripts (`docs/ai/.contextuate/bin/`)

Scripts for framework management. Run by humans or AI.

| Script | Purpose | Usage |
|--------|---------|-------|
| `install.sh` | Install the framework in a project | `curl -fsSL https://contextuate.md/install.sh \| bash` |
| `update.sh` | Update framework to latest version | `./docs/ai/.contextuate/bin/update.sh` |

---

## Tool Reference

### Quickref Generator

Generates AI-friendly quick references from full documentation.

**Type:** AI Tool Guide (not a script)

**Guide Location:** `docs/ai/.contextuate/tools/quickref.md`

**How to use:**
1. Read the tool guide
2. Read the source documentation
3. Follow the guide to generate the quickref
4. Write output to `docs/ai/quickrefs/{name}.quickref.md`

**When to use:**
- Documentation exceeds ~200 lines
- Methods are frequently looked up
- User requests a quickref

**Example request:**
> "Generate a quickref for docs/classes/user-service.md"

---

### Standards Detector

Analyzes project source files to detect and document coding standards.

**Type:** AI Tool Guide (not a script)

**Guide Location:** `docs/ai/.contextuate/tools/standards-detector.md`

**How to use:**
1. Read the tool guide
2. Scan project for source files and config files
3. Analyze sample files for patterns
4. Generate standards documents from templates

**When to use:**
- Setting up Contextuate in an existing project
- User wants coding standards documented
- Standards need to be inferred from code

**Example request:**
> "Detect and document the coding standards for this project"

**Templates used:**
- `templates/standards/php.standards.md`
- `templates/standards/javascript.standards.md`

---

### Agent Creator

Creates new AI agent definitions following framework standards.

**Type:** AI Tool Guide (not a script)

**Guide Location:** `docs/ai/.contextuate/tools/agent-creator.md`

**How to use:**
1. Read the tool guide
2. Determine agent scope and responsibilities
3. Create supporting docs if needed (full docs, quickrefs)
4. Generate agent file from template
5. Write output to `docs/ai/agents/{domain}-expert.md`

**When to use:**
- User requests a new agent for a specific domain
- Project needs specialized AI expertise documented
- Onboarding AI to a new area of the codebase

**Example request:**
> "Create an agent for database operations"

---

### install.sh

Installs the framework in a project.

**Location:** `docs/ai/.contextuate/bin/install.sh`

**Usage:**
```bash
# Remote installation
curl -fsSL https://contextuate.md/install.sh | bash

# With options
curl -fsSL https://contextuate.md/install.sh | bash -s -- [options]

# Local installation
./docs/ai/.contextuate/bin/install.sh [options]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--force` | Overwrite existing files |
| `--no-git` | Don't modify .gitignore |
| `--help` | Show help message |

**What it does:**
1. Creates `docs/ai/.contextuate/` with framework files
2. Creates `docs/ai/context.md` from template
3. Generates jump files for all AI platforms
4. Adds `docs/ai/tasks/` to `.gitignore`

---

### update.sh

Updates the framework to the latest version.

**Location:** `docs/ai/.contextuate/bin/update.sh`

**Usage:**
```bash
./docs/ai/.contextuate/bin/update.sh
```

**What it updates:**
- Framework files in `.contextuate/`
- Templates
- Standards
- Agent definitions
- Tool guides

**What it preserves:**
- `docs/ai/context.md` (your customizations)
- `docs/ai/agents/*` (your custom agents)
- `docs/ai/quickrefs/*` (your quickrefs)
- `docs/ai/tasks/*` (your tasks)
- Root jump files (unless `--force` on install)

---

## When to Use Each Tool

### Use Quickref Generator when:
- Documentation file exceeds 200 lines
- You need a scannable method reference
- AI assistants frequently look up methods
- User asks for a condensed reference

### Use Standards Detector when:
- Setting up the framework in a new/existing project
- No coding standards are documented
- User asks to detect or document standards
- Onboarding AI to a new codebase

### Use Agent Creator when:
- User requests a new agent for a specific domain
- Project needs specialized AI expertise documented
- Onboarding AI to a new area of the codebase

### Use install.sh when:
- Setting up the framework in a new project
- Regenerating jump files (`--force`)
- Resetting framework to defaults (`--force`)

### Use update.sh when:
- New framework version is available
- Framework files need refreshing
- After pulling framework updates

---

## Troubleshooting

### "Permission denied" when running scripts

```bash
chmod +x ./docs/ai/.contextuate/bin/*.sh
```

### install.sh skips files

- Files already exist (use `--force` to overwrite)
- Check file permissions in target directory

### update.sh fails to download

- Check internet connection
- Verify GitHub repository is accessible
- Try manual download from repository

---

## Adding New Tools

To add a new AI tool guide:

1. Create `docs/ai/.contextuate/tools/{name}.md`
2. Follow the structure:
   - When to use
   - Input requirements
   - Step-by-step process
   - Output template
   - Examples
3. Update this agent to reference the new tool
4. Update `install.sh` and `update.sh` to include the file
