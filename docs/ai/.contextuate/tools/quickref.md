# Quickref Generation Tool

> **Type:** AI Tool Guide
> **Purpose:** Instructions for generating AI-friendly quick references from documentation

---

## When to Use This Tool

Use this tool when:
- Full documentation exceeds ~200 lines
- A condensed reference would help AI assistants
- Methods/APIs are frequently looked up
- User requests a quickref be generated

---

## Input

**Required:** Path to source documentation file

**Example request:**
> "Generate a quickref for docs/classes/user-service.md"

---

## Process

### Step 1: Read the Source Documentation

Read the full documentation file to understand:
- What the class/service/API does
- All public methods and their signatures
- Properties and their types
- Common usage patterns

### Step 2: Extract Key Information

For each **method**, capture:
- Method name
- Full parameter signature with types
- One-line description (max 10 words)

For each **property**, capture:
- Property name
- Type
- One-line description

For **common usage**, capture:
- The single most common usage pattern
- Keep it to 3-5 lines of code max

### Step 3: Organize by Category

Group methods logically:
- By functionality (CRUD, validation, transformation)
- By access pattern (getters, setters, actions)
- By frequency of use (common first)

### Step 4: Generate Quickref

Write to `docs/ai/quickrefs/{name}.quickref.md` using the template below.

---

## Output Template

```markdown
# {Name} - Quick Reference

> **Source:** [{source-filename}](../../{path-to-source})
> **Generated:** {YYYY-MM-DD}

---

## Methods

### {Category 1}

| Method | Description |
|--------|-------------|
| `methodName( param: type )` | Brief description |
| `methodName( a: type, b: type ): returnType` | Brief description |

### {Category 2}

| Method | Description |
|--------|-------------|
| `methodName( param: type )` | Brief description |

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `propertyName` | type | Brief description |

---

## Common Usage

\`\`\`{language}
// Most common pattern - keep brief
\`\`\`

---

*See [full documentation](../../{path-to-source}) for details.*
```

---

## Guidelines

### Keep It Scannable
- One line per method/property
- No verbose explanations
- Tables for structured data
- Max 100 lines total

### Method Signatures
- Include all parameters with types
- Include return type if not void
- Use consistent formatting:
  ```
  methodName( param1: type, param2: type ): returnType
  ```

### Descriptions
- Max 10 words per description
- Start with verb: "Gets...", "Sets...", "Returns...", "Creates..."
- No periods at end

### Categories
- 2-5 categories typically
- Use clear, short category names
- Most-used methods first

### Common Usage
- Single most common pattern
- 3-5 lines of code max
- Must be copy-paste ready

---

## Example

**Source:** `docs/classes/cache-service.md` (300 lines)

**Generated quickref:**

```markdown
# Cache Service - Quick Reference

> **Source:** [cache-service.class.md](../../classes/cache-service.class.md)
> **Generated:** 2024-01-15

---

## Methods

### Core Operations

| Method | Description |
|--------|-------------|
| `get( key: string ): mixed` | Gets cached value by key |
| `set( key: string, value: mixed, ttl?: int )` | Sets value with optional TTL |
| `delete( key: string ): bool` | Removes key from cache |
| `has( key: string ): bool` | Checks if key exists |
| `clear()` | Removes all cached values |

### Batch Operations

| Method | Description |
|--------|-------------|
| `getMultiple( keys: array ): array` | Gets multiple values at once |
| `setMultiple( items: array, ttl?: int )` | Sets multiple values |
| `deleteMultiple( keys: array ): bool` | Removes multiple keys |

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `defaultTtl` | int | Default time-to-live in seconds |
| `prefix` | string | Key prefix for namespacing |

---

## Common Usage

\`\`\`php
$value = $cache->get( 'user:123' );
if( !$value ){
    $value = $this->loadUser( 123 );
    $cache->set( 'user:123', $value, 3600 );
}
\`\`\`

---

*See [full documentation](../../classes/cache-service.class.md) for details.*
```

---

## Naming Convention

- **File:** `{source-name}.quickref.md`
- **Location:** `docs/ai/quickrefs/`

Examples:
- `docs/classes/user-service.md` → `docs/ai/quickrefs/user-service.quickref.md`
- `docs/api/auth-endpoints.md` → `docs/ai/quickrefs/auth-endpoints.quickref.md`
