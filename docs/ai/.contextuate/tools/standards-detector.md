# Standards Detector Tool

> **Type:** AI Tool Guide
> **Purpose:** Analyze project files to detect and document coding standards

---

## When to Use This Tool

Use this tool when:
- Setting up Contextuate in an existing project
- User wants coding standards documented
- Standards need to be inferred from existing code
- Creating project-specific coding standards

---

## Input

**Required:** Access to project source files

**Optional:**
- Specific languages to analyze
- Specific directories to focus on
- Existing style config files (.eslintrc, .prettierrc, phpcs.xml, etc.)

---

## Process

### Step 1: Detect Languages

Scan project for source files and identify languages:

| Extension | Language |
|-----------|----------|
| `.php` | PHP |
| `.js`, `.jsx` | JavaScript |
| `.ts`, `.tsx` | TypeScript |
| `.py` | Python |
| `.go` | Go |
| `.rb` | Ruby |
| `.java` | Java |
| `.cs` | C# |
| `.rs` | Rust |

### Step 2: Check for Config Files

Look for existing style configuration:

| File | Tool | Language |
|------|------|----------|
| `.eslintrc*` | ESLint | JS/TS |
| `.prettierrc*` | Prettier | JS/TS/CSS |
| `phpcs.xml` | PHP_CodeSniffer | PHP |
| `.php-cs-fixer.php` | PHP CS Fixer | PHP |
| `pyproject.toml` | Black/Ruff | Python |
| `.editorconfig` | EditorConfig | All |
| `tslint.json` | TSLint | TypeScript |
| `.rubocop.yml` | RuboCop | Ruby |

If config files exist, extract rules from them first.

### Step 3: Analyze Sample Files

For each detected language, read 3-5 representative files:
- Prefer files in `src/`, `app/`, `lib/` directories
- Choose files with 50-200 lines (not too short, not too long)
- Include different file types (classes, utilities, controllers)

### Step 4: Detect Patterns

For each file, analyze:

#### Formatting
- **Indentation:** Count leading whitespace characters
  - All tabs → tabs
  - 2 spaces → 2-space indent
  - 4 spaces → 4-space indent
- **Braces:** Check if `{` appears on same line or next line
- **Spacing:** Check patterns inside parentheses, after keywords
- **Semicolons (JS):** Present at end of statements?
- **Quotes (JS):** Single or double quotes predominant?

#### Naming
- **Classes:** PascalCase, snake_case, etc.
- **Functions:** camelCase, snake_case, etc.
- **Variables:** camelCase, snake_case, etc.
- **Constants:** UPPER_CASE, camelCase, etc.

#### Structure
- **Import organization:** Grouped? Alphabetized?
- **File organization:** Class structure patterns

#### Documentation
- **DocBlocks:** Present? Format?
- **Comments:** Inline style?

### Step 5: Resolve Conflicts

If patterns vary across files:
1. Count occurrences of each pattern
2. Use majority pattern (60%+ agreement)
3. Note inconsistencies for user review
4. Flag if no clear majority exists

### Step 6: Generate Standards Document

Use the appropriate template from `templates/standards/`:
- `php.standards.md` for PHP
- `javascript.standards.md` for JS/TS

Fill in detected values, replacing `{placeholders}`.

---

## Output

Create files in `docs/ai/standards/` (user-customizable location):

```
docs/ai/standards/
├── php.standards.md        # If PHP detected
├── javascript.standards.md # If JS/TS detected
└── {language}.standards.md # For other languages
```

These user standards take priority over framework defaults in `docs/ai/.context/templates/standards/`.
See [../standards/coding-standards.md](../standards/coding-standards.md) for resolution order.

---

## Analysis Checklist

### PHP Analysis

```php
// Check for:
// 1. Indentation
$line = "    code";  // 4 spaces
$line = "\tcode";    // tab

// 2. Brace style
class Foo {          // same-line
class Foo            // next-line
{

// 3. Parentheses spacing
function foo( $param )   // spaces inside
function foo($param)     // no spaces

// 4. Array syntax
$arr = array();      // long syntax
$arr = [];           // short syntax

// 5. Visibility
public $prop;        // explicit
var $prop;           // implicit (old style)

// 6. Type hints
function foo( string $a ): int    // yes
function foo( $a )                 // no
```

### JavaScript Analysis

```javascript
// Check for:
// 1. Semicolons
const x = 1;    // with semicolon
const x = 1     // without

// 2. Quotes
'single quotes'
"double quotes"

// 3. Variable declaration
const x = 1;    // const preferred
let x = 1;      // let used
var x = 1;      // var used (legacy)

// 4. Arrow functions
const fn = () => {};     // arrow
function fn() {}         // declaration

// 5. Trailing commas
{ a: 1, b: 2, }   // trailing
{ a: 1, b: 2 }    // no trailing

// 6. Object shorthand
{ name: name }    // verbose
{ name }          // shorthand
```

---

## Example Output

After analyzing a PHP project:

```markdown
# PHP Coding Standards

> **Language:** PHP
> **Generated:** 2024-01-15
> **Detected from:** src/Services/, src/Controllers/

---

## Formatting

### Indentation
- **Style:** tabs
- **Size:** 1 tab

### Braces
- **Classes/Functions:** same-line
- **Control structures:** same-line

### Spacing
- **Inside parentheses:** yes - `function( $param )`
- **After keywords:** yes - `if( $x )`
- **Array brackets:** no-spaces - `$arr['key']`

---

## Naming Conventions

### Classes
- **Style:** PascalCase
- **Example:** `UserService`, `PaymentController`

### Methods
- **Style:** camelCase
- **Example:** `getUserById()`, `processPayment()`

### Variables
- **Style:** snake_case
- **Example:** `$user_name`, `$is_active`

...
```

---

## Edge Cases

### Mixed Standards
If project has inconsistent standards:
1. Document the majority pattern
2. Add a note about inconsistencies
3. Suggest the user choose one

```markdown
> **Note:** Mixed indentation detected (60% tabs, 40% 2-space).
> Documenting tabs as primary. Consider standardizing.
```

### No Clear Pattern
If no pattern emerges:
1. Leave placeholder in template
2. Note that manual input is needed

```markdown
### Variables
- **Style:** {inconsistent - needs manual specification}
```

### Config File Conflicts
If config files disagree with actual code:
1. Prefer config file settings (they're intentional)
2. Note that code may not follow config
3. Suggest running linter to fix

---

## Reporting

After generating standards, report:

1. Languages detected
2. Files analyzed
3. Config files found
4. Standards documents created
5. Any inconsistencies or manual input needed

```
Standards Detection Complete
============================
Languages: PHP, JavaScript
Files analyzed: 12
Config files: .eslintrc.js, .editorconfig

Created:
  - docs/ai/standards/php.standards.md
  - docs/ai/standards/javascript.standards.md

Notes:
  - PHP: Inconsistent variable naming (70% snake_case)
  - JS: No semicolons detected, matches .eslintrc
```
