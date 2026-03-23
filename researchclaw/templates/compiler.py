"""LaTeX compilation and error repair utilities (IMP-18).

Provides ``compile_latex()`` which attempts ``pdflatex`` compilation,
parses the log for common errors, applies automated fixes, and retries
up to 3 times.  Designed to run inside ``_package_deliverables()`` so
that the final paper.tex in ``deliverables/`` is compile-tested.

If pdflatex is not installed the module gracefully returns a failure
report without raising.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# BUG-201: Cyrillic → Latin transliteration for author names from Semantic Scholar.
# pdflatex without T2A font encoding chokes on Cyrillic (e.g. "А. И. Колесников").
_CYRILLIC_TO_LATIN_MAP: dict[str, str] = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E",
    "Ё": "E", "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y", "К": "K",
    "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "Kh", "Ц": "Ts",
    "Ч": "Ch", "Ш": "Sh", "Щ": "Shch", "Ъ": "", "Ы": "Y", "Ь": "",
    "Э": "E", "Ю": "Yu", "Я": "Ya",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
}


@dataclass
class CompileResult:
    """Outcome of a LaTeX compilation attempt."""

    success: bool
    log_excerpt: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    attempts: int = 0


def compile_latex(
    tex_path: Path,
    *,
    max_attempts: int = 3,
    timeout: int = 120,
) -> CompileResult:
    """Compile *tex_path* with pdflatex, auto-fixing common errors.

    Parameters
    ----------
    tex_path:
        Path to the ``.tex`` file.  Must be inside a directory that also
        contains ``references.bib`` and any required ``.sty`` files.
    max_attempts:
        Maximum compile→fix cycles.
    timeout:
        Seconds before killing a stuck pdflatex process.

    Returns
    -------
    CompileResult
        Contains success flag, log excerpt, errors found, and fixes applied.
    """
    if not shutil.which("pdflatex"):
        return CompileResult(
            success=False,
            log_excerpt="pdflatex not found on PATH",
            errors=["pdflatex not installed"],
        )

    result = CompileResult(success=False)
    work_dir = tex_path.parent
    tex_name = tex_path.name
    bib_stem = tex_name.rsplit(".", 1)[0]

    # Pre-flight: sanitize .bib file (escape bare & in field values)
    # Find bib filename from \bibliography{...} in the tex source
    _tex_src = tex_path.read_text(encoding="utf-8", errors="replace")
    _bib_match = re.search(r"\\bibliography\{([^}]+)\}", _tex_src)
    _bib_name = _bib_match.group(1) if _bib_match else bib_stem
    _sanitize_bib_file(work_dir / f"{_bib_name}.bib")

    # BUG-197: Pre-flight — strip invisible/problematic Unicode from .tex.
    # Characters like U+202F (NARROW NO-BREAK SPACE) cause pdflatex to emit
    # broken UTF-8 in error messages, which crashes subprocess text decoding
    # and prevents the bibtex + multi-pass pipeline from completing.
    _sanitize_tex_unicode(tex_path)

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt

        # --- Full 3-pass compilation: pdflatex → bibtex → pdflatex × 2 ---
        # Pass 1: generate .aux (needed by bibtex). Use nonstopmode (NOT
        # halt-on-error) so .aux is written even when there are non-fatal
        # errors like missing figures or overfull hboxes.
        log_text, pass1_ok = _run_pdflatex(work_dir, tex_name, timeout)
        if log_text is None:
            result.errors.append(f"pdflatex failed on pass 1 (attempt {attempt})")
            break

        # BibTeX: always run after pass 1 — it only needs .aux + .bib.
        # Previously gated behind pass1 success, which meant citations were
        # always [?] when the first pass had non-fatal errors.
        _run_bibtex(work_dir, bib_stem, timeout=60)

        # Passes 2-3: resolve cross-references and bibliography
        for _pass in (2, 3):
            pass_log, _ = _run_pdflatex(work_dir, tex_name, timeout)
            if pass_log is not None:
                log_text = pass_log  # keep final pass log for error analysis

        # Parse the final log for errors/warnings
        errors, warnings = _parse_log(log_text)
        result.warnings = warnings
        result.log_excerpt = log_text[-2000:] if len(log_text) > 2000 else log_text

        # Check for fatal errors only — non-fatal ones (overfull hbox,
        # missing figure in draft) don't prevent a valid PDF.
        fatal = [e for e in errors if _is_fatal_error(e)]
        result.errors = errors

        if not fatal:
            result.success = True
            logger.info("IMP-18: LaTeX compiled successfully on attempt %d", attempt)
            break

        # Try to auto-fix fatal errors
        tex_text = tex_path.read_text(encoding="utf-8")
        fixed_text, fixes = fix_common_latex_errors(tex_text, errors)
        if fixes:
            result.fixes_applied.extend(fixes)
            tex_path.write_text(fixed_text, encoding="utf-8")
            logger.info(
                "IMP-18: Applied %d fixes on attempt %d: %s",
                len(fixes),
                attempt,
                fixes,
            )
        else:
            # No fixes available — stop retrying
            logger.warning(
                "IMP-18: Compilation failed on attempt %d with %d unfixable errors",
                attempt,
                len(fatal),
            )
            break

    return result


def fix_common_latex_errors(
    tex_text: str, errors: list[str]
) -> tuple[str, list[str]]:
    """Apply automated fixes for common LaTeX errors.

    Returns ``(fixed_text, list_of_fix_descriptions)``.
    """
    fixes: list[str] = []
    fixed = tex_text

    # --- Pre-error-loop fixes: structural repairs that prevent compilation ---

    # Fix escaped braces in tabular column specs: \{lcccc\} → {lcccc}
    if re.search(r"\\begin\{tabular\}\\\{", fixed):
        fixed = re.sub(
            r"\\begin\{tabular\}\\\{([^}]*?)\\\}",
            r"\\begin{tabular}{\1}",
            fixed,
        )
        fixes.append("Fixed escaped braces in tabular column specs")

    # Fix escaped & inside tabular data rows: \& → & (column separator).
    # The converter's _escape_latex escapes & globally; inside tabular
    # environments the & must remain unescaped as the column separator.
    if "\\begin{tabular}" in fixed and "\\&" in fixed:
        fixed, n_tab_amp = _fix_escaped_ampersand_in_tabular(fixed)
        if n_tab_amp:
            fixes.append(f"Un-escaped \\& in {n_tab_amp} tabular data row(s)")

    # Fix escaped \} at end of \caption{...}: \caption{text.\}} → \caption{text.}
    if re.search(r"\\caption\{.*?\\\}", fixed):
        fixed = re.sub(
            r"(\\caption\{[^}]*?)\\\}",
            r"\1}",
            fixed,
        )
        fixes.append("Fixed escaped \\} in \\caption arguments")

    # Collapse multiple consecutive \clearpage into one
    if re.search(r"(\\clearpage\s*){2,}", fixed):
        fixed = re.sub(r"(\\clearpage\s*){2,}", "\\\\clearpage\n", fixed)
        fixes.append("Collapsed multiple \\clearpage commands")

    # Remove \textbf{Figure N.} paragraphs that follow \end{figure}
    dup_cap = re.search(
        r"(\\end\{figure\})\s*\n\s*\\textbf\{Figure\s+\d+",
        fixed,
    )
    if dup_cap:
        fixed = re.sub(
            r"(\\end\{figure\})\s*\n\s*\\textbf\{Figure\s+\d+[.:].*?\}\s*\n",
            r"\1\n",
            fixed,
        )
        fixes.append("Removed duplicate bold Figure captions after \\end{figure}")

    # BUG-189: Fix Python-style pseudocode inside algorithmic environments.
    # LLM generates `# comment` (LaTeX macro param char) and `var_name`
    # (unescaped underscore) inside \STATE commands — causes cascading errors.
    _algo_pat = re.compile(
        r"(\\begin\{algorithmic\}.*?\\end\{algorithmic\})", re.DOTALL
    )
    def _fix_algo_block(m: re.Match) -> str:
        block = m.group(0)
        lines = block.split("\n")
        out: list[str] = []
        for line in lines:
            if line.strip().startswith(("\\begin{", "\\end{")):
                out.append(line)
                continue
            # Replace # (Python comment) with \COMMENT{...}
            if "#" in line and "\\#" not in line:
                line = re.sub(r"#\s*(.*)$", r"\\COMMENT{\1}", line)
            # Escape bare underscores not already in math mode
            # Don't touch \STATE, \IF, \FOR, etc. commands
            parts = re.split(r"(\\\w+\{[^}]*\}|\$[^$]+\$)", line)
            fixed_parts = []
            for part in parts:
                if part.startswith("\\") or part.startswith("$"):
                    fixed_parts.append(part)
                else:
                    fixed_parts.append(re.sub(r'(?<!\\)_', r'\\_', part))
            line = "".join(fixed_parts)
            out.append(line)
        return "\n".join(out)

    if _algo_pat.search(fixed):
        fixed = _algo_pat.sub(_fix_algo_block, fixed)
        fixes.append("Fixed Python-style pseudocode in algorithmic environment")

    for err in errors:
        err_lower = err.lower()

        # Undefined control sequence: remove the offending command
        if "undefined control sequence" in err_lower:
            # Extract the command name from error like "! Undefined control sequence. \foo"
            m = re.search(r"\\([a-zA-Z]+)", err)
            if m:
                cmd = m.group(1)
                # Don't remove standard commands
                _safe_to_remove = {
                    "textsc", "textsl", "mathbb", "mathcal",
                    "bm", "boldsymbol",
                }
                if cmd in _safe_to_remove:
                    # Replace \cmd{text} → text
                    fixed = re.sub(
                        rf"\\{cmd}\{{([^}}]*)\}}", r"\1", fixed
                    )
                    fixes.append(f"Removed undefined \\{cmd}")

        # Missing $ inserted — likely double-escaped underscore \\_
        if "missing $ inserted" in err_lower:
            # BUG-182: Collapse double-escaped underscores \\_ to \_
            if "\\\\_" in fixed:
                fixed = fixed.replace("\\\\_", "\\_")
                fixes.append("Collapsed double-escaped underscores")
            # Also fix bare underscores outside math mode
            # (conservative — only in obvious identifier patterns)
            fixed = re.sub(
                r"(?<!\\)(?<!\$)_(?=[A-Za-z])", r"\\_", fixed
            )
            if fixed != tex_text:
                fixes.append("Escaped bare underscores outside math")

        # \tilde outside math mode — classify as non-fatal, PDF still generates

        # Encoding error for \k (Polish ogonek) → remove
        if "\\k unavailable" in err_lower or "command \\k" in err_lower:
            fixed = re.sub(r"\\k\{([^}]*)\}", r"\1", fixed)
            fixed = re.sub(r"\\k\b", "", fixed)
            fixes.append("Removed unsupported \\k command")

        # BUG-197: Unicode character "not set up for use with LaTeX"
        # Extract the hex codepoint and replace all instances in the tex.
        # The error line is "! LaTeX Error: Unicode character X (U+XXXX)".
        if "unicode character" in err_lower and "(u+" in err_lower:
            cp_match = re.search(r"\(U\+([0-9A-Fa-f]{4,})\)", err)
            if cp_match:
                cp = int(cp_match.group(1), 16)
                char = chr(cp)
                if char in fixed:
                    # Whitespace-like → ASCII space; others → remove
                    import unicodedata
                    cat = unicodedata.category(char)
                    replacement = " " if cat.startswith("Z") else ""
                    fixed = fixed.replace(char, replacement)
                    fixes.append(
                        f"Replaced Unicode U+{cp_match.group(1)} "
                        f"({'space' if replacement == ' ' else 'removed'})"
                    )

        # File not found
        if "file" in err_lower and "not found" in err_lower:
            m = re.search(r"File `([^']+)' not found", err)
            if m:
                missing_file = m.group(1)
                if missing_file.endswith(".sty"):
                    # Comment out the usepackage line
                    pkg = missing_file.replace(".sty", "")
                    fixed = re.sub(
                        rf"\\usepackage(\[[^\]]*\])?\{{{pkg}\}}",
                        f"% IMP-18: Removed missing package {pkg}",
                        fixed,
                    )
                    fixes.append(f"Removed missing package {pkg}")

        # Too many unprocessed floats / Float(s) lost
        if "too many unprocessed floats" in err_lower or "float(s) lost" in err_lower:
            # BUG-109 fix: Add \extrafloats and \clearpage for float overflow
            if "\\extrafloats" not in fixed:
                fixed = fixed.replace(
                    "\\begin{document}",
                    "\\begin{document}\n\\extrafloats{200}",
                )
                fixes.append("Added \\extrafloats{200} for float overflow")
            # BUG-109b: \textwidth in 2-column causes oversized floats to be lost
            if "\\resizebox{\\textwidth}" in fixed:
                fixed = fixed.replace(
                    "\\resizebox{\\textwidth}",
                    "\\resizebox{\\columnwidth}",
                )
                fixes.append("Replaced \\textwidth with \\columnwidth in resizebox")
            # Relax float placement from [ht] or [t] to [htbp!]
            fixed = re.sub(
                r"\\begin\{(table|figure)\}\[h?t\]",
                r"\\begin{\1}[htbp!]",
                fixed,
            )
            fixes.append("Relaxed float placement to [htbp!]")
            # Add \clearpage before first table as last resort
            fixed = fixed.replace(
                "\\begin{table}",
                "\\clearpage\n\\begin{table}",
                1,
            )
            fixes.append("Added \\clearpage for float overflow")

        # Misplaced alignment tab &
        if "misplaced alignment tab" in err_lower:
            # Usually from & outside tabular — escape stray &
            pass  # Hard to auto-fix without context

    return fixed, fixes


def _parse_log(log_text: str) -> tuple[list[str], list[str]]:
    """Parse pdflatex log output for errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    for line in log_text.split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        if line_stripped.startswith("!"):
            errors.append(line_stripped)
        elif "LaTeX Warning:" in line_stripped:
            warnings.append(line_stripped)
        # BUG-R6-26: Use elif to avoid duplicating "!" lines
        elif "Undefined control sequence" in line_stripped:
            errors.append(line_stripped)
        elif "Missing" in line_stripped and "inserted" in line_stripped:
            errors.append(line_stripped)
        elif "File" in line_stripped and "not found" in line_stripped:
            errors.append(line_stripped)
        # BUG-R6-21: Detect "Float(s) lost" and "Too many unprocessed floats"
        # even when they don't start with "!"
        elif "float(s) lost" in line_lower:
            errors.append(line_stripped)
        elif "too many unprocessed floats" in line_lower:
            errors.append(line_stripped)

    return errors, warnings


@dataclass
class QualityCheckResult:
    """Results of post-compilation quality checks."""

    unresolved_refs: list[str] = field(default_factory=list)
    unresolved_cites: list[str] = field(default_factory=list)
    overfull_hboxes: list[str] = field(default_factory=list)
    underfull_hboxes: list[str] = field(default_factory=list)
    page_count: int = 0
    orphan_figures: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    warnings_summary: list[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        return bool(self.unresolved_refs or self.unresolved_cites)


def check_compiled_quality(
    tex_path: Path,
    *,
    page_limit: int = 10,
) -> QualityCheckResult:
    """Run post-compilation quality checks on a LaTeX document.

    Parses the .log file and .tex source for:
    - Unresolved references (??)
    - Unresolved citations
    - Overfull/underfull hboxes
    - Page count vs limit
    - Orphan figures (defined but never referenced, or vice versa)
    """
    result = QualityCheckResult()
    work_dir = tex_path.parent
    stem = tex_path.stem

    # --- Parse .log file ---
    log_path = work_dir / f"{stem}.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in log_text.split("\n"):
            line_s = line.strip()
            # Unresolved references
            if "LaTeX Warning: Reference" in line_s and "undefined" in line_s:
                result.unresolved_refs.append(line_s)
            # Unresolved citations
            if "LaTeX Warning: Citation" in line_s and "undefined" in line_s:
                result.unresolved_cites.append(line_s)
            # Overfull hboxes (only flag significant ones > 1pt)
            if "Overfull \\hbox" in line_s:
                m = re.search(r"(\d+\.?\d*)pt", line_s)
                if m and float(m.group(1)) > 1.0:
                    result.overfull_hboxes.append(line_s)
            # Underfull hboxes (badness >= 5000)
            if "Underfull \\hbox" in line_s and "badness" in line_s:
                m = re.search(r"badness (\d+)", line_s)
                if m and int(m.group(1)) >= 5000:
                    result.underfull_hboxes.append(line_s)

    # --- Count pages from .aux or .log ---
    aux_path = work_dir / f"{stem}.aux"
    if aux_path.exists():
        aux_text = aux_path.read_text(encoding="utf-8", errors="replace")
        # Look for \newlabel{LastPage}{{N}{...}}
        m = re.search(r"\\newlabel\{LastPage\}\{\{(\d+)\}", aux_text)
        if m:
            result.page_count = int(m.group(1))
    if result.page_count == 0 and log_path.exists():
        # Fallback: count "Output written on ... (N pages)"
        m = re.search(r"Output written on .* \((\d+) page", log_text)
        if m:
            result.page_count = int(m.group(1))

    # --- Cross-reference validation ---
    tex_text = tex_path.read_text(encoding="utf-8", errors="replace")
    # Find all \label{fig:X}
    fig_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", tex_text))
    # Find all \ref{fig:X}
    fig_refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", tex_text))
    # Orphan labels (defined but never referenced)
    result.orphan_labels = sorted(fig_labels - fig_refs)
    # Orphan references (referenced but never defined)
    result.orphan_figures = sorted(fig_refs - fig_labels)

    # --- Build warnings summary ---
    if result.unresolved_refs:
        result.warnings_summary.append(
            f"{len(result.unresolved_refs)} unresolved reference(s)"
        )
    if result.unresolved_cites:
        result.warnings_summary.append(
            f"{len(result.unresolved_cites)} unresolved citation(s)"
        )
    if result.overfull_hboxes:
        result.warnings_summary.append(
            f"{len(result.overfull_hboxes)} overfull hbox(es) > 1pt"
        )
    if result.page_count > page_limit:
        result.warnings_summary.append(
            f"Page count {result.page_count} exceeds limit {page_limit}"
        )
    if result.orphan_figures:
        result.warnings_summary.append(
            f"{len(result.orphan_figures)} referenced but undefined figure(s): "
            + ", ".join(result.orphan_figures[:3])
        )
    if result.orphan_labels:
        result.warnings_summary.append(
            f"{len(result.orphan_labels)} defined but unreferenced figure(s): "
            + ", ".join(result.orphan_labels[:3])
        )

    return result


def remove_missing_figures(tex_text: str, stage_dir: Path) -> tuple[str, list[str]]:
    """Remove \\begin{figure}...\\end{figure} blocks that reference missing images.

    Returns ``(fixed_text, list_of_removed_paths)``.
    """
    removed: list[str] = []

    def _check_fig(m: re.Match) -> str:
        block = m.group(0)
        img_match = re.search(r"\\includegraphics.*?\{([^}]+)\}", block)
        if img_match:
            img_rel = img_match.group(1)
            img_path = stage_dir / img_rel
            if not img_path.exists():
                # Try prefix-matching: fig_main_results.png → fig_main_results_comparison.png
                parent = img_path.parent
                stem = img_path.stem  # e.g. "fig_main_results"
                if parent.exists():
                    candidates = sorted(parent.glob(f"{stem}*.png"))
                    if len(candidates) == 1:
                        new_rel = str(candidates[0].relative_to(stage_dir))
                        logger.info(
                            "Auto-mapped missing figure: %s → %s",
                            img_rel, new_rel,
                        )
                        return block.replace(img_rel, new_rel)
                logger.warning(
                    "Removing figure block with missing image: %s",
                    img_rel,
                )
                removed.append(img_rel)
                return ""  # Remove the entire figure block
        return block

    fixed = re.sub(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        _check_fig,
        tex_text,
        flags=re.DOTALL,
    )

    # Clean up orphan \ref{fig:X} that point to removed/nonexistent figures.
    # These render as "??" in the PDF.
    if removed:
        remaining_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", fixed))
        all_fig_refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", fixed))
        orphan = all_fig_refs - remaining_labels
        for oref in orphan:
            # Replace "Figure \ref{fig:X}" or "Fig. \ref{fig:X}" with empty
            fixed = re.sub(
                rf"(?:Figure|Fig\.?)\s*~?\\ref\{{{re.escape(oref)}\}}",
                "(figure omitted)",
                fixed,
            )
            # Replace standalone \ref{fig:X}
            fixed = fixed.replace(f"\\ref{{{oref}}}", "(ref omitted)")

    return fixed, removed


def _sanitize_tex_unicode(tex_path: Path) -> None:
    """Strip problematic Unicode characters from .tex source.

    BUG-197: Characters like U+202F (NARROW NO-BREAK SPACE), U+2009 (THIN
    SPACE), U+00A0 (NO-BREAK SPACE), and other non-ASCII whitespace cause
    pdflatex to emit broken UTF-8 in error messages, which crashes Python's
    ``subprocess.run(text=True)`` and prevents the bibtex + multi-pass
    pipeline from completing.  These characters appear when LLMs copy-paste
    text from web sources or academic papers.

    The safe replacement is a normal ASCII space for whitespace-like chars,
    and empty string for invisible control chars.
    """
    if not tex_path.exists():
        return
    try:
        text = tex_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    # Whitespace-like Unicode → ASCII space
    _UNICODE_SPACES = (
        "\u00a0",  # NO-BREAK SPACE
        "\u202f",  # NARROW NO-BREAK SPACE (BUG-197 trigger)
        "\u2009",  # THIN SPACE
        "\u2007",  # FIGURE SPACE
        "\u2008",  # PUNCTUATION SPACE
        "\u200a",  # HAIR SPACE
        "\u205f",  # MEDIUM MATHEMATICAL SPACE
        "\u3000",  # IDEOGRAPHIC SPACE
    )
    # Invisible control characters → remove
    _INVISIBLE_CHARS = (
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\ufeff",  # BOM / ZERO-WIDTH NO-BREAK SPACE
        "\u200b",  # ZERO-WIDTH SPACE
        "\u200c",  # ZERO-WIDTH NON-JOINER
        "\u200d",  # ZERO-WIDTH JOINER
        "\u00ad",  # SOFT HYPHEN
        "\u2060",  # WORD JOINER
        "\u2028",  # LINE SEPARATOR
        "\u2029",  # PARAGRAPH SEPARATOR
    )

    changed = False
    for ch in _UNICODE_SPACES:
        if ch in text:
            text = text.replace(ch, " ")
            changed = True
    for ch in _INVISIBLE_CHARS:
        if ch in text:
            text = text.replace(ch, "")
            changed = True

    # BUG-201: Transliterate any Cyrillic that leaked into .tex (from bib
    # entries inlined by bibtex, or from LLM-generated text).
    _has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in text)
    if _has_cyrillic:
        for cyr, lat in _CYRILLIC_TO_LATIN_MAP.items():
            if cyr in text:
                text = text.replace(cyr, lat)
        changed = True

    if changed:
        tex_path.write_text(text, encoding="utf-8")
        logger.info("BUG-197: Sanitized problematic Unicode in %s", tex_path.name)


def _sanitize_bib_file(bib_path: Path) -> None:
    """Sanitize .bib files: escape bare ``&`` and strip invisible Unicode.

    BibTeX treats ``&`` as a special character; journal names like
    "Science & Technology" must use ``\\&``.

    BUG-180: Invisible Unicode characters (U+200E LEFT-TO-RIGHT MARK,
    U+200F RIGHT-TO-LEFT MARK, U+FEFF BOM, U+200B ZERO-WIDTH SPACE,
    U+200C/U+200D joiners, U+00AD soft hyphen) can appear in
    copy-pasted author names and break pdflatex.
    """
    if not bib_path.exists():
        return
    try:
        text = bib_path.read_text(encoding="utf-8")
    except Exception:
        return

    # BUG-180: Strip invisible Unicode characters
    _INVISIBLE_CHARS = (
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\ufeff",  # BOM / ZERO-WIDTH NO-BREAK SPACE
        "\u200b",  # ZERO-WIDTH SPACE
        "\u200c",  # ZERO-WIDTH NON-JOINER
        "\u200d",  # ZERO-WIDTH JOINER
        "\u00ad",  # SOFT HYPHEN
        "\u2060",  # WORD JOINER
        "\u2028",  # LINE SEPARATOR
        "\u2029",  # PARAGRAPH SEPARATOR
    )
    for ch in _INVISIBLE_CHARS:
        if ch in text:
            text = text.replace(ch, "")

    # BUG-201: Transliterate Cyrillic characters to Latin equivalents.
    # Russian author names (e.g. "А. И. Колесников") from Semantic Scholar
    # cause "! LaTeX Error: Unicode character" when pdflatex runs without T2A
    # font encoding.  Transliterating preserves name readability.
    _orig_text = text
    for cyr, lat in _CYRILLIC_TO_LATIN_MAP.items():
        if cyr in text:
            text = text.replace(cyr, lat)

    # BUG-217: Strip literal escape sequences (\n, \r, \t) in bib field values.
    # These appear when API responses embed Python-style escapes into titles.
    # A literal `\n` is never a valid BibTeX/LaTeX command and causes
    # "Undefined control sequence" errors during compilation.
    text = re.sub(r"\\n(?=\s)", " ", text)
    text = re.sub(r"\\r(?=\s)", "", text)
    text = re.sub(r"\\t(?=\s)", " ", text)

    lines = text.split("\n")
    changed = text != _orig_text
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Only fix field-value lines (e.g.  journal = {Science & Technology},)
        # Skip @type{ lines, key lines, and URL/DOI fields (BUG-DA8-12)
        if "=" in stripped and "{" in stripped and "&" in stripped and "\\&" not in stripped:
            _field_name = stripped.split("=", 1)[0].strip().lower()
            if _field_name in ("url", "doi", "howpublished", "eprint"):
                continue  # Don't escape & in URLs
            lines[i] = line.replace("&", "\\&")
            changed = True

    new_text = "\n".join(lines)
    if new_text != text or changed:
        bib_path.write_text(new_text, encoding="utf-8")
        logger.info("Sanitized bib file %s", bib_path.name)


def _fix_escaped_ampersand_in_tabular(tex: str) -> tuple[str, int]:
    """Replace ``\\&`` with ``&`` inside tabular environments.

    Only touches data rows (between \\toprule/\\midrule/\\bottomrule)
    to avoid corrupting ``\\&`` in regular text.  Returns the fixed text
    and the count of rows fixed.
    """
    count = 0

    def _fix_tabular(m: re.Match[str]) -> str:
        nonlocal count
        block = m.group(0)
        if "\\&" not in block:
            return block
        # Only un-escape \& on lines that look like data rows (contain \\)
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if "\\&" in line and "\\\\" in line:
                lines[i] = line.replace("\\&", "&")
                count += 1
        return "\n".join(lines)

    tex = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        _fix_tabular,
        tex,
        flags=re.DOTALL,
    )
    return tex, count


def _run_pdflatex(
    work_dir: Path,
    tex_name: str,
    timeout: int = 120,
) -> tuple[str | None, bool]:
    """Run a single pdflatex pass with ``-interaction=nonstopmode``.

    Returns ``(log_text, success)``.  *log_text* is ``None`` only on
    hard failures (timeout, binary missing).

    BUG-197: Uses bytes mode with manual UTF-8 decoding (errors="replace")
    instead of ``text=True``.  pdflatex stdout can contain broken UTF-8
    sequences (e.g. from U+202F NARROW NO-BREAK SPACE error messages),
    which cause ``UnicodeDecodeError`` with ``text=True`` and kill the
    entire compilation pipeline — bibtex never runs, all citations [?].
    """
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_name],
            cwd=work_dir,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("pdflatex timed out after %ds", timeout)
        return None, False
    except FileNotFoundError:
        return None, False
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    log_text = stdout + "\n" + stderr
    return log_text, proc.returncode == 0


# Fatal error patterns — these prevent a valid PDF from being generated.
# Non-fatal issues (overfull hbox, missing figure, float warnings) still
# produce a usable PDF and should NOT trigger the auto-fix retry loop.
_FATAL_ERROR_PATTERNS = [
    "runaway argument",
    "emergency stop",
    "fatal error",
    "undefined control sequence",
    "missing $ inserted",
    "extra alignment tab",
    "misplaced alignment tab",
    "missing \\begin{document}",
    "file `" ,  # file not found (sty, cls)
    "file not found",
]


def _is_fatal_error(err: str) -> bool:
    """Return True if *err* represents a fatal LaTeX error."""
    err_lower = err.lower()
    # "!" prefix errors are almost always fatal
    if err.startswith("!"):
        # Non-fatal "!" errors — PDF is still generated
        if "overfull" in err_lower or "underfull" in err_lower:
            return False
        if "float(s) lost" in err_lower:
            return False
        if "too many unprocessed floats" in err_lower:
            return False
        # amsmath commands outside math mode — PDF still generates
        if "allowed only in math mode" in err_lower:
            return False
        # Encoding errors for special characters — PDF still generates
        if "unavailable in encoding" in err_lower:
            return False
        # BUG-197: Unicode character errors (e.g. U+202F NARROW NO-BREAK
        # SPACE "not set up for use with LaTeX") — pdflatex skips the
        # character and generates a valid PDF.  Treating these as fatal
        # prevents the retry loop from succeeding and blocks bibtex.
        # The error line is "! LaTeX Error: Unicode character X (U+XXXX)"
        # — the "not set up" text is on a continuation line.
        if "unicode character" in err_lower:
            return False
        return True
    for pat in _FATAL_ERROR_PATTERNS:
        if pat in err_lower:
            return True
    return False


def _run_bibtex(work_dir: Path, stem: str, timeout: int = 60) -> bool:
    """Run bibtex if the binary exists.  Returns True on success.

    BUG-197: Uses bytes mode with manual UTF-8 decoding (errors="replace")
    to avoid ``UnicodeDecodeError`` from non-ASCII bib content.  Logs
    failures so that silent bibtex issues are diagnosable.
    """
    if not shutil.which("bibtex"):
        logger.warning("bibtex not found on PATH — citations will be [?]")
        return False
    try:
        proc = subprocess.run(
            ["bibtex", stem],
            cwd=work_dir,
            capture_output=True,
            timeout=timeout,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            logger.warning(
                "bibtex returned %d: %s",
                proc.returncode,
                (stdout + stderr).strip()[:500],
            )
            return False
        # Log bibtex output at debug level for diagnostics
        if stdout.strip():
            logger.debug("bibtex output: %s", stdout.strip()[:300])
        # Verify .bbl was actually generated
        bbl_path = work_dir / f"{stem}.bbl"
        if not bbl_path.exists():
            logger.warning("bibtex ran but %s.bbl was not generated", stem)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("bibtex timed out after %ds", timeout)
        return False
    except FileNotFoundError:
        return False
