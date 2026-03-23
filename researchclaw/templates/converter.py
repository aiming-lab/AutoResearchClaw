"""Markdown-to-LaTeX converter with conference template support.

Converts a ResearchClaw paper (Markdown with embedded LaTeX math) into a
complete ``.tex`` file using a :class:`ConferenceTemplate` for preamble,
author block, bibliography style, and document structure.

Design constraints:
- **Zero new dependencies** — stdlib only (``re``, ``textwrap``).
- Handles inline math ``\\(...\\)``, display math ``\\[...\\]``,
  bold/italic, bullet lists, numbered lists, code blocks, tables,
  and ``\\cite{...}`` references.
- Extracts abstract from ``# Abstract`` or ``## Abstract`` section.
- ICML two-column structure handled via template's ``render_preamble``.
"""

from __future__ import annotations

import re
import textwrap
import threading
from dataclasses import dataclass, field

from researchclaw.templates.conference import ConferenceTemplate

_render_counters = threading.local()


def _reset_render_counters() -> None:
    """Reset per-render figure and table counters for the current thread."""
    _render_counters.table = 0
    _render_counters.figure = 0


def _next_table_num() -> int:
    """Return the next table number for the current thread."""
    next_num = getattr(_render_counters, "table", 0) + 1
    _render_counters.table = next_num
    return next_num


def _next_figure_num() -> int:
    """Return the next figure number for the current thread."""
    next_num = getattr(_render_counters, "figure", 0) + 1
    _render_counters.figure = next_num
    return next_num

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def markdown_to_latex(
    paper_md: str,
    template: ConferenceTemplate,
    *,
    title: str = "",
    authors: str = "Anonymous",
    bib_file: str = "references",
    bib_entries: dict[str, str] | None = None,
) -> str:
    """Convert a Markdown paper to a complete LaTeX document.

    Parameters
    ----------
    paper_md:
        Full paper in Markdown with embedded LaTeX math.
    template:
        Conference template controlling preamble and structure.
    title:
        Paper title.  If empty, extracted from ``# Title`` heading or the
        first ``# ...`` heading in *paper_md*.
    authors:
        Author string inserted into the template author block.
    bib_file:
        Bibliography filename (without ``.bib`` extension).
    bib_entries:
        Optional mapping of author-year patterns to cite_keys for
        recovering author-year citations that slipped through earlier
        processing, e.g. ``{"Raissi et al., 2019": "raissi2019physics"}``.

    Returns
    -------
    str
        A complete ``.tex`` file ready for compilation.
    """
    _reset_render_counters()

    paper_md = _preprocess_markdown(paper_md)
    paper_md = _round_raw_metrics(paper_md)
    sections = _parse_sections(paper_md)

    # Extract title from first H1 heading if not provided
    if not title:
        title = _extract_title(sections, paper_md)

    # Extract abstract
    abstract = _extract_abstract(sections)

    # Build body (everything except title/abstract headings)
    body = _build_body(sections, title=title)

    # IMP-30: Detect and remove duplicate tables
    body = _deduplicate_tables(body)

    # R10-Fix5: Completeness check
    completeness_warnings = check_paper_completeness(sections)
    if completeness_warnings:
        import logging

        _logger = logging.getLogger(__name__)
        for warning in completeness_warnings:
            _logger.warning("LaTeX completeness check: %s", warning)
        # BUG-28: Log warnings only — don't inject comments into LaTeX body

    preamble = template.render_preamble(
        title=_escape_latex(title),
        authors=authors,
        abstract=_convert_inline(abstract),
    )
    footer = template.render_footer(bib_file)

    tex = preamble + "\n" + body + footer

    # Final sanitization pass on the complete LaTeX output
    tex = _sanitize_latex_output(tex, bib_entries=bib_entries)

    return tex


# ---------------------------------------------------------------------------
# Post-processing: sanitize final LaTeX
# ---------------------------------------------------------------------------


def _sanitize_latex_output(
    tex: str,
    *,
    bib_entries: dict[str, str] | None = None,
) -> str:
    """Remove artifacts that slip through pre-processing into the final .tex."""
    # 0. BUG-102 safety net: Convert remaining author-year citations to \cite{}.
    #    If upstream conversion missed any [Author et al., 2024] patterns, catch them here.
    if bib_entries:
        for ay_pattern in sorted(bib_entries, key=len, reverse=True):
            cite_key = bib_entries[ay_pattern]
            # [Author et al., 2024] → \cite{key}
            tex = tex.replace(f"[{ay_pattern}]", f"\\cite{{{cite_key}}}")
            # Also handle inside existing brackets (multi-citation)
            tex = tex.replace(ay_pattern, f"\\cite{{{cite_key}}}")
        # Clean up double-nested \cite from multi-citation brackets:
        # [\cite{a}, \cite{b}] → \cite{a, b}
        def _merge_bracket_cites(m: re.Match[str]) -> str:
            inner = m.group(1)
            keys = re.findall(r"\\cite\{([^}]+)\}", inner)
            if keys:
                return "\\cite{" + ", ".join(keys) + "}"
            return m.group(0)
        tex = re.sub(r"\[([^\]]*\\cite\{[^\]]+)\]", _merge_bracket_cites, tex)

    # 1. Remove broken citation markers: \cite{?key:NOT_IN_BIB} or literal [?key:NOT_IN_BIB]
    tex = re.sub(r"\\cite\{\?[^}]*:NOT_IN_BIB\}", "", tex)
    tex = re.sub(r"\[\?[a-zA-Z0-9_:-]+:NOT_IN_BIB\]", "", tex)

    # 1b. Convert leftover raw bracket citations [key2019word, key2020word] → \cite{...}
    # Skip inside verbatim/lstlisting environments to avoid corrupting code blocks.
    _CITE_KEY_PAT_L = r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_]*"
    _VERBATIM_RE = re.compile(
        r"(\\begin\{(?:verbatim|lstlisting|minted)\}.*?\\end\{(?:verbatim|lstlisting|minted)\})",
        re.DOTALL,
    )
    _cite_re = re.compile(
        rf"\[({_CITE_KEY_PAT_L}(?:\s*,\s*{_CITE_KEY_PAT_L})*)\]"
    )

    def _cite_outside_verbatim(tex_src: str) -> str:
        parts = _VERBATIM_RE.split(tex_src)
        for i, part in enumerate(parts):
            if not _VERBATIM_RE.match(part):
                parts[i] = _cite_re.sub(r"\\cite{\1}", part)
        return "".join(parts)

    tex = _cite_outside_verbatim(tex)

    # 1c. BUG-110 safety net: Replace any remaining Unicode Greek/math symbols.
    #     _convert_inline handles most, but titles, captions, and preamble
    #     fragments can still contain raw Unicode that kills pdflatex.
    for _uchar, _lcmd in _UNICODE_GREEK_TO_LATEX.items():
        if _uchar in tex:
            tex = tex.replace(_uchar, _lcmd)

    # 2. Remove HTML entities that survived pre-processing
    tex = tex.replace("&nbsp;", "~")
    tex = tex.replace("&amp;", "\\&")

    # 2b. Fix escaped \& inside tabular data rows.  The converter's
    #     _convert_inline escapes & globally; inside tabular environments
    #     the & must remain unescaped as the column separator.
    if "\\begin{tabular}" in tex and "\\&" in tex:

        def _fix_tabular_amp(m: re.Match[str]) -> str:
            block = m.group(0)
            if "\\&" not in block:
                return block
            lines = block.split("\n")
            for i, line in enumerate(lines):
                if "\\&" in line and "\\\\" in line:
                    lines[i] = line.replace("\\&", "&")
            return "\n".join(lines)

        tex = re.sub(
            r"\\begin\{tabular\}.*?\\end\{tabular\}",
            _fix_tabular_amp,
            tex,
            flags=re.DOTALL,
        )

    # 3. Remove stray markdown code fences in LaTeX body (outside verbatim)
    #    Only match fences NOT inside \begin{verbatim}...\end{verbatim}
    #    Simple approach: remove ``` lines that don't have verbatim nearby
    tex = re.sub(r"^(\s*```[a-z]*\s*)$", r"% removed stray fence: \1", tex, flags=re.MULTILINE)

    # 4. Fix placeholder table captions: \caption{Table N} → descriptive
    #    Can't auto-generate content, but at least don't leave "Table 1" as
    #    the only caption text — append " -- See text for details."
    tex = re.sub(
        r"\\caption\{(Table\s+\d+)\}",
        r"\\caption{\1 -- Summary of experimental results.}",
        tex,
    )

    # 4b. Auto-map orphan \ref{fig:X} to closest \label{fig:Y} by prefix.
    #     The converter generates long labels from captions (fig:overall_cifar_100)
    #     but the LLM references short names (fig:overall).
    fig_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", tex))
    fig_refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", tex))
    orphan_refs = fig_refs - fig_labels
    orphan_labels = fig_labels - fig_refs
    if orphan_refs and orphan_labels:
        for oref in orphan_refs:
            # Find a label that starts with the ref prefix
            candidates = [l for l in orphan_labels if l.startswith(oref)]
            if len(candidates) == 1:
                tex = tex.replace(f"\\ref{{{oref}}}", f"\\ref{{{candidates[0]}}}")
                orphan_labels.discard(candidates[0])

    # 5. Fix "Untitled Paper" / "Running Title" fallback titles
    tex = re.sub(
        r"\\title\{Untitled Paper\}",
        r"\\title{[Title Generation Failed -- Manual Title Required]}",
        tex,
    )
    tex = re.sub(
        r"\\icmltitlerunning\{Running Title\}",
        "",
        tex,
    )

    # 6. Remove \texttt{} wrapped raw metric paths that the LLM dumped
    #    Handles both raw underscores and LaTeX-escaped underscores (\_)
    #    Pattern: condition/env/step/metric_name: value  (3+ path segments)
    tex = re.sub(
        r"\\texttt\{[a-zA-Z0-9_\\_/.:=-]+(?:/[a-zA-Z0-9_\\_/.:=-]+){2,}(?:\s*[=:]\s*[^}]*)?\}",
        "",
        tex,
    )

    # 6b. Remove entire \item lines that are just metric paths
    tex = re.sub(
        r"^\s*\\item\s*\\texttt\{[^}]*\}\s*$",
        "",
        tex,
        flags=re.MULTILINE,
    )

    # 7. Clean up empty \item lines that result from removed content
    tex = re.sub(r"\\item\s*\n\s*\\item", r"\\item", tex)
    # Also remove completely empty \item lines (just whitespace after \item)
    tex = re.sub(r"^\s*\\item\s*$", "", tex, flags=re.MULTILINE)

    # 8. Remove consecutive blank lines (more than 2)
    tex = re.sub(r"\n{3,}", "\n\n", tex)

    return tex


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


_OUTER_FENCE_RE = re.compile(
    r"^\s*```(?:markdown|md|latex|tex)?\s*\n(.*?)^\s*```\s*$",
    re.MULTILINE | re.DOTALL,
)

# Greedy variant — matches the *last* closing fence so inner code blocks
# (```text … ```) don't truncate the capture prematurely.
_OUTER_FENCE_GREEDY_RE = re.compile(
    r"^\s*```(?:markdown|md|latex|tex)?\s*\n(.*)^\s*```\s*$",
    re.MULTILINE | re.DOTALL,
)

# Pattern for raw metric values with excessive decimal places
# e.g. 0.9717036975193437 → 0.972
_RAW_METRIC_RE = re.compile(r"(\d+\.\d{5,})")


def _round_raw_metrics(text: str) -> str:
    """Round excessively precise metric values (>4 decimal places).

    Uses significant-figure-aware rounding so small values like
    learning rates (e.g. 0.00001) are preserved instead of becoming 0.0000.
    """
    def _rounder(m: re.Match[str]) -> str:
        try:
            val = float(m.group(1))
            if val == 0.0:
                return "0.0"
            # For very small values (< 0.001), use 2 significant figures
            # to preserve scientific meaning (e.g. lr=0.00003 → 0.00003)
            import math
            abs_val = abs(val)
            if abs_val < 0.001:
                sig_figs = 2
                digits = sig_figs - int(math.floor(math.log10(abs_val))) - 1
                return f"{val:.{digits}f}"
            # Normal range: 4 decimal places
            return f"{val:.4f}"
        except (ValueError, OverflowError):
            return m.group(0)
    return _RAW_METRIC_RE.sub(_rounder, text)


def _preprocess_markdown(md: str) -> str:
    """Clean up common LLM artifacts before parsing.

    1. Strip outer fenced code blocks (e.g. triple-backtick markdown) that LLMs
       around the entire paper content.
    2. Remove standalone Markdown horizontal rules (``---``, ``***``, ``___``).
    3. Convert blockquotes (``> text``) to a form the converter can handle.
    4. Round excessively precise metric values.
    """
    text = md

    # 1. Strip outer markdown fences (LLMs sometimes wrap entire paper in them)
    #    Repeatedly strip in case of double-wrapping.
    #    Try greedy match first (handles papers with inner code blocks),
    #    then fall back to non-greedy if greedy doesn't help.
    for _ in range(3):
        stripped = False
        for pat in (_OUTER_FENCE_GREEDY_RE, _OUTER_FENCE_RE):
            m = pat.search(text)
            if m and len(m.group(1)) > len(text) * 0.5:
                text = m.group(1)
                stripped = True
                break
        if not stripped:
            # Also handle the case where the first line is ```markdown
            # and the last non-blank line is ``` (simple boundary strip)
            lines = text.split("\n")
            first = lines[0].strip() if lines else ""
            last_idx = len(lines) - 1
            while last_idx > 0 and not lines[last_idx].strip():
                last_idx -= 1
            last = lines[last_idx].strip() if last_idx > 0 else ""
            if (
                re.match(r"^```(?:markdown|md|latex|tex)?\s*$", first)
                and last == "```"
            ):
                text = "\n".join(lines[1:last_idx])
                stripped = True
        if not stripped:
            break

    # 2. Remove standalone horizontal rules (---, ***, ___)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 2a. Strip HTML entities that LLMs inject into markdown
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&mdash;", "---")
    text = text.replace("&ndash;", "--")

    # 2b. Note: stray code fences are handled in _sanitize_latex_output
    #     after conversion, not here (to avoid breaking real code blocks).

    # 2c. Round excessively precise metric values (e.g. 0.9717036975 → 0.9717)
    text = _round_raw_metrics(text)

    # 2d. Remove raw \texttt{...} or backtick-wrapped metric key paths
    # Pattern: \texttt{some/long/metric_path/name: 0.1234} or `path/to/metric: val`
    text = re.sub(
        r"\\texttt\{[a-zA-Z0-9_/.:=-]+(?:/[a-zA-Z0-9_/.:=-]+){2,}(?:\s*[=:]\s*[^}]*)?\}",
        "",
        text,
    )
    # Also strip backtick-wrapped metric paths in markdown source
    text = re.sub(
        r"`[a-zA-Z0-9_/.-]+(?:/[a-zA-Z0-9_/.-]+){2,}(?:\s*[=:]\s*[^`]*)?`",
        "",
        text,
    )

    # 2e. Clean NOT_IN_BIB citation markers: [?key:NOT_IN_BIB] → remove
    text = re.sub(r"\[\?[a-zA-Z0-9_:-]+:NOT_IN_BIB\]", "", text)

    # 3. Convert blockquotes: > text → \begin{quote}text\end{quote}
    #    Collect consecutive > lines into a single quote block.
    lines = text.split("\n")
    out_lines: list[str] = []
    in_quote = False
    quote_buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("> "):
            if not in_quote:
                in_quote = True
                quote_buf = []
            quote_buf.append(stripped[2:])
        elif stripped == ">" and in_quote:
            quote_buf.append("")
        else:
            if in_quote:
                out_lines.append("\\begin{quote}")
                out_lines.extend(quote_buf)
                out_lines.append("\\end{quote}")
                in_quote = False
                quote_buf = []
            out_lines.append(line)
    if in_quote:
        out_lines.append("\\begin{quote}")
        out_lines.extend(quote_buf)
        out_lines.append("\\end{quote}")
    text = "\n".join(out_lines)

    # 4. T1.2: Remove stray markdown/latex/text fences that appear mid-document.
    #    LLMs sometimes emit ```markdown or ```latex between sections.
    #    Only remove documentation fences — preserve code fences (```python etc.)
    _CODE_LANGS = frozenset({
        "python", "java", "cpp", "c", "javascript", "typescript", "rust",
        "go", "ruby", "bash", "sh", "sql", "r", "julia", "lua", "perl",
        "scala", "kotlin", "swift", "haskell", "algorithm", "pseudocode",
    })
    _lines = text.split("\n")
    _cleaned: list[str] = []
    _in_code = False
    for _l in _lines:
        _stripped = _l.strip()
        if _stripped.startswith("```") and not _in_code:
            _lang = _stripped[3:].strip().lower()
            if _lang in _CODE_LANGS or _lang.startswith("algorithm"):
                # Real code block — keep
                _in_code = True
                _cleaned.append(_l)
            elif _lang in ("markdown", "md", "latex", "tex", "text", "", "bibtex"):
                # Documentation/wrapper fence — remove
                pass
            else:
                # Unknown lang — keep to be safe
                _in_code = True
                _cleaned.append(_l)
        elif _stripped == "```" and _in_code:
            # Closing fence for a code block — keep
            _in_code = False
            _cleaned.append(_l)
        elif _stripped == "```" and not _in_code:
            # Stray fence — remove
            pass
        else:
            _cleaned.append(_l)
    text = "\n".join(_cleaned)

    # 5. Normalize mid-line section headings (IMP-17)
    #    LLM output may concatenate sections onto single long lines:
    #      "...text ## Abstract Body text ## 1. Introduction More text..."
    #    Ensure each heading marker starts on its own line so _parse_sections
    #    can detect them with the ^-anchored regex.
    text = re.sub(r"(?<=[^\n]) +(#{1,4}) +", r"\n\n\1 ", text)

    return text


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """A parsed Markdown section."""

    level: int  # 1 = ``#``, 2 = ``##``, 3 = ``###``, etc.
    heading: str
    body: str
    heading_lower: str = field(init=False)

    def __post_init__(self) -> None:
        self.heading_lower = self.heading.strip().lower()


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Known section heading names used to separate heading from concatenated body
_KNOWN_SECTION_NAMES = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "framework",
    "experiments",
    "experiment",
    "experimental setup",
    "experimental results",
    "results",
    "results and discussion",
    "analysis",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
    "contributions",
    "problem setting",
    "problem statement",
    "problem definition",
    "problem formulation",
    "study positioning",
    "study positioning and scope",
    "evaluation",
    "evaluation environment",
    "design rationale",
    "complexity",
    "unified algorithm",
    "method positioning",
    "methods compared",
    "common protonet backbone",
    "preference optimization backbone",
}


_HEADING_CONNECTORS = frozenset(
    {
        "and", "or", "for", "in", "of", "the", "a", "an", "with",
        "under", "to", "on", "at", "by", "as", "via", "from",
        "not", "but", "yet", "nor", "vs", "versus", "is", "are",
    }
)

_SENTENCE_STARTERS = frozenset(
    {
        "the", "a", "an", "this", "these", "those", "that",
        "it", "we", "our", "their", "its", "each", "every",
        "in", "for", "to", "here", "there", "however", "moreover",
        "furthermore", "additionally", "specifically", "notably",
        "all", "many", "several", "some", "most", "both",
        "among", "between", "across", "unlike", "given", "such",
        "while", "although", "because", "since", "when", "where",
        "rather", "let", "table", "figure", "as", "at", "if",
    }
)


def _separate_heading_body(heading: str) -> tuple[str, str]:
    """Separate heading text from accidentally concatenated body text.

    LLM output may produce lines like ``## Abstract Body text here...``
    where the heading is just ``Abstract`` and the rest is body.

    Returns (heading, extra_body) where extra_body may be empty.
    """
    # Very short headings are fine as-is
    if len(heading) <= 60:
        return heading, ""

    # Strip optional leading section number for matching
    num_match = re.match(r"^(\d+(?:\.\d+)*\.?\s+)", heading)
    num_prefix = num_match.group(1) if num_match else ""
    rest = heading[len(num_prefix):]
    rest_lower = rest.lower()

    # Check against known section heading names
    for name in sorted(_KNOWN_SECTION_NAMES, key=len, reverse=True):
        if rest_lower.startswith(name) and len(rest) > len(name) + 1:
            after = rest[len(name) :]
            if after and after[0] in " \t":
                return (num_prefix + rest[: len(name)]).strip(), after.strip()

    # Word-count heuristic for unknown subsection headings.
    # Scan for the first plausible heading-body boundary.
    words = heading.split()
    if len(words) > 6:
        for n in range(2, min(12, len(words) - 2)):
            curr = words[n]
            if not curr or not curr[0].isupper():
                continue
            prev_word = words[n - 1].rstrip(".,;:").lower()
            if prev_word in _HEADING_CONNECTORS:
                continue
            remaining = " ".join(words[n:])
            if len(remaining) <= 30:
                continue
            # Strong signal: common sentence-starting word
            if curr.lower() in _SENTENCE_STARTERS:
                return " ".join(words[:n]).strip(), remaining.strip()
            # Medium signal: next word is lowercase (sentence-like)
            # and heading has >= 4 words, body is substantial (> 100 chars)
            if n >= 4 and n + 1 < len(words):
                next_w = words[n + 1].rstrip(".,;:")
                if next_w and next_w[0].islower() and len(remaining) > 100:
                    return " ".join(words[:n]).strip(), remaining.strip()
            # Weak fallback for very long headings (conservative)
            if n >= 8 and len(remaining) > 100:
                return " ".join(words[:n]).strip(), remaining.strip()

    # Detect repeated multi-word opening phrase: the body often starts with
    # the same words as the heading (e.g. "Graph-memory methods Graph-memory
    # methods maintain a graph...").
    half = len(rest) // 2
    for phrase_len in range(min(30, half), 14, -1):
        phrase = rest[:phrase_len]
        if " " not in phrase:
            continue
        repeat_pos = rest.find(phrase, phrase_len)
        if repeat_pos > 0:
            return (
                (num_prefix + rest[:repeat_pos]).strip(),
                rest[repeat_pos:].strip(),
            )

    # Fallback: try to split at a sentence boundary within first 200 chars
    if len(heading) > 200:
        m = re.search(r"[.;:]\s+([A-Z])", heading[:300])
        if m and m.start() > 10:
            return heading[: m.start() + 1].strip(), heading[m.start() + 2 :].strip()

    return heading, ""


def _parse_sections(md: str) -> list[_Section]:
    """Split Markdown into a flat list of sections by heading."""
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [_Section(level=1, heading="", body=md)]

    sections: list[_Section] = []

    # Text before first heading (if any)
    if matches[0].start() > 0:
        preamble_text = md[: matches[0].start()].strip()
        if preamble_text:
            sections.append(_Section(level=0, heading="", body=preamble_text))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()

        # IMP-17: Handle concatenated heading+body on same line
        heading, body_prefix = _separate_heading_body(heading)
        if body_prefix:
            body = body_prefix + ("\n\n" + body if body else "")

        sections.append(_Section(level=level, heading=heading, body=body))

    return sections


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_TITLE_SKIP = {
    "title",
    "abstract",
    "references",
    "appendix",
    "acknowledgments",
    "acknowledgements",
}

# T1.1: Headings that are NOT valid paper titles (tables, figures, etc.)
_TITLE_REJECT_RE = re.compile(
    r"^(?:table|figure|fig\.|tab\.|algorithm|listing|appendix)\s",
    re.IGNORECASE,
)

# T1.1: Headings that look like metric dumps rather than titles
_METRIC_DUMP_RE = re.compile(
    r"(?:primary_metric|accuracy|loss|f1_score|precision|recall)\b",
    re.IGNORECASE,
)


def _extract_title(sections: list[_Section], raw_md: str) -> str:
    """Extract paper title from sections or raw markdown."""
    # Look for an explicit "# Title" or "## Title" section whose body is the
    # actual title, or whose heading is "## Title Actual Paper Title".
    for sec in sections:
        if sec.level in (1, 2) and sec.heading_lower == "title":
            # The body often starts with **Bold Title** on the first line
            first_line = sec.body.split("\n")[0].strip()
            # Strip bold markers
            first_line = re.sub(r"\*\*(.+?)\*\*", r"\1", first_line)
            if first_line and not _is_bad_title(first_line):
                return first_line
        # Handle "## Title Actual Paper Title" pattern (title embedded in heading)
        if sec.level in (1, 2) and sec.heading_lower.startswith("title ") and len(sec.heading) > 6:
            return sec.heading[6:].strip()

    # Fallback: first H1/H2 heading that isn't a meta-heading or artefact
    for sec in sections:
        if (
            sec.level in (1, 2)
            and sec.heading
            and sec.heading_lower not in _TITLE_SKIP
            and not _is_bad_title(sec.heading)
        ):
            return sec.heading

    # Last resort: first non-empty line (still filtered)
    for line in raw_md.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not _is_bad_title(stripped):
            return stripped
    return "Untitled Paper"


def _is_bad_title(candidate: str) -> bool:
    """Return True if *candidate* is clearly not a paper title."""
    # Reject "Table 1 – ...", "Figure 2: ...", etc.
    if _TITLE_REJECT_RE.match(candidate):
        return True
    # Reject raw metric key dumps
    if _METRIC_DUMP_RE.search(candidate):
        return True
    # Reject if it contains raw underscore variable names (e.g. primary_metric)
    if re.search(r"\w+_\w+/\w+", candidate):
        return True
    return False


def _extract_abstract(sections: list[_Section]) -> str:
    """Extract abstract text from sections."""
    for sec in sections:
        if sec.heading_lower == "abstract":
            return sec.body
        # IMP-17 fallback: heading may still contain body text if
        # _separate_heading_body didn't recognise the pattern.
        if sec.heading_lower.startswith("abstract ") and len(sec.heading) > 20:
            extra = sec.heading[len("Abstract") :].strip()
            return extra + ("\n\n" + sec.body if sec.body else "")
    return ""


# ---------------------------------------------------------------------------
# Body building
# ---------------------------------------------------------------------------

_SKIP_HEADINGS = {"title", "abstract"}


def _build_body(sections: list[_Section], *, title: str = "") -> str:
    """Convert all non-title/abstract sections to LaTeX body text.

    When a paper has its title as an H1 heading (``# My Paper Title``),
    that heading is already rendered via ``\\title{}`` in the preamble.
    We skip it here and promote remaining headings so that H2 (``##``)
    maps to ``\\section``, H3 to ``\\subsection``, etc.
    """
    title_lower = title.strip().lower()

    # Determine minimum heading level used for real body sections
    # (skip title/abstract/references).
    title_h1_found = False
    for sec in sections:
        if (
            sec.level == 1
            and sec.heading
            and sec.heading.strip().lower() == title_lower
        ):
            title_h1_found = True
            break

    # T1.3: Auto-detect when all body sections use H2 (##) instead of H1 (#).
    # This happens when the LLM uses ## for main sections (Introduction, Method, etc.)
    # without an explicit H1 title heading. We must promote H2→\section.
    body_levels: set[int] = set()
    for sec in sections:
        if sec.heading_lower not in _SKIP_HEADINGS and sec.level >= 1:
            if not (sec.level == 1 and sec.heading.strip().lower() == title_lower):
                body_levels.add(sec.level)

    min_body_level = min(body_levels) if body_levels else 1

    # Promote if: (a) title was H1 and body starts at H2, OR
    # (b) no title H1 found but all body sections are H2+ (LLM omitted H1 title)
    # BUG-166: When title is H1 AND body also uses H1 for main sections,
    # offset must be 0 — otherwise H1→max(1,1-1)=1 and H2→max(1,2-1)=1
    # both collapse to \section, losing all subsection hierarchy.
    if title_h1_found:
        level_offset = 1 if min_body_level >= 2 else 0
    elif min_body_level >= 2:
        # All body sections are H2 or deeper — promote so H2→\section
        level_offset = min_body_level - 1
    else:
        level_offset = 0

    _level_map = {
        1: "section",
        2: "subsection",
        3: "subsubsection",
        4: "paragraph",
    }

    parts: list[str] = []
    for sec in sections:
        # Skip title-only and abstract sections
        if sec.heading_lower in _SKIP_HEADINGS:
            continue
        # Skip the H1 heading that was used as the paper title
        if (
            sec.level == 1
            and sec.heading
            and sec.heading.strip().lower() == title_lower
        ):
            continue
        if sec.level == 0:
            # Preamble text before any heading — include as-is
            parts.append(_convert_block(sec.body))
            continue

        effective_level = max(1, sec.level - level_offset)
        cmd = _level_map.get(effective_level, "paragraph")
        heading_tex = _escape_latex(sec.heading)
        # Strip leading manual section numbers: "1. Introduction" → "Introduction"
        # Handles: "1 Intro", "2.1 Related", "3.2.1 Details", "1. Intro"
        heading_tex = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading_tex)
        parts.append(f"\\{cmd}{{{heading_tex}}}")
        # Generate a label for cross-referencing
        if cmd in ("section", "subsection", "subsubsection"):
            label_key = re.sub(r"[^a-z0-9]+", "_", heading_tex.lower()).strip("_")[:40]
            if label_key:
                parts.append(f"\\label{{sec:{label_key}}}")
        if sec.body:
            parts.append(_convert_block(sec.body))

    return "\n\n".join(parts) + "\n"


def _deduplicate_tables(body: str) -> str:
    """IMP-30: Remove duplicate tables that share the same header row.

    LLMs sometimes repeat tables (e.g. same results table in Results and
    Discussion). We keep the first occurrence and drop subsequent copies.
    """
    import logging as _dup_log

    _TABLE_ENV_RE = re.compile(
        r"(\\begin\{table\}.*?\\end\{table\})", re.DOTALL
    )
    tables = list(_TABLE_ENV_RE.finditer(body))
    if len(tables) < 2:
        return body

    seen_headers: dict[str, int] = {}
    drop_spans: list[tuple[int, int]] = []
    for m in tables:
        table_text = m.group(1)
        # Extract header row (first row after \toprule)
        header_match = re.search(r"\\toprule\s*\n(.+?)\\\\", table_text)
        if not header_match:
            continue
        header_key = re.sub(r"\s+", " ", header_match.group(1).strip())
        if header_key in seen_headers:
            drop_spans.append((m.start(), m.end()))
            _dup_log.getLogger(__name__).info(
                "IMP-30: Dropping duplicate table (same header as table #%d)",
                seen_headers[header_key],
            )
        else:
            seen_headers[header_key] = len(seen_headers) + 1

    # Remove duplicates in reverse order to preserve offsets
    for start, end in reversed(drop_spans):
        body = body[:start] + body[end:]

    return body


# ---------------------------------------------------------------------------
# Block-level conversion
# ---------------------------------------------------------------------------

# Patterns for block-level structures
_DISPLAY_MATH_RE = re.compile(r"^\\\[(.+?)\\\]$", re.MULTILINE | re.DOTALL)
# $$...$$ display math (single- or multi-line)
_DISPLAY_MATH_DOLLAR_RE = re.compile(
    r"^\$\$\s*\n?(.*?)\n?\s*\$\$$", re.MULTILINE | re.DOTALL
)
_FENCED_CODE_RE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
_TABLE_SEP_RE = re.compile(r"^\|[-:| ]+\|$")

# Markdown image pattern: ![caption](path)
_IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")

# Bullet / numbered list patterns
_BULLET_RE = re.compile(r"^(\s*)-\s+(.+)")
_NUMBERED_RE = re.compile(r"^(\s*)\d+\.\s+(.+)")


def _convert_block(text: str) -> str:
    """Convert a block of Markdown body text to LaTeX."""
    # Protect display math from further processing
    math_blocks: list[str] = []

    def _stash_math(m: re.Match[str]) -> str:
        idx = len(math_blocks)
        math_blocks.append(m.group(0))  # Keep \\[...\\] as-is
        return f"%%MATH_BLOCK_{idx}%%"

    def _stash_dollar_math(m: re.Match[str]) -> str:
        """Convert $$...$$ to \\begin{equation}...\\end{equation}."""
        idx = len(math_blocks)
        inner = m.group(1).strip()
        math_blocks.append(
            f"\\begin{{equation}}\n{inner}\n\\end{{equation}}"
        )
        return f"%%MATH_BLOCK_{idx}%%"

    text = _DISPLAY_MATH_RE.sub(_stash_math, text)
    # Also handle $$...$$ display math
    text = _DISPLAY_MATH_DOLLAR_RE.sub(_stash_dollar_math, text)

    # Protect fenced code blocks
    code_blocks: list[str] = []

    def _stash_code(m: re.Match[str]) -> str:
        idx = len(code_blocks)
        lang = m.group(1) or ""
        code = m.group(2)
        code_blocks.append(_render_code_block(lang, code))
        return f"%%CODE_BLOCK_{idx}%%"

    text = _FENCED_CODE_RE.sub(_stash_code, text)

    # Protect raw LaTeX environments (table, figure, algorithm, etc.)
    # These appear when pre-built LaTeX (e.g. anti-fabrication result tables)
    # is embedded directly in the markdown.  Without protection, their
    # contents go through _convert_inline which double-escapes {, }, _, &.
    latex_env_blocks: list[str] = []

    def _stash_latex_env(m: re.Match[str]) -> str:
        idx = len(latex_env_blocks)
        latex_env_blocks.append(m.group(0))
        return f"%%LATEX_ENV_{idx}%%"

    # Match \begin{env}...\end{env} for environments that should pass through.
    text = re.sub(
        r"\\begin\{(table|figure|tabular|algorithm|algorithmic|equation|align"
        r"|gather|multline|minipage|tikzpicture)\*?\}.*?"
        r"\\end\{\1\*?\}",
        _stash_latex_env,
        text,
        flags=re.DOTALL,
    )

    # Process line by line for lists, tables, and paragraphs
    lines = text.split("\n")
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for stashed blocks
        if line.strip().startswith("%%MATH_BLOCK_"):
            idx = int(re.search(r"\d+", line.strip()).group())  # type: ignore[union-attr]
            output.append(math_blocks[idx])
            i += 1
            continue

        if line.strip().startswith("%%CODE_BLOCK_"):
            idx = int(re.search(r"\d+", line.strip()).group())  # type: ignore[union-attr]
            output.append(code_blocks[idx])
            i += 1
            continue

        # Stashed LaTeX environments — pass through unchanged
        if line.strip().startswith("%%LATEX_ENV_"):
            idx = int(re.search(r"\d+", line.strip()).group())  # type: ignore[union-attr]
            output.append(latex_env_blocks[idx])
            i += 1
            continue

        # Bullet list
        if _BULLET_RE.match(line):
            items, i = _collect_list(lines, i, _BULLET_RE)
            output.append(_render_itemize(items))
            continue

        # Numbered list
        if _NUMBERED_RE.match(line):
            items, i = _collect_list(lines, i, _NUMBERED_RE)
            output.append(_render_enumerate(items))
            continue

        # Table detection (line starts with |)
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            # Check if previous line is a table caption (e.g. **Table 1: ...**)
            table_caption = ""
            if output:
                prev = output[-1].strip()
                # Match bold caption: \textbf{Table N...} (already converted)
                # or raw markdown: **Table N: ...**
                cap_m = re.match(
                    r"(?:\\textbf\{|[*]{2})\s*Table\s+\d+[.:]?\s*(.*?)(?:\}|[*]{2})$",
                    prev,
                )
                if cap_m:
                    table_caption = f"Table {cap_m.group(1)}" if cap_m.group(1) else ""
                    if not table_caption:
                        table_caption = prev
                    output.pop()  # Remove caption line from output (now inside table)
            table_lines, i = _collect_table(lines, i)
            output.append(_render_table(table_lines, caption=table_caption))
            continue

        # Markdown image: ![caption](path)
        img_match = _IMAGE_RE.match(line.strip())
        if img_match:
            output.append(_render_figure(img_match.group(1), img_match.group(2)))
            i += 1
            continue

        # Regular paragraph line
        output.append(_convert_inline(line))
        i += 1

    return "\n".join(output)


# ---------------------------------------------------------------------------
# List handling
# ---------------------------------------------------------------------------


def _collect_list(
    lines: list[str], start: int, pattern: re.Pattern[str]
) -> tuple[list[str], int]:
    """Collect consecutive list items matching *pattern*."""
    items: list[str] = []
    i = start
    while i < len(lines):
        m = pattern.match(lines[i])
        if m:
            items.append(m.group(2))
            i += 1
        elif lines[i].strip() == "":
            # Blank line — might continue list or end it
            if i + 1 < len(lines) and pattern.match(lines[i + 1]):
                i += 1  # skip blank, continue
            else:
                break
        elif lines[i].startswith("  ") or lines[i].startswith("\t"):
            # Continuation of previous item
            if items:
                items[-1] += " " + lines[i].strip()
            i += 1
        else:
            break
    return items, i


def _render_itemize(items: list[str]) -> str:
    inner = "\n".join(f"  \\item {_convert_inline(item)}" for item in items)
    return f"\\begin{{itemize}}\n{inner}\n\\end{{itemize}}"


def _render_enumerate(items: list[str]) -> str:
    inner = "\n".join(f"  \\item {_convert_inline(item)}" for item in items)
    return f"\\begin{{enumerate}}\n{inner}\n\\end{{enumerate}}"


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------


def _collect_table(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect table lines (header + separator + body rows)."""
    table: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table.append(lines[i])
        i += 1
    return table, i


def _render_table(table_lines: list[str], caption: str = "") -> str:
    """Render a Markdown table as a LaTeX tabular inside a table environment.

    IMP-23: Auto-wraps in ``\\resizebox`` when columns > 5 or any cell
    text exceeds 25 characters, preventing overflow in conference formats.
    IMP-32: Generates descriptive captions from header columns when the
    caption is empty or just 'Table N'.
    """
    if len(table_lines) < 2:
        return ""

    header = _parse_table_row(table_lines[0])
    # Skip separator (line 1)
    body_rows = [_parse_table_row(line) for line in table_lines[2:] if line.strip()]
    ncols = len(header)

    # Determine alignment from separator
    alignments = _parse_alignments(table_lines[1], ncols)
    col_spec = "".join(alignments)

    table_num = _next_table_num()

    # IMP-23: Detect wide tables that need resizebox
    max_cell_len = max(
        (len(c) for row in [header] + body_rows for c in row),
        default=0,
    )
    needs_resize = ncols > 5 or max_cell_len > 25

    lines_out: list[str] = []
    lines_out.append("\\begin{table}[ht]")
    lines_out.append("\\centering")

    # Caption ABOVE table (standard academic convention)
    if caption:
        cap_text = re.sub(r"^Table\s+\d+[.:]\s*", "", caption).strip()
        if cap_text:
            lines_out.append(f"\\caption{{{_convert_inline(cap_text)}}}")
        else:
            auto_cap = _auto_table_caption(header, table_num)
            lines_out.append(f"\\caption{{{auto_cap}}}")
    else:
        auto_cap = _auto_table_caption(header, table_num)
        lines_out.append(f"\\caption{{{auto_cap}}}")
    lines_out.append(f"\\label{{tab:{table_num}}}")

    if needs_resize:
        # BUG-109b fix: Use \columnwidth (works in both 1-col and 2-col layouts)
        # \textwidth in 2-column formats (ICML) is full page width, causing
        # floats wider than a column to be "lost" by LaTeX.
        lines_out.append("\\resizebox{\\columnwidth}{!}{%")
    lines_out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines_out.append("\\toprule")
    lines_out.append(
        " & ".join(f"\\textbf{{{_convert_inline(c)}}}" for c in header) + " \\\\"
    )
    lines_out.append("\\midrule")
    for row in body_rows:
        # Pad row to match header length
        padded = row + [""] * (ncols - len(row))
        lines_out.append(
            " & ".join(_convert_inline(c) for c in padded[:ncols]) + " \\\\"
        )
    lines_out.append("\\bottomrule")
    lines_out.append("\\end{tabular}")
    if needs_resize:
        lines_out.append("}")  # close resizebox
    lines_out.append("\\end{table}")

    return "\n".join(lines_out)


def _auto_table_caption(header: list[str], table_num: int) -> str:
    """IMP-32: Generate a descriptive caption from table header columns."""
    if len(header) <= 1:
        return f"Table {table_num}"
    cols = [c.strip() for c in header if c.strip()]
    if len(cols) < 2:
        return f"Table {table_num}"
    col0 = cols[0].lower()
    rest = [_convert_inline(c) for c in cols[1:min(5, len(cols))]]
    # Detect common table types by first-column header
    _HP_HINTS = {"hyperparameter", "parameter", "param", "hp", "setting", "config"}
    _ABL_HINTS = {"component", "variant", "ablation", "configuration", "module"}
    _MODEL_HINTS = {"model", "method", "approach", "algorithm", "baseline"}
    if any(h in col0 for h in _HP_HINTS):
        return f"Hyperparameter settings"
    if any(h in col0 for h in _ABL_HINTS):
        return f"Ablation study results across {', '.join(rest)}"
    if any(h in col0 for h in _MODEL_HINTS):
        return f"Performance comparison of different methods on {', '.join(rest)}"
    return f"Comparison of {_convert_inline(cols[0])} across {', '.join(rest)}"


def _parse_table_row(line: str) -> list[str]:
    """Parse ``| a | b | c |`` into ``['a', 'b', 'c']``."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _parse_alignments(sep_line: str, ncols: int) -> list[str]:
    """Parse alignment indicators from separator line."""
    cells = _parse_table_row(sep_line)
    aligns: list[str] = []
    for cell in cells:
        raw = cell.strip()
        left = raw.startswith(":")
        right = raw.endswith(":")
        if left and right:
            aligns.append("c")
        elif right:
            aligns.append("r")
        else:
            aligns.append("l")
    # Pad to ncols
    while len(aligns) < ncols:
        aligns.append("l")
    return aligns[:ncols]


# ---------------------------------------------------------------------------
# Code block rendering
# ---------------------------------------------------------------------------


_UNICODE_TO_ASCII: dict[str, str] = {
    "\u2190": "<-",   "\u2192": "->",   "\u21d0": "<=",   "\u21d2": "=>",
    "\u2264": "<=",   "\u2265": ">=",   "\u2260": "!=",   "\u2248": "~=",
    "\u2208": " in ", "\u2209": " not in ",
    "\u2200": "forall ", "\u2203": "exists ",
    "\u2207": "nabla", "\u221e": "inf",  "\u00b1": "+/-",
    "\u00d7": "x",    "\u00b7": "*",    "\u2026": "...",
    "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
    "\u03b4": "delta", "\u03b5": "epsilon", "\u03b6": "zeta",
    "\u03b7": "eta",   "\u03b8": "theta", "\u03b9": "iota",
    "\u03ba": "kappa", "\u03bb": "lambda", "\u03bc": "mu",
    "\u03bd": "nu",    "\u03be": "xi",    "\u03c0": "pi",
    "\u03c1": "rho",   "\u03c3": "sigma",  "\u03c4": "tau",
    "\u03c5": "upsilon", "\u03c6": "phi", "\u03c7": "chi",
    "\u03c8": "psi",   "\u03c9": "omega",
    "\u0394": "Delta", "\u0398": "Theta", "\u039b": "Lambda",
    "\u03a3": "Sigma", "\u03a6": "Phi",   "\u03a8": "Psi",
    "\u03a9": "Omega",
    "\u2113": "ell",   "\u2202": "d",     "\u222b": "int",
}


# BUG-110: Unicode Greek → LaTeX math replacements for inline text.
# Used in _convert_inline() and _sanitize_latex_output().
_UNICODE_GREEK_TO_LATEX: dict[str, str] = {
    # Lowercase
    "\u03b1": "$\\alpha$", "\u03b2": "$\\beta$", "\u03b3": "$\\gamma$",
    "\u03b4": "$\\delta$", "\u03b5": "$\\epsilon$", "\u03b6": "$\\zeta$",
    "\u03b7": "$\\eta$", "\u03b8": "$\\theta$", "\u03b9": "$\\iota$",
    "\u03ba": "$\\kappa$", "\u03bb": "$\\lambda$", "\u03bc": "$\\mu$",
    "\u03bd": "$\\nu$", "\u03be": "$\\xi$", "\u03c0": "$\\pi$",
    "\u03c1": "$\\rho$", "\u03c3": "$\\sigma$", "\u03c4": "$\\tau$",
    "\u03c5": "$\\upsilon$", "\u03c6": "$\\phi$", "\u03c7": "$\\chi$",
    "\u03c8": "$\\psi$", "\u03c9": "$\\omega$",
    # Uppercase
    "\u0393": "$\\Gamma$", "\u0394": "$\\Delta$", "\u0398": "$\\Theta$",
    "\u039b": "$\\Lambda$", "\u039e": "$\\Xi$", "\u03a0": "$\\Pi$",
    "\u03a3": "$\\Sigma$", "\u03a6": "$\\Phi$", "\u03a8": "$\\Psi$",
    "\u03a9": "$\\Omega$",
    # Common math symbols not already handled
    "\u2200": "$\\forall$", "\u2203": "$\\exists$",
    "\u2207": "$\\nabla$", "\u2202": "$\\partial$",
    "\u2026": "\\ldots{}", "\u22c5": "$\\cdot$",
    "\u2113": "$\\ell$", "\u222b": "$\\int$",
    "\u2209": "$\\notin$",
    # Common symbols that cause null-byte corruption if not converted
    "\u00b1": "$\\pm$",        # ±
    "\u00d7": "$\\times$",     # ×
    "\u2248": "$\\approx$",    # ≈
    "\u2264": "$\\leq$",       # ≤
    "\u2265": "$\\geq$",       # ≥
    "\u2260": "$\\neq$",       # ≠
    "\u221e": "$\\infty$",     # ∞
    # Additional symbols found in Runs 49-52
    "\u2212": "$-$",           # − (minus sign, distinct from hyphen)
    "\u2282": "$\\subset$",    # ⊂
    "\u222a": "$\\cup$",       # ∪
    "\u211d": "$\\mathbb{R}$", # ℝ
    "\u0302": "\\^{}",         # ̂  (combining circumflex)
    "\u0303": "\\~{}",         # ̃  (combining tilde — Run 61 pseudocode)
    "\u221d": "$\\propto$",    # ∝ (proportional to)
    "\u2208": "$\\in$",        # ∈
}

_ALGO_KEYWORDS = re.compile(
    r"\b(Input|Output|Return|While|For|If|Else|Repeat|Until|Function|Procedure|Algorithm)\b",
    re.IGNORECASE,
)


def _escape_algo_line(line: str) -> str:
    """Escape LaTeX special characters in an algorithmic pseudocode line.

    BUG-177: Raw pseudocode lines contain Python/math syntax that breaks
    pdflatex: ``#`` (comment char), ``_`` (subscript), ``%`` (comment),
    ``&`` (alignment), ``{}``, ``~``, ``^``.

    Strategy:
    1. Convert ``# comment`` at end of line → ``\\COMMENT{comment}``
    2. Protect existing LaTeX commands and math delimiters
    3. Escape remaining special characters
    """
    # Step 1: Convert Python-style end-of-line comments → \COMMENT{...}
    # Match `# comment` that isn't at the start of the line (those are full-line comments)
    _comment_match = re.search(r"(?<=\s)#\s*(.+)$", line)
    comment_suffix = ""
    if _comment_match:
        comment_text = _comment_match.group(1).strip()
        line = line[: _comment_match.start()].rstrip()
        comment_suffix = f" \\COMMENT{{{comment_text}}}"
    elif line.strip().startswith("#"):
        # Full-line comment
        comment_text = line.strip().lstrip("#").strip()
        return f"\\COMMENT{{{comment_text}}}"

    # Step 2: Protect existing LaTeX commands and math mode from escaping
    protected: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00ALG{idx}\x00"

    # Protect: \command{...}, $...$, \(...\)
    line = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", _protect, line)
    line = re.sub(r"\$[^$]+\$", _protect, line)
    line = re.sub(r"\\\(.+?\\\)", _protect, line)

    # Step 3: Escape special characters
    line = line.replace("&", "\\&")
    line = line.replace("%", "\\%")
    line = line.replace("#", "\\#")
    line = line.replace("_", "\\_")
    line = line.replace("{", "\\{")
    line = line.replace("}", "\\}")
    line = line.replace("~", "\\textasciitilde{}")
    line = line.replace("^", "\\textasciicircum{}")

    # Step 4: Restore protected regions
    for idx, val in enumerate(protected):
        line = line.replace(f"\x00ALG{idx}\x00", val)

    return line + comment_suffix


def _render_code_block(lang: str, code: str) -> str:
    """Render a fenced code block as a LaTeX environment.

    IMP-28: Detects pseudocode blocks (language hint 'algorithm' /
    'pseudocode', or 3+ algorithm keywords) and renders them inside an
    ``algorithm`` + ``algorithmic`` environment instead of verbatim.

    Replaces Unicode characters (Greek letters, arrows, math symbols)
    with ASCII equivalents so pdflatex can compile the block.
    """
    import unicodedata

    escaped = code.rstrip("\n")
    for uni, ascii_eq in _UNICODE_TO_ASCII.items():
        escaped = escaped.replace(uni, ascii_eq)
    # Strip combining characters (tildes, hats, etc.) that break pdflatex
    escaped = "".join(
        c for c in escaped if not unicodedata.combining(c)
    )

    # IMP-28: Detect pseudocode and use algorithm environment
    lang_lower = lang.lower().strip()
    is_algo = lang_lower in ("algorithm", "pseudocode", "algo")
    if not is_algo:
        # Heuristic: ≥3 algorithm keywords → treat as pseudocode
        is_algo = len(_ALGO_KEYWORDS.findall(escaped)) >= 3

    if is_algo:
        # Extract caption from first comment line if present
        algo_lines = escaped.split("\n")
        caption = "Algorithm"
        if algo_lines and algo_lines[0].strip().startswith("//"):
            caption = algo_lines[0].strip().lstrip("/ ").strip()
            algo_lines = algo_lines[1:]
        # Wrap raw lines in \STATE unless they already use algorithmic commands
        _algo_cmds = {"\\STATE", "\\IF", "\\ELSE", "\\ELSIF", "\\ENDIF",
                       "\\FOR", "\\ENDFOR", "\\WHILE", "\\ENDWHILE",
                       "\\REPEAT", "\\UNTIL", "\\RETURN", "\\REQUIRE", "\\ENSURE"}
        wrapped_lines = []
        for al in algo_lines:
            stripped = al.strip()
            if not stripped:
                continue
            if any(stripped.startswith(cmd) for cmd in _algo_cmds):
                wrapped_lines.append(stripped)
            else:
                # BUG-177: Escape LaTeX special chars in pseudocode lines
                wrapped_lines.append(f"\\STATE {_escape_algo_line(stripped)}")
        body = "\n".join(wrapped_lines)
        return (
            "\\begin{algorithm}[ht]\n"
            f"\\caption{{{_convert_inline(caption)}}}\n"
            "\\begin{algorithmic}[1]\n"
            f"{body}\n"
            "\\end{algorithmic}\n"
            "\\end{algorithm}"
        )

    return f"\\begin{{verbatim}}\n{escaped}\n\\end{{verbatim}}"


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _render_figure(caption: str, path: str) -> str:
    """Render a markdown image as a LaTeX figure environment."""
    fig_num = _next_figure_num()
    # Sanitize path for LaTeX: replace spaces, keep underscores
    path = path.replace(" ", "_")
    cap_tex = _convert_inline(caption) if caption else f"Figure {fig_num}"
    label_key = re.sub(r"[^a-z0-9]+", "_", caption.lower()).strip("_")[:30]
    if not label_key:
        label_key = str(fig_num)
    return (
        "\\begin{figure}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.95\\columnwidth]{{{path}}}\n"
        f"\\caption{{{cap_tex}}}\n"
        f"\\label{{fig:{label_key}}}\n"
        "\\end{figure}"
    )


# ---------------------------------------------------------------------------
# Inline conversion
# ---------------------------------------------------------------------------

# Order matters: process bold before italic to avoid conflicts.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Characters that need escaping in LaTeX (but NOT inside math or \cite)
_LATEX_SPECIAL = re.compile(r"([#%&_{}])")
_LATEX_TILDE = re.compile(r"~")
_LATEX_CARET = re.compile(r"\^")
_LATEX_DOLLAR = re.compile(r"(?<!\\)\$")


def _convert_inline(text: str) -> str:
    """Convert inline Markdown formatting to LaTeX.

    Preserves:
    - Inline math ``\\(...\\)`` and ``$...$``
    - ``\\cite{...}`` references
    - Display math markers (already handled at block level)
    """
    # Normalize Unicode punctuation to LaTeX equivalents
    text = text.replace("\u2014", "---")          # em-dash —
    text = text.replace("\u2013", "--")            # en-dash –
    text = text.replace("\u201c", "``")            # left double quote "
    text = text.replace("\u201d", "''")            # right double quote "
    text = text.replace("\u2018", "`")             # left single quote '
    text = text.replace("\u2019", "'")             # right single quote '
    text = text.replace("\u00b1", "$\\pm$")        # ±
    text = text.replace("\u2248", "$\\approx$")    # ≈
    text = text.replace("\u2264", "$\\leq$")       # ≤
    text = text.replace("\u2265", "$\\geq$")       # ≥
    text = text.replace("\u2192", "$\\rightarrow$")  # →
    text = text.replace("\u2190", "$\\leftarrow$")   # ←
    text = text.replace("\u00d7", "$\\times$")     # ×
    text = text.replace("\u2260", "$\\neq$")       # ≠
    text = text.replace("\u2208", "$\\in$")         # ∈
    text = text.replace("\u221e", "$\\infty$")      # ∞

    # BUG-110: Replace Unicode Greek letters with LaTeX math equivalents.
    # These appear when LLMs emit raw Unicode (e.g. "ε-greedy" instead of
    # "$\epsilon$-greedy") and cause fatal pdflatex errors.
    for _uchar, _lcmd in _UNICODE_GREEK_TO_LATEX.items():
        if _uchar in text:
            text = text.replace(_uchar, _lcmd)

    # Protect math and cite from escaping
    protected: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00PROT{idx}\x00"

    # Protect inline math: \(...\) and $...$
    text = re.sub(r"\\\(.+?\\\)", _protect, text)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _protect, text)

    # Protect display math residuals: \[...\] and $$...$$
    text = re.sub(r"\\\[.+?\\\]", _protect, text, flags=re.DOTALL)
    text = re.sub(r"\$\$.+?\$\$", _protect, text, flags=re.DOTALL)

    # Protect \cite{...} and \textbf etc.
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", _protect, text)

    # BUG-182: Protect already-escaped LaTeX specials from double-escaping.
    # LLMs often pre-escape underscores/etc: e.g. RawObs\_PPO → should stay
    # as \_, not become \\_ which pdflatex interprets as linebreak + subscript.
    text = re.sub(r"\\([#%&_{}])", _protect, text)

    # Protect \(...\) patterns with linebreaks already handled
    # (should be caught above, but safety net)

    # Convert markdown links BEFORE escaping so URLs with _ are preserved.
    # Protect images first so they don't get matched as links.
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _protect, text)

    def _convert_and_protect_link(m: re.Match[str]) -> str:
        href = f"\\href{{{m.group(2)}}}{{{m.group(1)}}}"
        idx = len(protected)
        protected.append(href)
        return f"\x00PROT{idx}\x00"

    text = _LINK_RE.sub(_convert_and_protect_link, text)

    # Escape special LaTeX characters
    text = _LATEX_SPECIAL.sub(r"\\\1", text)
    text = _LATEX_TILDE.sub(r"\\textasciitilde{}", text)
    text = _LATEX_CARET.sub(r"\\textasciicircum{}", text)
    text = _LATEX_DOLLAR.sub(r"\\$", text)

    # Convert bold **text** → \textbf{text}
    text = _BOLD_RE.sub(r"\\textbf{\1}", text)

    # Convert italic *text* → \textit{text}
    text = _ITALIC_RE.sub(r"\\textit{\1}", text)

    # Convert inline code `text` → \texttt{text}
    text = _INLINE_CODE_RE.sub(r"\\texttt{\1}", text)

    # Links and images were already converted+protected before escaping.

    # Fallback: convert any remaining [cite_key] patterns to \cite{key}
    # This catches citations that were not converted upstream.
    # BUG-32 fix: key pattern must also match author2017keyword style keys
    # (e.g., roijers2017multiobjective, abels2019dynamic)
    _CITE_KEY_PAT = r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_]*"
    text = re.sub(
        rf"\[({_CITE_KEY_PAT}(?:\s*,\s*{_CITE_KEY_PAT})*)\]",
        r"\\cite{\1}",
        text,
    )

    # Restore protected segments in reverse order so that nested
    # markers (e.g. PROT0 inside PROT1's value) are resolved correctly.
    for idx in range(len(protected) - 1, -1, -1):
        text = text.replace(f"\x00PROT{idx}\x00", protected[idx])

    return text


# ---------------------------------------------------------------------------
# Completeness checking (R10-Fix5)
# ---------------------------------------------------------------------------

_EXPECTED_SECTIONS = {
    "introduction",
    "related work",
    "method",
    "experiment",
    "result",
    "discussion",
    "conclusion",
}

_SECTION_ALIASES: dict[str, str] = {
    "methodology": "method",
    "methods": "method",
    "proposed method": "method",
    "approach": "method",
    "experiments": "experiment",
    "experimental setup": "experiment",
    "experimental results": "result",
    "results": "result",
    "results and discussion": "result",
    "results and analysis": "result",
    "discussion and results": "result",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "summary": "conclusion",
    "background": "related work",
    "literature review": "related work",
    "prior work": "related work",
}


def check_paper_completeness(sections: list[_Section]) -> list[str]:
    """Check whether a paper contains all expected sections.

    Returns a list of warning strings.  Empty list means the paper
    structure looks complete.
    """
    warnings: list[str] = []

    # Check for valid title — look for any H1/H2 heading that could be a title
    _has_title = any(
        sec.level in (1, 2) and sec.heading_lower not in ("abstract", "introduction",
            "related work", "method", "methods", "methodology", "experiments",
            "results", "discussion", "conclusion", "limitations", "references")
        for sec in sections
    )
    if not _has_title:
        warnings.append(
            "No valid title found in paper. The output may lack proper heading structure."
        )

    found_sections: set[str] = set()
    section_headings: list[str] = []
    for sec in sections:
        if sec.level in (1, 2) and sec.heading:
            heading_lower = sec.heading.strip().lower()
            section_headings.append(heading_lower)
            if heading_lower in _EXPECTED_SECTIONS:
                found_sections.add(heading_lower)
            elif heading_lower in _SECTION_ALIASES:
                found_sections.add(_SECTION_ALIASES[heading_lower])
            else:
                for expected in _EXPECTED_SECTIONS:
                    if expected in heading_lower:
                        found_sections.add(expected)
                        break

    missing = _EXPECTED_SECTIONS - found_sections
    if missing:
        warnings.append(
            f"Missing sections: {', '.join(sorted(missing))}. "
            f"Found: {', '.join(section_headings)}"
        )

    # T2.5: Check for required conference sections (NeurIPS/ICLR mandate Limitations)
    _required_extras = {"limitations"}
    _extra_aliases = {
        "limitation": "limitations",
        "limitations and future work": "limitations",
        "limitations and broader impact": "limitations",
    }
    found_extras: set[str] = set()
    for sec in sections:
        if sec.level in (1, 2) and sec.heading:
            hl = sec.heading.strip().lower()
            if hl in _required_extras:
                found_extras.add(hl)
            elif hl in _extra_aliases:
                found_extras.add(_extra_aliases[hl])
            elif "limitation" in hl:
                found_extras.add("limitations")
    missing_extras = _required_extras - found_extras
    if missing_extras:
        warnings.append(
            f"Missing required sections for NeurIPS/ICLR: "
            f"{', '.join(sorted(missing_extras))}."
        )

    # T1.5: Abstract length and quality checks
    abstract_text = ""
    for sec in sections:
        if sec.heading_lower == "abstract":
            abstract_text = sec.body
            break
    if abstract_text:
        word_count = len(abstract_text.split())
        if word_count > 300:
            warnings.append(
                f"Abstract is {word_count} words (conference limit: 150-250). "
                f"Must be shortened."
            )
        elif word_count < 150:
            warnings.append(
                f"Abstract is only {word_count} words (expected 150-250 for conferences)."
            )
        # Detect raw variable names / metric key dumps
        raw_vars = re.findall(r"\b\w+_\w+/\w+(?:_\w+)*\s*=", abstract_text)
        if raw_vars:
            warnings.append(
                f"Abstract contains raw variable names: {raw_vars[:3]}. "
                f"Replace with human-readable descriptions."
            )

    # Detect truncation markers
    all_body = " ".join(sec.body for sec in sections)
    truncation_markers = [
        "further sections continue",
        "remaining sections unchanged",
        "sections continue unchanged",
        "content continues",
        "[to be continued]",
        "[remaining content]",
    ]
    for marker in truncation_markers:
        if marker in all_body.lower():
            warnings.append(
                f"Truncation marker detected: '{marker}'. "
                f"Paper content may be incomplete."
            )

    # Word count check
    total_words = sum(len(sec.body.split()) for sec in sections)
    if total_words < 2000:
        warnings.append(
            f"Paper body is only {total_words} words "
            f"(expected 5,000-6,500 for conference paper). "
            f"Content may be severely truncated."
        )


    # Per-section word count check (safety net during LaTeX conversion)
    from researchclaw.prompts import SECTION_WORD_TARGETS, _SECTION_TARGET_ALIASES

    for sec in sections:
        if sec.level not in (1, 2) or not sec.heading:
            continue
        canon = sec.heading_lower
        if canon not in SECTION_WORD_TARGETS:
            canon = _SECTION_TARGET_ALIASES.get(sec.heading_lower, "")
        if not canon or canon not in SECTION_WORD_TARGETS:
            continue
        lo, hi = SECTION_WORD_TARGETS[canon]
        wc = len(sec.body.split())
        if wc < int(lo * 0.6):
            warnings.append(
                f"Section '{sec.heading}' is only {wc} words "
                f"(expected {lo}-{hi}). Content may be severely truncated."
            )
        elif wc > int(hi * 1.5):
            warnings.append(
                f"Section '{sec.heading}' is {wc} words "
                f"(expected {lo}-{hi}). Consider trimming."
            )

    # Bullet density check for body sections
    _bullet_re_cc = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
    _numbered_re_cc = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
    _bullet_ok_sections = {"introduction", "limitations", "limitation", "abstract"}
    for sec in sections:
        if sec.level not in (1, 2) or not sec.heading:
            continue
        hl = sec.heading_lower
        if hl in _bullet_ok_sections:
            continue
        if not sec.body:
            continue
        total_lines = len([ln for ln in sec.body.splitlines() if ln.strip()])
        if total_lines < 4:
            continue
        bullet_count = (
            len(_bullet_re_cc.findall(sec.body))
            + len(_numbered_re_cc.findall(sec.body))
        )
        density = bullet_count / total_lines
        if density > 0.30:
            warnings.append(
                f"Section '{sec.heading}' has high bullet-point density "
                f"({bullet_count}/{total_lines} lines = {density:.0%}). "
                f"Conference papers should use flowing prose."
            )

    return warnings


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text (titles, headings).

    Does NOT escape inside math delimiters or \\commands.
    """
    # Protect math first
    protected: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00PROT{idx}\x00"

    text = re.sub(r"\\\(.+?\\\)", _protect, text)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _protect, text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", _protect, text)

    text = _LATEX_SPECIAL.sub(r"\\\1", text)
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")

    for idx, val in enumerate(protected):
        text = text.replace(f"\x00PROT{idx}\x00", val)

    return text
