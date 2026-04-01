"""Post-generation paper verification gate.

Extracts all numeric values from a generated LaTeX paper, compares them
against the ``VerifiedRegistry``, and rejects the paper if unverified
numbers appear in strict sections (Results, Experiments, Tables).

This is the **hard, deterministic** defense against fabrication.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from researchclaw.pipeline.verified_registry import VerifiedRegistry

logger = logging.getLogger(__name__)

# Numbers that are always allowed (years, common constants, etc.)
_ALWAYS_ALLOWED: set[float] = {
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
    0.5, 0.01, 0.001, 0.0001, 0.1, 0.05, 0.95, 0.99,
    2024.0, 2025.0, 2026.0, 2027.0,
    8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
    224.0, 299.0, 384.0,  # Common image sizes
    # BUG-192: Common hyperparameter values
    0.0003, 3e-4, 0.0005, 5e-4, 0.002, 2e-3,  # learning rates
    0.2, 0.3, 0.25, 0.7, 0.6, 0.8,  # clip epsilon, dropout, gradient clip, GCE q, common HP
    0.9, 0.999, 0.9999,  # Adam betas, momentum
    0.02, 0.03,  # weight init std
    1e-5, 1e-6, 1e-8,  # epsilon, weight decay
    300.0, 400.0, 500.0,  # epochs
    4096.0, 8192.0,  # larger batch sizes / hidden dims
}

# Regex for extracting decimal numbers (including negative, scientific notation)
# NOTE: lookbehind/lookahead must NOT exclude { } — numbers inside \textbf{91.5}
# must still be extracted.  We only exclude letters, underscore, and backslash.
_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z_\\])"   # Not preceded by letter, underscore, or backslash
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"(?![a-zA-Z_])"      # Not followed by letter or underscore
)

# Section header patterns (LaTeX)
_SECTION_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}",
    re.IGNORECASE,
)

# Patterns to SKIP (numbers inside these are not checked)
_SKIP_PATTERNS = [
    re.compile(r"\\cite\{[^}]*\}"),
    re.compile(r"\\ref\{[^}]*\}"),
    re.compile(r"\\label\{[^}]*\}"),
    re.compile(r"\\bibliographystyle\{[^}]*\}"),
    re.compile(r"\\bibliography\{[^}]*\}"),
    re.compile(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}"),
    re.compile(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}"),
    re.compile(r"% .*$", re.MULTILINE),  # Comments
    re.compile(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", re.DOTALL),
    re.compile(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", re.DOTALL),
    re.compile(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", re.DOTALL),
    re.compile(r"\\url\{[^}]*\}"),
    re.compile(r"\\href\{[^}]*\}\{[^}]*\}"),
    re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}"),
    re.compile(r"\\resizebox\{[^}]*\}\{[^}]*\}"),
    re.compile(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", re.DOTALL),
    re.compile(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", re.DOTALL),
]

# Strict sections — unverified numbers cause REJECT
_STRICT_SECTIONS: set[str] = {
    "results",
    "experiments",
    "experimental results",
    "evaluation",
    "ablation",
    "ablation study",
    "quantitative",
    "main results",
    "experimental setup",  # Catches wrong epoch/lr claims
}

# Lenient sections — unverified numbers cause WARNING only
_LENIENT_SECTIONS: set[str] = {
    "introduction",
    "related work",
    "discussion",
    "conclusion",
    "future work",
    "background",
    "preliminaries",
}


@dataclass
class UnverifiedNumber:
    """A number in the paper that doesn't match any verified value."""

    value: float
    line_number: int
    context: str  # Surrounding text (truncated)
    section: str  # Section name
    in_table: bool  # Whether inside a table environment


@dataclass
class FabricatedCondition:
    """A condition name found in paper but not in experiment data."""

    name: str
    line_number: int
    context: str


@dataclass
class VerificationResult:
    """Outcome of paper verification."""

    passed: bool
    severity: str  # "PASS" | "WARN" | "REJECT"
    unverified_numbers: list[UnverifiedNumber] = field(default_factory=list)
    fabricated_conditions: list[FabricatedCondition] = field(default_factory=list)
    strict_violations: int = 0
    lenient_violations: int = 0
    total_numbers_checked: int = 0
    total_numbers_verified: int = 0
    config_warnings: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def fabrication_rate(self) -> float:
        """Fraction of numbers that are unverified."""
        if self.total_numbers_checked == 0:
            return 0.0
        return len(self.unverified_numbers) / self.total_numbers_checked


def verify_paper(
    tex_text: str,
    registry: VerifiedRegistry,
    *,
    tolerance: float = 0.01,
    strict_sections: set[str] | None = None,
    lenient_sections: set[str] | None = None,
) -> VerificationResult:
    """Verify that all numbers in the paper are grounded in experiment data.

    Parameters
    ----------
    tex_text:
        The full LaTeX source of the paper.
    registry:
        The verified value registry built from experiment data.
    tolerance:
        Relative tolerance for number matching (default 1%).
    strict_sections:
        Section names where unverified numbers cause REJECT.
    lenient_sections:
        Section names where unverified numbers cause WARNING only.

    Returns
    -------
    VerificationResult
        Contains pass/fail status, list of unverified numbers, and summary.
    """
    if strict_sections is None:
        strict_sections = _STRICT_SECTIONS
    if lenient_sections is None:
        lenient_sections = _LENIENT_SECTIONS

    result = VerificationResult(passed=True, severity="PASS")

    # 1. Parse sections
    sections = _parse_sections(tex_text)

    # 2. Find all tables (for in_table flag)
    table_ranges = _find_table_ranges(tex_text)

    # 3. Create skip mask (positions to ignore)
    skip_mask = _build_skip_mask(tex_text)

    # 4. Extract and verify numbers
    lines = tex_text.split("\n")
    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        section = _section_at_line(sections, line_idx)
        section_lower = section.lower() if section else ""

        in_table = any(
            start <= line_idx <= end and is_results
            for start, end, is_results in table_ranges
        )

        for m in _NUMBER_RE.finditer(line):
            num_str = m.group(1)
            char_pos = _line_offset(lines, line_idx) + m.start()

            # Skip if inside a skip zone
            if skip_mask[char_pos]:
                continue

            try:
                value = float(num_str)
            except ValueError:
                continue

            if not math.isfinite(value):
                continue

            result.total_numbers_checked += 1

            # Always-allowed numbers
            if value in _ALWAYS_ALLOWED:
                result.total_numbers_verified += 1
                continue

            # Integer-like small numbers (likely indices, counts, etc.)
            # BUG-23 P1: In strict sections or tables, only auto-pass very small
            # integers (≤5) — larger counts (e.g. "20 datasets") could be fabricated.
            is_strict_ctx = _is_strict_section(section_lower, strict_sections) or in_table
            _int_limit = 5 if is_strict_ctx else 20
            if value == int(value) and abs(value) <= _int_limit:
                result.total_numbers_verified += 1
                continue

            # Check against registry
            if registry.is_verified(value, tolerance=tolerance):
                result.total_numbers_verified += 1
                continue

            # UNVERIFIED — classify severity by section
            ctx = line.strip()[:120]
            unv = UnverifiedNumber(
                value=value,
                line_number=line_num,
                context=ctx,
                section=section or "(preamble)",
                in_table=in_table,
            )
            result.unverified_numbers.append(unv)

            is_strict = _is_strict_section(section_lower, strict_sections)
            if is_strict or in_table:
                result.strict_violations += 1
            else:
                result.lenient_violations += 1

    # 5. Check for fabricated conditions
    result.fabricated_conditions = _check_condition_names(tex_text, registry, lines)

    # 5b. BUG-23 P2: Check training config claims (epochs, dataset, etc.)
    result.config_warnings = _check_training_config(tex_text, registry)

    # 6. Determine severity
    if result.strict_violations > 0 or len(result.fabricated_conditions) > 0:
        result.passed = False
        result.severity = "REJECT"
    elif result.lenient_violations > 0:
        result.passed = True
        result.severity = "WARN"
    else:
        result.passed = True
        result.severity = "PASS"

    # 7. Build summary
    result.summary = _build_summary(result)
    logger.info("Paper verification: %s", result.summary)

    return result


def verify_paper_file(
    tex_path: Path,
    registry: VerifiedRegistry,
    **kwargs,
) -> VerificationResult:
    """Convenience: verify from a file path."""
    tex_text = tex_path.read_text(encoding="utf-8")
    return verify_paper(tex_text, registry, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_sections(tex_text: str) -> list[tuple[int, str]]:
    """Parse section headings and their line positions.

    Returns list of (line_index, section_name) sorted by line_index.
    """
    sections: list[tuple[int, str]] = []
    lines = tex_text.split("\n")
    for i, line in enumerate(lines):
        m = _SECTION_RE.search(line)
        if m:
            sections.append((i, m.group(1).strip()))
    return sections


def _section_at_line(sections: list[tuple[int, str]], line_idx: int) -> str | None:
    """Return the section name that contains the given line."""
    current = None
    for sec_line, sec_name in sections:
        if sec_line <= line_idx:
            current = sec_name
        else:
            break
    return current


_STRICT_EXEMPT_KEYWORDS: set[str] = {
    "dataset", "setup", "protocol", "hyperparameter", "implementation",
    "hardware", "infrastructure", "notation", "preliminaries",
}


def _is_strict_section(section_lower: str, strict_set: set[str]) -> bool:
    """Check if a section name matches any strict section pattern.

    BUG-R49-02: Sections like "Datasets and Evaluation Protocol" contain
    the keyword "evaluation" but describe protocol parameters, not results.
    Such sections are exempted when they also contain a setup/protocol keyword.
    """
    if not section_lower:
        return False
    for strict_name in strict_set:
        if strict_name in section_lower:
            # Check for exemption: if the section also contains a
            # setup/protocol keyword, it's not a results section.
            if any(kw in section_lower for kw in _STRICT_EXEMPT_KEYWORDS):
                return False
            return True
    return False


def _find_table_ranges(tex_text: str) -> list[tuple[int, int, bool]]:
    """Find line ranges of table environments.

    Returns ``(start_line, end_line, is_results_table)`` tuples.
    Hyperparameter / configuration tables (detected by ``\\caption`` keywords)
    are marked ``is_results_table=False`` so the verifier skips strict checks
    on their numeric content (BUG-192).
    """
    _HP_CAPTION_KW = {
        "hyperparameter", "hyper-parameter", "configuration", "config",
        "setting", "training detail", "implementation detail",
    }
    ranges: list[tuple[int, int, bool]] = []
    lines = tex_text.split("\n")
    in_table = False
    start = 0
    for i, line in enumerate(lines):
        if r"\begin{table" in line:
            in_table = True
            start = i
        elif r"\end{table" in line and in_table:
            # Scan table block for \caption to determine type
            table_block = "\n".join(lines[start : i + 1]).lower()
            is_hp = any(kw in table_block for kw in _HP_CAPTION_KW)
            ranges.append((start, i, not is_hp))
            in_table = False
    return ranges


def _build_skip_mask(tex_text: str) -> list[bool]:
    """Build a per-character boolean mask of positions to skip."""
    mask = [False] * len(tex_text)
    for pattern in _SKIP_PATTERNS:
        for m in pattern.finditer(tex_text):
            for pos in range(m.start(), m.end()):
                if pos < len(mask):
                    mask[pos] = True
    return mask


def _line_offset(lines: list[str], line_idx: int) -> int:
    """Return the character offset of the start of a line."""
    offset = 0
    for i in range(line_idx):
        offset += len(lines[i]) + 1  # +1 for newline
    return offset


def _check_condition_names(
    tex_text: str,
    registry: VerifiedRegistry,
    lines: list[str],
) -> list[FabricatedCondition]:
    """Check if the paper mentions condition names that never ran."""
    fabricated: list[FabricatedCondition] = []

    # Only check if we have known conditions
    if not registry.condition_names:
        return fabricated

    # Build pattern of known condition names (exact match in text)
    # Look for condition-like names that appear in tables or bold text
    # This is heuristic — we look for unknown names that look like conditions
    known_lower = {name.lower() for name in registry.condition_names}

    # Common generic terms that should NOT be flagged as fabricated conditions
    _GENERIC_TERMS = {
        "method", "metric", "condition", "---", "",
        "model", "approach", "variant", "architecture",
        "ours", "average", "mean", "std", "total",
        "baseline", "proposed", "ablation", "default",
        "results", "table", "figure", "section",
    }

    def _is_candidate(name: str) -> bool:
        """Check if a cleaned name looks like a real condition name."""
        return bool(
            name
            and name.lower() not in known_lower
            and name.lower() not in _GENERIC_TERMS
            and not name.startswith("\\")
            and len(name) > 1
            and not name.isdigit()
            # BUG-DA8-15: Reject numeric-looking strings (e.g. "91.5" from \textbf{91.5})
            and not re.match(r'^[\d.eE+\-]+$', name)
        )

    def _clean_latex(s: str) -> str:
        s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
        return s.replace("\\_", "_").strip()

    _seen_names: set[str] = set()

    # 1. Extract potential condition names from TABLE ROWS
    for i, line in enumerate(lines):
        if "&" in line and "\\\\" in line:
            cells = line.split("&")
            if cells:
                cand_clean = _clean_latex(cells[0].strip().rstrip("\\").strip())
                if _is_candidate(cand_clean) and cand_clean.lower() not in _seen_names:
                    _seen_names.add(cand_clean.lower())
                    fabricated.append(
                        FabricatedCondition(
                            name=cand_clean,
                            line_number=i + 1,
                            context=line.strip()[:120],
                        )
                    )

    # 2. BUG-23 P2: Also check PROSE — bold/italic condition mentions in
    #    Results/Experiments sections that don't match known conditions.
    _strict_sections_lower = {
        "results", "experiments", "experimental results",
        "evaluation", "ablation", "comparison",
    }
    sections = _parse_sections(tex_text)
    for i, line in enumerate(lines):
        section = _section_at_line(sections, i)
        if not section or section.lower() not in _strict_sections_lower:
            continue
        # Find \textbf{CondName} or \textit{CondName} in prose
        for m in re.finditer(r"\\text(?:bf|it)\{([^}]+)\}", line):
            cand_clean = _clean_latex(m.group(1)).strip()
            # Only flag multi-word or snake_case names that look like conditions
            if (
                _is_candidate(cand_clean)
                and ("_" in cand_clean or cand_clean[0].isupper())
                and cand_clean.lower() not in _seen_names
            ):
                _seen_names.add(cand_clean.lower())
                fabricated.append(
                    FabricatedCondition(
                        name=cand_clean,
                        line_number=i + 1,
                        context=line.strip()[:120],
                    )
                )

    return fabricated


def _check_training_config(
    tex_text: str,
    registry: VerifiedRegistry,
) -> list[str]:
    """BUG-23 P2: Check if paper claims about training config match reality.

    Extracts epoch counts from paper text and compares against known
    training_config from the registry. Returns list of warning strings.
    """
    warnings: list[str] = []

    # Extract "trained for N epochs" or "N epochs" claims
    epoch_claims = re.findall(
        r"(?:trained?\s+(?:for\s+)?|over\s+|(?:for|with)\s+)(\d+)\s+epoch",
        tex_text,
        re.IGNORECASE,
    )
    if epoch_claims and registry.training_config:
        actual_steps = registry.training_config.get("TRAINING_STEPS")
        actual_epochs = registry.training_config.get("epochs")
        if actual_epochs is not None:
            for claim in epoch_claims:
                claimed = int(claim)
                if abs(claimed - actual_epochs) > max(5, actual_epochs * 0.3):
                    warnings.append(
                        f"Paper claims {claimed} epochs but experiment ran {int(actual_epochs)} epochs"
                    )
        elif actual_steps is not None:
            # Can't compare epochs to steps directly, but flag very large claims
            for claim in epoch_claims:
                claimed = int(claim)
                if claimed > 500:
                    warnings.append(
                        f"Paper claims {claimed} epochs — verify against actual training steps ({int(actual_steps)})"
                    )

    # Check condition count claims ("N conditions" / "N methods" / "N baselines")
    count_claims = re.findall(
        r"(\d+)\s+(?:condition|method|baseline|approach|variant)s?\b",
        tex_text,
        re.IGNORECASE,
    )
    if count_claims and registry.condition_names:
        actual_count = len(registry.condition_names)
        for claim in count_claims:
            claimed = int(claim)
            if claimed > actual_count + 1:
                warnings.append(
                    f"Paper claims {claimed} conditions/methods but only {actual_count} ran"
                )

    if warnings:
        logger.warning("Training config validation: %s", warnings)
    return warnings


def _build_summary(result: VerificationResult) -> str:
    """Build human-readable summary."""
    parts = [f"severity={result.severity}"]
    parts.append(
        f"checked={result.total_numbers_checked}, "
        f"verified={result.total_numbers_verified}, "
        f"unverified={len(result.unverified_numbers)}"
    )
    if result.strict_violations:
        parts.append(f"strict_violations={result.strict_violations}")
    if result.fabricated_conditions:
        names = [fc.name for fc in result.fabricated_conditions[:3]]
        parts.append(f"fabricated_conditions={names}")
    if result.config_warnings:
        parts.append(f"config_warnings={len(result.config_warnings)}")
    return "; ".join(parts)
