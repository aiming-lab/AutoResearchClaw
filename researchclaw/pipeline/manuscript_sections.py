"""CommonMark-aware manuscript section parsing and deterministic merging.

This module is intentionally independent of pipeline execution and LLM code.
It provides the lossless document model needed by the sectional Stage 19
redesign without changing the current Stage 19 path.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml
from markdown_it import MarkdownIt


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructureIssue:
    """A deterministic structural ambiguity found in a manuscript."""

    code: str
    message: str
    ordinal: int | None = None


class ManuscriptStructureError(ValueError):
    """Raised when strict parsing finds an ambiguous manuscript structure."""

    def __init__(self, issues: tuple[StructureIssue, ...]):
        self.issues = issues
        detail = "; ".join(issue.message for issue in issues)
        super().__init__(detail or "invalid manuscript structure")


@dataclass(frozen=True)
class ManuscriptSection:
    """A heading and its body, preserved exactly as they appeared in source."""

    section_id: str
    title: str
    level: int
    ordinal: int
    path: tuple[str, ...]
    heading_source: str
    body: str
    start_line: int
    end_line: int
    original_sha256: str

    @property
    def source(self) -> str:
        return self.heading_source + self.body


@dataclass(frozen=True)
class ManuscriptDocument:
    """Lossless section representation of a Markdown manuscript."""

    preamble: str
    sections: tuple[ManuscriptSection, ...]
    source_sha256: str
    source_length: int
    structure_issues: tuple[StructureIssue, ...] = ()


@dataclass(frozen=True)
class _Heading:
    title: str
    level: int
    container_level: int
    start_line: int
    heading_end_line: int


def parse_manuscript(
    source: str,
    *,
    strict: bool = True,
) -> ManuscriptDocument:
    """Parse Markdown headings without changing any source bytes.

    CommonMark token maps determine heading boundaries, so heading-like text in
    fenced code blocks is not treated as structure. ``strict=False`` still
    reports structural issues and is intended for diagnostics and exact
    round-trip of legacy manuscripts. Sectional revision must use strict mode.
    Tokenizer contract failures always raise in both modes. Line offsets are
    zero-based and ``end_line`` is exclusive.
    """

    if not isinstance(source, str):
        raise TypeError("source must be a string")

    lines = split_commonmark_lines_keepends(source)
    front_matter_lines = _front_matter_line_count(lines)
    markdown_source = "".join(lines[front_matter_lines:])
    headings = _extract_headings(markdown_source, line_offset=front_matter_lines)
    issues = _structure_issues(headings)

    if not headings:
        issues = issues + (
            StructureIssue(
                code="no_headings",
                message="manuscript has no CommonMark headings",
            ),
        )
        document = ManuscriptDocument(
            preamble=source,
            sections=(),
            source_sha256=_sha256(source),
            source_length=len(source),
            structure_issues=issues,
        )
        if strict:
            raise ManuscriptStructureError(issues)
        return document

    paths = _heading_paths(headings, normalized=False)
    sections: list[ManuscriptSection] = []
    for ordinal, (heading, path) in enumerate(zip(headings, paths, strict=True)):
        next_start = (
            headings[ordinal + 1].start_line
            if ordinal + 1 < len(headings)
            else len(lines)
        )
        heading_source = "".join(
            lines[heading.start_line : heading.heading_end_line]
        )
        body = "".join(lines[heading.heading_end_line : next_start])
        section_source = heading_source + body
        sections.append(
            ManuscriptSection(
                section_id=_section_id(ordinal, path),
                title=heading.title,
                level=heading.level,
                ordinal=ordinal,
                path=path,
                heading_source=heading_source,
                body=body,
                start_line=heading.start_line,
                end_line=next_start,
                original_sha256=_sha256(section_source),
            )
        )

    document = ManuscriptDocument(
        preamble="".join(lines[: headings[0].start_line]),
        sections=tuple(sections),
        source_sha256=_sha256(source),
        source_length=len(source),
        structure_issues=issues,
    )
    if strict and issues:
        raise ManuscriptStructureError(issues)
    return document


def parse_manuscript_file(
    path: Path,
    *,
    strict: bool = True,
    encoding: str = "utf-8",
) -> ManuscriptDocument:
    """Read a manuscript without newline translation and parse it."""

    with path.open("r", encoding=encoding, newline="") as handle:
        return parse_manuscript(handle.read(), strict=strict)


def merge_manuscript(
    document: ManuscriptDocument,
    replacement_bodies: Mapping[str, str] | None = None,
) -> str:
    """Merge replacement section bodies while preserving headings and order.

    Replacement content is keyed by stable ``section_id``. Unknown IDs and
    non-string bodies are rejected. Phase B validators will determine whether a
    proposed replacement body is semantically admissible before calling this
    deterministic merger.
    """

    replacements = dict(replacement_bodies or {})
    if replacements and document.structure_issues:
        raise ManuscriptStructureError(document.structure_issues)

    pristine_parts = [document.preamble]
    for section in document.sections:
        if _sha256(section.source) != section.original_sha256:
            raise ManuscriptStructureError(
                (
                    StructureIssue(
                        code="section_hash_mismatch",
                        message=f"section {section.section_id} content hash changed",
                        ordinal=section.ordinal,
                    ),
                )
            )
        pristine_parts.append(section.source)
    pristine = "".join(pristine_parts)
    if (
        len(pristine) != document.source_length
        or _sha256(pristine) != document.source_sha256
    ):
        raise ManuscriptStructureError(
            (
                StructureIssue(
                    code="round_trip_mismatch",
                    message="document content no longer matches its source hash",
                ),
            )
        )
    if not replacements:
        return pristine

    known_ids = {section.section_id for section in document.sections}
    unknown_ids = sorted(set(replacements) - known_ids)
    if unknown_ids:
        raise KeyError(f"unknown section IDs: {', '.join(unknown_ids)}")
    for section_id, body in replacements.items():
        if not isinstance(body, str):
            raise TypeError(f"replacement body for {section_id} must be a string")

    parts = [document.preamble]
    for section in document.sections:
        parts.append(section.heading_source)
        parts.append(replacements.get(section.section_id, section.body))
    return "".join(parts)


def _extract_headings(
    source: str,
    *,
    line_offset: int = 0,
) -> tuple[_Heading, ...]:
    tokens = MarkdownIt("commonmark").parse(source)
    headings: list[_Heading] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        if token.map is None or len(token.map) != 2:
            raise ManuscriptStructureError(
                (
                    StructureIssue(
                        code="heading_map_missing",
                        message="CommonMark heading token has no source map",
                    ),
                )
            )
        try:
            level = int(token.tag.removeprefix("h"))
        except ValueError as exc:
            raise ManuscriptStructureError(
                (
                    StructureIssue(
                        code="heading_level_invalid",
                        message=f"invalid heading tag: {token.tag}",
                    ),
                )
            ) from exc
        title = ""
        if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
            title = tokens[index + 1].content.strip()
        headings.append(
            _Heading(
                title=title,
                level=level,
                container_level=int(token.level),
                start_line=int(token.map[0]) + line_offset,
                heading_end_line=int(token.map[1]) + line_offset,
            )
        )
    return tuple(headings)


def _front_matter_line_count(lines: list[str]) -> int:
    if not lines or lines[0].rstrip("\r\n") != "---":
        return 0
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") in {"---", "..."}:
            candidate = "".join(lines[1:index])
            try:
                parsed = yaml.safe_load(candidate)
            except yaml.YAMLError:
                continue
            if parsed is None or isinstance(parsed, dict):
                return index + 1
    return 0


def split_commonmark_lines_keepends(source: str) -> list[str]:
    """Split only on CommonMark CR/LF sequences and preserve terminators."""

    lines: list[str] = []
    start = 0
    index = 0
    while index < len(source):
        char = source[index]
        if char == "\n":
            lines.append(source[start : index + 1])
            index += 1
            start = index
            continue
        if char == "\r":
            end = index + 2 if source[index + 1 : index + 2] == "\n" else index + 1
            lines.append(source[start:end])
            index = end
            start = index
            continue
        index += 1
    if start < len(source):
        lines.append(source[start:])
    return lines


def _heading_paths(
    headings: tuple[_Heading, ...],
    *,
    normalized: bool,
) -> tuple[tuple[str, ...], ...]:
    parent_titles: dict[int, str] = {}
    paths: list[tuple[str, ...]] = []
    for heading in headings:
        for level in tuple(parent_titles):
            if level >= heading.level:
                del parent_titles[level]
        title = _normalize_title(heading.title) if normalized else heading.title
        path = tuple(
            parent_titles[level]
            for level in sorted(parent_titles)
            if level < heading.level
        ) + (title,)
        paths.append(path)
        parent_titles[heading.level] = title
    return tuple(paths)


def _structure_issues(headings: tuple[_Heading, ...]) -> tuple[StructureIssue, ...]:
    issues: list[StructureIssue] = []
    seen_paths: dict[tuple[str, ...], int] = {}
    previous_level: int | None = None
    normalized_paths = _heading_paths(headings, normalized=True)

    for ordinal, (heading, normalized_path) in enumerate(
        zip(headings, normalized_paths, strict=True)
    ):
        if not heading.title:
            issues.append(
                StructureIssue(
                    code="empty_heading",
                    message=f"heading {ordinal} has no title",
                    ordinal=ordinal,
                )
            )
        if heading.container_level != 0:
            issues.append(
                StructureIssue(
                    code="nested_heading",
                    message=(
                        f"heading {ordinal} is nested at token level "
                        f"{heading.container_level}"
                    ),
                    ordinal=ordinal,
                )
            )
        if previous_level is not None and heading.level > previous_level + 1:
            issues.append(
                StructureIssue(
                    code="heading_level_jump",
                    message=(
                        f"heading {ordinal} jumps from h{previous_level} "
                        f"to h{heading.level}"
                    ),
                    ordinal=ordinal,
                )
            )
        if normalized_path in seen_paths:
            first = seen_paths[normalized_path]
            issues.append(
                StructureIssue(
                    code="duplicate_heading_path",
                    message=(
                        f"heading {ordinal} duplicates canonical path from "
                        f"heading {first}: {' / '.join(normalized_path)}"
                    ),
                    ordinal=ordinal,
                )
            )
        else:
            seen_paths[normalized_path] = ordinal
        previous_level = heading.level

    return tuple(issues)


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(normalized.split())


def _section_id(ordinal: int, path: tuple[str, ...]) -> str:
    canonical = "\x1f".join(_normalize_title(part) for part in path)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    ascii_title = (
        unicodedata.normalize("NFKD", path[-1])
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title).strip("-") or "section"
    return f"s{ordinal:03d}-{slug[:40]}-{digest}"
