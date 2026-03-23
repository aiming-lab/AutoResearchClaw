"""LaTeX formatting adapter for Overleaf compatibility."""

from __future__ import annotations

import re
from pathlib import Path


class LatexFormatter:
    """Adapt pipeline LaTeX output for Overleaf conventions."""

    @staticmethod
    def normalize_paths(content: str, figures_prefix: str = "figures/") -> str:
        """Normalize figure paths to use Overleaf-style relative paths."""
        # Replace absolute or deep-nested paths with flat figures/ prefix
        content = re.sub(
            r"\\includegraphics(\[.*?\])?\{[^}]*?([^/}]+\.(?:png|pdf|jpg|eps))\}",
            lambda m: f"\\includegraphics{m.group(1) or ''}{{{figures_prefix}{m.group(2)}}}",
            content,
        )
        return content

    @staticmethod
    def ensure_document_class(content: str) -> str:
        """Ensure the file has a \\documentclass declaration."""
        if "\\documentclass" not in content:
            content = "\\documentclass{article}\n" + content
        return content

    @staticmethod
    def strip_local_comments(content: str) -> str:
        """Remove AutoResearchClaw-internal comments from LaTeX."""
        lines = content.splitlines(keepends=True)
        return "".join(
            line for line in lines
            if not line.strip().startswith("% RESEARCHCLAW:")
        )

    @staticmethod
    def fix_encoding(content: str) -> str:
        """Ensure UTF-8 input encoding package is declared."""
        if "\\usepackage[utf8]{inputenc}" not in content and "\\usepackage{inputenc}" not in content:
            # Insert after documentclass
            content = re.sub(
                r"(\\documentclass.*?\n)",
                r"\1\\usepackage[utf8]{inputenc}\n",
                content,
                count=1,
            )
        return content

    def format_for_overleaf(self, tex_path: Path) -> str:
        """Apply all formatting steps to a LaTeX file."""
        content = tex_path.read_text(encoding="utf-8")
        content = self.ensure_document_class(content)
        content = self.fix_encoding(content)
        content = self.normalize_paths(content)
        content = self.strip_local_comments(content)
        return content
