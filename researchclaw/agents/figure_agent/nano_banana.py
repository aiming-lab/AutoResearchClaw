"""Nano Banana Agent — generates conceptual/architectural images via Gemini.

Uses Google's Gemini native image generation (Nano Banana) to create
non-data figures such as:
  - Model architecture diagrams
  - Method pipeline flowcharts
  - System overview illustrations
  - Concept/intuition diagrams

These figures complement the Code-to-Viz agent which handles data-driven
charts (bar plots, line charts, heatmaps, etc.).

Requires: ``pip install google-genai Pillow``
API key:  Set ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` env var, or
          pass via config.

References:
  - Nano Banana docs: https://ai.google.dev/gemini-api/docs/image-generation
  - Gemini 3.1 Flash Image Preview: high-efficiency, high-volume
  - Gemini 3 Pro Image Preview: professional asset production
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult
from researchclaw.utils.sanitize import sanitize_figure_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-2.5-flash-image"
_FALLBACK_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]

_ACADEMIC_STYLE_PROMPT = (
    "The image should be in a clean, professional ACADEMIC style suitable "
    "for a top-tier AI/ML research paper (NeurIPS, ICML, ICLR). "
    "Use a white or light background. Use clear labels and annotations. "
    "Avoid excessive decoration. Use a consistent color palette. "
    "Text should be legible at column width (~3.25 inches). "
    "Style: technical illustration, vector-like, clean lines."
)


class NanoBananaAgent(BaseAgent):
    """Generates conceptual/architectural figures using Gemini image generation.

    This agent uses the Gemini API (Nano Banana) to create publication-quality
    conceptual figures that complement data-driven charts from Code-to-Viz.
    """

    name = "nano_banana"

    def __init__(
        self,
        llm: Any,
        *,
        gemini_api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        output_dir: str | Path | None = None,
        aspect_ratio: str = "16:9",
        use_sdk: bool | None = None,  # None = auto-detect
    ) -> None:
        super().__init__(llm)
        self._api_key = (
            gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self._model = model
        self._output_dir = Path(output_dir) if output_dir else None
        self._aspect_ratio = aspect_ratio

        # Detect SDK availability
        self._use_sdk = use_sdk
        if self._use_sdk is None:
            try:
                import google.genai  # noqa: F401
                self._use_sdk = True
            except ImportError:
                self._use_sdk = False

        if not self._api_key:
            logger.warning(
                "No Gemini API key found. Set GEMINI_API_KEY or "
                "GOOGLE_API_KEY env var for Nano Banana image generation."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Generate images for figure decisions marked as 'image' backend.

        Context keys:
            image_figures (list[dict]): Decisions from FigureDecisionAgent
                                       with backend="image"
            topic (str): Research topic
            output_dir (str|Path): Output directory for images
        """
        image_figures = context.get("image_figures", [])
        topic = context.get("topic", "")
        output_dir = Path(
            context.get("output_dir", self._output_dir or "charts")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if not image_figures:
            # BUG-DA8-10: Use _make_result() to properly track LLM counters
            return self._make_result(
                success=True,
                data={"generated": [], "count": 0},
            )

        if not self._api_key:
            return self._make_result(
                success=False,
                error="No Gemini API key configured for Nano Banana",
                data={"generated": [], "count": 0},
            )

        generated: list[dict[str, Any]] = []

        for i, fig in enumerate(image_figures):
            figure_id = sanitize_figure_id(
                fig.get("figure_id", f"conceptual_{i + 1}")
            )
            description = fig.get("description", "")
            figure_type = fig.get("figure_type", "architecture_diagram")
            section = fig.get("section", "Method")

            # Build prompt for Gemini
            prompt = self._build_prompt(
                description=description,
                figure_type=figure_type,
                section=section,
                topic=topic,
            )

            # Generate image
            output_path = output_dir / f"{figure_id}.png"

            try:
                success = self._generate_image(
                    prompt=prompt,
                    output_path=output_path,
                )

                if success:
                    generated.append({
                        "figure_id": figure_id,
                        "figure_type": figure_type,
                        "section": section,
                        "description": description,
                        "output_path": str(output_path),
                        "path": str(output_path),
                        "title": description[:80] if description else f"Figure {figure_id}",
                        "caption": description or "",
                        "prompt": prompt,
                        "success": True,
                        "backend": "nano_banana",
                    })
                    logger.info(
                        "Generated %s: %s", figure_id, output_path
                    )
                else:
                    generated.append({
                        "figure_id": figure_id,
                        "success": False,
                        "error": "Generation returned no image",
                        "backend": "nano_banana",
                    })

            except Exception as e:
                logger.warning(
                    "Failed to generate %s via Nano Banana: %s",
                    figure_id, e,
                )
                generated.append({
                    "figure_id": figure_id,
                    "success": False,
                    "error": str(e),
                    "backend": "nano_banana",
                })

        success_count = sum(1 for g in generated if g.get("success"))

        return self._make_result(
            success=success_count > 0,
            data={
                "generated": generated,
                "count": success_count,
                "total_attempted": len(image_figures),
            },
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        *,
        description: str,
        figure_type: str,
        section: str,
        topic: str,
    ) -> str:
        """Build a Gemini prompt for academic figure generation."""
        type_guidelines = self._get_type_guidelines(figure_type)

        prompt = (
            f"Create a professional academic figure for the '{section}' "
            f"section of a research paper about: {topic}\n\n"
            f"Figure description: {description}\n\n"
            f"Style guidelines:\n{type_guidelines}\n\n"
            f"{_ACADEMIC_STYLE_PROMPT}\n\n"
            f"The figure must be publication-ready for a top-tier "
            f"AI/ML conference paper."
        )

        return prompt

    @staticmethod
    def _get_type_guidelines(figure_type: str) -> str:
        """Get specific guidelines for each figure type."""
        guidelines = {
            "architecture_diagram": (
                "- Show the model layers, connections, and data flow\n"
                "- Use boxes for layers/modules with clear labels\n"
                "- Use arrows to show data flow direction\n"
                "- Include dimensions/shapes where relevant\n"
                "- Group related components with dashed borders\n"
                "- Use a consistent left-to-right or top-to-bottom flow"
            ),
            "method_flowchart": (
                "- Show the step-by-step process flow\n"
                "- Use rounded rectangles for processes\n"
                "- Use diamonds for decision points\n"
                "- Use arrows with labels for transitions\n"
                "- Number the steps if sequential\n"
                "- Highlight key/novel steps with color"
            ),
            "pipeline_overview": (
                "- Show the full pipeline from input to output\n"
                "- Use distinct visual blocks for each stage\n"
                "- Include example inputs/outputs at each stage\n"
                "- Use consistent arrow style for data flow\n"
                "- Label each stage clearly\n"
                "- Show parallel/branching paths if applicable"
            ),
            "concept_illustration": (
                "- Illustrate the key concept or intuition\n"
                "- Use simple, clean diagrams\n"
                "- Include before/after or problem/solution comparison\n"
                "- Use visual metaphors where appropriate\n"
                "- Keep it simple enough to understand at a glance"
            ),
            "system_diagram": (
                "- Show the overall system architecture\n"
                "- Include all major components and their interactions\n"
                "- Use standard UML-like notation where appropriate\n"
                "- Show data stores, APIs, and external services\n"
                "- Include protocols/data formats for connections"
            ),
            "attention_visualization": (
                "- Show attention weights or patterns\n"
                "- Use heatmap-style coloring for attention scores\n"
                "- Include input/output sequences\n"
                "- Label attention heads if multi-head attention\n"
                "- Use clear color scale legend"
            ),
            "comparison_illustration": (
                "- Show side-by-side comparison of approaches\n"
                "- Highlight key differences with visual cues\n"
                "- Use consistent styling across comparisons\n"
                "- Include labels for each approach\n"
                "- Use checkmarks/crosses for feature comparison"
            ),
        }
        return guidelines.get(figure_type, guidelines["concept_illustration"])

    # ------------------------------------------------------------------
    # Image generation backends
    # ------------------------------------------------------------------

    def _generate_image(
        self,
        prompt: str,
        output_path: Path,
    ) -> bool:
        """Generate image via Gemini API.

        Tries google-genai SDK first, falls back to REST API.
        """
        if self._use_sdk:
            return self._generate_via_sdk(prompt, output_path)
        return self._generate_via_rest(prompt, output_path)

    def _generate_via_sdk(
        self,
        prompt: str,
        output_path: Path,
    ) -> bool:
        """Generate image using google-genai SDK."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._api_key)

            response = client.models.generate_content(
                model=self._model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=self._aspect_ratio,
                    ),
                ),
            )

            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    image.save(str(output_path))
                    return True

            logger.warning("Gemini SDK returned no image data")
            return False

        except ImportError:
            logger.warning("google-genai SDK not installed, falling back to REST")
            self._use_sdk = False
            return self._generate_via_rest(prompt, output_path)
        except Exception as e:
            logger.warning("Gemini SDK error: %s, falling back to REST", e)
            return self._generate_via_rest(prompt, output_path)

    def _generate_via_rest(
        self,
        prompt: str,
        output_path: Path,
    ) -> bool:
        """Generate image using Gemini REST API (no SDK dependency)."""
        # Validate model name to prevent URL injection
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", self._model):
            logger.error("Invalid Gemini model name: %r", self._model)
            return False

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._model}:generateContent"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": self._aspect_ratio,
                },
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # Extract image from response
            candidates = result.get("candidates", [])
            if not candidates:
                logger.warning("Gemini REST API returned no candidates")
                return False

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData", {})
                if inline_data.get("mimeType", "").startswith("image/"):
                    image_bytes = base64.b64decode(inline_data["data"])
                    output_path.write_bytes(image_bytes)
                    return True

            logger.warning("Gemini REST API returned no image parts")
            return False

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            logger.warning("Gemini REST API error %d: %s", e.code, body)
            return False
        except Exception as e:
            logger.warning("Gemini REST API error: %s", e)
            return False
