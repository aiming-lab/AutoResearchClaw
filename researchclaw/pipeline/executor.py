from __future__ import annotations

import json
import logging
import math
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.hardware import HardwareProfile, detect_hardware, ensure_torch_available, is_metric_name
from researchclaw.llm import create_llm_client
from researchclaw.llm.client import LLMClient
from researchclaw.prompts import PromptManager
from researchclaw.pipeline.stages import (
    NEXT_STAGE,
    Stage,
    StageStatus,
    TransitionEvent,
    TransitionOutcome,
    advance,
    gate_required,
)
from researchclaw.pipeline.contracts import CONTRACTS, StageContract
from researchclaw.experiment.validator import (
    CodeValidation,
    format_issues_for_llm,
    validate_code,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain detection — maps research topic to academic domain & venue context
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, tuple[list[str], str, str]] = {
    # domain_id: (keywords, display_name, top_venues)
    "ml": (
        ["machine learning", "deep learning", "neural network", "transformer",
         "reinforcement learning", "GAN", "diffusion model", "LLM", "language model",
         "computer vision", "NLP", "representation learning", "self-supervised",
         "federated learning", "meta-learning", "continual learning", "few-shot",
         "knowledge distillation", "attention mechanism", "fine-tuning", "RLHF",
         "vision transformer", "ViT", "BERT", "GPT", "autoencoder"],
        "machine learning",
        "NeurIPS, ICML, ICLR",
    ),
    "physics": (
        ["quantum", "thermodynamic", "electrodynamic", "particle physics",
         "condensed matter", "statistical mechanics", "cosmology", "astrophysics",
         "plasma", "optics", "photonics", "relativity", "gravitational",
         "PDE", "PINN", "physics-informed", "Burgers", "Navier-Stokes",
         "Darcy flow", "Schrödinger", "scientific computing", "operator learning",
         "neural operator", "Fourier neural", "DeepONet"],
        "physics",
        "Physical Review Letters, Nature Physics, JHEP",
    ),
    "chemistry": (
        ["molecular", "catalysis", "polymer", "organic chemistry", "inorganic",
         "electrochemistry", "spectroscopy", "crystallography", "drug discovery",
         "protein folding", "computational chemistry", "DFT", "force field"],
        "chemistry",
        "JACS, Nature Chemistry, Angewandte Chemie",
    ),
    "economics": (
        ["econometric", "macroeconomic", "microeconomic", "game theory",
         "market", "fiscal policy", "monetary", "behavioral economics",
         "causal inference", "panel data", "regression discontinuity",
         "instrumental variable", "supply chain", "auction"],
        "economics",
        "AER, Econometrica, QJE, Review of Economic Studies",
    ),
    "mathematics": (
        ["theorem", "proof", "conjecture", "topology", "algebra",
         "number theory", "combinatorics", "differential equation",
         "stochastic process", "functional analysis", "manifold",
         "Riemannian", "category theory", "graph theory",
         "neural ODE", "dynamical system", "Lorenz", "chaotic",
         "Lyapunov", "attractor", "ODE solver", "trajectory prediction"],
        "mathematics",
        "Annals of Mathematics, Inventiones Mathematicae, JAMS",
    ),
    "engineering": (
        ["robotics", "control system", "signal processing", "FPGA",
         "embedded system", "VLSI", "antenna", "fluid dynamics", "CFD",
         "finite element", "structural", "mechatronics", "autonomous"],
        "engineering",
        "IEEE Transactions, ASME journals, AIAA",
    ),
    "biology": (
        ["genomics", "proteomics", "transcriptomics", "CRISPR",
         "single-cell", "phylogenetic", "ecology", "neuroscience",
         "bioinformatics", "sequencing", "gene expression", "epigenetic"],
        "biology",
        "Nature, Science, Cell, PNAS",
    ),
}


def _detect_domain(topic: str, domains: tuple[str, ...] = ()) -> tuple[str, str, str]:
    """Detect research domain from topic string and config domains.

    Returns ``(domain_id, display_name, top_venues)``.
    Falls back to ``("ml", "machine learning", "NeurIPS, ICML, ICLR")``.
    """
    # If user explicitly specified domains, check them first
    for d in domains:
        d_lower = d.lower().strip()
        for did, (kws, dname, venues) in _DOMAIN_KEYWORDS.items():
            if d_lower in (did, dname) or any(k in d_lower for k in kws[:3]):
                return did, dname, venues

    # Auto-detect from topic text
    topic_lower = topic.lower()
    best_did, best_score = "ml", 0
    for did, (kws, dname, venues) in _DOMAIN_KEYWORDS.items():
        score = sum(1 for k in kws if k.lower() in topic_lower)
        if score > best_score:
            best_score = score
            best_did = did

    did = best_did
    _, dname, venues = _DOMAIN_KEYWORDS[did]
    return did, dname, venues


def _is_ml_domain(domain_id: str) -> bool:
    """Check if the detected domain is ML/AI."""
    return domain_id == "ml"


@dataclass(frozen=True)
class StageResult:
    """Outcome of executing a single stage."""

    stage: Stage
    status: StageStatus
    artifacts: tuple[str, ...]
    error: str | None = None
    decision: str = "proceed"
    evidence_refs: tuple[str, ...] = ()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_fallback_queries(topic: str) -> list[str]:
    """Extract meaningful search queries from a long topic string.

    Instead of using the raw topic as a query (which is often 200+ chars
    and returns garbage from search engines), extract noun phrases and
    domain keywords. Returns 5-10 targeted queries.
    """
    # Split on common delimiters and extract meaningful chunks
    chunks = re.split(r"[,:;()\[\]]+", topic)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 8]
    cleaned_chunks = []
    for c in chunks:
        c = re.sub(
            r"^(and|or|the|a|an|in|of|for|with|across|multiple|three|various)\s+",
            "", c, flags=re.IGNORECASE,
        )
        c = c.strip()
        if len(c) > 8:
            cleaned_chunks.append(c)
    chunks = cleaned_chunks

    # Extract key terms (words that look like domain terms, not stopwords)
    _stop = {
        "the", "and", "for", "with", "from", "that", "this", "into",
        "over", "across", "multiple", "three", "result", "comprehensive",
        "using", "based", "between", "various", "different", "several",
        "parameter", "parameters", "analysis", "approach", "method",
        "framework", "frameworks",
    }
    words = topic.lower().split()
    key_terms = [w for w in words if len(w) > 3 and w not in _stop]

    queries: list[str] = []

    # Strategy 1: Use meaningful chunks (up to 60 chars each)
    for chunk in chunks[:4]:
        if len(chunk) > 60:
            chunk = " ".join(chunk.split()[:6])
        if chunk and chunk not in queries:
            queries.append(chunk)

    # Strategy 2: Bigrams of key terms
    clean_terms = [t for t in key_terms if re.match(r"^[a-z]", t) and ":" not in t]
    for i in range(min(len(clean_terms) - 1, 4)):
        bigram = f"{clean_terms[i]} {clean_terms[i + 1]}"
        if bigram not in queries:
            queries.append(bigram)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique.append(q.strip())

    # Ensure we have at least a few useful queries
    topic_short = topic[:60].strip()
    for suffix in ("survey", "review", "benchmark", "state of the art", "recent advances"):
        if len(unique) >= 5:
            break
        candidate = f"{topic_short} {suffix}".strip()
        if candidate.lower() not in seen:
            seen.add(candidate.lower())
            unique.append(candidate)

    return unique[:10]


def _write_stage_meta(
    stage_dir: Path, stage: Stage, run_id: str, result: StageResult
) -> None:
    next_stage = NEXT_STAGE[stage]
    meta = {
        "stage_id": f"{int(stage):02d}-{stage.name.lower()}",
        "run_id": run_id,
        "status": result.status.value,
        "decision": result.decision,
        "output_artifacts": list(result.artifacts),
        "evidence_refs": list(result.evidence_refs),
        "error": result.error,
        "ts": _utcnow_iso(),
        "next_stage": int(next_stage) if next_stage is not None else None,
    }
    (stage_dir / "decision.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


_SANDBOX_SAFE_PACKAGES = {
    "numpy", "scipy", "torch", "sklearn", "matplotlib",
    "pandas", "seaborn", "tqdm", "gymnasium", "gym",
}


def _ensure_sandbox_deps(code: str, python_path: str) -> list[str]:
    """P7: Scan code imports and auto-install missing common packages."""
    import subprocess as _sp

    imports: set[str] = set()
    for line in code.splitlines():
        m = re.match(r"^(?:from|import)\s+(\w+)", line.strip())
        if m:
            imports.add(m.group(1))

    to_check = imports & _SANDBOX_SAFE_PACKAGES
    if not to_check:
        return []

    py = python_path
    py_path = Path(py)
    if not py_path.is_absolute():
        py_path = Path.cwd() / py_path

    installed: list[str] = []
    for pkg in sorted(to_check):
        try:
            r = _sp.run(
                [str(py_path), "-c", f"import {pkg}"],
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                pip_name = "scikit-learn" if pkg == "sklearn" else pkg
                logger.info("Sandbox: installing missing dependency '%s'", pip_name)
                _sp.run(
                    [str(py_path), "-m", "pip", "install", pip_name, "--quiet"],
                    capture_output=True, timeout=120,
                )
                installed.append(pip_name)
        except Exception as exc:
            logger.warning("Sandbox: failed to check/install '%s': %s", pkg, exc)

    if installed:
        logger.info("Sandbox: auto-installed packages: %s", ", ".join(installed))
    return installed


def _read_prior_artifact(run_dir: Path, filename: str) -> str | None:
    # R14-2: Sort so non-versioned dirs (stage-13) come before versioned (stage-13_v1).
    # Within the same stage number, prefer the latest (non-versioned) copy.
    def _stage_sort_key(p: Path) -> tuple[str, int]:
        name = p.name
        # Extract base stage name and version
        if "_v" in name:
            base, _, ver = name.rpartition("_v")
            try:
                return (base, -int(ver))  # Versioned: lower priority (negative version)
            except ValueError:
                return (name, -999)
        return (name, 0)  # Non-versioned: highest priority

    for stage_subdir in sorted(run_dir.glob("stage-*"), key=_stage_sort_key, reverse=True):
        candidate = stage_subdir / filename
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        if filename.endswith("/") and (stage_subdir / filename.rstrip("/")).is_dir():
            return str(stage_subdir / filename.rstrip("/"))
    return None


def _find_prior_file(run_dir: Path, filename: str) -> Path | None:
    """Like ``_read_prior_artifact`` but returns the *Path* instead of content."""
    def _stage_sort_key(p: Path) -> tuple[str, int]:
        name = p.name
        if "_v" in name:
            base, _, ver = name.rpartition("_v")
            try:
                return (base, -int(ver))
            except ValueError:
                return (name, -999)
        return (name, 0)

    for stage_subdir in sorted(run_dir.glob("stage-*"), key=_stage_sort_key, reverse=True):
        candidate = stage_subdir / filename
        if candidate.is_file():
            return candidate
    return None


def _load_hardware_profile(run_dir: Path) -> dict[str, Any] | None:
    """Load hardware_profile.json from a prior stage (usually stage-01)."""
    raw = _read_prior_artifact(run_dir, "hardware_profile.json")
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_yaml_block(text: str) -> str:
    """Extract YAML from text that may contain ACP noise.

    Strips [thinking] blocks, insight blocks, and other ACP artifacts
    before looking for YAML in markdown fences or raw text.
    """
    # Strip ACP noise: [thinking]..., insight blocks, [plan]...
    cleaned = re.sub(
        r"\[thinking\].*?(?=\n```|\n[A-Z]|\Z)",
        "", text, flags=re.DOTALL,
    )
    cleaned = re.sub(r"\[plan\].*?\n\n", "", cleaned, flags=re.DOTALL)

    # Try markdown fences first (most reliable) — on cleaned text
    if "```yaml" in cleaned:
        return cleaned.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```yml" in cleaned:
        return cleaned.split("```yml", 1)[1].split("```", 1)[0].strip()
    if "```" in cleaned:
        block = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        if block:
            return block

    # Try the original text too (in case cleaning removed too much)
    if "```yaml" in text:
        return text.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```yml" in text:
        return text.split("```yml", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0].strip()
        if block:
            return block

    # Last resort: try to find YAML-like content (lines starting with key:)
    yaml_lines: list[str] = []
    in_yaml = False
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not in_yaml and re.match(r"^[a-z_]+:", stripped):
            in_yaml = True
        if in_yaml:
            if stripped and not stripped.startswith("#"):
                yaml_lines.append(line)
            elif not stripped and yaml_lines:
                yaml_lines.append(line)
    if yaml_lines:
        return "\n".join(yaml_lines).strip()

    return text.strip()


def _safe_json_loads(text: str, default: Any) -> Any:
    """Parse JSON from text, handling noisy ACP output.

    Tries multiple strategies: direct parse, markdown fence extraction,
    balanced brace matching (largest dict wins), and array brackets.
    """
    if not text or not text.strip():
        return default

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        pass

    # Strategy 2: Find JSON in markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: Find outermost balanced braces
    brace_depth = 0
    start = -1
    candidates: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    # Try candidates from largest to smallest
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 4: Same for array [ ]
    bracket_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if bracket_depth == 0:
                start = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1

    return default


_METACLAW_SKILLS_DIR = str(Path.home() / ".metaclaw" / "skills")


def _get_evolution_overlay(run_dir: Path | None, stage_name: str) -> str:
    """Load evolution lessons + MetaClaw skills for prompt injection.

    Combines intra-run lessons (from current run's evolution dir) with
    cross-run arc-* skills (from ~/.metaclaw/skills/).

    Returns empty string if no relevant lessons/skills exist or on any error.
    """
    if run_dir is None:
        return ""
    try:
        from researchclaw.evolution import EvolutionStore

        store = EvolutionStore(run_dir / "evolution")
        return store.build_overlay(
            stage_name, max_lessons=5, skills_dir=_METACLAW_SKILLS_DIR
        )
    except Exception:  # noqa: BLE001
        return ""


def _chat_with_prompt(
    llm: LLMClient,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
    retries: int = 0,
    strip_thinking: bool = True,
) -> Any:
    """Send a chat request with optional retry on timeout/transient errors.

    Parameters
    ----------
    retries:
        Number of extra attempts after the first failure (0 = no retry).
        Uses exponential backoff: 2s, 4s, 8s, ...
    strip_thinking:
        If True (default for pipeline usage), strip ``<think>`` tags from
        the LLM response.  This prevents chain-of-thought leakage from
        breaking YAML / JSON / LaTeX parsers downstream.
    """
    import time

    messages = [{"role": "user", "content": user}]
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            if json_mode and max_tokens is not None:
                return llm.chat(messages, system=system, json_mode=True, max_tokens=max_tokens, strip_thinking=strip_thinking)
            if json_mode:
                return llm.chat(messages, system=system, json_mode=True, strip_thinking=strip_thinking)
            if max_tokens is not None:
                return llm.chat(messages, system=system, max_tokens=max_tokens, strip_thinking=strip_thinking)
            return llm.chat(messages, system=system, strip_thinking=strip_thinking)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    1 + retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                raise last_exc from None
    raise last_exc  # type: ignore[misc]  # unreachable but satisfies type checker


def _generate_neurips_checklist(
    has_experiments: bool = True,
    has_theory: bool = False,
    has_code: bool = True,
) -> str:
    """Generate a NeurIPS-style paper checklist appendix in markdown.

    This checklist is based on the NeurIPS 2025 submission requirements.
    It is appended to the paper before LaTeX conversion.
    """
    items = [
        ("Claims", "Do the main claims accurately reflect the paper's contributions and scope?", "Yes"),
        ("Limitations", "Does the paper discuss limitations of the work?", "Yes"),
    ]
    if has_theory:
        items.append(
            ("Theory", "Are all assumptions stated and proofs included?", "Yes")
        )
    items.extend([
        ("Experiments reproducibility", "Does the paper fully disclose experimental settings?", "Yes" if has_experiments else "NA"),
        ("Code and data", "Is code or data provided for reproducibility?", "Yes" if has_code else "No"),
        ("Experimental details", "Are training details and hyperparameters specified?", "Yes" if has_experiments else "NA"),
        ("Error bars", "Are error bars or confidence intervals reported?", "Yes" if has_experiments else "NA"),
        ("Compute resources", "Are compute requirements documented?", "Yes" if has_experiments else "NA"),
        ("Code of ethics", "Does the work comply with the code of ethics?", "Yes"),
        ("Broader impacts", "Are potential negative societal impacts discussed?", "Yes"),
        ("Licenses", "Are licenses for used assets respected?", "Yes"),
        ("New assets", "Are newly released assets documented?", "NA"),
        ("Human subjects", "Were IRB approvals obtained if applicable?", "NA"),
    ])

    lines = [
        "## NeurIPS Paper Checklist",
        "",
    ]
    for label, question, answer in items:
        lines.append(f"**{label}**: {question}")
        lines.append(f"Answer: [{answer}]")
        lines.append("")

    return "\n".join(lines)


def _extract_paper_title(md_text: str) -> str:
    """Extract paper title from markdown text for LaTeX generation.

    Prioritises H1 headings that appear *before* the abstract section and
    look like real titles (>= 4 words, starts with uppercase).  This avoids
    picking up pseudocode comments or algorithm step labels.
    """
    import re as _re

    # Limit search to content before Abstract heading
    abstract_pos = _re.search(
        r"^#{1,2}\s+(Abstract|ABSTRACT)", md_text, _re.MULTILINE
    )
    search_region = md_text[: abstract_pos.start()] if abstract_pos else md_text[:3000]

    _SKIP = {"title", "abstract", "references", "appendix"}
    candidates: list[str] = []

    for line in search_region.splitlines():
        line = line.strip()
        # Match H1 or H2 headings
        hm = _re.match(r"^(#{1,2})\s+(.+)$", line)
        if hm:
            heading = hm.group(2).strip()
            heading_lower = heading.lower()
            # Handle "## Title Actual Paper Title" pattern
            if heading_lower.startswith("title ") and len(heading) > 6:
                heading = heading[6:].strip()
                heading_lower = heading.lower()
            if heading_lower in _SKIP:
                continue
            candidates.append(heading)
        # Bold title line (e.g. **My Paper Title**)
        m = _re.match(r"\*\*(.+?)\*\*$", line)
        if m and len(m.group(1).split()) >= 3:
            candidates.append(m.group(1))

    # Prefer candidates that look like real titles (>= 4 words, capitalised)
    for c in candidates:
        words = c.split()
        if len(words) >= 4 and c[0].isupper():
            return c

    # Fallback: any candidate
    if candidates:
        return candidates[0]

    return "Untitled Paper"


def _generate_framework_diagram_prompt(
    paper_text: str,
    config: "RCConfig",
    *,
    llm: "LLMClient | None" = None,
) -> str:
    """Generate a text-to-image prompt for a methodology framework diagram.

    Reads the paper's method section and produces a detailed prompt suitable
    for AI image generators (DALL-E, Midjourney, etc.).  The prompt describes
    an academic-style architecture/framework overview figure.

    Returns the prompt as a Markdown string, or empty string on failure.
    """
    import re as _re

    # Extract method/approach section from paper
    _method_section = ""
    _method_patterns = [
        r"(?:^#{1,3}\s+(?:Method(?:ology)?|Approach|Proposed\s+(?:Method|Framework|Approach)|Our\s+Method|Technical\s+Approach|Model\s+Architecture).*?)(?=^#{1,3}\s+|\Z)",
    ]
    for _pat in _method_patterns:
        _match = _re.search(_pat, paper_text, _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
        if _match:
            _method_section = _match.group(0)[:3000]
            break

    if not _method_section:
        # Fallback: use abstract + first 1500 chars
        _abs_match = _re.search(
            r"(?:^#{1,2}\s+Abstract\s*\n)(.*?)(?=^#{1,2}\s+|\Z)",
            paper_text, _re.MULTILINE | _re.DOTALL | _re.IGNORECASE,
        )
        _method_section = (_abs_match.group(1)[:1500] if _abs_match else paper_text[:2000])

    title = _extract_paper_title(paper_text)
    topic = config.research.topic

    # Use LLM to generate the prompt if available
    if llm is not None:
        _system = (
            "You are an expert academic figure designer. Generate a detailed text-to-image "
            "prompt for creating a methodology framework/architecture overview diagram.\n\n"
            "Requirements:\n"
            "- Academic style: clean, professional, suitable for a top-tier ML conference paper\n"
            "- Color palette: sophisticated and harmonious (suggest specific hex colors, "
            "prefer muted blues #4477AA, teals #44AA99, warm accents #CCBB44, soft purples #AA3377)\n"
            "- Layout: left-to-right or top-to-bottom data flow, with clearly labeled components\n"
            "- Components: boxes/modules with rounded corners, directional arrows, clear labels\n"
            "- Information density: high but not cluttered — each box should have a short label\n"
            "- Text on figure: minimal, only component names and key annotations\n"
            "- Background: white or very light grey\n"
            "- Style: vector-art look, flat design with subtle shadows, NO photorealism\n\n"
            "Output ONLY the prompt text (no markdown headers, no explanations). "
            "The prompt should be 150-300 words, highly specific and actionable."
        )
        _user = (
            f"Paper title: {title}\n"
            f"Research topic: {topic}\n\n"
            f"Method section excerpt:\n{_method_section}\n\n"
            "Generate a detailed text-to-image prompt for the methodology framework diagram."
        )
        try:
            resp = _chat_with_prompt(llm, _system, _user, max_tokens=1024)
            _llm_prompt = resp.content.strip()
            if len(_llm_prompt) > 50:
                return (
                    f"# Framework Diagram Prompt\n\n"
                    f"**Paper**: {title}\n\n"
                    f"## Image Generation Prompt\n\n"
                    f"{_llm_prompt}\n\n"
                    f"## Usage Instructions\n\n"
                    f"1. Copy the prompt above into an AI image generator "
                    f"(DALL-E 3, Midjourney, Ideogram, etc.)\n"
                    f"2. Generate the image at high resolution (2048x1024 or similar landscape)\n"
                    f"3. Save as `framework_diagram.png` in the same `charts/` folder\n"
                    f"4. Insert into the paper's Method section using:\n"
                    f"   - LaTeX: `\\includegraphics[width=\\textwidth]{{charts/framework_diagram.png}}`\n"
                    f"   - Markdown: `![Framework Overview](charts/framework_diagram.png)`\n"
                )
        except Exception:
            logger.debug("Framework prompt LLM generation failed, using template")

    # Fallback: template-based prompt without LLM
    _components = []
    _component_patterns = [
        (r"(?:encoder|decoder|transformer|attention|convolution|MLP|GNN|ResNet|ViT)", "Neural Network Module"),
        (r"(?:loss|objective|criterion|training|optimization)", "Training/Optimization"),
        (r"(?:data|dataset|input|preprocessing|augmentation)", "Data Pipeline"),
        (r"(?:output|prediction|inference|evaluation)", "Output/Evaluation"),
    ]
    _method_lower = _method_section.lower()
    for pat, label in _component_patterns:
        if _re.search(pat, _method_lower):
            _components.append(label)

    if not _components:
        _components = ["Input Processing", "Core Model", "Training Loop", "Evaluation"]

    return (
        f"# Framework Diagram Prompt\n\n"
        f"**Paper**: {title}\n\n"
        f"## Image Generation Prompt\n\n"
        f"Create a clean, academic-style methodology framework diagram for a research paper "
        f"titled \"{title}\". "
        f"The diagram should show a left-to-right data flow pipeline with these main components: "
        f"{', '.join(_components)}. "
        f"Use a professional color palette with muted blues (#4477AA), teals (#44AA99), "
        f"warm yellows (#CCBB44), and soft purples (#AA3377) on a white background. "
        f"Each component should be a rounded rectangle with a short label inside. "
        f"Connect components with clean directional arrows. "
        f"Add subtle shadows for depth. Flat vector-art style, no photorealism. "
        f"High information density but visually clean. "
        f"Suitable for a top-tier machine learning conference paper (ICML/NeurIPS/ICLR). "
        f"Landscape orientation, 2048x1024 resolution.\n\n"
        f"## Usage Instructions\n\n"
        f"1. Copy the prompt above into an AI image generator "
        f"(DALL-E 3, Midjourney, Ideogram, etc.)\n"
        f"2. Generate the image at high resolution (2048x1024 or similar landscape)\n"
        f"3. Save as `framework_diagram.png` in the same `charts/` folder\n"
        f"4. Insert into the paper's Method section using:\n"
        f"   - LaTeX: `\\includegraphics[width=\\textwidth]{{charts/framework_diagram.png}}`\n"
        f"   - Markdown: `![Framework Overview](charts/framework_diagram.png)`\n"
    )


def _safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    return name[:100] or "unnamed"


def _collect_experiment_results(
    run_dir: Path,
    metric_key: str = "",
    metric_direction: str = "maximize",
) -> dict[str, Any]:
    """Aggregate experiment metrics from runs/ directory across prior stages.

    Returns a dict with ``runs``, ``metrics_summary``, ``best_run``,
    ``latex_table``, and optionally ``structured_results``.
    """
    runs_data: list[dict[str, Any]] = []
    structured_results: Any = None

    # Scan all stage dirs for runs/ subdirectory
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        # Check for structured results.json first
        results_json = stage_subdir / "results.json"
        if results_json.exists() and structured_results is None:
            try:
                structured_results = json.loads(
                    results_json.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue  # Already handled above
            parsed = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if isinstance(parsed, dict) and "metrics" in parsed:
                # Also check for structured_results inside run payload
                if "structured_results" in parsed and structured_results is None:
                    structured_results = parsed["structured_results"]
                runs_data.append(parsed)
            elif isinstance(parsed, dict) and "key_metrics" in parsed:
                # Simulated mode uses key_metrics
                parsed["metrics"] = parsed.pop("key_metrics")
                runs_data.append(parsed)

    if not runs_data:
        result: dict[str, Any] = {"runs": [], "metrics_summary": {}, "best_run": None, "latex_table": ""}
        if structured_results is not None:
            result["structured_results"] = structured_results
        return result

    # Aggregate metrics across runs
    all_metric_keys: set[str] = set()
    for r in runs_data:
        m = r.get("metrics") or {}
        if isinstance(m, dict):
            all_metric_keys.update(m.keys())

    metrics_summary: dict[str, dict[str, float | None]] = {}
    for key in sorted(all_metric_keys):
        values = []
        for r in runs_data:
            m = r.get("metrics") or {}
            if isinstance(m, dict) and key in m:
                try:
                    values.append(float(m[key]))
                except (ValueError, TypeError):
                    pass
        if values:
            metrics_summary[key] = {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "mean": round(sum(values) / len(values), 6),
                "count": len(values),
            }

    # Find best run using metric_key and metric_direction
    best_run: dict[str, Any] | None = None
    if runs_data:

        def _primary_metric(r: dict[str, Any]) -> float:
            m = r.get("metrics") or {}
            if isinstance(m, dict):
                # Try specific metric_key first
                if metric_key and metric_key in m:
                    try:
                        return float(m[metric_key])
                    except (ValueError, TypeError):
                        pass
                # Fallback to first metric
                for v in m.values():
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
            return 0.0

        _cmp = min if metric_direction == "minimize" else max
        best_run = _cmp(runs_data, key=_primary_metric)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Experiment Results}",
    ]
    if metrics_summary:
        cols = sorted(metrics_summary.keys())
        header = "Metric & Min & Max & Mean & N \\\\"
        latex_lines.append(r"\begin{tabular}{l" + "r" * 4 + "}")
        latex_lines.append(r"\hline")
        latex_lines.append(header)
        latex_lines.append(r"\hline")
        for col in cols:
            s = metrics_summary[col]
            row = f"{col} & {s['min']:.4f} & {s['max']:.4f} & {s['mean']:.4f} & {s['count']} \\\\"
            latex_lines.append(row)
        latex_lines.append(r"\hline")
        latex_lines.append(r"\end{tabular}")
    else:
        latex_lines.append(r"\begin{tabular}{l}")
        latex_lines.append("No experiment data available \\\\")
        latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    # R18-1: Extract paired statistical comparisons from stdout
    from researchclaw.experiment.sandbox import extract_paired_comparisons

    paired_comparisons: list[dict[str, object]] = []
    for r in runs_data:
        stdout = r.get("stdout", "")
        if stdout:
            paired_comparisons.extend(extract_paired_comparisons(stdout))

    collected: dict[str, Any] = {
        "runs": runs_data,
        "metrics_summary": metrics_summary,
        "best_run": best_run,
        "latex_table": "\n".join(latex_lines),
    }
    if paired_comparisons:
        collected["paired_comparisons"] = paired_comparisons
    if structured_results is not None:
        collected["structured_results"] = structured_results
    return collected


def _read_kb_files(
    kb_root: str,
    filenames: list[str],
    max_chars_each: int = 3000,
) -> list[tuple[str, str]]:
    """Read files from the knowledge base directory.

    Returns a list of (filename, content) pairs for files that exist and are
    non-empty.  Contents are truncated to *max_chars_each* to prevent the KB
    from dominating the context window when the combined preamble is large.
    """
    if not kb_root:
        return []
    kb_path = Path(kb_root)
    if not kb_path.is_dir():
        return []
    results: list[tuple[str, str]] = []
    for fname in filenames:
        fpath = kb_path / fname
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8").strip()
            if text:
                results.append((fname, text[:max_chars_each]))
        except OSError:
            pass
    return results


def _build_context_preamble(
    config: RCConfig,
    run_dir: Path,
    *,
    include_goal: bool = False,
    include_hypotheses: bool = False,
    include_synthesis: bool = False,
    include_exp_plan: bool = False,
    include_analysis: bool = False,
    include_decision: bool = False,
    include_experiment_data: bool = False,
    include_kb: bool = False,
) -> str:
    parts = [
        "## Research Context",
        f"**Topic**: {config.research.topic}",
        f"**Domains**: {', '.join(config.research.domains) if config.research.domains else 'general'}",
    ]
    if include_goal:
        goal = _read_prior_artifact(run_dir, "goal.md")
        if goal:
            parts.append(f"\n### Goal\n{goal[:2200]}")
    if include_hypotheses:
        hyp = _read_prior_artifact(run_dir, "hypotheses.md")
        if hyp:
            parts.append(f"\n### Hypotheses\n{hyp[:2200]}")
    if include_synthesis:
        synthesis = _read_prior_artifact(run_dir, "synthesis.md")
        if synthesis:
            parts.append(f"\n### Synthesis\n{synthesis[:2200]}")
    if include_exp_plan:
        plan = _read_prior_artifact(run_dir, "exp_plan.yaml")
        if plan:
            parts.append(f"\n### Experiment Plan\n{plan[:2000]}")
    if include_analysis:
        analysis = _read_prior_artifact(run_dir, "analysis.md")
        if analysis:
            parts.append(f"\n### Result Analysis\n{analysis[:2500]}")
    if include_decision:
        decision = _read_prior_artifact(run_dir, "decision.md")
        if decision:
            parts.append(f"\n### Research Decision\n{decision[:1500]}")
    if include_experiment_data:
        hw_profile = _load_hardware_profile(run_dir)
        if hw_profile:
            hw_lines = ["### Hardware Environment"]
            for hk, hv in hw_profile.items():
                hw_lines.append(f"- **{hk}**: {hv}")
            parts.append("\n" + "\n".join(hw_lines))
        exp_summary = _read_prior_artifact(run_dir, "experiment_summary.json")
        if exp_summary:
            summary = _safe_json_loads(exp_summary, {})
            if isinstance(summary, dict) and summary.get("metrics_summary"):
                parts.append("\n### Experiment Results (Quantitative)")
                ms = summary["metrics_summary"]
                for mk, mv in ms.items():
                    if isinstance(mv, dict):
                        parts.append(
                            f"- **{mk}**: mean={mv.get('mean', '?')}, "
                            f"min={mv.get('min', '?')}, max={mv.get('max', '?')}, n={mv.get('count', '?')}"
                        )
                if summary.get("latex_table"):
                    parts.append(
                        f"\n### LaTeX Table\n```latex\n{summary['latex_table']}\n```"
                    )
    if include_kb:
        # Inject pre-existing knowledge base files so stages that design or
        # execute experiments can use real data rather than generating synthetic
        # stand-ins.  Files are listed in priority order; missing files are
        # silently skipped.  Each file is capped at 3000 chars to keep the
        # preamble manageable.
        _kb_files = _read_kb_files(
            getattr(config.knowledge_base, "root", ""),
            [
                "experiments_summary.md",
                "data_inventory.md",
                "compute_inventory.md",
                "claims_summary.md",
                "insights_summary.md",
            ],
        )
        if _kb_files:
            parts.append("\n## Knowledge Base (pre-existing project results)")
            parts.append(
                "IMPORTANT: The following files document REAL experimental results "
                "and REAL data that already exist in this project. Use these results "
                "directly rather than generating synthetic data or placeholder values."
            )
            for _kb_fname, _kb_text in _kb_files:
                parts.append(f"\n### {_kb_fname}\n{_kb_text}")
    return "\n".join(parts)


# --- P1-1: Topic keyword extraction for domain pre-filter ---
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "under",
        "over",
        "using",
        "based",
        "via",
        "toward",
        "towards",
        "new",
        "novel",
        "approach",
        "method",
        "study",
        "research",
        "paper",
        "work",
        "propose",
        "proposed",
    }
)


def _extract_topic_keywords(
    topic: str, domains: tuple[str, ...] | list[str] = ()
) -> list[str]:
    """Extract meaningful keywords from the research topic + domain list.

    Returns lowercased keyword list (2+ chars, no stop words).
    Used by the domain pre-filter to drop obviously irrelevant papers.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", topic.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
    # Add domain names as keywords
    for d in domains:
        for part in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", d.lower()):
            if part not in _STOP_WORDS and len(part) >= 2:
                keywords.append(part)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


# --- P1-2: Topic constraint block for paper generation stages ---
def _topic_constraint_block(topic: str) -> str:
    """Return a hard constraint instruction that anchors paper content to the topic.

    Prevents the common LLM failure mode of drifting off-topic or
    presenting environmental/infrastructure issues as research contributions.
    """
    return (
        "\n\n=== HARD TOPIC CONSTRAINT ===\n"
        f"The paper MUST be about: {topic}\n"
        "PROHIBITED content (unless user explicitly specifies case-study mode):\n"
        "- Do NOT treat environment setup, dependency installation, or infrastructure "
        "failures as a research contribution.\n"
        "- Do NOT present debugging logs, system errors, or configuration issues "
        "as experimental findings.\n"
        "- Do NOT drift to tangential topics not directly related to the stated topic.\n"
        "- Every section MUST connect back to the core research question.\n"
        "- The Abstract and Introduction MUST clearly state the research problem "
        f"derived from: {topic}\n"
        "- The Method section MUST describe a technical approach, not a workflow.\n"
        "- The Results section MUST report quantitative outcomes of experiments, "
        "not environment status.\n"
        "=== END CONSTRAINT ===\n"
    )


def _detect_runtime_issues(sandbox_result: Any) -> str:
    """Detect NaN/Inf in metrics and extract stderr warnings from sandbox run.

    Returns a formatted string describing all runtime issues, or empty string
    if no issues are found.
    """
    import math

    issues: list[str] = []

    # Check metrics for NaN/Inf
    metrics = getattr(sandbox_result, "metrics", {}) or {}
    for key, val in metrics.items():
        try:
            fval = float(val)
            if math.isnan(fval):
                issues.append(f"METRIC NaN: '{key}' returned NaN — likely a division by zero or invalid computation in code")
            elif math.isinf(fval):
                issues.append(f"METRIC Inf: '{key}' returned Infinity — likely overflow or unbounded computation")
        except (TypeError, ValueError):
            pass

    # Check stdout for NaN values (word boundary to avoid matching "Nanotechnology" etc.)
    stdout = getattr(sandbox_result, "stdout", "") or ""
    _nan_re = re.compile(r"\bnan\b", re.IGNORECASE)
    if _nan_re.search(stdout):
        nan_lines = [
            line.strip()
            for line in stdout.splitlines()
            if _nan_re.search(line)
        ]
        if nan_lines:
            issues.append(
                f"NaN values detected in output:\n" + "\n".join(nan_lines[:10])
            )

    # Extract meaningful warnings from stderr
    stderr = getattr(sandbox_result, "stderr", "") or ""
    if stderr.strip():
        warning_lines = []
        for line in stderr.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Keep RuntimeWarning, ValueError, ZeroDivisionError, etc.
            if any(
                kw in line_stripped
                for kw in (
                    "Warning",
                    "Error",
                    "Traceback",
                    "Exception",
                    "divide",
                    "overflow",
                    "invalid value",
                    "NaN",
                    "inf",
                )
            ):
                warning_lines.append(line_stripped)
        if warning_lines:
            issues.append(
                "Runtime warnings/errors from stderr:\n"
                + "\n".join(warning_lines[:15])
            )

    # Check for identical metric values across all entries in stdout
    # (e.g., all algorithms reporting convergence_rate=1.0)
    stdout = getattr(sandbox_result, "stdout", "") or ""
    if stdout:
        from collections import Counter

        metric_values_by_name: dict[str, list[float]] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.rsplit(":", 1)
            if len(parts) != 2:
                continue
            try:
                fval = float(parts[1].strip())
            except (ValueError, TypeError):
                continue
            # Extract metric suffix (e.g. "convergence_rate" from "UCB (Stochastic) convergence_rate")
            name = parts[0].strip()
            metric_suffix = name.split()[-1] if name.split() else name
            metric_values_by_name.setdefault(metric_suffix, []).append(fval)

        for metric_name, vals in metric_values_by_name.items():
            if len(vals) >= 3:
                unique = set(vals)
                if len(unique) <= 2:
                    issues.append(
                        f"DUMMY METRIC: '{metric_name}' has only {len(unique)} unique value(s) "
                        f"across {len(vals)} entries ({unique}) — likely a placeholder. "
                        f"Implement real measurement logic (e.g., track iterations to convergence)."
                    )

    # R5-3: Check for diverging loss values (fast-fail indicator)
    for key, val in metrics.items():
        try:
            fval = float(val)
            if "loss" in key.lower() and fval > 100:
                issues.append(
                    f"DIVERGING LOSS: '{key}' = {fval} (>100) — the optimization is "
                    f"diverging. Reduce learning rate, check gradient computation, "
                    f"or add gradient clipping."
                )
        except (TypeError, ValueError):
            pass

    if not issues:
        return ""

    return (
        "## Runtime Issues Detected\n\n"
        "The experiment code ran but produced problematic results. "
        "Fix the ROOT CAUSE of these issues in the code:\n\n"
        + "\n\n".join(f"- {issue}" for issue in issues)
    )


def _parse_metrics_from_stdout(stdout: str) -> dict[str, Any]:
    """Parse ``name: value`` metric lines from experiment stdout.

    Handles formats like ``UCB (Stochastic) cumulative_regret: 361.9233``
    and simple ``loss: 0.0042``.  Returns a flat dict of metric_name → value.

    Filters out log/status lines (e.g. "Running experiments for support set
    size: 1") using :func:`is_metric_name`.
    """
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        # Split on the LAST colon to handle names with colons
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        name_part = parts[0].strip()
        value_part = parts[1].strip()
        # Filter out log lines that look like status messages
        if not is_metric_name(name_part):
            continue
        try:
            fval = float(value_part)
            # Use the full name (e.g. "UCB (Stochastic) cumulative_regret")
            metrics[name_part] = fval
        except (ValueError, TypeError):
            pass
    return metrics


def _extract_code_block(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return content.strip()


def _extract_multi_file_blocks(content: str) -> dict[str, str]:
    """Parse LLM response containing multiple files with filename markers.

    Expected format::

        ```filename:main.py
        import model
        ...
        ```

        ```filename:model.py
        class MyModel:
        ...
        ```

    Also handles common LLM format variations:
    - ````` ```python filename:main.py````` (space before filename)
    - ````` ``` filename:main.py````` (space after backticks)
    - ``filename:main.py`` on next line after backticks
    - ``# FILE: main.py`` comment markers inside code blocks

    Falls back to treating the entire code block as ``main.py`` if no
    ``filename:`` markers are found.

    Returns a dict mapping filename → code content.
    """
    # R13-2: Multiple patterns to handle LLM format variations
    patterns = [
        # Original: ```filename:xxx.py or ```python filename:xxx.py
        re.compile(
            r"```(?:python\s+)?filename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ``` filename:xxx.py (space after backticks)
        re.compile(
            r"```\s+filename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ```python\nfilename:xxx.py (filename on next line)
        re.compile(
            r"```(?:python)?\s*\nfilename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ```python\n# filename: xxx.py (comment marker)
        re.compile(
            r"```(?:python)?\s*\n#\s*(?:FILE|filename)\s*:\s*(\S+\.py)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
    ]

    matches: list[tuple[str, str]] = []
    for pattern in patterns:
        matches = pattern.findall(content)
        if matches:
            break

    if matches:
        files: dict[str, str] = {}
        for fname, code in matches:
            fname = fname.strip()
            # Security: prevent path traversal
            if ".." in fname or fname.startswith("/"):
                continue
            # Normalise to flat filenames (strip leading ./ or subdirs for safety)
            fname = fname.replace("\\", "/").split("/")[-1]
            if fname and fname.endswith(".py"):
                files[fname] = code.strip()
        if files:
            # Ensure there is a main.py entry point
            if "main.py" not in files:
                # Pick the first file as main.py
                first_key = next(iter(files))
                files["main.py"] = files.pop(first_key)
            return files

    # Fallback: single code block → main.py
    code = _extract_code_block(content)
    if code.strip():
        return {"main.py": code}
    return {}


def _parse_jsonl_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _safe_json_loads(line, {})
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_json_context(
    directory: Path,
    *,
    max_files: int = 30,
    max_total_chars: int = 50_000,
) -> str:
    """Collect JSON context from a directory, with size limits.

    Large fields like ``stderr`` and ``stdout`` are stripped to avoid
    exceeding LLM token limits (the raw experiment output can be 5 MB+).
    """
    chunks: list[str] = []
    total = 0
    for file_path in sorted(directory.glob("*.json"))[:max_files]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Strip verbose fields that bloat the context
        if isinstance(data, dict):
            for key in ("stderr", "stdout", "raw_output", "traceback"):
                if key in data and isinstance(data[key], str) and len(data[key]) > 500:
                    data[key] = data[key][:500] + f"\n... [truncated, {len(data[key])} chars total]"
        chunk = json.dumps(data, indent=2, ensure_ascii=False)
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 200:
                chunks.append(chunk[:remaining] + "\n... [truncated]")
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks)


def _default_hypotheses(topic: str) -> str:
    return f"""# Hypotheses

## H1
Increasing protocol control for {topic} improves metric stability across random seeds.

## H2
Adding robustness-aware objectives for {topic} improves out-of-domain performance without major in-domain regression.

## H3
The combined approach outperforms either component under fixed compute budget.

## Generated
{_utcnow_iso()}
"""


def _default_paper_outline(topic: str) -> str:
    return f"""# Paper Outline

## 1. Title
Focused title on {topic}

## 2. Abstract
- Problem framing
- Method overview
- Key quantitative result

## 3. Introduction
- Motivation
- Gap statement
- Contributions

## 4. Related Work
- Method families
- Evaluation practices

## 5. Method
- Problem setup
- Model/algorithm
- Complexity and constraints

## 6. Experiments
- Datasets and metrics
- Baselines and ablations
- Reproducibility protocol

## 7. Results
- Main table
- Robustness analysis
- Failure cases

## 8. Discussion
- Practical implications
- Limitations

## 9. Conclusion
- Findings and next steps

Generated: {_utcnow_iso()}
"""


def _default_quality_report(threshold: float) -> dict[str, Any]:
    # When LLM fails, return below-threshold score to force revision
    score = max(1.0, float(threshold) - 2.0) if threshold > 0 else 5.0
    score = max(1.0, min(10.0, score))
    verdict = "revise"
    return {
        "score_1_to_10": round(score, 2),
        "verdict": verdict,
        "criteria": {
            "novelty": round(min(10.0, score + 0.3), 2),
            "methodological_rigor": round(score, 2),
            "clarity": round(max(1.0, score - 0.2), 2),
            "reproducibility": round(min(10.0, score + 0.1), 2),
        },
        "strengths": [
            "Stage-by-stage evidence chain preserved",
            "Experiment artifacts are generated and archived",
        ],
        "weaknesses": [
            "Statistical significance may need stronger reporting",
            "Broader external validity remains partially evaluated",
        ],
        "required_actions": [
            "Report confidence intervals and seed variance",
            "Include at least one stronger external baseline",
        ],
        "generated": _utcnow_iso(),
    }


def _execute_topic_init(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    topic = config.research.topic
    domains = (
        ", ".join(config.research.domains) if config.research.domains else "general"
    )
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "topic_init")
        sp = _pm.for_stage(
            "topic_init",
            evolution_overlay=_overlay,
            topic=topic,
            domains=domains,
            project_name=config.project.name,
            quality_threshold=config.research.quality_threshold,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        goal_md = resp.content
    else:
        goal_md = f"""# Research Goal

## Topic
{topic}

## Scope
Investigate the topic with emphasis on reproducible methods and measurable outcomes.

## SMART Goal
- Specific: Build a focused research plan for {topic}
- Measurable: Produce literature shortlist, hypotheses, experiment plan, and final paper
- Achievable: Complete through staged pipeline with gate checks
- Relevant: Aligned with project {config.project.name}
- Time-bound: Constrained by pipeline execution budget

## Constraints
- Quality threshold: {config.research.quality_threshold}
- Daily paper target: {config.research.daily_paper_count}

## Success Criteria
- At least 2 falsifiable hypotheses
- Executable experiment code and results analysis
- Revised paper passing quality gate

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "goal.md").write_text(goal_md, encoding="utf-8")

    # --- Hardware detection (GPU / MPS / CPU) ---
    hw = detect_hardware()
    (stage_dir / "hardware_profile.json").write_text(
        json.dumps(hw.to_dict(), indent=2), encoding="utf-8"
    )
    if hw.warning:
        logger.warning("Hardware advisory: %s", hw.warning)
    else:
        logger.info("Hardware detected: %s (%s, %s MB VRAM)", hw.gpu_name, hw.gpu_type, hw.vram_mb)

    # --- Optionally ensure PyTorch is available ---
    if hw.has_gpu and config.experiment.mode == "sandbox":
        torch_ok = ensure_torch_available(config.experiment.sandbox.python_path, hw.gpu_type)
        if torch_ok:
            logger.info("PyTorch is available for sandbox experiments")
        else:
            logger.warning("PyTorch could not be installed; sandbox will use CPU-only packages")
    elif hw.has_gpu and config.experiment.mode == "docker":
        logger.info("Docker sandbox: PyTorch pre-installed in container image")

    return StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md", "hardware_profile.json"),
        evidence_refs=("stage-01/goal.md", "stage-01/hardware_profile.json"),
    )


def _execute_problem_decompose(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    goal_text = _read_prior_artifact(run_dir, "goal.md") or ""
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "problem_decompose")
        sp = _pm.for_stage(
            "problem_decompose",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            goal_text=goal_text,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        body = resp.content
    else:
        body = f"""# Problem Decomposition

## Source
Derived from `goal.md` for topic: {config.research.topic}

## Sub-questions
1. Which problem settings and benchmarks define current SOTA?
2. Which methodological gaps remain unresolved?
3. Which hypotheses are testable under realistic constraints?
4. Which datasets and metrics best discriminate method quality?
5. Which failure modes can invalidate expected gains?

## Priority Ranking
1. Problem framing and benchmark setup
2. Gap identification and hypothesis formulation
3. Experiment and metric design
4. Failure analysis and robustness checks

## Risks
- Ambiguous task definition
- Dataset leakage or metric mismatch

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "problem_tree.md").write_text(body, encoding="utf-8")

    # IMP-35: Topic/title quality pre-evaluation
    # Quick LLM check: is the topic well-scoped for a conference paper?
    if llm is not None:
        try:
            _eval_resp = llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this research topic for a top ML conference paper. "
                            "Score 1-10 on: (a) novelty, (b) specificity, (c) feasibility. "
                            "If overall score < 5, suggest a refined topic.\n\n"
                            f"Topic: {config.research.topic}\n\n"
                            "Reply as JSON: {\"novelty\": N, \"specificity\": N, "
                            "\"feasibility\": N, \"overall\": N, \"suggestion\": \"...\"}"
                        ),
                    }
                ],
                system=(
                    f"You are a senior {_detect_domain(config.research.topic, config.research.domains)[1]} "
                    f"researcher evaluating research topic quality."
                ),
            )
            _eval_data = _safe_json_loads(_eval_resp.content, {})
            if isinstance(_eval_data, dict):
                overall = _eval_data.get("overall", 10)
                if isinstance(overall, (int, float)) and overall < 5:
                    logger.warning(
                        "IMP-35: Topic quality score %s/10 — consider refining: %s",
                        overall,
                        _eval_data.get("suggestion", ""),
                    )
                else:
                    logger.info("IMP-35: Topic quality score %s/10", overall)
                (stage_dir / "topic_evaluation.json").write_text(
                    json.dumps(_eval_data, indent=2), encoding="utf-8"
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-35: Topic evaluation skipped (non-blocking)")

    return StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.DONE,
        artifacts=("problem_tree.md",),
        evidence_refs=("stage-02/problem_tree.md",),
    )


def _execute_search_strategy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    topic = config.research.topic
    plan: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "search_strategy")
        sp = _pm.for_stage("search_strategy", evolution_overlay=_overlay, topic=topic, problem_tree=problem_tree)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict):
            yaml_text = str(payload.get("search_plan_yaml", "")).strip()
            if yaml_text:
                try:
                    parsed = yaml.safe_load(_extract_yaml_block(yaml_text))
                except yaml.YAMLError:
                    parsed = None
                if isinstance(parsed, dict):
                    plan = parsed
            src = payload.get("sources", [])
            if isinstance(src, list):
                sources = [item for item in src if isinstance(item, dict)]
    if plan is None:
        # Build smart fallback queries by extracting key terms from topic
        # instead of using the raw (often very long) topic string.
        _fallback_queries = _build_fallback_queries(topic)
        plan = {
            "topic": topic,
            "generated": _utcnow_iso(),
            "search_strategies": [
                {
                    "name": "keyword_core",
                    "queries": _fallback_queries[:5],
                    "sources": ["arxiv", "semantic_scholar", "openreview"],
                    "max_results_per_query": 60,
                },
                {
                    "name": "backward_forward_citation",
                    "queries": _fallback_queries[5:10] or _fallback_queries[:3],
                    "sources": ["semantic_scholar", "google_scholar"],
                    "depth": 1,
                },
            ],
            "filters": {
                "min_year": 2020,
                "language": ["en"],
                "peer_review_preferred": True,
            },
            "deduplication": {"method": "title_doi_hash", "fuzzy_threshold": 0.9},
        }
    if not sources:
        sources = [
            {
                "id": "arxiv",
                "name": "arXiv",
                "type": "api",
                "url": "https://export.arxiv.org/api/query",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "type": "api",
                "url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
        ]
    if config.openclaw_bridge.use_web_fetch:
        for src in sources:
            try:
                response = adapters.web_fetch.fetch(str(src.get("url", "")))
                src["status"] = (
                    "verified"
                    if response.status_code in (200, 301, 302, 405)
                    else "unreachable"
                )
                src["http_status"] = response.status_code
            except Exception:  # noqa: BLE001
                src["status"] = "unknown"
    (stage_dir / "search_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    (stage_dir / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "count": len(sources), "generated": _utcnow_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # F1.5: Extract queries from plan for Stage 4 real literature search
    queries_list: list[str] = []
    year_min = 2020
    if isinstance(plan, dict):
        strategies = plan.get("search_strategies", [])
        if isinstance(strategies, list):
            for strat in strategies:
                if isinstance(strat, dict):
                    qs = strat.get("queries", [])
                    if isinstance(qs, list):
                        queries_list.extend(str(q) for q in qs if q)
        filters = plan.get("filters", {})
        if isinstance(filters, dict) and filters.get("min_year"):
            try:
                year_min = int(filters["min_year"])
            except (ValueError, TypeError):
                pass

    # --- Sanitize queries: shorten overly long queries ---
    # LLMs often produce the full topic title as a query, which is too long for
    # arXiv and Semantic Scholar (they work best with 3-8 keyword queries).
    _stop = {
        "a", "an", "the", "of", "for", "in", "on", "and", "or", "with",
        "to", "by", "from", "its", "is", "are", "was", "be", "as", "at",
        "via", "using", "based", "study", "analysis", "empirical",
        "towards", "toward", "into", "exploring", "comparison", "tasks",
        "effectiveness", "investigation", "comprehensive", "novel",
    }

    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from text, removing stop words."""
        return [
            w for w in re.split(r"[^a-zA-Z0-9]+", text)
            if w.lower() not in _stop and len(w) > 1
        ]

    _MAX_QUERY_LEN = 60  # characters — beyond this, shorten to keywords
    _SEARCH_SUFFIXES = ["benchmark", "survey", "seminal", "state of the art"]

    def _shorten_query(q: str, max_kw: int = 6) -> str:
        """Shorten a query to *max_kw* keywords, preserving any trailing suffix."""
        q_stripped = q.strip()
        # Check if query ends with a known search suffix
        suffix = ""
        q_core = q_stripped
        for sfx in _SEARCH_SUFFIXES:
            if q_stripped.lower().endswith(sfx):
                suffix = sfx
                q_core = q_stripped[: -len(sfx)].strip()
                break
        # Extract keywords from the core part
        kws = _extract_keywords(q_core)
        shortened = " ".join(kws[:max_kw])
        if suffix:
            shortened = f"{shortened} {suffix}"
        return shortened

    if queries_list:
        sanitized: list[str] = []
        for q in queries_list:
            if len(q) > _MAX_QUERY_LEN:
                shortened = _shorten_query(q)
                if shortened.strip():
                    sanitized.append(shortened)
            else:
                sanitized.append(q)
        queries_list = sanitized

    if not queries_list:
        # Build diverse keyword queries from the topic
        _words = _extract_keywords(topic)
        kw_primary = " ".join(_words[:6])
        kw_short = " ".join(_words[:4])
        queries_list = [
            kw_primary,
            f"{kw_short} benchmark",
            f"{kw_short} survey",
        ]

    # Ensure minimum query diversity — if dedup leaves too few, add variants
    _all_kw = _extract_keywords(topic)
    _seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries_list:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in _seen_q:
            _seen_q.add(q_lower)
            unique_queries.append(q.strip())
    # If we have fewer than 5 unique queries, generate supplemental keyword variants
    if len(unique_queries) < 5 and len(_all_kw) >= 3:
        supplements = [
            " ".join(_all_kw[:4]) + " survey",
            " ".join(_all_kw[:4]) + " benchmark",
            " ".join(_all_kw[1:5]),  # shifted window for diversity
            " ".join(_all_kw[:3]) + " comparison",
            " ".join(_all_kw[:3]) + " deep learning",
            " ".join(_all_kw[2:6]),  # another shifted window
        ]
        for s in supplements:
            s_lower = s.strip().lower()
            if s_lower not in _seen_q:
                _seen_q.add(s_lower)
                unique_queries.append(s.strip())
            if len(unique_queries) >= 8:
                break
    queries_list = unique_queries
    (stage_dir / "queries.json").write_text(
        json.dumps({"queries": queries_list, "year_min": year_min}, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.SEARCH_STRATEGY,
        status=StageStatus.DONE,
        artifacts=("search_plan.yaml", "sources.json", "queries.json"),
        evidence_refs=(
            "stage-03/search_plan.yaml",
            "stage-03/sources.json",
            "stage-03/queries.json",
        ),
    )


def _expand_search_queries(queries: list[str], topic: str) -> list[str]:
    """Expand search queries for broader literature coverage.

    Generates additional queries by extracting key phrases from the topic
    and creating focused sub-queries. This ensures we find papers even when
    the original queries are too narrow or specific for arXiv.
    """
    expanded = list(queries)  # keep originals
    seen = {q.lower().strip() for q in queries}

    # Extract key phrases from topic by splitting on common delimiters
    # e.g. "Comparing A, B, and C on X with Y" → ["A", "B", "C", "X", "Y"]
    topic_words = topic.split()

    # Generate shorter, broader queries from the topic
    if len(topic_words) > 5:
        # First 5 words as a broader query
        broad = " ".join(topic_words[:5])
        if broad.lower().strip() not in seen:
            expanded.append(broad)
            seen.add(broad.lower().strip())

        # Last 5 words as another perspective
        tail = " ".join(topic_words[-5:])
        if tail.lower().strip() not in seen:
            expanded.append(tail)
            seen.add(tail.lower().strip())

    # Add "survey" and "benchmark" variants of the topic
    for suffix in ("survey", "benchmark", "comparison"):
        # Take first 4 content words + suffix
        short_topic = " ".join(topic_words[:4])
        variant = f"{short_topic} {suffix}"
        if variant.lower().strip() not in seen:
            expanded.append(variant)
            seen.add(variant.lower().strip())

    return expanded


def _execute_literature_collect(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 4: Collect literature — prefer real APIs, fallback to LLM."""
    topic = config.research.topic

    # Read queries.json from Stage 3 (F1.5 output)
    queries_text = _read_prior_artifact(run_dir, "queries.json")
    queries_data = _safe_json_loads(queries_text or "{}", {})
    queries: list[str] = queries_data.get("queries", [topic])
    year_min: int = queries_data.get("year_min", 2020)

    # --- Try real API search first ---
    candidates: list[dict[str, Any]] = []
    bibtex_entries: list[str] = []
    real_search_succeeded = False

    try:
        from researchclaw.literature.search import (
            search_papers_multi_query,
            papers_to_bibtex,
        )

        # Expand queries for broader coverage
        expanded_queries = _expand_search_queries(queries, config.research.topic)
        logger.info(
            "[literature] Searching %d queries (expanded from %d) "
            "across OpenAlex → S2 → arXiv…",
            len(expanded_queries),
            len(queries),
        )
        papers = search_papers_multi_query(
            expanded_queries,
            limit_per_query=40,
            year_min=year_min,
            s2_api_key=config.llm.s2_api_key,
        )
        if papers:
            real_search_succeeded = True
            # Count by source
            src_counts: dict[str, int] = {}
            for p in papers:
                src_counts[p.source] = src_counts.get(p.source, 0) + 1
                d = p.to_dict()
                d["collected_at"] = _utcnow_iso()
                candidates.append(d)
                bibtex_entries.append(p.to_bibtex())
            src_str = ", ".join(f"{s}: {n}" for s, n in src_counts.items())
            logger.info(
                "[literature] Found %d papers (%s)", len(papers), src_str
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[rate-limit] Literature search failed — falling back to LLM",
            exc_info=True,
        )

    # --- Inject foundational/seminal papers ---
    try:
        from researchclaw.data import load_seminal_papers
        seminal = load_seminal_papers(topic)
        if seminal:
            _existing_titles = {c.get("title", "").lower() for c in candidates}
            _injected = 0
            for sp in seminal:
                if sp.get("title", "").lower() not in _existing_titles:
                    candidates.append({
                        "id": f"seminal-{sp.get('cite_key', '')}",
                        "title": sp.get("title", ""),
                        "source": "seminal_library",
                        "url": "",
                        "year": sp.get("year", 2020),
                        "abstract": f"Foundational paper on {', '.join(sp.get('keywords', [])[:3])}.",
                        "authors": [{"name": sp.get("authors", "")}],
                        "cite_key": sp.get("cite_key", ""),
                        "venue": sp.get("venue", ""),
                        "collected_at": _utcnow_iso(),
                    })
                    _injected += 1
            if _injected:
                logger.info("Stage 4: Injected %d seminal papers from seed library", _injected)
    except Exception:  # noqa: BLE001
        logger.debug("Seminal paper injection skipped", exc_info=True)

    # --- Fallback: LLM-generated candidates ---
    if not candidates and llm is not None:
        plan_text = _read_prior_artifact(run_dir, "search_plan.yaml") or ""
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_collect")
        sp = _pm.for_stage("literature_collect", evolution_overlay=_overlay, topic=topic, plan_text=plan_text)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            candidates = [row for row in payload["candidates"] if isinstance(row, dict)]

    # --- Web search augmentation (Tavily/DDG + Google Scholar + Crawl4AI) ---
    web_context_parts: list[str] = []
    if config.web_search.enabled:
        try:
            from researchclaw.web.agent import WebSearchAgent
            import os

            tavily_key = config.web_search.tavily_api_key or os.environ.get(
                config.web_search.tavily_api_key_env, ""
            )
            web_agent = WebSearchAgent(
                tavily_api_key=tavily_key,
                enable_scholar=config.web_search.enable_scholar,
                enable_crawling=config.web_search.enable_crawling,
                enable_pdf=config.web_search.enable_pdf_extraction,
                max_web_results=config.web_search.max_web_results,
                max_scholar_results=config.web_search.max_scholar_results,
                max_crawl_urls=config.web_search.max_crawl_urls,
            )
            web_result = web_agent.search_and_extract(
                topic, search_queries=queries,
            )

            # Convert Google Scholar papers into candidates
            for sp in web_result.scholar_papers:
                _existing_titles = {
                    str(c.get("title", "")).lower().strip() for c in candidates
                }
                if sp.title.lower().strip() not in _existing_titles:
                    lit_paper = sp.to_literature_paper()
                    d = lit_paper.to_dict()
                    d["collected_at"] = _utcnow_iso()
                    candidates.append(d)
                    bibtex_entries.append(lit_paper.to_bibtex())

            # Save web search context for downstream stages
            web_context = web_result.to_context_string(max_length=20_000)
            if web_context.strip():
                (stage_dir / "web_context.md").write_text(
                    web_context, encoding="utf-8"
                )
                web_context_parts.append(web_context)

            # Save full web search metadata
            (stage_dir / "web_search_result.json").write_text(
                json.dumps(web_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

            logger.info(
                "[web-search] Added %d scholar papers, %d web results, %d crawled pages",
                len(web_result.scholar_papers),
                len(web_result.web_results),
                len(web_result.crawled_pages),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[web-search] Web search augmentation failed — continuing with academic APIs only",
                exc_info=True,
            )

    # --- Ultimate fallback: placeholder data ---
    real_search_succeeded = bool(candidates)
    if not candidates:
        logger.warning("Stage 4: All literature searches failed — using placeholder papers")
        candidates = [
            {
                "id": f"candidate-{idx + 1}",
                "title": f"[Placeholder] Study {idx + 1} on {topic}",
                "source": "arxiv" if idx % 2 == 0 else "semantic_scholar",
                "url": f"https://example.org/{_safe_filename(topic.lower())}/{idx + 1}",
                "year": 2024,
                "abstract": f"This candidate investigates {topic} and reports preliminary findings.",
                "collected_at": _utcnow_iso(),
                "is_placeholder": True,
            }
            for idx in range(max(20, config.research.daily_paper_count or 20))
        ]

    # Write candidates
    out = stage_dir / "candidates.jsonl"
    _write_jsonl(out, candidates)

    # BUG-50 fix: Generate BibTeX from candidates when real search failed
    # (LLM/placeholder fallback paths don't populate bibtex_entries)
    if not bibtex_entries and candidates:
        for c in candidates:
            if c.get("is_placeholder"):
                continue
            _ck = c.get("cite_key", "")
            if not _ck:
                # Derive cite_key from first author surname + year
                _authors = c.get("authors", [])
                _surname = "unknown"
                if isinstance(_authors, list) and _authors:
                    _a0 = _authors[0] if isinstance(_authors[0], str) else (_authors[0].get("name", "") if isinstance(_authors[0], dict) else "")
                    _surname = _a0.split()[-1].lower() if _a0.strip() else "unknown"
                _yr = c.get("year", 2024)
                _title_word = "".join(
                    w[0] for w in str(c.get("title", "study")).split()[:3]
                ).lower()
                _ck = f"{_surname}{_yr}{_title_word}"
            _title = c.get("title", "Untitled")
            _year = c.get("year", 2024)
            _author_str = ""
            _raw_authors = c.get("authors", [])
            if isinstance(_raw_authors, list):
                _names = []
                for _a in _raw_authors:
                    if isinstance(_a, str):
                        _names.append(_a)
                    elif isinstance(_a, dict):
                        _names.append(_a.get("name", ""))
                _author_str = " and ".join(n for n in _names if n)
            bibtex_entries.append(
                f"@article{{{_ck},\n"
                f"  title={{{_title}}},\n"
                f"  author={{{_author_str or 'Unknown'}}},\n"
                f"  year={{{_year}}},\n"
                f"  url={{{c.get('url', '')}}},\n"
                f"}}"
            )
        logger.info(
            "Stage 4: Generated %d BibTeX entries from candidates (fallback)",
            len(bibtex_entries),
        )

    # Write references.bib (F2.4)
    artifacts = ["candidates.jsonl"]
    if web_context_parts:
        artifacts.append("web_context.md")
    if (stage_dir / "web_search_result.json").exists():
        artifacts.append("web_search_result.json")
    if bibtex_entries:
        bib_content = "\n\n".join(bibtex_entries) + "\n"
        (stage_dir / "references.bib").write_text(bib_content, encoding="utf-8")
        artifacts.append("references.bib")
        logger.info(
            "Stage 4: Wrote %d BibTeX entries to references.bib", len(bibtex_entries)
        )

    # Write search metadata
    (stage_dir / "search_meta.json").write_text(
        json.dumps(
            {
                "real_search": real_search_succeeded,
                "queries_used": queries,
                "year_min": year_min,
                "total_candidates": len(candidates),
                "bibtex_entries": len(bibtex_entries),
                "ts": _utcnow_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.append("search_meta.json")

    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


def _execute_literature_screen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""

    # --- P1-1: keyword relevance pre-filter ---
    # Before LLM screening, drop papers whose title+abstract share no keywords
    # with the research topic.  This catches cross-domain noise cheaply.
    topic_keywords = _extract_topic_keywords(
        config.research.topic, config.research.domains
    )
    filtered_rows: list[dict[str, Any]] = []
    dropped_count = 0
    for raw_line in candidates_text.strip().splitlines():
        row = _safe_json_loads(raw_line, {})
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()
        text_blob = f"{title} {abstract}"
        overlap = sum(1 for kw in topic_keywords if kw in text_blob)
        # T2.2: Relaxed from ≥2 to ≥1 keyword hit — previous threshold was
        # too aggressive (94% rejection rate).  Single-keyword matches are
        # still screened by the LLM in the next step.
        if overlap >= 1:
            row["keyword_overlap"] = overlap
            filtered_rows.append(row)
        else:
            dropped_count += 1
    # If pre-filter dropped everything, fall back to original (safety valve)
    if not filtered_rows:
        filtered_rows = _parse_jsonl_rows(candidates_text)
    # Rebuild candidates_text from filtered rows
    candidates_text = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in filtered_rows
    )
    logger.info(
        "Domain pre-filter: kept %d, dropped %d (keywords: %s)",
        len(filtered_rows),
        dropped_count,
        topic_keywords[:8],
    )

    shortlist: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_screen")
        sp = _pm.for_stage(
            "literature_screen",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            domains=", ".join(config.research.domains)
            if config.research.domains
            else "general",
            quality_threshold=config.research.quality_threshold,
            candidates_text=candidates_text,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("shortlist"), list):
            shortlist = [row for row in payload["shortlist"] if isinstance(row, dict)]
    # T2.2: Ensure minimum shortlist size of 15 for adequate related work
    _MIN_SHORTLIST = 15
    if not shortlist:
        rows = (
            filtered_rows[:_MIN_SHORTLIST]
            if filtered_rows
            else _parse_jsonl_rows(candidates_text)[:_MIN_SHORTLIST]
        )
        for idx, item in enumerate(rows):
            item["relevance_score"] = round(0.75 - idx * 0.02, 3)
            item["quality_score"] = round(0.72 - idx * 0.015, 3)
            item["keep_reason"] = "Template screened entry"
            shortlist.append(item)
    elif len(shortlist) < _MIN_SHORTLIST:
        # T2.2: LLM returned too few — supplement from filtered candidates
        existing_titles = {
            str(s.get("title", "")).lower().strip() for s in shortlist
        }
        for row in filtered_rows:
            if len(shortlist) >= _MIN_SHORTLIST:
                break
            title_lower = str(row.get("title", "")).lower().strip()
            if title_lower and title_lower not in existing_titles:
                row.setdefault("relevance_score", 0.5)
                row.setdefault("quality_score", 0.5)
                row.setdefault("keep_reason", "Supplemented to meet minimum shortlist")
                shortlist.append(row)
                existing_titles.add(title_lower)
        logger.info(
            "Stage 5: Supplemented shortlist to %d papers (minimum: %d)",
            len(shortlist), _MIN_SHORTLIST,
        )
    out = stage_dir / "shortlist.jsonl"
    _write_jsonl(out, shortlist)
    return StageResult(
        stage=Stage.LITERATURE_SCREEN,
        status=StageStatus.DONE,
        artifacts=("shortlist.jsonl",),
        evidence_refs=("stage-05/shortlist.jsonl",),
    )


def _execute_knowledge_extract(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    shortlist = _read_prior_artifact(run_dir, "shortlist.jsonl") or ""

    # Inject web context from Stage 4 if available
    web_context = _read_prior_artifact(run_dir, "web_context.md") or ""
    if web_context:
        shortlist = shortlist + "\n\n--- Web Search Context ---\n" + web_context[:10_000]

    cards_dir = stage_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "knowledge_extract")
        sp = _pm.for_stage("knowledge_extract", evolution_overlay=_overlay, shortlist=shortlist)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            cards = [item for item in payload["cards"] if isinstance(item, dict)]
    if not cards:
        rows = _parse_jsonl_rows(shortlist)
        for idx, paper in enumerate(rows[:6]):
            title = str(paper.get("title", f"Paper {idx + 1}"))
            cards.append(
                {
                    "card_id": f"card-{idx + 1}",
                    "title": title,
                    "problem": f"How to improve {config.research.topic}",
                    "method": "Template method summary",
                    "data": "Template dataset",
                    "metrics": "Template metric",
                    "findings": "Template key finding",
                    "limitations": "Template limitation",
                    "citation": str(paper.get("url", "")),
                    "cite_key": str(paper.get("cite_key", "")),
                }
            )
    for idx, card in enumerate(cards):
        card_id = _safe_filename(str(card.get("card_id", f"card-{idx + 1}")))
        parts = [f"# {card.get('title', card_id)}", ""]
        for key in (
            "cite_key",
            "problem",
            "method",
            "data",
            "metrics",
            "findings",
            "limitations",
            "citation",
        ):
            parts.append(f"## {key.title()}")
            parts.append(str(card.get(key, "")))
            parts.append("")
        (cards_dir / f"{card_id}.md").write_text("\n".join(parts), encoding="utf-8")
    return StageResult(
        stage=Stage.KNOWLEDGE_EXTRACT,
        status=StageStatus.DONE,
        artifacts=("cards/",),
        evidence_refs=("stage-06/cards/",),
    )


def _execute_synthesis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    cards_path = _read_prior_artifact(run_dir, "cards/") or ""
    cards_context = ""
    if cards_path:
        snippets: list[str] = []
        for path in sorted(Path(cards_path).glob("*.md"))[:24]:
            snippets.append(path.read_text(encoding="utf-8"))
        cards_context = "\n\n".join(snippets)
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "synthesis")
        sp = _pm.for_stage(
            "synthesis",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            cards_context=cards_context,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
            max_tokens=sp.max_tokens or 8192,
        )
        synthesis_md = resp.content
    else:
        synthesis_md = f"""# Synthesis

## Cluster Overview
- Cluster A: Representation methods
- Cluster B: Training strategies
- Cluster C: Evaluation robustness

## Gap 1
Limited consistency across benchmark protocols.

## Gap 2
Under-reported failure behavior under distribution shift.

## Prioritized Opportunities
1. Unified experimental protocol
2. Robustness-aware evaluation suite

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "synthesis.md").write_text(synthesis_md, encoding="utf-8")
    return StageResult(
        stage=Stage.SYNTHESIS,
        status=StageStatus.DONE,
        artifacts=("synthesis.md",),
        evidence_refs=("stage-07/synthesis.md",),
    )


def _multi_perspective_generate(
    llm: LLMClient,
    roles: dict[str, dict[str, str]],
    variables: dict[str, str],
    perspectives_dir: Path,
) -> dict[str, str]:
    """Generate outputs from multiple debate perspectives.

    Each role has its own system/user prompt. Outputs are saved to
    *perspectives_dir* and returned as ``{role_name: response_text}``.
    """
    from researchclaw.prompts import _render  # noqa: PLC0415

    perspectives_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for role_name, role_prompts in roles.items():
        try:
            system = _render(role_prompts["system"], variables)
            user = _render(role_prompts["user"], variables)
            resp = llm.chat(
                [{"role": "user", "content": user}],
                system=system,
            )
            results[role_name] = resp.content
            (perspectives_dir / f"{role_name}.md").write_text(
                resp.content, encoding="utf-8"
            )
            logger.info("Debate perspective '%s' generated (%d chars)", role_name, len(resp.content))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debate perspective '%s' failed: %s", role_name, exc)
    if len(results) < 2:
        logger.error("Multi-perspective debate: only %d/%d roles succeeded", len(results), len(roles))
    return results


def _synthesize_perspectives(
    llm: LLMClient,
    perspectives: dict[str, str],
    sub_prompt_name: str,
    prompts: PromptManager,
) -> str:
    """Synthesize multiple perspective outputs into a unified result."""
    parts = []
    for role_name, text in perspectives.items():
        parts.append(f"### Perspective: {role_name}\n{text}")
    combined = "\n\n---\n\n".join(parts)
    sp = prompts.sub_prompt(sub_prompt_name, perspectives=combined)
    resp = llm.chat(
        [{"role": "user", "content": sp.user}],
        system=sp.system,
    )
    return resp.content


def _execute_hypothesis_gen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    synthesis = _read_prior_artifact(run_dir, "synthesis.md") or ""
    if llm is not None:
        _pm = prompts or PromptManager()
        from researchclaw.prompts import DEBATE_ROLES_HYPOTHESIS  # noqa: PLC0415

        # --- Multi-perspective debate ---
        perspectives_dir = stage_dir / "perspectives"
        variables = {"topic": config.research.topic, "synthesis": synthesis}
        perspectives = _multi_perspective_generate(
            llm, DEBATE_ROLES_HYPOTHESIS, variables, perspectives_dir
        )
        # --- Synthesize into final hypotheses ---
        hypotheses_md = _synthesize_perspectives(
            llm, perspectives, "hypothesis_synthesize", _pm
        )
    else:
        hypotheses_md = _default_hypotheses(config.research.topic)
    (stage_dir / "hypotheses.md").write_text(hypotheses_md, encoding="utf-8")

    # --- Novelty check (non-blocking) ---
    novelty_artifacts: tuple[str, ...] = ()
    try:
        from researchclaw.literature.novelty import check_novelty  # noqa: PLC0415

        candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""
        papers_seen = _parse_jsonl_rows(candidates_text) if candidates_text else []
        novelty_report = check_novelty(
            topic=config.research.topic,
            hypotheses_text=hypotheses_md,
            papers_already_seen=papers_seen,
            s2_api_key=getattr(config.llm, "s2_api_key", ""),
        )
        (stage_dir / "novelty_report.json").write_text(
            json.dumps(novelty_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        novelty_artifacts = ("novelty_report.json",)
        logger.info(
            "Novelty check: score=%.3f  assessment=%s  recommendation=%s",
            novelty_report["novelty_score"],
            novelty_report["assessment"],
            novelty_report["recommendation"],
        )
    except Exception:  # noqa: BLE001
        logger.warning("Novelty check failed (non-blocking)", exc_info=True)

    return StageResult(
        stage=Stage.HYPOTHESIS_GEN,
        status=StageStatus.DONE,
        artifacts=("hypotheses.md",) + novelty_artifacts,
        evidence_refs=("stage-08/hypotheses.md",),
    )


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    preamble = _build_context_preamble(
        config, run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_kb=True,  # inject pre-existing KB so real data paths reach the LLM
    )
    plan: dict[str, Any] | None = None

    # ── Domain detection ──────────────────────────────────────────────────
    # Detect the research domain early so we can adapt experiment design
    # and code generation. For ML domains, existing behavior is unchanged.
    _domain_profile = None
    try:
        from researchclaw.domains.detector import detect_domain as _detect_domain_adv
        _domain_profile = _detect_domain_adv(
            topic=config.research.topic,
            hypotheses=hypotheses,
        )
        logger.info(
            "Domain detected: %s (%s)",
            _domain_profile.display_name,
            _domain_profile.domain_id,
        )
        # Persist domain profile for Stage 10
        import json as _json_dd
        (stage_dir / "domain_profile.json").write_text(
            _json_dd.dumps({
                "domain_id": _domain_profile.domain_id,
                "display_name": _domain_profile.display_name,
                "experiment_paradigm": _domain_profile.experiment_paradigm,
                "core_libraries": _domain_profile.core_libraries,
                "gpu_required": _domain_profile.gpu_required,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Domain detection unavailable", exc_info=True)
    if llm is not None:
        _pm = prompts or PromptManager()
        # Pass dataset_guidance block for experiment design
        try:
            _dg_block = _pm.block("dataset_guidance")
        except (KeyError, Exception):  # noqa: BLE001
            _dg_block = ""
        # I-08: Inject RL step guidance for RL topics
        _rl_kws = ("reinforcement learning", "ppo", "sac", "td3", "ddpg",
                    "dqn", "mujoco", "continuous control", "actor-critic",
                    "policy gradient", "exploration bonus")
        if any(kw in config.research.topic.lower() for kw in _rl_kws):
            try:
                _dg_block += _pm.block("rl_step_guidance")
            except Exception:  # noqa: BLE001
                pass
        # F-01: Inject framework docs for experiment design
        try:
            from researchclaw.data import detect_frameworks, load_framework_docs
            _fw_ids = detect_frameworks(config.research.topic, hypotheses)
            if _fw_ids:
                _fw_docs = load_framework_docs(_fw_ids, max_chars=4000)
                if _fw_docs:
                    _dg_block += _fw_docs
        except Exception:  # noqa: BLE001
            pass
        _overlay = _get_evolution_overlay(run_dir, "experiment_design")
        sp = _pm.for_stage(
            "experiment_design",
            evolution_overlay=_overlay,
            preamble=preamble,
            hypotheses=hypotheses,
            dataset_guidance=_dg_block,
            time_budget_sec=config.experiment.time_budget_sec,
            metric_key=config.experiment.metric_key,
            metric_direction=config.experiment.metric_direction,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        raw_yaml = _extract_yaml_block(resp.content)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            parsed = None
        # Fallback: reasoning models sometimes emit the YAML without fences
        # or wrapped in prose. Try parsing the whole response as YAML.
        if not isinstance(parsed, dict):
            try:
                parsed = yaml.safe_load(resp.content)
            except yaml.YAMLError:
                pass
        # Last fallback: try to find any YAML-like dict in the response
        if not isinstance(parsed, dict):
            import re as _re_yaml

            # Look for lines starting with known keys
            _yaml_lines = []
            _capturing = False
            for line in resp.content.splitlines():
                if _re_yaml.match(
                    r"^(baselines|proposed_methods|ablations|datasets|"
                    r"metrics|objectives|risks|compute_budget)\s*:",
                    line,
                ):
                    _capturing = True
                if _capturing:
                    if line.strip() == "" or line.startswith("```"):
                        continue
                    if line.startswith("#") or line.startswith("**"):
                        continue
                    _yaml_lines.append(line)
            if _yaml_lines:
                try:
                    parsed = yaml.safe_load("\n".join(_yaml_lines))
                except yaml.YAMLError:
                    pass
        if isinstance(parsed, dict):
            plan = parsed
        else:
            logger.warning(
                "Stage 09: LLM response could not be parsed as YAML "
                "(len=%d, first 200 chars: %s). Content extraction method "
                "returned: %s",
                len(resp.content),
                resp.content[:200],
                raw_yaml[:200] if raw_yaml else "<empty>",
            )
            # BUG-12: Retry with a stricter, shorter prompt
            if llm is not None:
                logger.info("Stage 09: Retrying with strict YAML-only prompt...")
                _retry_prompt = (
                    "Output ONLY valid YAML. No prose, no markdown fences, no explanation.\n"
                    f"Topic: {config.research.topic}\n"
                    "Required keys: baselines, proposed_methods, ablations, "
                    "datasets, metrics, objectives, risks, compute_budget.\n"
                    "Each key maps to a list of strings."
                )
                _retry_resp = _chat_with_prompt(
                    llm,
                    "You output ONLY valid YAML. Nothing else.",
                    _retry_prompt,
                    max_tokens=4096,
                )
                try:
                    _retry_parsed = yaml.safe_load(_retry_resp.content)
                    if isinstance(_retry_parsed, dict):
                        plan = _retry_parsed
                        logger.info("Stage 09: Strict YAML retry succeeded.")
                except yaml.YAMLError:
                    pass

    # BUG-12: Fallback 4 — extract method/baseline names from Stage 8 hypotheses
    if plan is None:
        _hyp_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        if _hyp_text:
            import re as _re_hyp
            # Extract method-like names from hypothesis text
            _method_candidates = _re_hyp.findall(
                r"(?:proposed|our|novel|new)\s+(?:method|approach|algorithm|framework|model)[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            _baseline_candidates = _re_hyp.findall(
                r"(?:baseline|compare|existing|standard|traditional)\s+(?:method|approach|model)?[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            if _method_candidates or _baseline_candidates:
                logger.info(
                    "Stage 09: Extracted names from hypotheses: methods=%s, baselines=%s",
                    _method_candidates[:3], _baseline_candidates[:3],
                )
                plan = {
                    "topic": config.research.topic,
                    "generated": _utcnow_iso(),
                    "objectives": ["Evaluate hypotheses with controlled experiments"],
                    "datasets": ["primary_dataset"],
                    "baselines": _baseline_candidates[:3] or ["baseline_1", "baseline_2"],
                    "proposed_methods": _method_candidates[:3] or ["proposed_method"],
                    "ablations": ["without_key_component", "simplified_version"],
                    "metrics": [config.experiment.metric_key, "secondary_metric"],
                    "risks": ["validity threats", "confounding variables"],
                    "compute_budget": {"max_gpu": 1, "max_hours": 4},
                }

    if plan is None:
        # BUG-12: Use domain-aware names instead of fully generic placeholders
        _topic_prefix = config.research.topic.split()[0] if config.research.topic else "method"
        logger.warning(
            "Stage 09: LLM failed to produce valid experiment plan YAML. "
            "Using topic-derived fallback."
        )
        plan = {
            "topic": config.research.topic,
            "generated": _utcnow_iso(),
            "objectives": ["Evaluate hypotheses with controlled experiments"],
            "datasets": ["primary_dataset", "secondary_dataset"],
            "baselines": [f"{_topic_prefix}_baseline_1", f"{_topic_prefix}_baseline_2"],
            "proposed_methods": [f"{_topic_prefix}_proposed", f"{_topic_prefix}_variant"],
            "ablations": ["without_key_component", "simplified_version"],
            "metrics": [config.experiment.metric_key, "secondary_metric"],
            "risks": ["validity threats", "confounding variables"],
            "compute_budget": {"max_gpu": 1, "max_hours": 4},
        }
    # ── BA: BenchmarkAgent — intelligent dataset/baseline selection ──────
    _benchmark_plan = None
    # BUG-40: Skip BenchmarkAgent for non-ML domains — it has no relevant
    # benchmarks for physics/chemistry/mathematics/etc. and would inject
    # wrong datasets (e.g., CIFAR-10 for PDE topics).
    _ba_domain_id, _, _ = _detect_domain(
        config.research.topic,
        tuple(config.research.domains) if config.research.domains else (),
    )
    _ba_domain_ok = _ba_domain_id == "ml"
    if not _ba_domain_ok:
        logger.info(
            "BenchmarkAgent skipped: domain '%s' is not ML (topic: %s)",
            _ba_domain_id, config.research.topic[:80],
        )
    if (
        _ba_domain_ok
        and config.experiment.benchmark_agent.enabled
        and config.experiment.mode in ("sandbox", "docker")
        and llm is not None
    ):
        try:
            from researchclaw.agents.benchmark_agent import BenchmarkOrchestrator
            from researchclaw.agents.benchmark_agent.orchestrator import (
                BenchmarkAgentConfig as _BACfg,
            )

            _ba_cfg_raw = config.experiment.benchmark_agent
            _ba_cfg = _BACfg(
                enabled=_ba_cfg_raw.enabled,
                enable_hf_search=_ba_cfg_raw.enable_hf_search,
                max_hf_results=_ba_cfg_raw.max_hf_results,
                enable_web_search=_ba_cfg_raw.enable_web_search,
                max_web_results=_ba_cfg_raw.max_web_results,
                web_search_min_local=_ba_cfg_raw.web_search_min_local,
                tier_limit=_ba_cfg_raw.tier_limit,
                min_benchmarks=_ba_cfg_raw.min_benchmarks,
                min_baselines=_ba_cfg_raw.min_baselines,
                prefer_cached=_ba_cfg_raw.prefer_cached,
                max_iterations=_ba_cfg_raw.max_iterations,
            )

            _hw = _load_hardware_profile(run_dir)
            _ba = BenchmarkOrchestrator(
                llm,
                config=_ba_cfg,
                gpu_memory_mb=(
                    _hw.get("gpu_memory_mb", 49000) if _hw else 49000
                ),
                time_budget_sec=config.experiment.time_budget_sec,
                network_policy=(
                    config.experiment.docker.network_policy
                    if config.experiment.mode == "docker"
                    else "full"
                ),
                stage_dir=stage_dir / "benchmark_agent",
            )
            _benchmark_plan = _ba.orchestrate({
                "topic": config.research.topic,
                "hypothesis": hypotheses,
                "experiment_plan": plan.get("objectives", "") if isinstance(plan, dict) else "",
            })

            # Inject BenchmarkAgent selections into experiment plan
            if isinstance(plan, dict) and _benchmark_plan.selected_benchmarks:
                plan["datasets"] = [
                    b["name"] for b in _benchmark_plan.selected_benchmarks
                ]
                # Normalize existing baselines to list of strings
                # BUG-35: LLM may emit baselines as dict, list of dicts,
                # or list of strings — normalize all to list[str].
                _baselines_from_plan = plan.get("baselines", [])
                if isinstance(_baselines_from_plan, dict):
                    _baselines_from_plan = list(_baselines_from_plan.keys())
                elif isinstance(_baselines_from_plan, list):
                    _baselines_from_plan = [
                        item["name"] if isinstance(item, dict) else str(item)
                        for item in _baselines_from_plan
                    ]
                else:
                    _baselines_from_plan = []
                plan["baselines"] = [
                    bl["name"] for bl in _benchmark_plan.selected_baselines
                ] + _baselines_from_plan
                # Deduplicate baselines
                plan["baselines"] = list(dict.fromkeys(plan["baselines"]))

            logger.info(
                "BenchmarkAgent: %d benchmarks, %d baselines selected (%d LLM calls, %.1fs)",
                len(_benchmark_plan.selected_benchmarks),
                len(_benchmark_plan.selected_baselines),
                _benchmark_plan.total_llm_calls,
                _benchmark_plan.elapsed_sec,
            )
        except Exception as _ba_exc:
            logger.warning("BenchmarkAgent failed (non-fatal): %s", _ba_exc)

    # Save benchmark plan for code_generation stage
    if _benchmark_plan is not None:
        try:
            (stage_dir / "benchmark_plan.json").write_text(
                json.dumps(_benchmark_plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    plan.setdefault("topic", config.research.topic)
    (stage_dir / "exp_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.DONE,
        artifacts=("exp_plan.yaml",),
        evidence_refs=("stage-09/exp_plan.yaml",),
    )


def _execute_code_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    metric = config.experiment.metric_key
    max_repair = 5  # BUG-14: Increased from 3 to give more chances for critical bugs
    files: dict[str, str] = {}
    validation_log: list[str] = []

    # --- Detect available packages for sandbox ---
    _pm = prompts or PromptManager()

    # --- Hardware-aware package hint ---
    hw_profile = _load_hardware_profile(run_dir)
    if config.experiment.mode in ("sandbox", "docker"):
        if config.experiment.mode == "docker":
            pkg_prefix = "docker mode"
            _net_policy = config.experiment.docker.network_policy
            _base_pkgs = (
                ", torchvision, torchaudio, matplotlib, seaborn, scipy, "
                "tqdm, torchdiffeq, gymnasium, networkx, PyYAML, Pillow, "
                "transformers, datasets, accelerate, peft, bitsandbytes, "
                "timm, einops, torchmetrics, h5py"
            )
            if _net_policy == "none":
                pkg_extras = _base_pkgs + " (ONLY pre-installed packages — NO pip install available)"
            elif _net_policy in ("setup_only", "pip_only"):
                pkg_extras = _base_pkgs + ", and additional pip-installable packages via requirements.txt"
            else:
                pkg_extras = _base_pkgs + ", and additional pip-installable packages (auto-detected from imports)"
        else:
            pkg_prefix = "sandbox mode"
            pkg_extras = ""
        if hw_profile and hw_profile.get("has_gpu"):
            gpu_type = hw_profile.get("gpu_type", "cuda")
            gpu_name = hw_profile.get("gpu_name", "GPU")
            tier = hw_profile.get("tier", "limited")
            if tier == "high":
                device_hint = f"torch.device('{gpu_type}')"
                pkg_hint = (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {gpu_name} ({gpu_type}). You MAY use PyTorch with GPU acceleration.\n"
                    f"Use `device = {device_hint}` for tensor operations.\n"
                )
            else:  # limited (low VRAM NVIDIA or MPS)
                device_hint = f"torch.device('{gpu_type}')"
                pkg_hint = (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {gpu_name} ({gpu_type}) — LIMITED performance.\n"
                    f"Use `device = {device_hint}` but design LIGHTWEIGHT experiments:\n"
                    f"- Small models (<1M parameters)\n"
                    f"- Few epochs (<=20)\n"
                    f"- Small datasets (<=10K samples)\n"
                    f"- Avoid large batch sizes\n"
                )
        else:
            pkg_hint = _pm.block("pkg_hint_sandbox")
    else:
        pkg_hint = ""

    # --- Compute budget hint ---
    time_budget_sec = config.experiment.time_budget_sec
    try:
        compute_budget = _pm.block("compute_budget").replace(
            "{time_budget_sec}", str(time_budget_sec)
        )
    except Exception:  # noqa: BLE001
        compute_budget = (
            f"\n## Compute Budget Constraint\n"
            f"- Total execution time limit: {time_budget_sec} seconds\n"
            f"- Design experiments that complete within this budget\n"
            f"- Implement a time guard: stop gracefully at 80% of budget\n"
        )

    # --- Dataset guidance + setup script + HP reporting (docker/sandbox modes) ---
    extra_guidance = ""

    # Inject KB files that list pre-built binaries and existing datasets so the
    # code agent can call real executables instead of generating synthetic data.
    _kb_for_code = _read_kb_files(
        getattr(config.knowledge_base, "root", ""),
        ["compute_inventory.md", "data_inventory.md", "experiments_summary.md"],
        max_chars_each=2000,
    )
    if _kb_for_code:
        extra_guidance += (
            "\n\n## Pre-existing Project Resources (USE THESE — do not generate synthetic data)\n"
            "The following files describe real data and pre-built binaries available for this experiment.\n"
        )
        for _kb_fn, _kb_txt in _kb_for_code:
            extra_guidance += f"\n### {_kb_fn}\n{_kb_txt}\n"

    _net_policy = getattr(getattr(config, "docker", None), "network_policy", "setup_only")
    if config.experiment.mode in ("sandbox", "docker"):
        _net_policy = (
            config.experiment.docker.network_policy
            if config.experiment.mode == "docker"
            else "none"  # sandbox mode has no network
        )
        if _net_policy == "none":
            # Network disabled: inject strict offline-only guidance
            try:
                extra_guidance += _pm.block("network_disabled_guidance")
            except Exception:  # noqa: BLE001
                pass
        elif _net_policy == "full":
            try:
                extra_guidance += _pm.block("dataset_guidance")
                extra_guidance += _pm.block("network_full_guidance")
            except Exception:  # noqa: BLE001
                pass
        else:
            # setup_only or pip_only — existing behavior
            try:
                extra_guidance += _pm.block("dataset_guidance")
            except Exception:  # noqa: BLE001
                pass
            if config.experiment.mode == "docker":
                try:
                    extra_guidance += _pm.block("setup_script_guidance")
                except Exception:  # noqa: BLE001
                    pass
        try:
            extra_guidance += _pm.block("hp_reporting")
        except Exception:  # noqa: BLE001
            pass
        # I-06: Multi-seed enforcement for all experiments
        try:
            extra_guidance += _pm.block("multi_seed_enforcement")
        except Exception:  # noqa: BLE001
            pass

    # --- BA: Inject BenchmarkAgent plan from Stage 9 ---
    _bp_path = None
    for _s9_dir in sorted(run_dir.glob("stage-09*"), reverse=True):
        _candidate = _s9_dir / "benchmark_plan.json"
        if _candidate.exists():
            _bp_path = _candidate
            break
    if _bp_path is not None:
        try:
            import json as _json_bp
            _bp_data = _json_bp.loads(_bp_path.read_text(encoding="utf-8"))
            # Reconstruct the prompt block
            from researchclaw.agents.benchmark_agent.orchestrator import BenchmarkPlan
            _bp = BenchmarkPlan(
                selected_benchmarks=_bp_data.get("selected_benchmarks", []),
                selected_baselines=_bp_data.get("selected_baselines", []),
                data_loader_code=_bp_data.get("data_loader_code", ""),
                baseline_code=_bp_data.get("baseline_code", ""),
                experiment_notes=_bp_data.get("experiment_notes", ""),
            )
            _bp_block = _bp.to_prompt_block()
            if _bp_block:
                extra_guidance += (
                    "\n\n## BenchmarkAgent Selections (USE THESE)\n"
                    "The following datasets, baselines, and code snippets were "
                    "automatically selected and validated by the BenchmarkAgent. "
                    "You MUST use these selections in your experiment code.\n\n"
                    + _bp_block
                )
                logger.info(
                    "BA: Injected benchmark plan (%d benchmarks, %d baselines)",
                    len(_bp.selected_benchmarks), len(_bp.selected_baselines),
                )
        except Exception as _bp_exc:
            logger.debug("BA: Failed to load benchmark plan: %s", _bp_exc)

    # --- P2.2+P2.3: LLM training topic detection and guidance ---
    _llm_keywords = (
        "language model", "llm", "fine-tun", "lora", "qlora", "peft",
        "instruction tun", "rlhf", "dpo", "sft", "alignment",
        "transformer train", "causal lm", "chat model", "qwen", "llama",
        "mistral", "phi-", "gemma", "pretraining", "tokeniz",
    )
    topic_lower = config.research.topic.lower()
    is_llm_topic = any(kw in topic_lower for kw in _llm_keywords)

    # --- I-08: RL topic detection and step guidance ---
    _rl_keywords = (
        "reinforcement learning", "policy gradient", "ppo", "sac", "td3",
        "ddpg", "dqn", "a2c", "a3c", "mujoco", "locomotion", "continuous control",
        "reward shaping", "exploration", "multi-agent rl", "marl", "curriculum rl",
        "imitation learning", "inverse rl", "offline rl", "model-based rl",
        "actor-critic", "reinforce", "gym", "gymnasium",
    )
    is_rl_topic = any(kw in topic_lower for kw in _rl_keywords)
    if is_rl_topic:
        try:
            extra_guidance += _pm.block("rl_step_guidance")
        except Exception:  # noqa: BLE001
            pass

    # --- F-01: Framework API doc injection (auto-detected) ---
    try:
        from researchclaw.data import detect_frameworks, load_framework_docs
        _hypothesis_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        _fw_ids = detect_frameworks(
            config.research.topic, _hypothesis_text, exp_plan or ""
        )
        if _fw_ids:
            _fw_docs = load_framework_docs(_fw_ids, max_chars=8000)
            if _fw_docs:
                extra_guidance += _fw_docs
                logger.info("F-01: Injected framework docs for: %s", _fw_ids)
    except Exception:  # noqa: BLE001
        logger.debug("F-01: Framework doc injection skipped", exc_info=True)

    if is_llm_topic and config.experiment.mode == "docker":
        try:
            extra_guidance += _pm.block("llm_training_guidance")
        except Exception:  # noqa: BLE001
            pass
        try:
            extra_guidance += _pm.block("llm_eval_guidance")
        except Exception:  # noqa: BLE001
            pass
        # P2.3: Warn if time budget is too short for LLM training
        if time_budget_sec < 3600:
            extra_guidance += (
                "\n## COMPUTE BUDGET WARNING\n"
                f"Current time_budget_sec={time_budget_sec} is likely TOO SHORT "
                f"for LLM fine-tuning. Typical LoRA training needs 1-4 hours. "
                f"Design a LIGHTWEIGHT experiment:\n"
                f"- Use a small dataset (<=5000 samples)\n"
                f"- Train for 1-3 epochs only\n"
                f"- Use small batch size (1-2) with gradient accumulation\n"
                f"- Use 4-bit quantization (QLoRA) to minimize memory\n"
                f"- Limit max_seq_length to 512-1024\n"
                f"- If possible, use a smaller model (<=7B parameters)\n"
            )

    # --- Domain-specific guidance injection for non-ML domains ---
    try:
        from researchclaw.domains.detector import detect_domain as _dd_s10, is_ml_domain as _is_ml_s10
        _dp = _dd_s10(topic=config.research.topic)
        if not _is_ml_s10(_dp):
            from researchclaw.domains.prompt_adapter import get_adapter as _ga
            _adapter = _ga(_dp)
            _blocks = _adapter.get_code_generation_blocks({})
            if _blocks.compute_budget:
                compute_budget = _blocks.compute_budget
            if _blocks.dataset_guidance:
                extra_guidance = _blocks.dataset_guidance + "\n" + extra_guidance
            if _blocks.code_generation_hints:
                extra_guidance += "\n" + _blocks.code_generation_hints
            if _blocks.output_format_guidance:
                extra_guidance += "\n" + _blocks.output_format_guidance
            logger.info("Injected domain-specific guidance for %s", _dp.domain_id)
    except Exception:  # noqa: BLE001
        logger.debug("Domain guidance injection skipped", exc_info=True)

    # --- Code generation: Beast Mode → CodeAgent → Legacy single-shot ---
    _code_agent_active = False
    _beast_mode_used = False
    _code_max_tokens = 8192

    # ── Beast Mode: OpenCode external agent (optional) ─────────────────
    _oc_cfg = config.experiment.opencode
    if _oc_cfg.enabled:
        from researchclaw.pipeline.opencode_bridge import (
            OpenCodeBridge,
            OpenCodeResult,
            count_historical_failures,
            score_complexity,
        )

        _hist_failures = count_historical_failures(run_dir)
        _cplx = score_complexity(
            exp_plan=exp_plan,
            topic=config.research.topic,
            historical_failures=_hist_failures,
            threshold=_oc_cfg.complexity_threshold,
        )

        # Persist complexity analysis
        (stage_dir / "complexity_analysis.json").write_text(
            json.dumps(
                {
                    "score": _cplx.score,
                    "signals": _cplx.signals,
                    "recommendation": _cplx.recommendation,
                    "reason": _cplx.reason,
                    "threshold": _oc_cfg.complexity_threshold,
                    "historical_failures": _hist_failures,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if _cplx.recommendation == "beast_mode":
            _proceed = _oc_cfg.auto
            if not _proceed:
                # Non-auto mode: check for HITL adapter
                if adapters.hitl is not None:
                    try:
                        _proceed = adapters.hitl.confirm(
                            f"Beast Mode: complexity={_cplx.score:.2f} "
                            f"(threshold={_oc_cfg.complexity_threshold}). "
                            f"Route to OpenCode?"
                        )
                    except Exception:  # noqa: BLE001
                        logger.info(
                            "Beast mode: HITL adapter unavailable, skipping "
                            "(set opencode.auto=true for non-interactive runs)"
                        )
                else:
                    logger.info(
                        "Beast mode: no HITL adapter, skipping "
                        "(set opencode.auto=true for non-interactive runs)"
                    )

            if _proceed:
                _oc_model = _oc_cfg.model or config.llm.primary_model
                _bridge = OpenCodeBridge(
                    model=_oc_model,
                    llm_base_url=config.llm.base_url,
                    api_key_env=config.llm.api_key_env,
                    llm_provider=config.llm.provider,
                    timeout_sec=_oc_cfg.timeout_sec,
                    max_retries=_oc_cfg.max_retries,
                    workspace_cleanup=_oc_cfg.workspace_cleanup,
                )

                logger.info(
                    "Beast mode: ENGAGED (complexity=%.2f, model=%s)",
                    _cplx.score,
                    _oc_model,
                )

                _oc_result: OpenCodeResult = _bridge.generate(
                    stage_dir=stage_dir,
                    topic=config.research.topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint + "\n" + compute_budget,
                    extra_guidance=extra_guidance,
                    time_budget_sec=config.experiment.time_budget_sec,
                )

                # Persist beast mode log
                (stage_dir / "beast_mode_log.json").write_text(
                    json.dumps(
                        {
                            "success": _oc_result.success,
                            "elapsed_sec": _oc_result.elapsed_sec,
                            "files": list(_oc_result.files.keys()),
                            "error": _oc_result.error,
                            "complexity_score": _cplx.score,
                            "model": _oc_model,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if _oc_result.success and _oc_result.files:
                    files = _oc_result.files
                    _beast_mode_used = True
                    _code_agent_active = True  # skip legacy path
                    logger.info(
                        "Beast mode: SUCCESS — %d files in %.1fs",
                        len(files),
                        _oc_result.elapsed_sec,
                    )
                else:
                    logger.warning(
                        "Beast mode: FAILED (%s) — falling back to CodeAgent",
                        _oc_result.error or "unknown error",
                    )
        else:
            logger.info(
                "Beast mode: complexity=%.2f (threshold=%.2f), not triggered",
                _cplx.score,
                _oc_cfg.complexity_threshold,
            )

    if not _beast_mode_used and config.experiment.code_agent.enabled and llm is not None:
        # ── F-02: Advanced Code Agent path ────────────────────────────────
        from researchclaw.pipeline.code_agent import CodeAgent as _CodeAgent

        _ca_cfg = config.experiment.code_agent
        # Ensure we have a proper config object
        if not hasattr(_ca_cfg, "enabled"):
            from researchclaw.pipeline.code_agent import (
                CodeAgentConfig as _CAConfig,
            )
            _ca_cfg = _CAConfig()

        # Sandbox factory (only for sandbox/docker modes)
        _sandbox_factory = None
        if config.experiment.mode in ("sandbox", "docker"):
            from researchclaw.experiment.factory import (
                create_sandbox as _csb,
            )
            _sandbox_factory = _csb

        if any(
            config.llm.primary_model.startswith(p)
            for p in ("gpt-5", "o3", "o4")
        ):
            _code_max_tokens = 16384

        # ── Domain detection + Code Search for non-ML domains ──────────
        _domain_profile = None
        _code_search_result = None
        try:
            from researchclaw.domains.detector import detect_domain as _dd
            from researchclaw.domains.detector import is_ml_domain as _is_ml
            _domain_profile = _dd(topic=config.research.topic)
            logger.info(
                "CodeAgent: domain=%s (%s)",
                _domain_profile.display_name,
                _domain_profile.domain_id,
            )
            # Run code search for non-ML domains (ML has enough built-in knowledge)
            if not _is_ml(_domain_profile):
                try:
                    from researchclaw.agents.code_searcher import CodeSearchAgent
                    _cs_agent = CodeSearchAgent(llm=llm)
                    _code_search_result = _cs_agent.search(
                        topic=config.research.topic,
                        domain=_domain_profile,
                    )
                    if _code_search_result and _code_search_result.patterns.has_content:
                        logger.info(
                            "Code search: %d patterns, %d repos found",
                            len(_code_search_result.patterns.api_patterns),
                            len(_code_search_result.repos_found),
                        )
                except Exception:  # noqa: BLE001
                    logger.debug("Code search unavailable", exc_info=True)
        except Exception:  # noqa: BLE001
            logger.debug("Domain detection unavailable", exc_info=True)

        _agent = _CodeAgent(
            llm=llm,
            prompts=_pm,
            config=_ca_cfg,
            stage_dir=stage_dir,
            sandbox_factory=_sandbox_factory,
            experiment_config=config.experiment,
            domain_profile=_domain_profile,
            code_search_result=_code_search_result,
        )
        _agent_result = _agent.generate(
            topic=config.research.topic,
            exp_plan=exp_plan,
            metric=metric,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            max_tokens=_code_max_tokens,
        )
        files = _agent_result.files
        _code_agent_active = True

        # Write agent artifacts
        (stage_dir / "code_agent_log.json").write_text(
            json.dumps(
                {
                    "log": _agent_result.validation_log,
                    "llm_calls": _agent_result.total_llm_calls,
                    "sandbox_runs": _agent_result.total_sandbox_runs,
                    "best_score": _agent_result.best_score,
                    "tree_nodes_explored": _agent_result.tree_nodes_explored,
                    "review_rounds": _agent_result.review_rounds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if _agent_result.architecture_spec:
            (stage_dir / "architecture_spec.yaml").write_text(
                _agent_result.architecture_spec, encoding="utf-8",
            )
        logger.info(
            "CodeAgent: %d LLM calls, %d sandbox runs, score=%.2f",
            _agent_result.total_llm_calls,
            _agent_result.total_sandbox_runs,
            _agent_result.best_score,
        )
    elif not _beast_mode_used and llm is not None:
        # ── Legacy single-shot generation ─────────────────────────────────
        topic = config.research.topic
        _md = config.experiment.metric_direction
        _md_hint = (
            f"`{_md}` — use direction={'lower' if _md == 'minimize' else 'higher'} "
            f"in METRIC_DEF. You MUST NOT use the opposite direction."
        )
        _overlay = _get_evolution_overlay(run_dir, "code_generation")
        sp = _pm.for_stage(
            "code_generation",
            evolution_overlay=_overlay,
            topic=topic,
            metric=metric,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
            metric_direction_hint=_md_hint,
        )
        # R13-3: Use higher max_tokens for reasoning models (they consume tokens
        # for internal chain-of-thought). Retry once with even higher limit on empty.
        _code_max_tokens = sp.max_tokens or 8192
        if any(config.llm.primary_model.startswith(p) for p in ("gpt-5", "o3", "o4")):
            _code_max_tokens = max(_code_max_tokens, 16384)

        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=_code_max_tokens,
        )
        files = _extract_multi_file_blocks(resp.content)
        if not files and not resp.content.strip():
            # Empty response — retry with higher token limit
            logger.warning(
                "R13-3: Empty LLM response for code_generation (len=%d, "
                "finish_reason=%s, tokens=%d). Retrying with 32768 tokens.",
                len(resp.content),
                resp.finish_reason,
                resp.total_tokens,
            )
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=32768,
            )
            files = _extract_multi_file_blocks(resp.content)
        if not files:
            logger.warning(
                "R13-2: _extract_multi_file_blocks returned empty. "
                "LLM response length=%d, first 300 chars: %s",
                len(resp.content),
                resp.content[:300],
            )

    # --- Fallback: generic numerical experiment ---
    if not files:
        files = {
            "main.py": (
                "import numpy as np\n"
                "\n"
                "np.random.seed(42)\n"
                "\n"
                "# Fallback experiment: parameter sweep on a synthetic objective\n"
                "# This runs when LLM code generation fails to produce valid code.\n"
                "dim = 10\n"
                "n_conditions = 3\n"
                "results = {}\n"
                "\n"
                "for cond_idx in range(n_conditions):\n"
                "    cond_name = f'condition_{cond_idx}'\n"
                "    scores = []\n"
                "    for seed in range(3):\n"
                "        rng = np.random.RandomState(seed + cond_idx * 100)\n"
                "        x = rng.randn(dim)\n"
                "        score = float(1.0 / (1.0 + np.sum(x ** 2)))\n"
                "        scores.append(score)\n"
                "    mean_score = float(np.mean(scores))\n"
                "    results[cond_name] = mean_score\n"
                f"    print(f'condition={{cond_name}} {metric}: {{mean_score:.6f}}')\n"
                "\n"
                "best = max(results, key=results.get)\n"
                f"print(f'{metric}: {{results[best]:.6f}}')\n"
            )
        }

    # --- Validate each file + auto-repair loop ---
    all_valid = True
    attempt = 0
    for fname, code in list(files.items()):
        validation = validate_code(code)
        repair_attempt = 0
        while not validation.ok and llm is not None and repair_attempt < max_repair:
            repair_attempt += 1
            attempt += 1
            issues_text = format_issues_for_llm(validation)
            validation_log.append(
                f"File {fname} attempt {repair_attempt}: {validation.summary()}"
            )
            logger.info(
                "Code validation failed for %s (attempt %d/%d): %s",
                fname,
                repair_attempt,
                max_repair,
                validation.summary(),
            )
            all_files_ctx = "\n\n".join(
                f"```filename:{f}\n{c}\n```" for f, c in files.items()
            )
            rp = _pm.sub_prompt(
                "code_repair",
                fname=fname,
                issues_text=issues_text,
                all_files_ctx=all_files_ctx,
            )
            resp = _chat_with_prompt(llm, rp.system, rp.user)
            files[fname] = _extract_code_block(resp.content)
            validation = validate_code(files[fname])
        if not validation.ok:
            all_valid = False
            # BUG-14: Log remaining issues prominently
            logger.warning(
                "Code validation FAILED for %s after %d repair attempts: %s",
                fname, max_repair, validation.summary(),
            )

    # BUG-14: Block on critical validation failures (syntax/import errors)
    if not all_valid:
        _has_critical = False
        for fname, code in files.items():
            _v = validate_code(code)
            if not _v.ok:
                for issue in _v.issues:
                    if issue.severity == "error" and issue.category in (
                        "syntax", "import",
                    ):
                        _has_critical = True
        if _has_critical:
            logger.error(
                "Stage 10: CRITICAL validation issues remain after %d repair "
                "attempts. Blocking stage.", max_repair,
            )
            (stage_dir / "validation_report.md").write_text(
                "# Code Validation Report\n\n"
                f"**Status**: BLOCKED — critical issues remain after {max_repair} repairs\n\n"
                + "\n".join(f"- {e}" for e in validation_log),
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.FAILED,
                artifacts=("validation_report.md",),
                evidence_refs=(),
            )

    # --- Write experiment directory ---
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for fname, code in files.items():
        (exp_dir / fname).write_text(code, encoding="utf-8")

    # --- Write validation report ---
    if validation_log or not all_valid:
        report_lines = ["# Code Validation Report\n"]
        if all_valid:
            report_lines.append(f"**Status**: PASSED after {attempt} total repair(s)\n")
        else:
            report_lines.append(
                f"**Status**: FAILED after {attempt} total repair attempt(s)\n"
            )
        for entry in validation_log:
            report_lines.append(f"- {entry}")
        (stage_dir / "validation_report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

    # --- R10-Fix6: Code complexity and quality check ---
    from researchclaw.experiment.validator import (
        auto_fix_unbound_locals,
        check_code_complexity,
        deep_validate_files,
    )

    # --- BUG-3 fix: Programmatic auto-fix for UnboundLocalError patterns ---
    _total_ub_fixes = 0
    for fname, code in list(files.items()):
        if fname.endswith(".py"):
            fixed_code, n_fixes = auto_fix_unbound_locals(code)
            if n_fixes > 0:
                files[fname] = fixed_code
                (exp_dir / fname).write_text(fixed_code, encoding="utf-8")
                _total_ub_fixes += n_fixes
                logger.info(
                    "Stage 10: auto-fixed %d UnboundLocalError risk(s) in %s",
                    n_fixes, fname,
                )
    if _total_ub_fixes:
        logger.info(
            "Stage 10: auto-fixed %d total UnboundLocalError risks", _total_ub_fixes
        )

    complexity_warnings: list[str] = []
    for fname, code in files.items():
        if fname.endswith(".py"):
            cw = check_code_complexity(code)
            for w in cw:
                complexity_warnings.append(f"[{fname}] {w}")
                logger.warning("Stage 10 code quality: [%s] %s", fname, w)

    # --- P1.1+P1.2: Deep quality analysis (class quality, scoping, API) ---
    deep_warnings = deep_validate_files(files)
    for w in deep_warnings:
        logger.warning("Stage 10 deep quality: %s", w)
    complexity_warnings.extend(deep_warnings)

    # --- P1.2: If critical deep issues found, attempt one repair cycle ---
    critical_deep = [w for w in deep_warnings if any(
        kw in w for kw in ("UnboundLocalError", "unregistered", "does not exist",
                           "empty or trivial subclass", "does NOT override",
                           "Import-usage mismatch", "NameError",
                           "was removed", "ptp()",
                           "copy-paste", "identical method signatures",
                           "identical AST", "NOT a real ablation")
    )]
    if critical_deep and llm is not None:
        logger.info(
            "Stage 10: %d critical code issues found — triggering repair cycle",
            len(critical_deep),
        )
        repair_issues = "\n".join(f"- {w}" for w in critical_deep)
        all_code_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        repair_prompt = (
            f"CRITICAL CODE QUALITY ISSUES FOUND:\n{repair_issues}\n\n"
            f"Fix ALL these issues in the code below. Return the complete "
            f"corrected files using ```filename:xxx.py format.\n\n"
            f"RULES:\n"
            f"- nn.Linear/nn.Conv must be created in __init__(), not forward()\n"
            f"- Variables used after if/else must be defined before the branch\n"
            f"- Use scipy.special.erf, not np.erf\n"
            f"- Ablation/variant classes must have genuinely different logic\n"
            f"- Every class must have a real implementation, not just `pass`\n"
            f"- Ablation classes MUST override the parent method that implements "
            f"the component being ablated (e.g., if ablating attention, override "
            f"the attention method with a simpler alternative like mean pooling)\n"
            f"- IMPORT CONSISTENCY: if you write `from X import Y`, call `Y()` "
            f"directly — NOT `X.Y()`. Mixing styles causes NameError.\n"
            f"- NumPy 2.0: ndarray.ptp() was removed — use arr.max()-arr.min()\n"
            f"- NumPy 2.0: np.bool/np.int/np.float removed — use builtins\n"
            f"- Pretrained models (EfficientNet, ResNet, ViT) expect 224×224 input "
            f"— add `transforms.Resize(224)` when using CIFAR (32×32) or similar\n"
            f"- Copy-paste ablation: if two classes have identical bodies, REWRITE "
            f"the ablation to genuinely remove/reduce a component (e.g., zero out "
            f"attention weights, halve hidden dimensions, remove a loss term)\n"
            f"- KD: teacher must be frozen, add projection layers if teacher_dim != "
            f"student_dim, use temperature T=4 for soft targets\n\n"
            f"Current code:\n{all_code_ctx}\n"
        )
        try:
            repair_resp = _chat_with_prompt(
                llm,
                _pm.prompts["code_generation"]["system"],
                repair_prompt,
                max_tokens=_code_max_tokens,
            )
            repaired = _extract_multi_file_blocks(repair_resp.content)
            if repaired and "main.py" in repaired:
                files = repaired
                for fname, code in files.items():
                    (exp_dir / fname).write_text(code, encoding="utf-8")
                # Re-check after repair
                deep_warnings_after = deep_validate_files(files)
                fixed = len(critical_deep) - len([
                    w for w in deep_warnings_after
                    if any(kw in w for kw in (
                        "UnboundLocalError", "unregistered", "does not exist",
                        "empty or trivial subclass", "does NOT override",
                        "Import-usage mismatch", "NameError",
                        "was removed", "ptp()",
                        "copy-paste", "identical method signatures",
                        "identical AST", "NOT a real ablation"
                    ))
                ])
                logger.info(
                    "Stage 10: Deep repair fixed %d/%d critical issues",
                    fixed, len(critical_deep),
                )
                complexity_warnings.append(
                    f"[REPAIR] Deep repair fixed {fixed}/{len(critical_deep)} "
                    f"critical issues"
                )
        except Exception as exc:
            logger.debug("Deep repair failed: %s", exc)

    if complexity_warnings:
        health: dict[str, Any] = {}
        health["code_complexity_warnings"] = complexity_warnings
        (stage_dir / "stage_health.json").write_text(
            json.dumps(health, indent=2), encoding="utf-8"
        )

    # --- P1.4: LLM Code Review (Stage 10.5) ---
    # Skip when CodeAgent is active — Phase 4 review already covers this.
    if llm is not None and not _code_agent_active:
        all_code_review = "\n\n".join(
            f"# --- {fname} ---\n{code}" for fname, code in files.items()
        )
        if len(all_code_review) > 12000:
            all_code_review = all_code_review[:12000] + "\n... [truncated]"
        review_prompt = (
            f"You are a senior researcher reviewing experiment code for a "
            f"research submission.\n\n"
            f"TOPIC: {config.research.topic}\n"
            f"EXPERIMENT PLAN:\n{exp_plan[:3000]}\n\n"
            f"CODE:\n```python\n{all_code_review}\n```\n\n"
            f"Review the code and return JSON with this EXACT structure:\n"
            f'{{"score": <1-10>, "issues": ['
            f'{{"severity": "critical|major|minor", '
            f'"description": "...", "fix": "..."}}], '
            f'"verdict": "pass|needs_fix"}}\n\n'
            f"Check specifically:\n"
            f"1. Does each algorithm/method have a DISTINCT implementation? "
            f"(Not just renamed copies)\n"
            f"2. Are ablation conditions genuinely different from the main method?\n"
            f"3. Are loss functions / training loops mathematically correct?\n"
            f"4. Will the code actually run without errors? Check variable scoping, "
            f"API usage, tensor shape compatibility.\n"
            f"5. Is the code complex enough for a research paper? (Not trivial)\n"
            f"6. Are experimental conditions fairly compared (same seeds, data)?\n"
            f"7. If using pretrained models (EfficientNet, ResNet, ViT), are input "
            f"images resized to the model's expected size (e.g., 224x224)? CIFAR "
            f"images are 32x32 and MUST be resized for pretrained models.\n"
            f"8. Are imports consistent? `from X import Y` must use `Y()`, not `X.Y()`.\n"
        )
        try:
            review_resp = llm.chat(
                [{"role": "user", "content": review_prompt}],
                system="You are a meticulous ML code reviewer. Be strict.",
                max_tokens=2048,
            )
            # Extract JSON from LLM response (may be wrapped in markdown fences)
            _review_text = review_resp.content if hasattr(review_resp, "content") else str(review_resp)
            # Strip markdown JSON fences if present
            _review_text = _review_text.strip()
            if _review_text.startswith("```"):
                _lines = _review_text.splitlines()
                _start = 1 if _lines[0].strip().startswith("```") else 0
                _end = len(_lines) - 1 if _lines[-1].strip() == "```" else len(_lines)
                _review_text = "\n".join(_lines[_start:_end])
            review_data = _safe_json_loads(_review_text, {})
            if isinstance(review_data, dict):
                review_score = review_data.get("score", 0)
                review_verdict = review_data.get("verdict", "unknown")
                review_issues = review_data.get("issues", [])

                # Write review report
                review_report = {
                    "score": review_score,
                    "verdict": review_verdict,
                    "issues": review_issues,
                    "timestamp": _utcnow_iso(),
                }
                (stage_dir / "code_review.json").write_text(
                    json.dumps(review_report, indent=2), encoding="utf-8"
                )

                # If critical issues found and score low, attempt fix
                critical_issues = [
                    i for i in review_issues
                    if isinstance(i, dict)
                    and i.get("severity") == "critical"
                ]
                if critical_issues and review_score <= 4:
                    logger.warning(
                        "Stage 10 code review: score=%d, %d critical issues — "
                        "attempting fix",
                        review_score, len(critical_issues),
                    )
                    fix_descriptions = "\n".join(
                        f"- [{i.get('severity', '?')}] {i.get('description', '?')}: "
                        f"{i.get('fix', 'no fix suggested')}"
                        for i in critical_issues
                    )
                    fix_prompt = (
                        f"Code review found {len(critical_issues)} CRITICAL issues "
                        f"(score: {review_score}/10):\n{fix_descriptions}\n\n"
                        f"Fix ALL critical issues. Return complete corrected files "
                        f"using ```filename:xxx.py format.\n\n"
                        f"Current code:\n"
                        + "\n\n".join(
                            f"```filename:{f}\n{c}\n```" for f, c in files.items()
                        )
                    )
                    try:
                        fix_resp = _chat_with_prompt(
                            llm,
                            _pm.prompts["code_generation"]["system"],
                            fix_prompt,
                            max_tokens=_code_max_tokens,
                        )
                        fixed_files = _extract_multi_file_blocks(fix_resp.content)
                        if fixed_files and "main.py" in fixed_files:
                            files = fixed_files
                            for fname, code in files.items():
                                (exp_dir / fname).write_text(code, encoding="utf-8")
                            logger.info(
                                "Stage 10: Code fixed after review "
                                "(was %d/10, %d critical issues)",
                                review_score, len(critical_issues),
                            )
                    except Exception as exc:
                        logger.debug("Review-fix failed: %s", exc)
        except Exception as exc:
            logger.debug("Code review failed: %s", exc)

    # --- FIX-3: Topic-experiment alignment check ---
    alignment_ok = True
    alignment_note = ""
    if llm is not None:
        # Concatenate all code for alignment check
        all_code_for_check = "\n\n".join(
            f"# --- {fname} ---\n{code}" for fname, code in files.items()
        )
        # Truncate to avoid token overflow
        if len(all_code_for_check) > 8000:
            all_code_for_check = all_code_for_check[:8000] + "\n... [truncated]"
        align_prompt = (
            f"Research topic: {config.research.topic}\n\n"
            f"Experiment code:\n```python\n{all_code_for_check}\n```\n\n"
            "TASK: Evaluate whether this experiment code actually tests the "
            "stated research topic. Answer with JSON:\n"
            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n\n'
            "Check specifically:\n"
            "- Does the code implement models/methods relevant to the topic?\n"
            "- If the topic mentions LLMs/transformers/language models, does "
            "the code use or simulate them (not just small MLPs)?\n"
            "- If the topic mentions a specific technique (e.g. curriculum "
            "learning, RLHF), does the code actually implement it?\n"
            "- Are the experimental conditions meaningfully different from each other?\n"
        )
        try:
            align_resp = llm.chat(
                [{"role": "user", "content": align_prompt}],
                system="You are a scientific code reviewer checking topic-experiment alignment.",
                max_tokens=1024,
            )
            align_data = _safe_json_loads(align_resp.content, {})
            if isinstance(align_data, dict) and not align_data.get("aligned", True):
                alignment_ok = False
                alignment_note = align_data.get("reason", "Misaligned")
                suggestions = align_data.get("suggestions", "")
                logger.warning(
                    "Stage 10: Topic-experiment MISALIGNMENT detected: %s",
                    alignment_note,
                )
                # Attempt one regeneration with explicit alignment instruction
                regen_prompt = (
                    f"The experiment code you previously generated does NOT align "
                    f"with the research topic.\n\n"
                    f"TOPIC: {config.research.topic}\n"
                    f"MISALIGNMENT: {alignment_note}\n"
                    f"SUGGESTIONS: {suggestions}\n\n"
                    f"REGENERATE the experiment code to DIRECTLY test the stated "
                    f"topic. The code MUST implement the core technique described "
                    f"in the topic, not a generic proxy.\n\n"
                    f"{pkg_hint}\n{compute_budget}\n"
                    f"PLAN:\n{exp_plan}\n\n"
                    f"Return multiple files using ```filename:xxx.py format."
                )
                regen_resp = _chat_with_prompt(
                    llm,
                    system=_pm.prompts["code_generation"]["system"],
                    user=regen_prompt,
                    max_tokens=_code_max_tokens,
                )
                regen_files = _extract_multi_file_blocks(regen_resp.content)
                if regen_files and "main.py" in regen_files:
                    files = regen_files
                    for fname, code in files.items():
                        (exp_dir / fname).write_text(code, encoding="utf-8")
                    alignment_ok = True
                    alignment_note = "Regenerated after alignment check"
                    logger.info("Stage 10: Code regenerated after alignment fix")
        except Exception as exc:
            logger.debug("Alignment check failed: %s", exc)

    # --- FIX-7: Ablation distinctness check ---
    main_code = files.get("main.py", "")
    if llm is not None and main_code and "condition" in main_code.lower():
        try:
            ablation_prompt = (
                f"Examine this experiment code:\n```python\n{main_code[:6000]}\n```\n\n"
                "Check if any experimental conditions (methods/ablations) have "
                "IDENTICAL configurations (same hyperparameters, same code paths). "
                "Answer JSON: "
                '{"has_duplicates": true/false, "details": "which conditions are identical"}'
            )
            abl_resp = llm.chat(
                [{"role": "user", "content": ablation_prompt}],
                system="You are a code reviewer checking experimental conditions.",
                max_tokens=512,
            )
            abl_data = _safe_json_loads(abl_resp.content, {})
            if isinstance(abl_data, dict) and abl_data.get("has_duplicates"):
                logger.warning(
                    "Stage 10: Duplicate ablation conditions detected: %s",
                    abl_data.get("details", ""),
                )
                (stage_dir / "ablation_warning.json").write_text(
                    json.dumps(abl_data, indent=2), encoding="utf-8"
                )
                # --- Attempt ablation repair ---
                all_code_ctx = "\n\n".join(
                    f"```filename:{f}\n{c}\n```" for f, c in files.items()
                )
                dup_details = abl_data.get("details", "unknown")
                abl_repair_prompt = (
                    f"ABLATION REPAIR REQUIRED — duplicate conditions detected:\n"
                    f"{dup_details}\n\n"
                    f"Rewrite the ablation/variant conditions so each one is "
                    f"GENUINELY DIFFERENT. Concrete strategies:\n"
                    f"- 'no_<component>': REMOVE the component entirely "
                    f"(e.g., replace attention with mean pooling, remove a loss term)\n"
                    f"- 'reduced_capacity': HALVE hidden dimensions or layers\n"
                    f"- Different conditions MUST produce different outputs on the "
                    f"same input. Add a startup assertion that runs one forward pass "
                    f"per condition on identical input and prints:\n"
                    f"  ABLATION_CHECK: <cond1> vs <cond2> outputs_differ=True\n\n"
                    f"Return ALL files using ```filename:xxx.py format.\n\n"
                    f"Current code:\n{all_code_ctx}\n"
                )
                try:
                    abl_repair_resp = _chat_with_prompt(
                        llm,
                        _pm.prompts["code_generation"]["system"],
                        abl_repair_prompt,
                        max_tokens=_code_max_tokens,
                    )
                    repaired_files = _extract_multi_file_blocks(
                        abl_repair_resp.content
                    )
                    if repaired_files and "main.py" in repaired_files:
                        files = repaired_files
                        for fname, code in files.items():
                            (exp_dir / fname).write_text(code, encoding="utf-8")
                        logger.info(
                            "Stage 10: Ablation repair applied — "
                            "rewrote duplicate conditions"
                        )
                except Exception as exc:
                    logger.debug("Ablation repair failed: %s", exc)
        except Exception:
            pass

    # --- Write spec ---
    file_list = ", ".join(f"`{f}`" for f in sorted(files.keys()))
    main_validation = validate_code(files.get("main.py", ""))
    _align_status = "ALIGNED" if alignment_ok else f"MISALIGNED: {alignment_note}"
    spec = f"""# Experiment Specification

## Topic
{config.research.topic}

## Project Structure
Multi-file experiment project with {len(files)} file(s): {file_list}

## Entry Point
`main.py` \u2014 executed directly via sandbox

## Outputs
- `main.py` emits metric lines in `name: value` format
- Primary metric key: `{metric}`

## Topic-Experiment Alignment
{_align_status}

## Constraints
- Time budget per run: {config.experiment.time_budget_sec}s
- Max iterations: {config.experiment.max_iterations}
- Self-contained execution (no external data, no network)
- Validated: {main_validation.summary()}

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "experiment_spec.md").write_text(spec, encoding="utf-8")

    artifacts = ["experiment/", "experiment_spec.md"]
    if (stage_dir / "validation_report.md").exists():
        artifacts.append("validation_report.md")

    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
    )


def _execute_resource_planning(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    schedule: dict[str, Any] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "resource_planning")
        sp = _pm.for_stage("resource_planning", evolution_overlay=_overlay, exp_plan=exp_plan)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        parsed = _safe_json_loads(resp.content, {})
        if isinstance(parsed, dict):
            schedule = parsed
    if schedule is None:
        schedule = {
            "tasks": [
                {
                    "id": "baseline",
                    "name": "Run baseline",
                    "depends_on": [],
                    "gpu_count": 1,
                    "estimated_minutes": 20,
                    "priority": "high",
                },
                {
                    "id": "proposed",
                    "name": "Run proposed method",
                    "depends_on": ["baseline"],
                    "gpu_count": 1,
                    "estimated_minutes": 30,
                    "priority": "high",
                },
            ],
            "total_gpu_budget": 1,
            "generated": _utcnow_iso(),
        }
    schedule.setdefault("generated", _utcnow_iso())
    (stage_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    return StageResult(
        stage=Stage.RESOURCE_PLANNING,
        status=StageStatus.DONE,
        artifacts=("schedule.json",),
        evidence_refs=("stage-11/schedule.json",),
    )


def _execute_experiment_run(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.experiment.factory import create_sandbox
    from researchclaw.experiment.runner import ExperimentRunner

    schedule_text = _read_prior_artifact(run_dir, "schedule.json") or "{}"
    # Try multi-file experiment directory first, fall back to single file
    exp_dir_path = _read_prior_artifact(run_dir, "experiment/")
    code_text = ""
    if exp_dir_path and Path(exp_dir_path).is_dir():
        main_path = Path(exp_dir_path) / "main.py"
        if main_path.exists():
            code_text = main_path.read_text(encoding="utf-8")
    if not code_text:
        code_text = _read_prior_artifact(run_dir, "experiment.py") or ""

    runs_dir = stage_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    mode = config.experiment.mode
    if mode in ("sandbox", "docker"):
        # P7: Auto-install missing dependencies before subprocess sandbox
        if mode == "sandbox":
            _all_code = code_text
            if exp_dir_path and Path(exp_dir_path).is_dir():
                for _pyf in Path(exp_dir_path).glob("*.py"):
                    _all_code += "\n" + _pyf.read_text(encoding="utf-8")
            _ensure_sandbox_deps(_all_code, config.experiment.sandbox.python_path)

        sandbox = create_sandbox(config.experiment, runs_dir / "sandbox")
        # Use run_project for multi-file, run for single-file
        if exp_dir_path and Path(exp_dir_path).is_dir():
            result = sandbox.run_project(
                Path(exp_dir_path), timeout_sec=config.experiment.time_budget_sec
            )
        else:
            result = sandbox.run(
                code_text, timeout_sec=config.experiment.time_budget_sec
            )
        # Try to read structured results.json from sandbox working dir
        structured_results: dict[str, Any] | None = None
        sandbox_project = runs_dir / "sandbox" / "_project"
        results_json_path = sandbox_project / "results.json"
        if results_json_path.exists():
            try:
                structured_results = json.loads(
                    results_json_path.read_text(encoding="utf-8")
                )
                # Copy results.json to runs dir for easy access
                (runs_dir / "results.json").write_text(
                    results_json_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, OSError):
                structured_results = None

        # If sandbox metrics are empty, try to parse from stdout
        effective_metrics = result.metrics
        if not effective_metrics and result.stdout:
            effective_metrics = _parse_metrics_from_stdout(result.stdout)

        # Determine run status: completed / partial (timed out with data) / failed
        # R6-2: Detect stdout failure signals even when exit code is 0
        _stdout_has_failure = bool(
            result.stdout
            and not effective_metrics
            and any(
                sig in result.stdout
                for sig in ("FAIL:", "NaN/divergence", "Traceback (most recent")
            )
        )
        if result.returncode == 0 and not result.timed_out and not _stdout_has_failure:
            run_status = "completed"
        elif result.timed_out and effective_metrics:
            run_status = "partial"
            logger.warning(
                "Experiment timed out but captured %d partial metrics",
                len(effective_metrics),
            )
        else:
            run_status = "failed"
            if _stdout_has_failure:
                logger.warning(
                    "Experiment exited cleanly but stdout contains failure signals"
                )

        # P1: Warn if experiment completed suspiciously fast (trivially easy benchmark)
        if run_status == "completed" and result.elapsed_sec and result.elapsed_sec < 5.0:
            logger.warning(
                "Stage 12: Experiment completed in %.2fs — benchmark may be trivially easy. "
                "Consider increasing task difficulty.",
                result.elapsed_sec,
            )

        run_payload: dict[str, Any] = {
            "run_id": "run-1",
            "task_id": "sandbox-main",
            "status": run_status,
            "metrics": effective_metrics,
            "elapsed_sec": result.elapsed_sec,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "completed_at": _utcnow_iso(),
        }
        if structured_results is not None:
            run_payload["structured_results"] = structured_results
        # Auto-generate results.json from parsed metrics if sandbox didn't produce one
        if structured_results is None and effective_metrics:
            auto_results = {"source": "stdout_parsed", "metrics": effective_metrics}
            (runs_dir / "results.json").write_text(
                json.dumps(auto_results, indent=2), encoding="utf-8"
            )
            logger.info("Stage 12: Auto-generated results.json from stdout metrics (%d keys)", len(effective_metrics))
        (runs_dir / "run-1.json").write_text(
            json.dumps(run_payload, indent=2), encoding="utf-8"
        )

        # R11-6: Time budget adequacy check
        if result.timed_out or (result.elapsed_sec and result.elapsed_sec > config.experiment.time_budget_sec * 0.9):
            # Parse stdout to estimate how many conditions/seeds completed
            _stdout = result.stdout or ""
            _completed_conditions = set()
            _completed_seeds = 0
            for _line in _stdout.splitlines():
                if "condition=" in _line and "seed=" in _line:
                    _completed_seeds += 1
                    _cond_match = re.match(r".*condition=(\S+)", _line)
                    if _cond_match:
                        _completed_conditions.add(_cond_match.group(1))
            _time_budget_warning = {
                "timed_out": result.timed_out,
                "elapsed_sec": result.elapsed_sec,
                "budget_sec": config.experiment.time_budget_sec,
                "conditions_completed": sorted(_completed_conditions),
                "total_seed_runs": _completed_seeds,
                "warning": (
                    f"Experiment used {result.elapsed_sec:.0f}s of "
                    f"{config.experiment.time_budget_sec}s budget. "
                    f"Only {len(_completed_conditions)} conditions completed "
                    f"({_completed_seeds} seed-runs). Consider increasing "
                    f"time_budget_sec for more complete results."
                ),
            }
            logger.warning(
                "Stage 12: %s", _time_budget_warning["warning"]
            )
            (stage_dir / "time_budget_warning.json").write_text(
                json.dumps(_time_budget_warning, indent=2), encoding="utf-8"
            )

        # FIX-8: Validate seed count from structured results
        if structured_results and isinstance(structured_results, dict):
            _sr_conditions = structured_results.get("conditions", structured_results.get("per_condition", {}))
            if isinstance(_sr_conditions, dict):
                for _cname, _cdata in _sr_conditions.items():
                    if isinstance(_cdata, dict):
                        _seeds_run = _cdata.get("seeds_run", _cdata.get("n_seeds", 0))
                        if isinstance(_seeds_run, (int, float)) and 0 < _seeds_run < 3:
                            logger.warning(
                                "Stage 12: Condition '%s' ran only %d seed(s) — "
                                "minimum 3 required for statistical validity",
                                _cname, int(_seeds_run),
                            )

    elif mode == "simulated":
        schedule = _safe_json_loads(schedule_text, {})
        tasks = schedule.get("tasks", []) if isinstance(schedule, dict) else []
        if not isinstance(tasks, list):
            tasks = []
        for idx, task in enumerate(tasks or [{"id": "task-1", "name": "simulated"}]):
            task_id = (
                str(task.get("id", f"task-{idx + 1}"))
                if isinstance(task, dict)
                else f"task-{idx + 1}"
            )
            payload = {
                "run_id": f"run-{idx + 1}",
                "task_id": task_id,
                "status": "simulated",
                "key_metrics": {
                    config.experiment.metric_key: round(0.3 + idx * 0.03, 4),
                    "secondary_metric": round(0.6 - idx * 0.04, 4),
                },
                "notes": "Simulated run result",
                "completed_at": _utcnow_iso(),
            }
            run_id = str(payload["run_id"])
            (runs_dir / f"{_safe_filename(run_id)}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    else:
        runner = ExperimentRunner(config.experiment, runs_dir / "workspace")
        history = runner.run_loop(code_text, run_id=f"exp-{run_dir.name}", llm=llm)
        runner.save_history(stage_dir / "experiment_history.json")
        for item in history.results:
            payload = {
                "run_id": f"run-{item.iteration}",
                "task_id": item.run_id,
                "status": "completed" if item.error is None else "failed",
                "metrics": item.metrics,
                "primary_metric": item.primary_metric,
                "improved": item.improved,
                "kept": item.kept,
                "elapsed_sec": item.elapsed_sec,
                "error": item.error,
                "completed_at": _utcnow_iso(),
            }
            run_id = str(payload["run_id"])
            (runs_dir / f"{_safe_filename(run_id)}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    return StageResult(
        stage=Stage.EXPERIMENT_RUN,
        status=StageStatus.DONE,
        artifacts=("runs/",),
        evidence_refs=("stage-12/runs/",),
    )


def _execute_iterative_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.experiment.factory import create_sandbox
    from researchclaw.experiment.validator import format_issues_for_llm, validate_code

    def _to_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    # R10-Fix3: Skip iterative refinement in simulated mode (no real execution)
    if config.experiment.mode == "simulated":
        logger.info(
            "Stage 13: Skipping iterative refinement in simulated mode "
            "(no real code execution available)"
        )
        import shutil

        final_dir = stage_dir / "experiment_final"
        # Copy latest experiment code as final (directory or single file)
        copied = False
        for stage_num in (12, 10):
            src_dir = run_dir / f"stage-{stage_num:02d}" / "experiment"
            if src_dir.is_dir():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(src_dir, final_dir)
                copied = True
                break
            # Also check for single experiment.py
            src_file = run_dir / f"stage-{stage_num:02d}" / "experiment.py"
            if src_file.is_file():
                (stage_dir / "experiment_final.py").write_text(
                    src_file.read_text(encoding="utf-8"), encoding="utf-8"
                )
                copied = True
                break

        log: dict[str, Any] = {
            "generated": _utcnow_iso(),
            "mode": "simulated",
            "skipped": True,
            "skip_reason": "Iterative refinement not meaningful in simulated mode",
            "metric_key": config.experiment.metric_key,
        }
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=("refinement_log.json",),
            evidence_refs=(),
        )

    metric_key = config.experiment.metric_key
    metric_direction = config.experiment.metric_direction

    # P9: Detect metric direction mismatch between config and experiment code.
    # The code-gen stage instructs experiments to print a line like:
    #   METRIC_DEF: primary_metric | direction=higher | desc=...
    # Log a warning if mismatch is detected, but trust the config value
    # (BUG-06 fix: no longer auto-override, since Stage 9 and 12 now
    # explicitly enforce config.metric_direction in prompts).
    _runs_dir_detect = _read_prior_artifact(run_dir, "runs/")
    if _runs_dir_detect and Path(_runs_dir_detect).is_dir():
        import re as _re_detect

        for _rf in sorted(Path(_runs_dir_detect).glob("*.json"))[:5]:
            try:
                _rp = _safe_json_loads(_rf.read_text(encoding="utf-8"), {})
                _stdout = _rp.get("stdout", "") if isinstance(_rp, dict) else ""
                _match = _re_detect.search(
                    r"METRIC_DEF:.*direction\s*=\s*(higher|lower)", _stdout
                )
                if _match:
                    _detected = _match.group(1)
                    _detected_dir = "maximize" if _detected == "higher" else "minimize"
                    if _detected_dir != metric_direction:
                        logger.warning(
                            "P9: Metric direction mismatch — config says '%s' but "
                            "experiment code declares 'direction=%s'. "
                            "Keeping config value '%s'. Code will be "
                            "corrected in next refinement cycle.",
                            metric_direction,
                            _detected,
                            metric_direction,
                        )
                    break
            except OSError:
                pass

    maximize = metric_direction == "maximize"

    def _is_better(candidate: float | None, current: float | None) -> bool:
        if candidate is None:
            return False
        if current is None:
            return True
        return candidate > current if maximize else candidate < current

    def _find_metric(metrics: dict[str, object], key: str) -> float | None:
        """R13-4: Find metric value with fuzzy key matching.

        Tries exact match first, then looks for aggregate keys that contain
        the metric name (e.g. 'primary_metric_mean' when key='primary_metric').
        """
        # Exact match
        val = _to_float(metrics.get(key))
        if val is not None:
            return val
        # Try aggregate/mean keys containing the metric name
        # Prefer keys ending with the metric name or containing '_mean'
        candidates: list[tuple[str, float]] = []
        for mk, mv in metrics.items():
            fv = _to_float(mv)
            if fv is None:
                continue
            if mk == key or mk.endswith(f"/{key}"):
                return fv  # Exact match via condition prefix
            if key in mk and ("mean" in mk or "avg" in mk):
                candidates.append((mk, fv))
            elif mk.endswith(f"_{key}") or mk.endswith(f"/{key}_mean"):
                candidates.append((mk, fv))
        if candidates:
            # Take the aggregate mean if available, otherwise first match
            for ck, cv in candidates:
                if "mean" in ck:
                    return cv
            return candidates[0][1]
        # Last resort: if there's an "overall" or root-level aggregate
        for mk, mv in metrics.items():
            fv = _to_float(mv)
            if fv is not None and key in mk and "/" not in mk and "seed" not in mk:
                return fv
        return None

    requested_iterations = int(getattr(config.experiment, "max_iterations", 10) or 10)
    max_iterations = max(1, min(requested_iterations, 10))

    # BUG-57: Wall-clock time cap for the entire refinement stage.
    # Default: 3× the per-iteration time budget (e.g., 2400s → 7200s = 2h).
    import time as _time_bug57
    _refine_start_time = _time_bug57.monotonic()
    _per_iter_budget = int(getattr(config.experiment, "time_budget_sec", 2400) or 2400)
    _max_refine_wall_sec = int(
        getattr(config.experiment, "max_refine_duration_sec", 0) or 0
    ) or (_per_iter_budget * 3)

    # --- Collect baseline metrics from prior runs ---
    runs_dir_path: Path | None = None
    runs_dir_text = _read_prior_artifact(run_dir, "runs/")
    if runs_dir_text:
        runs_dir_path = Path(runs_dir_text)

    run_summaries: list[str] = []
    baseline_metric: float | None = None
    if runs_dir_path is not None:
        for run_file in sorted(runs_dir_path.glob("*.json"))[:40]:
            payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if not isinstance(payload, dict):
                continue
            # R5-5: Truncate stdout/stderr for context efficiency
            summary = dict(payload)
            if "stdout" in summary and isinstance(summary["stdout"], str):
                lines = summary["stdout"].splitlines()
                if len(lines) > 30:
                    summary["stdout"] = (
                        f"[...truncated {len(lines) - 30} lines...]\n"
                        + "\n".join(lines[-30:])
                    )
                if len(summary["stdout"]) > 2000:
                    summary["stdout"] = summary["stdout"][-2000:]
            if "stderr" in summary and isinstance(summary["stderr"], str):
                lines = summary["stderr"].splitlines()
                if len(lines) > 50:
                    summary["stderr"] = "\n".join(lines[-50:])
                if len(summary["stderr"]) > 2000:
                    summary["stderr"] = summary["stderr"][-2000:]
            run_summaries.append(json.dumps(summary, ensure_ascii=False))
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                metrics = (
                    payload.get("key_metrics")
                    if isinstance(payload.get("key_metrics"), dict)
                    else {}
                )
            metric_val = (
                _find_metric(metrics, metric_key)
                if isinstance(metrics, dict)
                else None
            )
            if metric_val is None:
                metric_val = _to_float(payload.get("primary_metric"))
            if _is_better(metric_val, baseline_metric):
                baseline_metric = metric_val

    # --- Read experiment project (multi-file or single-file) ---
    # BUG-58: When PIVOT rolls back to Stage 13, prefer the best refined code
    # from a previous cycle (stage-13_vX/experiment_final/) over the original
    # unrefined code (stage-12/experiment/ or stage-10/experiment/).
    exp_dir_text: str | None = None
    _prev_refine_dirs = sorted(
        run_dir.glob("stage-13_v*/experiment_final"),
        key=lambda p: p.parent.name,
        reverse=True,  # latest version first
    )
    if _prev_refine_dirs and _prev_refine_dirs[0].is_dir():
        exp_dir_text = str(_prev_refine_dirs[0])
        logger.info(
            "BUG-58: Recovered best refined code from previous PIVOT cycle: %s",
            _prev_refine_dirs[0].parent.name,
        )
    if not exp_dir_text:
        exp_dir_text = _read_prior_artifact(run_dir, "experiment/")
    best_files: dict[str, str] = {}
    if exp_dir_text and Path(exp_dir_text).is_dir():
        for pyfile in sorted(Path(exp_dir_text).glob("*.py")):
            best_files[pyfile.name] = pyfile.read_text(encoding="utf-8")
    if not best_files:
        # Backward compat: single experiment.py
        original_code = _read_prior_artifact(run_dir, "experiment.py") or ""
        if original_code:
            best_files = {"main.py": original_code}

    # --- Detect if prior experiment timed out ---
    prior_timed_out = False
    prior_time_budget = config.experiment.time_budget_sec
    if runs_dir_path is not None:
        for run_file in sorted(runs_dir_path.glob("*.json"))[:5]:
            try:
                payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
                if isinstance(payload, dict) and payload.get("timed_out"):
                    prior_timed_out = True
                    break
            except OSError:
                pass

    best_metric = baseline_metric
    best_version = "experiment/"
    # BUG-58: Recover best_metric from previous PIVOT cycle
    if _prev_refine_dirs:
        _prev_log_path = _prev_refine_dirs[0].parent / "refinement_log.json"
        if _prev_log_path.is_file():
            _prev_log = _safe_json_loads(
                _prev_log_path.read_text(encoding="utf-8"), {}
            )
            if isinstance(_prev_log, dict):
                _prev_best = _prev_log.get("best_metric")
                if isinstance(_prev_best, (int, float)) and _is_better(
                    _prev_best, best_metric
                ):
                    best_metric = _prev_best
                    logger.info(
                        "BUG-58: Recovered best_metric=%.4f from previous PIVOT",
                        best_metric,
                    )
    no_improve_streak = 0
    consecutive_no_metrics = 0

    log: dict[str, Any] = {
        "generated": _utcnow_iso(),
        "mode": config.experiment.mode,
        "metric_key": metric_key,
        "metric_direction": metric_direction,
        "max_iterations_requested": requested_iterations,
        "max_iterations_executed": max_iterations,
        "baseline_metric": baseline_metric,
        "project_files": list(best_files.keys()),
        "iterations": [],
        "converged": False,
        "stop_reason": "max_iterations_reached",
    }

    # --- Helper: write files to a directory ---
    def _write_project(target_dir: Path, project_files: dict[str, str]) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in project_files.items():
            (target_dir / fname).write_text(code, encoding="utf-8")

    # --- Helper: format all files for LLM context ---
    def _files_to_context(project_files: dict[str, str]) -> str:
        parts = []
        for fname, code in sorted(project_files.items()):
            parts.append(f"```filename:{fname}\n{code}\n```")
        return "\n\n".join(parts)

    if llm is None:
        logger.info("Stage 13: LLM unavailable, saving original experiment as final")
        final_dir = stage_dir / "experiment_final"
        _write_project(final_dir, best_files)
        # Backward compat
        if "main.py" in best_files:
            (stage_dir / "experiment_final.py").write_text(
                best_files["main.py"], encoding="utf-8"
            )
        log.update(
            {
                "converged": True,
                "stop_reason": "llm_unavailable",
                "best_metric": best_metric,
                "best_version": "experiment_final/",
                "iterations": [
                    {
                        "iteration": 0,
                        "version_dir": "experiment_final/",
                        "source": "fallback_original",
                        "metric": best_metric,
                    }
                ],
            }
        )
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
        artifacts = ("refinement_log.json", "experiment_final/")
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=artifacts,
            evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
        )

    _pm = prompts or PromptManager()
    timeout_refine_attempts = 0

    # R7-3: Read experiment plan to detect condition coverage gaps
    _exp_plan_text = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    _condition_coverage_hint = ""
    if _exp_plan_text and run_summaries:
        # Check if stdout contains condition labels
        _all_stdout = " ".join(run_summaries)
        _has_condition_labels = "condition=" in _all_stdout
        if not _has_condition_labels and _exp_plan_text.strip():
            _condition_coverage_hint = (
                "\nCONDITION COVERAGE GAP DETECTED:\n"
                "The experiment plan specifies multiple conditions/treatments, "
                "but the output contains NO condition labels (no 'condition=...' in stdout).\n"
                "You MUST:\n"
                "1. Run ALL conditions/treatments from the experiment plan independently\n"
                "2. Label each metric output: `condition=<name> {metric_key}: <value>`\n"
                "3. Print a SUMMARY line comparing all conditions after completion\n"
                "This is the MOST IMPORTANT improvement — a single unlabeled metric stream "
                "cannot support any comparative conclusions.\n\n"
            )
            logger.info(
                "Stage 13: condition coverage gap detected, injecting multi-condition hint"
            )

    # P1: Track metrics history for saturation detection
    _metrics_history: list[float | None] = [baseline_metric]

    for iteration in range(1, max_iterations + 1):
        # BUG-57: Check wall-clock time before starting a new iteration
        _elapsed = _time_bug57.monotonic() - _refine_start_time
        if _elapsed > _max_refine_wall_sec:
            logger.warning(
                "Stage 13: Wall-clock time cap reached (%.0fs > %ds). "
                "Stopping refinement after %d iterations.",
                _elapsed, _max_refine_wall_sec, iteration - 1,
            )
            log["stop_reason"] = "wall_clock_time_cap"
            break
        logger.info("Stage 13: refinement iteration %d/%d (%.0fs elapsed, cap %ds)",
                    iteration, max_iterations, _elapsed, _max_refine_wall_sec)

        # P1: Detect metric saturation and inject difficulty upgrade hint
        _saturation_hint = ""
        _valid_metrics = [m for m in _metrics_history if m is not None]
        if len(_valid_metrics) >= 2:
            _last_two = _valid_metrics[-2:]
            _saturated = False
            # Use relative change rate instead of hard-coded thresholds
            _change_rate = abs(_last_two[-1] - _last_two[-2]) / max(abs(_last_two[-2]), 1e-8)
            if metric_direction == "minimize":
                _saturated = all(m <= 0.001 for m in _last_two) or (
                    _change_rate < 0.001 and _last_two[-1] < 0.01
                )
            else:
                _saturated = all(m >= 0.999 for m in _last_two) or (
                    _change_rate < 0.001 and _last_two[-1] > 0.99
                )
            if _saturated:
                _saturation_hint = (
                    "\n\nWARNING — BENCHMARK SATURATION DETECTED:\n"
                    "All methods achieve near-perfect scores, making the task too easy "
                    "to discriminate between methods.\n"
                    "YOU MUST increase benchmark difficulty in this iteration:\n"
                    "1. Increase the number of actions/decisions from 8 to at least 20\n"
                    "2. Increase the horizon from 12-18 to at least 50-100 steps\n"
                    "3. Increase noise level to at least 0.3-0.5\n"
                    "4. Add partial observability (agent cannot see full state)\n"
                    "5. Add delayed rewards (reward only at episode end)\n"
                    "6. Ensure random search achieves < 50% success rate\n"
                    "Without this change, the experiment produces meaningless results.\n"
                )
                logger.warning("Stage 13: metric saturation detected, injecting difficulty upgrade hint")

        files_context = _files_to_context(best_files)
        # BUG-10 fix: anchor refinement to original experiment plan
        _exp_plan_anchor = ""
        if _exp_plan_text.strip():
            _exp_plan_anchor = (
                "Original experiment plan (exp_plan.yaml):\n"
                "```yaml\n" + _exp_plan_text[:4000] + "\n```\n"
                "You MUST preserve ALL condition names from this plan.\n\n"
            )
        ip = _pm.sub_prompt(
            "iterative_improve",
            metric_key=metric_key,
            metric_direction=metric_direction,
            files_context=files_context,
            run_summaries=chr(10).join(run_summaries[:20]),
            condition_coverage_hint=_condition_coverage_hint,
            topic=config.research.topic,
            exp_plan_anchor=_exp_plan_anchor,
        )

        # --- Timeout-aware prompt injection ---
        user_prompt = ip.user + _saturation_hint
        if prior_timed_out and baseline_metric is None:
            timeout_refine_attempts += 1
            timeout_hint = (
                f"\n\nCRITICAL: The experiment TIMED OUT after {prior_time_budget}s "
                f"with NO results. You MUST drastically reduce the experiment scale:\n"
                f"- Reduce total runs to ≤50\n"
                f"- Reduce steps per run to ≤2000\n"
                f"- Remove conditions that are not essential\n"
                f"- Add time.time() checks to stop gracefully before timeout\n"
                f"- Print intermediate metrics frequently so partial data is captured\n"
                f"- Time budget is {prior_time_budget}s — design for ≤{int(prior_time_budget * 0.7)}s\n"
            )
            user_prompt = user_prompt + timeout_hint
            logger.warning(
                "Stage 13: injecting timeout-aware prompt (attempt %d)",
                timeout_refine_attempts,
            )

        response = _chat_with_prompt(
            llm,
            ip.system,
            user_prompt,
            max_tokens=ip.max_tokens or 8192,
        )
        extracted_files = _extract_multi_file_blocks(response.content)
        # If LLM returns only single block, treat as main.py update
        if not extracted_files:
            single_code = _extract_code_block(response.content)
            if single_code.strip():
                extracted_files = {"main.py": single_code}
        # R8-2: Merge with best_files to preserve supporting modules
        # (e.g., graphs.py, game.py) that the LLM didn't rewrite
        candidate_files = dict(best_files)
        if extracted_files:
            candidate_files.update(extracted_files)
        # If LLM returned nothing at all, candidate_files == best_files (unchanged)

        # Validate main.py
        main_code = candidate_files.get("main.py", "")
        validation = validate_code(main_code)
        issue_text = ""
        repaired = False

        if not validation.ok:
            issue_text = format_issues_for_llm(validation)
            logger.info(
                "Stage 13 iteration %d validation failed: %s",
                iteration,
                validation.summary(),
            )
            irp = _pm.sub_prompt(
                "iterative_repair",
                issue_text=issue_text,
                all_files_ctx=_files_to_context(candidate_files),
            )
            repair_response = _chat_with_prompt(llm, irp.system, irp.user)
            candidate_files["main.py"] = _extract_code_block(repair_response.content)
            validation = validate_code(candidate_files["main.py"])
            repaired = True

        # Save version directory
        version_dir = stage_dir / f"experiment_v{iteration}"
        _write_project(version_dir, candidate_files)

        iter_record: dict[str, Any] = {
            "iteration": iteration,
            "version_dir": f"experiment_v{iteration}/",
            "files": list(candidate_files.keys()),
            "validation_ok": validation.ok,
            "validation_summary": validation.summary(),
            "repaired": repaired,
            "metric": None,
            "improved": False,
        }
        if issue_text:
            iter_record["validation_issues"] = issue_text

        metric_val = None  # R6-3: initialize before conditional block
        if validation.ok and config.experiment.mode in ("sandbox", "docker"):
            # P7: Ensure deps for refined code (subprocess sandbox only)
            if config.experiment.mode == "sandbox":
                _refine_code = "\n".join(candidate_files.values())
                _ensure_sandbox_deps(_refine_code, config.experiment.sandbox.python_path)

            sandbox = create_sandbox(
                config.experiment,
                stage_dir / f"refine_sandbox_v{iteration}",
            )
            rerun = sandbox.run_project(
                version_dir,
                timeout_sec=config.experiment.time_budget_sec,
            )
            metric_val = _find_metric(rerun.metrics, metric_key)
            # R19-1: Store stdout (capped) so PAIRED lines survive for Stage 14
            _stdout_cap = rerun.stdout[:50000] if rerun.stdout else ""
            iter_record["sandbox"] = {
                "returncode": rerun.returncode,
                "metrics": rerun.metrics,
                "elapsed_sec": rerun.elapsed_sec,
                "timed_out": rerun.timed_out,
                "stderr": rerun.stderr[:2000] if rerun.stderr else "",
                "stdout": _stdout_cap,
            }
            iter_record["metric"] = metric_val

            # --- Track timeout in refine sandbox ---
            if rerun.timed_out:
                prior_timed_out = True
                timeout_refine_attempts += 1
                logger.warning(
                    "Stage 13 iteration %d: sandbox timed out after %.1fs",
                    iteration,
                    rerun.elapsed_sec,
                )
                # If still no metrics after timeout, use partial stdout metrics
                if not rerun.metrics and rerun.stdout:
                    from researchclaw.experiment.sandbox import parse_metrics as _parse_sb_metrics
                    partial = _parse_sb_metrics(rerun.stdout)
                    if partial:
                        iter_record["sandbox"]["metrics"] = partial
                        metric_val = _find_metric(partial, metric_key)
                        iter_record["metric"] = metric_val
                        logger.info(
                            "Stage 13 iteration %d: recovered %d partial metrics from timeout stdout",
                            iteration,
                            len(partial),
                        )

            # --- Detect runtime issues (NaN/Inf, stderr warnings) ---
            runtime_issues = _detect_runtime_issues(rerun)
            if runtime_issues:
                iter_record["runtime_issues"] = runtime_issues
                logger.info(
                    "Stage 13 iteration %d: runtime issues detected: %s",
                    iteration,
                    runtime_issues[:200],
                )
                # Attempt LLM repair with runtime context
                rrp = _pm.sub_prompt(
                    "iterative_repair",
                    issue_text=runtime_issues,
                    all_files_ctx=_files_to_context(candidate_files),
                )
                repair_resp = _chat_with_prompt(llm, rrp.system, rrp.user)
                repaired_files = _extract_multi_file_blocks(repair_resp.content)
                if not repaired_files:
                    single = _extract_code_block(repair_resp.content)
                    if single.strip():
                        repaired_files = dict(candidate_files)
                        repaired_files["main.py"] = single
                if repaired_files:
                    candidate_files = repaired_files
                    _write_project(version_dir, candidate_files)
                    # Re-run after runtime fix
                    sandbox2 = create_sandbox(
                        config.experiment,
                        stage_dir / f"refine_sandbox_v{iteration}_fix",
                    )
                    rerun2 = sandbox2.run_project(
                        version_dir,
                        timeout_sec=config.experiment.time_budget_sec,
                    )
                    metric_val = _find_metric(rerun2.metrics, metric_key)
                    iter_record["sandbox_after_fix"] = {
                        "returncode": rerun2.returncode,
                        "metrics": rerun2.metrics,
                        "elapsed_sec": rerun2.elapsed_sec,
                        "timed_out": rerun2.timed_out,
                    }
                    iter_record["metric"] = metric_val
                    iter_record["runtime_repaired"] = True

            if metric_val is not None:
                consecutive_no_metrics = 0
                # R6-1: Only count toward no_improve_streak when we have real metrics
                if _is_better(metric_val, best_metric):
                    best_metric = metric_val
                    best_files = dict(candidate_files)
                    best_version = f"experiment_v{iteration}/"
                    iter_record["improved"] = True
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            else:
                consecutive_no_metrics += 1
        elif validation.ok and best_version == "experiment/":
            best_files = dict(candidate_files)
            best_version = f"experiment_v{iteration}/"

        # P1: Track metric for saturation detection
        _metrics_history.append(metric_val)

        log["iterations"].append(iter_record)

        if consecutive_no_metrics >= 3:
            log["stop_reason"] = "consecutive_no_metrics"
            logger.warning("Stage 13: Aborting after %d consecutive iterations without metrics", consecutive_no_metrics)
            break

        if no_improve_streak >= 2:
            log["converged"] = True
            log["stop_reason"] = "no_improvement_for_2_iterations"
            logger.info(
                "Stage 13 converged after %d iterations (no improvement streak=%d)",
                iteration,
                no_improve_streak,
            )
            break

    # Write final experiment directory
    final_dir = stage_dir / "experiment_final"
    _write_project(final_dir, best_files)
    # Backward compat: also write experiment_final.py (copy of main.py)
    if "main.py" in best_files:
        (stage_dir / "experiment_final.py").write_text(
            best_files["main.py"], encoding="utf-8"
        )

    log["best_metric"] = best_metric
    log["best_version"] = best_version
    log["final_version"] = "experiment_final/"
    (stage_dir / "refinement_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )

    artifacts = ["refinement_log.json", "experiment_final/"]
    artifacts.extend(
        entry["version_dir"]
        for entry in log["iterations"]
        if isinstance(entry, dict) and isinstance(entry.get("version_dir"), str)
    )
    return StageResult(
        stage=Stage.ITERATIVE_REFINE,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
    )


def _execute_result_analysis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    # --- Collect experiment data ---
    exp_data = _collect_experiment_results(run_dir)
    runs_dir = _read_prior_artifact(run_dir, "runs/") or ""
    context = ""
    if runs_dir:
        context = _collect_json_context(Path(runs_dir), max_files=30)

    # --- R13-1: Merge Stage 13 (ITERATIVE_REFINE) results if available ---
    # Stage 13 stores richer per-condition metrics in refinement_log.json
    # that _collect_experiment_results() misses (it only scans runs/ dirs).
    _refine_log_text = _read_prior_artifact(run_dir, "refinement_log.json")
    if _refine_log_text:
        try:
            _refine_data = json.loads(_refine_log_text)
            _best_iter = None
            _best_ver = _refine_data.get("best_version", "")
            for _it in _refine_data.get("iterations", []):
                _sbx = _it.get("sandbox", {})
                _it_metrics = _sbx.get("metrics", {})
                if _it.get("version_dir", "") == _best_ver and _it_metrics:
                    _best_iter = _it
                    break
            # If no version match, take the first iteration with metrics
            if _best_iter is None:
                for _it in _refine_data.get("iterations", []):
                    _sbx = _it.get("sandbox", {})
                    if _sbx.get("metrics"):
                        _best_iter = _it
                        break
            if _best_iter is not None:
                _sbx = _best_iter.get("sandbox", {})
                _refine_metrics = _sbx.get("metrics", {})
                if _refine_metrics and (
                    not exp_data["metrics_summary"]
                    or len(_refine_metrics) > len(exp_data["metrics_summary"])
                ):
                    # Refinement has richer data — rebuild metrics_summary from it
                    _new_summary: dict[str, dict[str, float | None]] = {}
                    for _mk, _mv in _refine_metrics.items():
                        try:
                            _fv = float(_mv)
                            _new_summary[_mk] = {
                                "min": round(_fv, 6),
                                "max": round(_fv, 6),
                                "mean": round(_fv, 6),
                                "count": 1,
                            }
                        except (ValueError, TypeError):
                            pass
                    if _new_summary:
                        exp_data["metrics_summary"] = _new_summary
                        # Also update best_run with refinement data
                        exp_data["best_run"] = {
                            "run_id": "iterative-refine-best",
                            "task_id": "sandbox-main",
                            "status": "completed",
                            "metrics": {
                                k: v for k, v in _refine_metrics.items()
                            },
                            "elapsed_sec": _sbx.get("elapsed_sec", 0),
                            "stdout": "",  # omit for brevity
                            "stderr": _sbx.get("stderr", ""),
                            "timed_out": _sbx.get("timed_out", False),
                        }
                        # Rebuild latex table
                        _ltx = [
                            r"\begin{table}[h]", r"\centering",
                            r"\caption{Experiment Results (Best Refinement Iteration)}",
                            r"\begin{tabular}{lrrrr}", r"\hline",
                            r"Metric & Min & Max & Mean & N \\", r"\hline",
                        ]
                        for _col in sorted(_new_summary.keys()):
                            _s = _new_summary[_col]
                            _ltx.append(
                                f"{_col} & {_s['min']:.4f} & {_s['max']:.4f} "
                                f"& {_s['mean']:.4f} & {_s['count']} \\\\"
                            )
                        _ltx.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
                        exp_data["latex_table"] = "\n".join(_ltx)
                        # Count unique conditions (keys without 'seed' and not ending in _mean/_std)
                        _conditions = {
                            k for k in _refine_metrics
                            if "seed" not in k and not k.endswith("_std")
                        }
                        exp_data["runs"] = [exp_data["best_run"]]
                        # Store condition count for accurate reporting
                        exp_data["best_run"]["condition_count"] = len(_conditions)
                        if not context:
                            context = json.dumps(
                                {"refinement_best_metrics": _refine_metrics},
                                indent=2, default=str,
                            )
                        logger.info(
                            "R13-1: Merged %d metrics from refinement_log (best_metric=%.4f)",
                            len(_refine_metrics),
                            _refine_data.get("best_metric", 0),
                        )
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("R13-1: Failed to parse refinement_log.json, using Stage 12 data")

    # --- R19-2: Extract PAIRED comparisons from refinement stdout ---
    from researchclaw.experiment.sandbox import extract_paired_comparisons as _extract_paired

    _all_paired: list[dict[str, object]] = []
    # First: from _collect_experiment_results (Stage 12 runs/)
    if exp_data.get("paired_comparisons"):
        _all_paired.extend(exp_data["paired_comparisons"])
    # Second: from refinement_log iterations (Stage 13)
    if _refine_log_text:
        try:
            _rl = json.loads(_refine_log_text)
            for _it in _rl.get("iterations", []):
                for _sbx_key in ("sandbox", "sandbox_after_fix"):
                    _sbx_stdout = (_it.get(_sbx_key) or {}).get("stdout", "")
                    if _sbx_stdout:
                        _all_paired.extend(_extract_paired(_sbx_stdout))
        except (json.JSONDecodeError, OSError):
            pass

    # --- R19-3: Build structured condition_summaries from metrics ---
    _condition_summaries: dict[str, dict[str, Any]] = {}
    _ms = exp_data.get("metrics_summary", {})
    _best_metrics = {}
    if exp_data.get("best_run") and isinstance(exp_data["best_run"], dict):
        _best_metrics = exp_data["best_run"].get("metrics", {})

    # Group metrics by condition prefix (e.g., "ppo/primary_metric" → condition "ppo")
    for _mk, _mv in _best_metrics.items():
        parts = _mk.split("/")
        if len(parts) >= 2:
            cond = parts[0]
            metric_name = parts[-1]
            if cond not in _condition_summaries:
                _condition_summaries[cond] = {"metrics": {}}
            try:
                _condition_summaries[cond]["metrics"][metric_name] = float(_mv)
            except (ValueError, TypeError):
                pass

    # BUG-09 fix: If no condition summaries were built (metrics don't use
    # condition/metric format), try to extract from metrics_summary or
    # structured_results so FigureAgent has data to work with.
    if not _condition_summaries and _ms:
        # Try to parse condition data from metrics_summary keys
        for _mk, _mv in _ms.items():
            parts = _mk.split("/")
            if len(parts) >= 2:
                cond = parts[0]
                metric_name = parts[-1]
                if cond not in _condition_summaries:
                    _condition_summaries[cond] = {"metrics": {}}
                try:
                    _val = float(_mv) if not isinstance(_mv, dict) else None
                    if _val is not None:
                        _condition_summaries[cond]["metrics"][metric_name] = _val
                except (ValueError, TypeError):
                    pass
    if not _condition_summaries:
        # Last resort: build from structured_results condition keys
        _sr = exp_data.get("structured_results", {})
        if isinstance(_sr, dict):
            for _sk, _sv in _sr.items():
                if isinstance(_sv, dict) and _sk not in ("metadata", "config"):
                    _condition_summaries[_sk] = {"metrics": {}}
                    for _smk, _smv in _sv.items():
                        try:
                            _condition_summaries[_sk]["metrics"][_smk] = float(_smv)
                        except (ValueError, TypeError):
                            pass

    # R33: Build per-seed data structure (needed for CIs and paired tests below)
    _seed_data: dict[str, dict[int, float]] = {}  # {condition: {seed: value}}
    for _mk, _mv in _best_metrics.items():
        parts = _mk.split("/")
        # Pattern: condition/regime/seed_id/primary_metric
        if len(parts) >= 4 and parts[-1] == config.experiment.metric_key:
            cond = parts[0]
            try:
                seed_id = int(parts[2])
                val = float(_mv)
                _seed_data.setdefault(cond, {})[seed_id] = val
            except (ValueError, TypeError):
                pass

    # Enrich condition summaries with seed counts, success rates, and CIs
    for _ck, _cv in _condition_summaries.items():
        # Look for success_rate in metrics
        sr_key = f"{_ck}/success_rate"
        if sr_key in _best_metrics:
            try:
                _cv["success_rate"] = float(_best_metrics[sr_key])
            except (ValueError, TypeError):
                pass
        # Count seed-level entries to estimate n_seeds
        _seed_count = 0
        for _mk in _best_metrics:
            if _mk.startswith(f"{_ck}/") and "seed" in _mk.lower():
                _seed_count += 1
        if _seed_count > 0:
            _cv["n_seed_metrics"] = _seed_count

        # R33: Compute mean ± std and bootstrap 95% CI from per-seed data
        if _ck in _seed_data and len(_seed_data[_ck]) >= 3:
            _vals = list(_seed_data[_ck].values())
            import statistics as _stats_mod
            _mean = _stats_mod.mean(_vals)
            _std = _stats_mod.stdev(_vals)
            _cv["metrics"][f"{config.experiment.metric_key}_mean"] = round(_mean, 6)
            _cv["metrics"][f"{config.experiment.metric_key}_std"] = round(_std, 6)
            _cv["n_seeds"] = len(_vals)
            # Bootstrap 95% CI
            import random as _rng
            _rng.seed(42)
            _boot_means = []
            for _ in range(1000):
                _sample = [_rng.choice(_vals) for _ in range(len(_vals))]
                _boot_means.append(_stats_mod.mean(_sample))
            _boot_means.sort()
            _ci_low = round(_boot_means[int(0.025 * len(_boot_means))], 6)
            _ci_high = round(_boot_means[int(0.975 * len(_boot_means))], 6)
            # IMP-16: Sanity check — CI must contain the mean
            if _ci_low > _mean or _ci_high < _mean:
                logger.warning(
                    "Bootstrap CI [%.4f, %.4f] does not contain mean %.4f "
                    "for condition %s — replacing CI with mean ± 1.96*SE",
                    _ci_low, _ci_high, _mean, _ck,
                )
                _se = _std / (len(_vals) ** 0.5)
                _ci_low = round(_mean - 1.96 * _se, 6)
                _ci_high = round(_mean + 1.96 * _se, 6)
            _cv["ci95_low"] = _ci_low
            _cv["ci95_high"] = _ci_high

    # Count totals
    _total_conditions = len(_condition_summaries) if _condition_summaries else None
    _total_metrics = len(_best_metrics) if _best_metrics else None

    # --- R33: Pipeline-level paired computation as fallback ---
    # If the experiment code's PAIRED lines are sparse or suspicious (e.g.,
    # all identical t-stats), compute fresh paired tests from per-seed data.
    # (_seed_data was built above before condition summary enrichment)
    if len(_seed_data) >= 2:
        # Find common seeds across conditions
        _all_seeds_sets = [set(v.keys()) for v in _seed_data.values()]
        _common_seeds = set.intersection(*_all_seeds_sets) if _all_seeds_sets else set()

        if len(_common_seeds) >= 3:
            _cond_names_sorted = sorted(_seed_data.keys())
            _pipeline_paired: list[dict[str, object]] = []
            # Compare each condition against the first baseline (alphabetically)
            _baseline_cond = _cond_names_sorted[0]
            for _other_cond in _cond_names_sorted[1:]:
                _diffs = []
                for _sid in sorted(_common_seeds):
                    _diffs.append(
                        _seed_data[_other_cond][_sid] - _seed_data[_baseline_cond][_sid]
                    )
                if _diffs:
                    import statistics
                    _n = len(_diffs)
                    _mean_d = statistics.mean(_diffs)
                    _std_d = statistics.stdev(_diffs) if _n > 1 else 0.0
                    _t = (_mean_d / (_std_d / (_n ** 0.5))) if _std_d > 0 else 0.0
                    _df = _n - 1
                    # Two-tailed p-value using t-distribution
                    import math
                    try:
                        from scipy.stats import t as _t_dist
                        _p = float(2 * _t_dist.sf(abs(_t), _df))
                    except ImportError:
                        _p = 2 * (1 - 0.5 * (1 + math.erf(abs(_t) / (2 ** 0.5))))
                        if _df < 30:
                            _p = min(1.0, _p * (1 + 2.5 / max(_df, 1)))
                    _pipeline_paired.append({
                        "method": _other_cond,
                        "baseline": _baseline_cond,
                        "mean_diff": round(_mean_d, 6),
                        "std_diff": round(_std_d, 6),
                        "t_stat": round(_t, 4),
                        "p_value": round(_p, 6),
                        "n_seeds": _n,
                        "source": "pipeline_computed",
                    })

            # Use pipeline-computed if experiment code's are suspicious
            _exp_t_stats = {round(p.get("t_stat", 0), 4) for p in _all_paired}
            _all_identical = len(_exp_t_stats) <= 1 and len(_all_paired) > 1
            if _pipeline_paired and (_all_identical or len(_all_paired) < len(_pipeline_paired)):
                logger.info(
                    "R33: Using %d pipeline-computed paired tests (experiment code had %d, identical=%s)",
                    len(_pipeline_paired), len(_all_paired), _all_identical,
                )
                _all_paired = _pipeline_paired

    # --- P8: Detect identical conditions (broken ablations) ---
    _ablation_warnings: list[str] = []
    if _condition_summaries and len(_condition_summaries) >= 2:
        _cond_names = sorted(_condition_summaries.keys())
        for _i in range(len(_cond_names)):
            for _j in range(_i + 1, len(_cond_names)):
                _c1, _c2 = _cond_names[_i], _cond_names[_j]
                _s1 = _condition_summaries[_c1]
                _s2 = _condition_summaries[_c2]
                # Compare mean values for all shared metrics
                _shared_keys = set(_s1.keys()) & set(_s2.keys())
                if not _shared_keys:
                    continue
                _all_equal = True
                for _sk in _shared_keys:
                    _v1 = _s1[_sk].get("mean") if isinstance(_s1[_sk], dict) else _s1[_sk]
                    _v2 = _s2[_sk].get("mean") if isinstance(_s2[_sk], dict) else _s2[_sk]
                    if _v1 != _v2:
                        _all_equal = False
                        break
                if _all_equal and _shared_keys:
                    _warn = (
                        f"ABLATION FAILURE: Conditions '{_c1}' and '{_c2}' produce "
                        f"identical outputs across all {len(_shared_keys)} metrics. "
                        f"The ablation is invalid — the differentiating parameter "
                        f"is likely not used in the code."
                    )
                    _ablation_warnings.append(_warn)
                    logger.warning("P8: %s", _warn)

    # --- Write structured experiment summary ---
    summary_payload = {
        "metrics_summary": exp_data["metrics_summary"],
        "total_runs": len(exp_data["runs"]),
        "best_run": exp_data["best_run"],
        "latex_table": exp_data["latex_table"],
        "generated": _utcnow_iso(),
    }
    # R13-1: Detect zero-variance across conditions (all conditions identical primary metric)
    if _condition_summaries and len(_condition_summaries) >= 2:
        _primary_vals = []
        for _cs in _condition_summaries.values():
            if isinstance(_cs, dict):
                # Try 'metrics' dict first (actual structure), then 'primary_metric' fallback
                _metrics = _cs.get("metrics", {})
                if isinstance(_metrics, dict) and _metrics:
                    _pv_candidate = next(iter(_metrics.values()), None)
                    if isinstance(_pv_candidate, dict):
                        _pv_candidate = _pv_candidate.get("mean")
                    if isinstance(_pv_candidate, (int, float)):
                        _primary_vals.append(_pv_candidate)
                        continue
                _pm = _cs.get("primary_metric", {})
                _pv = _pm.get("mean") if isinstance(_pm, dict) else _pm
                if isinstance(_pv, (int, float)):
                    _primary_vals.append(_pv)
        if len(_primary_vals) >= 2 and len(set(_primary_vals)) == 1:
            _zv_warn = (
                f"ZERO VARIANCE: All {len(_primary_vals)} conditions have "
                f"identical primary_metric ({_primary_vals[0]}). "
                f"Experiment condition wiring is likely broken."
            )
            _ablation_warnings.append(_zv_warn)
            logger.warning("R13-1: %s", _zv_warn)

    if _ablation_warnings:
        summary_payload["ablation_warnings"] = _ablation_warnings
    if _all_paired:
        summary_payload["paired_comparisons"] = _all_paired
    if _condition_summaries:
        summary_payload["condition_summaries"] = _condition_summaries
        summary_payload["condition_metrics"] = _condition_summaries  # alias for quality gate
        summary_payload["total_conditions"] = _total_conditions
    if _total_metrics:
        summary_payload["total_metric_keys"] = _total_metrics
    (stage_dir / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )
    if exp_data["latex_table"]:
        (stage_dir / "results_table.tex").write_text(
            exp_data["latex_table"], encoding="utf-8"
        )

    # --- Build data-augmented prompt ---
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    data_context = ""
    if exp_data["metrics_summary"]:
        lines = ["\n## Quantitative Results"]
        for mk, mv in exp_data["metrics_summary"].items():
            if isinstance(mv, dict):
                lines.append(
                    f"- {mk}: mean={mv.get('mean', '?')}, min={mv.get('min', '?')}, "
                    f"max={mv.get('max', '?')}, n={mv.get('count', '?')}"
                )
        data_context = "\n".join(lines)

    # Append structured results if available
    if exp_data.get("structured_results"):
        structured_text = json.dumps(
            exp_data["structured_results"], indent=2, default=str
        )
        # Truncate to avoid blowing up context
        if len(structured_text) > 6000:
            structured_text = structured_text[:6000] + "\n... (truncated)"
        data_context += (
            f"\n\n## Structured Experiment Results (from results.json)\n"
            f"```json\n{structured_text}\n```"
        )

    # P8: Inject ablation warnings into data context
    if _ablation_warnings:
        data_context += "\n\nCRITICAL ABLATION WARNINGS:\n"
        for _aw in _ablation_warnings:
            data_context += f"- {_aw}\n"
        data_context += (
            "\nYou MUST address these in your analysis. Identical conditions "
            "mean the ablation design is broken and the comparison is meaningless.\n"
        )

    if llm is not None:
        _pm = prompts or PromptManager()
        from researchclaw.prompts import DEBATE_ROLES_ANALYSIS  # noqa: PLC0415

        # --- Multi-perspective debate ---
        perspectives_dir = stage_dir / "perspectives"
        variables = {
            "preamble": preamble,
            "data_context": data_context,
            "context": context,
        }
        perspectives = _multi_perspective_generate(
            llm, DEBATE_ROLES_ANALYSIS, variables, perspectives_dir
        )
        # --- Synthesize into unified analysis ---
        analysis = _synthesize_perspectives(
            llm, perspectives, "analysis_synthesize", _pm
        )
    else:
        # Template with real data if available
        ms = exp_data["metrics_summary"]
        metrics_block = ""
        if ms:
            for mk, mv in ms.items():
                if isinstance(mv, dict):
                    metrics_block += (
                        f"- **{mk}**: mean={mv.get('mean')}, "
                        f"min={mv.get('min')}, max={mv.get('max')}, n={mv.get('count')}\n"
                    )
        else:
            metrics_block = f"- Primary metric key: `{config.experiment.metric_key}`\n- No quantitative data yet.\n"

        analysis = f"""# Result Analysis

## Metrics Summary
{metrics_block}
## Comparative Findings
- Proposed approach results from {len(exp_data["runs"])} run(s) collected.

## Statistical Checks
- Recommend confidence interval and seed-wise variance reporting.

## Limitations
- Limited runs and synthetic constraints.

## Conclusion
- Proceed to decision stage with moderate confidence.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "analysis.md").write_text(analysis, encoding="utf-8")

    artifacts = ["analysis.md", "experiment_summary.json"]
    if (stage_dir / "results_table.tex").exists():
        artifacts.append("results_table.tex")

    # IMP-6 + FA: Generate charts early (Stage 14) so paper draft can reference them
    # Try FigureAgent first (multi-agent intelligent charts), fall back to visualize.py
    _figure_plan_saved = False
    if config.experiment.figure_agent.enabled and llm is not None:
        try:
            from researchclaw.agents.figure_agent import FigureOrchestrator
            from researchclaw.agents.figure_agent.orchestrator import FigureAgentConfig as _FACfg

            _fa_cfg = _FACfg(
                enabled=True,
                min_figures=config.experiment.figure_agent.min_figures,
                max_figures=config.experiment.figure_agent.max_figures,
                max_iterations=config.experiment.figure_agent.max_iterations,
                render_timeout_sec=config.experiment.figure_agent.render_timeout_sec,
                use_docker=config.experiment.figure_agent.use_docker,
                docker_image=config.experiment.figure_agent.docker_image,
                output_format=config.experiment.figure_agent.output_format,
                gemini_api_key=config.experiment.figure_agent.gemini_api_key,
                gemini_model=config.experiment.figure_agent.gemini_model,
                nano_banana_enabled=config.experiment.figure_agent.nano_banana_enabled,
                strict_mode=config.experiment.figure_agent.strict_mode,
                dpi=config.experiment.figure_agent.dpi,
            )
            _fa = FigureOrchestrator(llm, _fa_cfg, stage_dir=stage_dir)

            # Build conditions list from condition_summaries
            _fa_conditions = list(_condition_summaries.keys()) if _condition_summaries else []

            # BUG-09 fix: pass best_run metrics as fallback data if
            # structured_results is empty, so Planner has some data to chart
            _fa_exp_results = exp_data.get("structured_results", {})
            if not _fa_exp_results and _best_metrics:
                _fa_exp_results = {"best_run_metrics": _best_metrics}

            # Read paper draft for Decision Agent analysis
            _paper_draft = (
                _read_prior_artifact(run_dir, "paper_draft.md")
                or _read_prior_artifact(run_dir, "outline.md")
                or ""
            )

            _fa_plan = _fa.orchestrate({
                "experiment_results": _fa_exp_results,
                "condition_summaries": _condition_summaries,
                "metrics_summary": exp_data.get("metrics_summary", {}),
                "metric_key": config.experiment.metric_key,
                "conditions": _fa_conditions,
                "topic": _read_prior_artifact(run_dir, "topic.md") or config.research.topic,
                "hypothesis": _read_prior_artifact(run_dir, "hypotheses.md") or "",
                "paper_draft": _paper_draft,
                "output_dir": str(stage_dir / "charts"),
            })

            if _fa_plan.figure_count > 0:
                # Save figure plan for Stage 17 to read
                (stage_dir / "figure_plan.json").write_text(
                    json.dumps(_fa_plan.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
                _figure_plan_saved = True
                for _cf_name in _fa_plan.get_chart_files():
                    artifacts.append(f"charts/{_cf_name}")
                logger.info(
                    "Stage 14: FigureAgent generated %d charts (%d passed review, %.1fs)",
                    _fa_plan.figure_count,
                    _fa_plan.passed_count,
                    _fa_plan.elapsed_sec,
                )
            else:
                logger.warning("Stage 14: FigureAgent produced no charts, falling back")
        except Exception as _fa_exc:
            logger.warning("Stage 14: FigureAgent failed (%s), falling back to visualize.py", _fa_exc)

    # Fallback: legacy visualize.py chart generation
    if not _figure_plan_saved:
        try:
            from researchclaw.experiment.visualize import (
                generate_all_charts as _gen_charts_early,
            )

            _charts_dir = stage_dir / "charts"
            _early_charts = _gen_charts_early(
                run_dir,
                _charts_dir,
                metric_key=config.experiment.metric_key,
            )
            if _early_charts:
                for _cp in _early_charts:
                    artifacts.append(f"charts/{_cp.name}")
                logger.info(
                    "Stage 14: Generated %d early charts (legacy) for paper embedding",
                    len(_early_charts),
                )
        except Exception as _chart_exc:
            logger.warning("Stage 14: Early chart generation failed: %s", _chart_exc)

    return StageResult(
        stage=Stage.RESULT_ANALYSIS,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-14/{a}" for a in artifacts),
    )


def _parse_decision(text: str) -> str:
    """Extract PROCEED/PIVOT/REFINE from decision text.

    Looks for the first standalone keyword on its own line after a
    ``## Decision`` heading.  Falls back to a keyword scan of the first
    few lines after the heading, but only matches the keyword itself
    (not mentions inside explanatory prose like "PIVOT is not warranted").
    Returns lowercase ``"proceed"`` / ``"pivot"`` / ``"refine"``.
    Defaults to ``"proceed"`` if nothing matches.
    """
    import re as _re

    text_upper = text.upper()
    # Look in the first occurrence after "## Decision" heading
    decision_section = ""
    for keyword in ("## DECISION", "## Decision", "## decision"):
        if keyword.upper() in text_upper:
            idx = text_upper.index(keyword.upper())
            decision_section = text[idx : idx + 200]
            break
    search_text = decision_section or text[:500]

    # First try: look for a line that is just the keyword (possibly with
    # whitespace / markdown bold / trailing punctuation).
    for line in search_text.splitlines():
        stripped = line.strip().strip("*").strip("#").strip()
        if stripped.upper() in ("PROCEED", "PIVOT", "REFINE"):
            return stripped.lower()

    # Fallback: regex for standalone word boundaries so that
    # "PIVOT is not warranted" does NOT match as a decision.
    for kw in ("PIVOT", "REFINE", "PROCEED"):
        # Only match if the keyword appears as the FIRST keyword-class token
        # on its own (not embedded in a sentence saying "not PIVOT").
        pattern = _re.compile(
            r"(?:^|##\s*Decision\s*\n\s*)" + kw, _re.IGNORECASE | _re.MULTILINE
        )
        if pattern.search(search_text):
            return kw.lower()

    # Last resort: simple containment (original behavior)
    search_upper = search_text.upper()
    if "REFINE" in search_upper:
        return "refine"
    if "PIVOT" in search_upper:
        return "pivot"
    return "proceed"


def _execute_research_decision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""

    # P6: Detect degenerate REFINE cycles — inject warning if metrics stagnate
    _degenerate_hint = ""
    _refine_log = _read_prior_artifact(run_dir, "refinement_log.json")
    if _refine_log:
        try:
            _rl = json.loads(_refine_log)
            _iters = _rl.get("iterations", [])
            _metrics = [it.get("metric") for it in _iters if isinstance(it, dict)]
            _valid = [m for m in _metrics if m is not None]
            _all_saturated = _valid and all(m <= 0.001 or m >= 0.999 for m in _valid)
            _all_identical = len(set(_valid)) <= 1 and len(_valid) >= 2
            if _all_saturated or _all_identical:
                _degenerate_hint = (
                    "\n\nSYSTEM WARNING — DEGENERATE REFINE CYCLE DETECTED:\n"
                    f"Metrics across {len(_valid)} iterations: {_valid}\n"
                    "All iterations produce identical/saturated results. Further REFINE "
                    "cycles CANNOT fix this — the underlying benchmark design is too "
                    "easy/hard. You SHOULD choose PROCEED with a quality caveat rather "
                    "than REFINE again.\n"
                )
                logger.warning("P6: Degenerate refine cycle detected, injecting PROCEED hint")
        except (json.JSONDecodeError, OSError):
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "research_decision")
        sp = _pm.for_stage("research_decision", evolution_overlay=_overlay, analysis=analysis)
        _user = sp.user + _degenerate_hint
        resp = llm.chat(
            [{"role": "user", "content": _user}],
            system=sp.system,
        )
        decision_md = resp.content
    else:
        decision_md = f"""# Research Decision

## Decision
PROCEED

## Justification
Current evidence suggests measurable progress with actionable limitations.

## Next Actions
- Build detailed paper outline
- Expand ablation and uncertainty analysis in writing

Generated: {_utcnow_iso()}
"""
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")

    # --- Extract structured decision ---
    decision = _parse_decision(decision_md)

    # T3.1: Validate decision quality — check for minimum experiment rigor
    _quality_warnings: list[str] = []
    _dec_lower = decision_md.lower()
    if "baseline" not in _dec_lower and "control" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention baselines")
    if "seed" not in _dec_lower and "replicat" not in _dec_lower and "run" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention multi-seed/replicate runs")
    if "metric" not in _dec_lower and "accuracy" not in _dec_lower and "loss" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention evaluation metrics")
    if _quality_warnings:
        logger.warning("T3.1: Decision quality warnings: %s", _quality_warnings)

    decision_payload = {
        "decision": decision,
        "raw_text_excerpt": decision_md[:500],
        "quality_warnings": _quality_warnings,
        "generated": _utcnow_iso(),
    }
    (stage_dir / "decision_structured.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8"
    )
    logger.info("Research decision: %s", decision)

    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md", "decision_structured.json"),
        evidence_refs=("stage-15/decision.md",),
        decision=decision,
    )


def _execute_paper_outline(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
    decision = _read_prior_artifact(run_dir, "decision.md") or ""
    preamble = _build_context_preamble(
        config,
        run_dir,
        include_analysis=True,
        include_decision=True,
        include_experiment_data=True,
    )

    # WS-5.2: Read iteration feedback if available (multi-round iteration)
    feedback = ""
    iter_ctx_path = run_dir / "iteration_context.json"
    if iter_ctx_path.exists():
        try:
            ctx = json.loads(iter_ctx_path.read_text(encoding="utf-8"))
            iteration = ctx.get("iteration", 1)
            prev_score = ctx.get("quality_score")
            reviews_excerpt = ctx.get("reviews_excerpt", "")
            if iteration > 1 and reviews_excerpt:
                feedback = (
                    f"\n\n## Iteration {iteration} Feedback\n"
                    f"Previous quality score: {prev_score}/10\n"
                    f"Reviewer feedback to address:\n{reviews_excerpt[:2000]}\n"
                    f"\nYou MUST address these reviewer concerns in this revision.\n"
                )
        except (json.JSONDecodeError, KeyError):
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        # IMP-20: Pass academic style guide block for outline stage
        try:
            _asg = _pm.block("academic_style_guide")
        except (KeyError, Exception):
            _asg = ""
        _overlay = _get_evolution_overlay(run_dir, "paper_outline")
        sp = _pm.for_stage(
            "paper_outline",
            evolution_overlay=_overlay,
            preamble=preamble,
            topic_constraint=_pm.block("topic_constraint", topic=config.research.topic),
            feedback=feedback,
            analysis=analysis,
            decision=decision,
            academic_style_guide=_asg,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        outline = resp.content
        # Reasoning models may consume all tokens on CoT — retry with more
        if not outline.strip() and sp.max_tokens:
            logger.warning("Empty outline from LLM — retrying with 2x tokens")
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens * 2,
            )
            outline = resp.content
        if not outline.strip():
            logger.warning("LLM returned empty outline — using default")
            outline = _default_paper_outline(config.research.topic)
    else:
        outline = _default_paper_outline(config.research.topic)
    (stage_dir / "outline.md").write_text(outline, encoding="utf-8")
    return StageResult(
        stage=Stage.PAPER_OUTLINE,
        status=StageStatus.DONE,
        artifacts=("outline.md",),
        evidence_refs=("stage-16/outline.md",),
    )


def _collect_raw_experiment_metrics(run_dir: Path) -> tuple[str, bool]:
    """Collect raw experiment metric lines from stdout for paper writing.

    Returns a tuple of (formatted block, has_parsed_metrics).
    ``has_parsed_metrics`` is True when at least one run had a non-empty
    ``metrics`` dict in its JSON payload — a reliable signal of real data.
    """
    metric_lines: list[str] = []
    run_count = 0
    has_parsed_metrics = False

    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue

            # R10: Skip simulated data — only collect real experiment results
            if payload.get("status") == "simulated":
                continue

            run_count += 1

            # Extract from parsed metrics (check both 'metrics' and 'key_metrics')
            metrics = payload.get("metrics", {}) or payload.get("key_metrics", {})
            if isinstance(metrics, dict) and metrics:
                has_parsed_metrics = True
                for k, v in metrics.items():
                    metric_lines.append(f"  {k}: {v}")

            # Also extract from stdout for full detail
            # BUG-23: Filter out infrastructure lines that are NOT experiment results
            _INFRA_KEYS = {
                "SEED_COUNT", "TIME_ESTIMATE", "TRAINING_STEPS",
                "REGISTERED_CONDITIONS", "METRIC_DEF", "GPU_MEMORY",
                "BATCH_SIZE", "NUM_WORKERS", "TOTAL_PARAMS",
                "time_budget_sec", "max_epochs", "num_seeds",
            }
            stdout = payload.get("stdout", "")
            if stdout:
                for line in stdout.splitlines():
                    line = line.strip()
                    if ":" in line:
                        parts = line.rsplit(":", 1)
                        try:
                            float(parts[1].strip())
                            key_part = parts[0].strip().split("/")[-1]  # last segment
                            if key_part in _INFRA_KEYS:
                                continue  # skip infrastructure lines
                            metric_lines.append(f"  {line}")
                        except (ValueError, TypeError, IndexError):
                            pass

    # R19-4 + R23-1: Collect metrics from refinement_log.json (Stage 13).
    # If refinement has richer data than Stage 12 runs/, REPLACE Stage 12 data
    # to avoid confusing the paper writer with conflicting sources.
    _refine_lines: list[str] = []
    _refine_run_count = 0
    # Scan ALL refinement logs across versions, pick the richest
    _best_refine_metrics: dict[str, Any] = {}
    _best_refine_stdout = ""
    for _rl_path in sorted(run_dir.glob("stage-13*/refinement_log.json")):
        try:
            _rlog = json.loads(_rl_path.read_text(encoding="utf-8"))
            _best_ver = _rlog.get("best_version", "")
            for _it in _rlog.get("iterations", []):
                for _sbx_key in ("sandbox", "sandbox_after_fix"):
                    _sbx = _it.get(_sbx_key, {})
                    if not isinstance(_sbx, dict):
                        continue
                    _sbx_metrics = _sbx.get("metrics", {})
                    if isinstance(_sbx_metrics, dict) and len(_sbx_metrics) > len(_best_refine_metrics):
                        _best_refine_metrics = _sbx_metrics
                        _best_refine_stdout = _sbx.get("stdout", "")
        except (json.JSONDecodeError, OSError):
            pass

    if _best_refine_metrics and len(_best_refine_metrics) > len(metric_lines) // 2:
        # Refinement has richer data — REPLACE Stage 12 data to avoid conflicts
        metric_lines = []
        run_count = 1
        for k, v in _best_refine_metrics.items():
            metric_lines.append(f"  {k}: {v}")
        # Also extract PAIRED and metric lines from stdout
        if _best_refine_stdout:
            for _line in _best_refine_stdout.splitlines():
                _line = _line.strip()
                if _line.startswith("PAIRED:"):
                    metric_lines.append(f"  {_line}")
                elif ":" in _line:
                    parts = _line.rsplit(":", 1)
                    try:
                        float(parts[1].strip())
                        metric_lines.append(f"  {_line}")
                    except (ValueError, TypeError, IndexError):
                        pass
    elif _best_refine_metrics:
        # Refinement has some data but not richer — append to existing
        run_count += 1
        for k, v in _best_refine_metrics.items():
            metric_lines.append(f"  {k}: {v}")
        if _best_refine_stdout:
            for _line in _best_refine_stdout.splitlines():
                _line = _line.strip()
                if _line.startswith("PAIRED:"):
                    metric_lines.append(f"  {_line}")

    if not metric_lines:
        return "", has_parsed_metrics

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for line in metric_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    # BUG-29: Reformat raw metric lines into human-readable condition summaries
    # to prevent LLM from pasting raw path-style lines into the paper
    _grouped: dict[str, list[str]] = {}
    _ungrouped: list[str] = []
    for line in unique[:200]:
        stripped = line.strip()
        # Match pattern: condition/env/step/metric: value
        parts = stripped.split("/")
        if len(parts) >= 3 and ":" in parts[-1]:
            cond = parts[0]
            detail = "/".join(parts[1:])
            _grouped.setdefault(cond, []).append(f"  - {detail}")
        else:
            _ungrouped.append(stripped)

    formatted_lines: list[str] = []
    if _grouped:
        for cond, details in sorted(_grouped.items()):
            formatted_lines.append(f"## Condition: {cond}")
            formatted_lines.extend(details[:30])
    if _ungrouped:
        formatted_lines.extend(_ungrouped)

    return (
        f"\n\nACTUAL EXPERIMENT DATA (from {run_count} run(s) — use ONLY these numbers):\n"
        "```\n"
        + "\n".join(formatted_lines[:200])
        + "\n```\n"
        "CRITICAL: Every number in the Results table MUST come from the data above. "
        "Do NOT round excessively, do NOT invent numbers, do NOT change values. "
        f"The experiment ran {run_count} time(s) — state this accurately in the methodology.\n"
        "NEVER paste raw metric paths (like 'condition/env/step/metric: value') "
        "into the paper. Always convert to formatted LaTeX tables or inline prose.\n"
    ), has_parsed_metrics


def _write_paper_sections(
    *,
    llm: LLMClient,
    pm: PromptManager,
    run_dir: Path | None = None,
    preamble: str,
    topic_constraint: str,
    exp_metrics_instruction: str,
    citation_instruction: str,
    outline: str,
    model_name: str = "",
) -> str:
    """Write a conference-grade paper in 3 sequential LLM calls.

    Call 1: Title + Abstract + Introduction + Related Work
    Call 2: Method + Experiments (with full experiment data)
    Call 3: Results + Discussion + Limitations + Conclusion

    Each call receives prior sections for coherence.
    """
    # Render writing_structure block for injection
    try:
        _writing_structure = pm.block("writing_structure")
    except (KeyError, Exception):  # noqa: BLE001
        _writing_structure = ""

    _overlay = _get_evolution_overlay(run_dir, "paper_draft")
    system = pm.for_stage(
        "paper_draft",
        evolution_overlay=_overlay,
        preamble=preamble,
        topic_constraint=topic_constraint,
        exp_metrics_instruction=exp_metrics_instruction,
        citation_instruction=citation_instruction,
        writing_structure=_writing_structure,
        outline=outline,
    ).system

    sections: list[str] = []

    # --- R4-3: Title guidelines and abstract structure ---
    try:
        title_guidelines = pm.block("title_guidelines")
    except (KeyError, Exception):  # noqa: BLE001
        title_guidelines = ""
    try:
        abstract_structure = pm.block("abstract_structure")
    except (KeyError, Exception):  # noqa: BLE001
        abstract_structure = ""

    # IMP-20/25/31/24: Academic style, narrative, anti-hedging, anti-repetition
    try:
        academic_style_guide = pm.block("academic_style_guide")
    except (KeyError, Exception):  # noqa: BLE001
        academic_style_guide = ""
    try:
        narrative_writing_rules = pm.block("narrative_writing_rules")
    except (KeyError, Exception):  # noqa: BLE001
        narrative_writing_rules = ""
    try:
        anti_hedging_rules = pm.block("anti_hedging_rules")
    except (KeyError, Exception):  # noqa: BLE001
        anti_hedging_rules = ""
    try:
        anti_repetition_rules = pm.block("anti_repetition_rules")
    except (KeyError, Exception):  # noqa: BLE001
        anti_repetition_rules = ""

    # --- Call 1: Title + Abstract + Introduction + Related Work ---
    call1_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{citation_instruction}\n\n"
        f"{title_guidelines}\n\n"
        f"{academic_style_guide}\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n"
        f"{anti_repetition_rules}\n\n"
        "Write the following sections of a NeurIPS/ICML-quality paper in markdown. "
        "Follow the LENGTH REQUIREMENTS strictly:\n\n"
        "1. **Title** (HARD RULE: MUST be 14 words or fewer. Create a catchy method name "
        "first, then build the title: 'MethodName: Subtitle'. If your title exceeds 14 words, "
        "it will be automatically rejected. NEVER use 'Untitled Paper'.)\n"
        f"2. **Abstract** (150-220 words — HARD LIMIT. Do NOT exceed 220 words. "
        f"Do NOT include raw metric paths or 16-digit decimals.){abstract_structure}\n"
        "3. **Introduction** (800-1000 words): real-world motivation, problem statement, "
        "research gap analysis with citations, method overview, 3-4 contributions as bullet points, "
        "paper organization paragraph. MUST cite 8-12 references.\n"
        "4. **Related Work** (600-800 words): organized into 3-4 thematic subsections, each discussing "
        "4-5 papers with proper citations. Compare approaches, identify limitations, position this work.\n\n"
        f"Outline:\n{outline}\n\n"
        "Output markdown with ## headers. Do NOT include a References section.\n"
        "IMPORTANT: Start DIRECTLY with '## Title'. Do NOT include any preamble, "
        "data verification, condition listing, or metric enumeration before the title. "
        "The paper should read like a published manuscript, not a data report."
    )
    # R14-1: Higher token limit for reasoning models
    _paper_max_tokens = 12000
    if any(model_name.startswith(p) for p in ("gpt-5", "o3", "o4")):
        _paper_max_tokens = 24000

    # T3.5: Retry once on failure, use placeholder if still fails
    try:
        resp1 = _chat_with_prompt(llm, system, call1_user, max_tokens=_paper_max_tokens, retries=1)
        part1 = resp1.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 1 LLM call failed after retry — using placeholder")
        part1 = (
            "## Title\n[PLACEHOLDER — LLM call failed]\n\n"
            "## Abstract\n[This section could not be generated due to an LLM error. "
            "Please regenerate this stage.]\n\n"
            "## Introduction\n[PLACEHOLDER]\n\n"
            "## Related Work\n[PLACEHOLDER]"
        )
    sections.append(part1)
    logger.info("Stage 17: Part 1 (Title+Abstract+Intro+Related Work) — %d chars", len(part1))

    # --- Call 2: Method + Experiments ---
    call2_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{exp_metrics_instruction}\n\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n\n"
        # IMP-21: Citation instruction for Method + Experiments
        "CITATION REQUIREMENT: The Method section MUST cite at least 3-5 related "
        "technical papers (foundations your method builds on). The Experiments section "
        "MUST cite baseline method papers. Use [cite_key] syntax.\n"
        f"{citation_instruction}\n\n"
        "You are continuing a paper. The sections written so far are:\n\n"
        f"---\n{part1}\n---\n\n"
        "Now write the next sections, maintaining consistency with the above:\n\n"
        "5. **Method** (1000-1500 words): formal problem definition with mathematical notation "
        "($x$, $\\theta$, etc.), detailed algorithm description with equations, step-by-step procedure, "
        "complexity analysis, design rationale for key choices. Include algorithm pseudocode if applicable. "
        "Write as FLOWING PROSE — do NOT use bullet-point lists for method components.\n"
        "6. **Experiments** (800-1200 words): detailed experimental setup, datasets with statistics "
        "(size, splits, features), all baselines and their implementations, hyperparameter settings "
        "in a markdown table, evaluation metrics with mathematical definitions, hardware and runtime info.\n"
        "METHOD NAMES IN TABLES: Use SHORT abbreviations (4-8 chars) for method names "
        "in tables. Define abbreviation mappings in a footnote. "
        "NEVER put method names longer than 20 characters in table cells.\n\n"
        f"Outline:\n{outline}\n\n"
        "Output markdown with ## headers. Continue from where Part 1 ended."
    )
    try:
        resp2 = _chat_with_prompt(llm, system, call2_user, max_tokens=_paper_max_tokens, retries=1)
        part2 = resp2.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 2 LLM call failed after retry — using placeholder")
        part2 = (
            "## Method\n[PLACEHOLDER — LLM call failed. Please regenerate this stage.]\n\n"
            "## Experiments\n[PLACEHOLDER]"
        )
    sections.append(part2)
    logger.info("Stage 17: Part 2 (Method+Experiments) — %d chars", len(part2))

    # --- Call 3: Results + Discussion + Limitations + Conclusion ---
    call3_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{exp_metrics_instruction}\n\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n"
        f"{anti_repetition_rules}\n\n"
        # IMP-21: Citation instruction for Results + Discussion + Conclusion
        "CITATION REQUIREMENT: The Discussion section MUST cite at least 3-5 papers "
        "when comparing findings with prior work. The Conclusion may cite 1-2 "
        "foundational references.\n"
        f"{citation_instruction}\n\n"
        "You are completing a paper. The sections written so far are:\n\n"
        f"---\n{part1}\n\n{part2}\n---\n\n"
        "Now write the final sections, maintaining consistency:\n\n"
        "7. **Results** (600-800 words):\n"
        "   - START with an AGGREGATED results table (Table 1): rows = methods, columns = metrics.\n"
        "     Each cell = mean ± std across seeds. Bold the best value per column.\n"
        "     EVERY table MUST have a descriptive caption that allows understanding without "
        "     reading the main text. NEVER use just 'Table 1' as a caption.\n"
        "   - Follow with a PER-REGIME table (Table 2) breaking down by easy/hard regimes.\n"
        "   - Include a STATISTICAL COMPARISON table (Table 3): paired t-tests between key methods.\n"
        "   - NEVER dump raw per-seed numbers in the main text. Aggregate first, then discuss.\n"
        "   - MUST include at least 2 figures using markdown image syntax: ![Caption](charts/filename.png)\n"
        "     One figure MUST be a performance comparison chart. Figures MUST be referenced "
        "     in text: 'As shown in Figure 1, ...'\n"
        "8. **Discussion** (400-600 words): interpretation of key findings, unexpected results, "
        "comparison with prior work (CITE 3-5 papers here!), practical implications.\n"
        "9. **Limitations** (200-300 words): honest assessment of scope, dataset, methodology. "
        "ALL caveats consolidated HERE — nowhere else in the paper.\n"
        "10. **Conclusion** (100-200 words MAXIMUM — this is a HARD LIMIT): "
        "Summarize contributions in 2-3 sentences. State main finding in 1 sentence. "
        "Suggest 2-3 concrete future directions in 1-2 sentences. "
        "Do NOT repeat any specific numbers from Results. Do NOT restate the abstract. "
        "A good conclusion is SHORT and forward-looking.\n\n"
        "CRITICAL FORMATTING RULES FOR ALL SECTIONS:\n"
        "- Write as FLOWING PROSE paragraphs, NOT bullet-point lists\n"
        "- NEVER dump raw metric paths like 'config/method_name/seed_3/primary_metric'\n"
        "- All numbers must be rounded to 4 decimal places maximum\n"
        "- Every table MUST have a descriptive caption (not just 'Table 1')\n"
        "- Use \\begin{algorithm} or pseudocode notation, NOT \\begin{verbatim}\n\n"
        "Output markdown with ## headers. Do NOT include a References section."
    )
    try:
        resp3 = _chat_with_prompt(llm, system, call3_user, max_tokens=_paper_max_tokens, retries=1)
        part3 = resp3.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 3 LLM call failed after retry — using placeholder")
        part3 = (
            "## Results\n[PLACEHOLDER — LLM call failed. Please regenerate this stage.]\n\n"
            "## Discussion\n[PLACEHOLDER]\n\n"
            "## Limitations\n[PLACEHOLDER]\n\n"
            "## Conclusion\n[PLACEHOLDER]"
        )
    sections.append(part3)
    logger.info("Stage 17: Part 3 (Results+Discussion+Limitations+Conclusion) — %d chars", len(part3))

    # --- Truncation recovery: ensure Limitations and Conclusion were written ---
    # ACP ignores max_tokens; Claude can stop mid-section when context fills up.
    # Detect missing required sections and request a targeted completion call.
    _required_sections = ("## Limitations", "## Conclusion")
    _missing = [s for s in _required_sections if s.lower() not in part3.lower()]
    if _missing:
        logger.warning(
            "Stage 17: Part 3 truncated — missing sections: %s. Requesting completion.",
            ", ".join(_missing),
        )
        _partial_end = part3[-3000:] if len(part3) > 3000 else part3
        _completion_user = (
            f"{preamble}\n\n"
            "You are completing a scientific paper. The previous LLM call was truncated "
            "before finishing all required sections. Here is where the paper ends:\n\n"
            f"---\n{_partial_end}\n---\n\n"
            f"The paper is MISSING these sections: {', '.join(_missing)}\n\n"
            "Write ONLY the missing sections now, picking up exactly where the text above ends. "
            "Do NOT repeat any text that already appears above. "
            "Required sections to write:\n"
        )
        if "## Limitations" in _missing:
            _completion_user += (
                "- **Limitations** (200-300 words): honest assessment of scope, "
                "dataset, methodology. ALL caveats consolidated here.\n"
            )
        if "## Conclusion" in _missing:
            _completion_user += (
                "- **Conclusion** (100-200 words MAXIMUM — HARD LIMIT): "
                "Summarize contributions in 2-3 sentences. State main finding. "
                "Suggest 2-3 concrete future directions. Do NOT repeat numbers from Results.\n"
            )
        _completion_user += "\nOutput markdown with ## headers only for the missing sections."
        try:
            _resp_completion = _chat_with_prompt(
                llm, system, _completion_user, max_tokens=_paper_max_tokens, retries=1
            )
            _completion_text = _resp_completion.content.strip()
            if _completion_text:
                part3 = part3 + "\n\n" + _completion_text
                # Replace the last entry we appended
                sections[-1] = part3
                logger.info(
                    "Stage 17: Completion call added %d chars for missing sections",
                    len(_completion_text),
                )
        except Exception:  # noqa: BLE001
            logger.error("Stage 17: Completion call failed — truncated paper will proceed")

    # Combine all sections
    draft = "\n\n".join(sections)

    # R32: Strip data verification preamble that LLMs sometimes emit before
    # the actual paper.  The preamble typically starts with "## Tested Conditions"
    # or similar headings and ends before "## Title".
    import re as _re_strip
    _title_match = _re_strip.search(r"^## Title\b", draft, _re_strip.MULTILINE)
    if _title_match and _title_match.start() > 200:
        _stripped = draft[_title_match.start():]
        logger.info(
            "R32: Stripped %d-char preamble before '## Title'",
            _title_match.start(),
        )
        draft = _stripped

    total_words = len(draft.split())
    logger.info("Stage 17: Full draft — %d chars, ~%d words", len(draft), total_words)

    return draft


# ---------------------------------------------------------------------------
# Draft quality validation (section balance + bullet-point density)
# ---------------------------------------------------------------------------

# Sections where bullets/numbered lists are acceptable.
_BULLET_LENIENT_SECTIONS = frozenset({
    "introduction", "limitations", "limitation",
    "limitations and future work", "abstract",
})

# Main body sections used for balance ratio check.
_BALANCE_SECTIONS = frozenset({
    "introduction", "related work", "method", "experiments", "results",
    "discussion",
})


def _validate_draft_quality(
    draft: str,
    stage_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a paper draft for section balance and prose quality.

    Checks:
    1. Per-section word count vs ``SECTION_WORD_TARGETS``.
    2. Bullet-point / numbered-list density per section.
    3. Largest-to-smallest main-section word-count ratio.

    Returns a dict with ``section_analysis``, ``overall_warnings``, and
    ``revision_directives``.  Optionally writes ``draft_quality.json`` to
    *stage_dir*.
    """
    from researchclaw.prompts import SECTION_WORD_TARGETS, _SECTION_TARGET_ALIASES

    _heading_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(_heading_re.finditer(draft))

    sections_data: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(draft)
        body = draft[start:end].strip()
        sections_data.append({
            "heading": heading,
            "heading_lower": heading.strip().lower(),
            "level": level,
            "body": body,
        })

    section_analysis: list[dict[str, Any]] = []
    overall_warnings: list[str] = []
    revision_directives: list[str] = []
    main_section_words: dict[str, int] = {}

    _bullet_re = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
    _numbered_re = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)

    # BUG-24: Accumulate subsection (H3+) word counts into parent H2 sections
    _subsection_words: dict[str, int] = {}
    _current_parent = ""
    for sec in sections_data:
        if sec["level"] <= 2:
            _current_parent = sec["heading_lower"]
            _subsection_words.setdefault(_current_parent, 0)
        else:
            # Add subsection words to parent
            _subsection_words[_current_parent] = (
                _subsection_words.get(_current_parent, 0) + len(sec["body"].split())
            )

    for sec in sections_data:
        if sec["level"] > 2:
            continue
        heading_lower: str = sec["heading_lower"]
        body: str = sec["body"]
        # BUG-24: Include subsection words in the parent's word count
        word_count = len(body.split()) + _subsection_words.get(heading_lower, 0)
        canon = heading_lower
        if canon not in SECTION_WORD_TARGETS:
            canon = _SECTION_TARGET_ALIASES.get(heading_lower, "")
        entry: dict[str, Any] = {
            "heading": sec["heading"],
            "word_count": word_count,
            "canonical": canon,
        }
        if canon and canon in SECTION_WORD_TARGETS:
            lo, hi = SECTION_WORD_TARGETS[canon]
            entry["target"] = [lo, hi]
            if word_count < int(lo * 0.7):
                overall_warnings.append(
                    f"{sec['heading']} is severely under target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"EXPAND {sec['heading']} from {word_count} to {lo}+ words. "
                    f"Add substantive content \u2014 do NOT pad with filler."
                )
                entry["status"] = "severely_short"
            elif word_count < lo:
                overall_warnings.append(
                    f"{sec['heading']} is under target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"Expand {sec['heading']} from {word_count} to {lo}+ words."
                )
                entry["status"] = "short"
            elif word_count > int(hi * 1.3):
                overall_warnings.append(
                    f"{sec['heading']} exceeds target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"Compress {sec['heading']} from {word_count} to {hi} words or fewer."
                )
                entry["status"] = "long"
            else:
                entry["status"] = "ok"
        if body:
            total_lines = len([ln for ln in body.splitlines() if ln.strip()])
            bullet_lines = len(_bullet_re.findall(body)) + len(_numbered_re.findall(body))
            density = bullet_lines / total_lines if total_lines > 0 else 0.0
            entry["bullet_density"] = round(density, 2)
            threshold = 0.50 if heading_lower in _BULLET_LENIENT_SECTIONS else 0.25
            if density > threshold and total_lines >= 4:
                overall_warnings.append(
                    f"{sec['heading']} has {bullet_lines}/{total_lines} "
                    f"bullet/numbered lines ({density:.0%} density, "
                    f"threshold {threshold:.0%})"
                )
                revision_directives.append(
                    f"REWRITE {sec['heading']} as flowing academic prose. "
                    f"Convert bullet points to narrative paragraphs."
                )
                entry["bullet_status"] = "high"
            else:
                entry["bullet_status"] = "ok"
        canon_balance = canon or heading_lower
        if canon_balance in _BALANCE_SECTIONS:
            main_section_words[canon_balance] = word_count
        section_analysis.append(entry)

    if len(main_section_words) >= 2:
        wc_values = list(main_section_words.values())
        max_wc = max(wc_values)
        min_wc = min(wc_values)
        if min_wc > 0 and max_wc / min_wc > 3.0:
            largest = max(main_section_words, key=main_section_words.get)  # type: ignore[arg-type]
            smallest = min(main_section_words, key=main_section_words.get)  # type: ignore[arg-type]
            overall_warnings.append(
                f"Section imbalance: {largest} ({max_wc} words) vs "
                f"{smallest} ({min_wc} words) \u2014 ratio {max_wc / min_wc:.1f}x"
            )
            revision_directives.append(
                f"Rebalance sections: expand {smallest} and/or compress {largest} "
                f"to achieve more even section lengths."
            )

    # --- C-4/C-5: Citation count and recency checks ---
    _cite_pattern = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9]*)\]")
    cited_keys = set(_cite_pattern.findall(draft))
    if cited_keys:
        n_citations = len(cited_keys)
        if n_citations < 15:
            overall_warnings.append(
                f"Only {n_citations} unique citations found (target: >=15 for a full paper)"
            )
            revision_directives.append(
                f"Add more references — a top-venue paper typically cites 25-40 works. "
                f"Currently only {n_citations} unique citations."
            )
        # Check recency: count citations with year >= current_year - 2
        _year_pat = re.compile(r"(\d{4})")
        import datetime as _dt_cit
        _cur_year = _dt_cit.datetime.now().year
        recent_count = sum(
            1 for k in cited_keys
            for m in [_year_pat.search(k)]
            if m and int(m.group(1)) >= _cur_year - 2
        )
        recency_ratio = recent_count / n_citations if n_citations > 0 else 0.0
        if recency_ratio < 0.3 and n_citations >= 10:
            overall_warnings.append(
                f"Citation recency low: only {recent_count}/{n_citations} "
                f"({recency_ratio:.0%}) from last 3 years (target: >=30%%)"
            )

    # --- Abstract and Conclusion length enforcement ---
    for sec in sections_data:
        hl = sec["heading_lower"]
        body_text: str = sec["body"]
        wc = len(body_text.split())
        if hl == "abstract" and wc > 250:
            overall_warnings.append(
                f"Abstract is too long: {wc} words (target: 150-220 words)"
            )
            revision_directives.append(
                f"COMPRESS the Abstract from {wc} to 150-220 words. "
                f"Remove raw metric values, redundant context, and self-references."
            )
        if hl in ("conclusion", "conclusions", "conclusion and future work"):
            if wc > 300:
                overall_warnings.append(
                    f"Conclusion is too long: {wc} words (target: 100-200 words)"
                )
                revision_directives.append(
                    f"COMPRESS the Conclusion from {wc} to 100-200 words. "
                    f"Do NOT repeat specific metric values from Results. "
                    f"Summarize findings in 2-3 sentences, then 2-3 future directions."
                )

    # --- Raw metric path detection (log dumps in prose) ---
    _raw_path_re = re.compile(
        r"\\texttt\{[a-zA-Z0-9_/.-]+(?:/[a-zA-Z0-9_/.-]+){2,}",
    )
    raw_path_count = len(_raw_path_re.findall(draft))
    if raw_path_count > 3:
        overall_warnings.append(
            f"Raw metric paths in prose: {raw_path_count} instances of "
            f"\\texttt{{config/path/metric}} style dumps"
        )
        revision_directives.append(
            "REMOVE raw experiment log paths from prose. Replace "
            "\\texttt{config/metric/path} with human-readable metric names "
            "and summarize values in tables, not inline text."
        )

    # --- Writing quality lint ---
    _weasel_words = re.compile(
        r"\b(various|many|several|quite|fairly|really|very|rather|"
        r"somewhat|relatively|arguably|interestingly|importantly|"
        r"it is well known that|it is obvious that|clearly)\b",
        re.IGNORECASE,
    )
    _duplicate_words = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    weasel_count = len(_weasel_words.findall(draft))
    dup_matches = _duplicate_words.findall(draft)
    dup_count = len([d for d in dup_matches if d.lower() not in ("that", "had")])
    if weasel_count > 20:
        overall_warnings.append(
            f"High weasel-word count: {weasel_count} instances "
            f"(consider replacing vague words with precise language)"
        )
        revision_directives.append(
            "Replace vague hedging words (various, several, quite, fairly, "
            "rather, somewhat) with precise quantities or remove them."
        )
    if dup_count > 0:
        overall_warnings.append(
            f"Duplicate adjacent words found: {dup_count} instance(s) "
            f"(e.g., 'the the', 'is is')"
        )
        revision_directives.append(
            "Fix duplicate adjacent words (likely typos)."
        )

    # --- AI-slop / boilerplate detection ---
    _BOILERPLATE_PHRASES = [
        "delves into", "delve into", "it is worth noting",
        "it should be noted", "it is important to note",
        "leverage the power of", "leverages the power of",
        "in this paper, we propose", "in this work, we propose",
        "to the best of our knowledge",
        "in the realm of", "in the landscape of",
        "plays a crucial role", "plays a pivotal role",
        "groundbreaking", "cutting-edge", "state-of-the-art",
        "game-changing", "paradigm shift",
        "a myriad of", "a plethora of",
        "aims to bridge the gap", "bridge the gap",
        "shed light on", "sheds light on",
        "pave the way", "paves the way",
        "the advent of", "with the advent of",
        "in recent years", "in recent times",
        "has gained significant attention",
        "has attracted considerable interest",
        "has emerged as a promising",
        "a comprehensive overview",
        "a holistic approach", "holistic understanding",
        "showcasing the efficacy", "demonstrate the efficacy",
        "multifaceted", "underscores the importance",
        "navigate the complexities",
        "harness the potential", "harnessing the power",
        "it is imperative to", "it is crucial to",
        "a nuanced understanding", "nuanced approach",
        "robust and scalable", "seamlessly integrates",
        "the intricacies of", "intricate interplay",
        "facilitate a deeper understanding",
        "a testament to",
    ]
    draft_lower = draft.lower()
    boilerplate_hits: list[str] = []
    for phrase in _BOILERPLATE_PHRASES:
        count = draft_lower.count(phrase)
        if count > 0:
            boilerplate_hits.extend([phrase] * count)
    if len(boilerplate_hits) > 5:
        unique_phrases = sorted(set(boilerplate_hits))[:5]
        overall_warnings.append(
            f"AI boilerplate detected: {len(boilerplate_hits)} instances "
            f"of generic LLM phrases (e.g., {', '.join(repr(p) for p in unique_phrases[:3])})"
        )
        revision_directives.append(
            "REWRITE sentences containing AI-generated boilerplate phrases. "
            "Replace generic language (e.g., 'delves into', 'it is worth noting', "
            "'leverages the power of', 'plays a crucial role', 'paves the way') "
            "with precise, specific academic language."
        )

    # --- Related work depth check ---
    _rw_headings = {"related work", "related works", "background", "literature review"}
    rw_body = ""
    for sec in sections_data:
        if sec["heading_lower"] in _rw_headings and sec["level"] <= 2:
            rw_body = sec["body"]
            break
    if rw_body and len(rw_body.split()) > 50:
        _comparative_pats = re.compile(
            r"\b(unlike|in contrast|whereas|while .+ focus|"
            r"however|differ(?:s|ent)|our (?:method|approach) .+ instead|"
            r"we (?:instead|differ)|compared to|as opposed to|"
            r"goes beyond|extends|improves upon|addresses the limitation)\b",
            re.IGNORECASE,
        )
        sentences = [s.strip() for s in re.split(r"[.!?]+", rw_body) if s.strip()]
        comparative_sents = sum(1 for s in sentences if _comparative_pats.search(s))
        ratio = comparative_sents / len(sentences) if sentences else 0.0
        if ratio < 0.15 and len(sentences) >= 5:
            overall_warnings.append(
                f"Related Work is purely descriptive: only {comparative_sents}/{len(sentences)} "
                f"sentences ({ratio:.0%}) contain comparative language (target: >=15%)"
            )
            revision_directives.append(
                "REWRITE Related Work to critically compare with prior methods. "
                "Use phrases like 'unlike X, our approach...', 'in contrast to...', "
                "'while X focuses on... we address...' for at least 20% of sentences."
            )

    # --- Statistical rigor check (result sections) ---
    _results_headings = {"results", "experiments", "experimental results", "evaluation"}
    results_body = ""
    for sec in sections_data:
        if sec["heading_lower"] in _results_headings and sec["level"] <= 2:
            results_body += sec["body"] + "\n"
    if results_body and len(results_body.split()) > 100:
        has_std = bool(re.search(r"±|\\pm|\bstd\b|\\std\b|standard deviation", results_body, re.IGNORECASE))
        has_ci = bool(re.search(r"confidence interval|\bCI\b|95%|p-value|p\s*<", results_body, re.IGNORECASE))
        has_seeds = bool(re.search(r"(?:seed|run|trial)s?\s*[:=]\s*\d|averaged?\s+over\s+\d+\s+(?:seed|run|trial)", results_body, re.IGNORECASE))
        if not has_std and not has_ci and not has_seeds:
            overall_warnings.append(
                "No statistical measures found in results (no std, CI, p-values, or multi-seed reporting)"
            )
            revision_directives.append(
                "ADD error bars (±std), confidence intervals, or note the number of "
                "random seeds used. Single-run results without variance reporting "
                "are insufficient for top venues."
            )

    result: dict[str, Any] = {
        "section_analysis": section_analysis,
        "overall_warnings": overall_warnings,
        "revision_directives": revision_directives,
    }
    if stage_dir is not None:
        (stage_dir / "draft_quality.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if overall_warnings:
            logger.warning(
                "Draft quality: %d warning(s) \u2014 %s",
                len(overall_warnings),
                "; ".join(overall_warnings[:3]),
            )
        else:
            logger.info("Draft quality: all checks passed")
    return result


def _review_compiled_pdf(
    pdf_path: Path,
    llm: LLMClient,
    topic: str,
) -> dict[str, Any]:
    """Multi-dimensional LLM review of compiled paper (AI-Scientist style).

    Scores the paper on 7 academic review dimensions (1-10 each),
    identifies specific strengths/weaknesses, and provides an overall
    accept/reject recommendation with confidence.

    Returns a dict with dimensional scores, issues, and decision.
    """
    if not pdf_path.exists():
        return {}

    # Use source-based review since not all models support vision
    tex_path = pdf_path.with_suffix(".tex")
    if not tex_path.exists():
        return {}

    tex_content = tex_path.read_text(encoding="utf-8")[:12000]

    review_prompt = (
        "You are a senior Area Chair at a top AI conference (NeurIPS/ICML/ICLR) "
        "reviewing a paper submission. Provide a rigorous, structured review.\n\n"
        f"PAPER TOPIC: {topic}\n\n"
        f"LaTeX source:\n```latex\n{tex_content}\n```\n\n"
        "REVIEW INSTRUCTIONS:\n"
        "Score each dimension 1-10 (1=unacceptable, 5=borderline, 8=strong accept, "
        "10=best paper candidate). Be critical but fair.\n\n"
        "DIMENSIONS:\n"
        "1. SOUNDNESS: Are claims well-supported? Is methodology correct? "
        "Are there logical gaps or unsupported claims?\n"
        "2. PRESENTATION: Is the writing clear, flowing, and professional? "
        "Are there grammar errors, bullet lists in prose sections, or "
        "boilerplate phrases? Is it free of AI-generated slop?\n"
        "3. CONTRIBUTION: Is the contribution significant? Does it advance "
        "the field beyond incremental improvement?\n"
        "4. ORIGINALITY: Is the approach novel? Does it differentiate clearly "
        "from prior work?\n"
        "5. CLARITY: Are the method and results easy to understand? Are figures "
        "and tables well-designed with descriptive captions?\n"
        "6. SIGNIFICANCE: Would the community benefit from this work? Does it "
        "open new research directions?\n"
        "7. REPRODUCIBILITY: Are experimental details sufficient to reproduce "
        "results? Are hyperparameters, datasets, and metrics clearly stated?\n\n"
        "Also evaluate:\n"
        "- Are all figures referenced in the text?\n"
        "- Are tables properly formatted (booktabs style, no vertical rules)?\n"
        "- Does the related work critically compare, not just list papers?\n"
        "- Are statistical measures (std, CI, multiple seeds) reported?\n"
        "- Is there a clear limitations section?\n\n"
        "Return a JSON object:\n"
        "{\n"
        '  "soundness": N,\n'
        '  "presentation": N,\n'
        '  "contribution": N,\n'
        '  "originality": N,\n'
        '  "clarity": N,\n'
        '  "significance": N,\n'
        '  "reproducibility": N,\n'
        '  "overall_score": N,\n'
        '  "confidence": N,\n'
        '  "decision": "accept" or "reject",\n'
        '  "strengths": ["strength1", "strength2", ...],\n'
        '  "weaknesses": ["weakness1", "weakness2", ...],\n'
        '  "critical_issues": ["issue requiring revision", ...],\n'
        '  "minor_issues": ["formatting/typo issues", ...],\n'
        '  "summary": "2-3 sentence overall assessment"\n'
        "}\n"
    )

    try:
        resp = llm.chat(
            messages=[{"role": "user", "content": review_prompt}],
            system=(
                "You are a meticulous, critical academic reviewer. "
                "You have reviewed 100+ papers at top venues. "
                "Score honestly — most papers deserve 4-6, not 7-9. "
                "Flag any sign of AI-generated boilerplate."
            ),
        )
        review_data = _safe_json_loads(resp.content, {})
        if isinstance(review_data, dict) and "overall_score" in review_data:
            # Compute weighted aggregate if individual scores present
            dim_scores = {
                k: review_data.get(k, 0)
                for k in (
                    "soundness", "presentation", "contribution",
                    "originality", "clarity", "significance",
                    "reproducibility",
                )
            }
            valid = {k: v for k, v in dim_scores.items() if isinstance(v, (int, float)) and v > 0}
            if valid:
                review_data["mean_score"] = round(sum(valid.values()) / len(valid), 2)
            return review_data
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF review LLM call failed: %s", exc)

    return {}


def _check_ablation_effectiveness(
    exp_summary: dict[str, Any],
    threshold: float = 0.05,
) -> list[str]:
    """P7: Check if ablation results are within *threshold* of baseline.

    Returns a list of warning strings for ineffective ablations.
    """
    warnings: list[str] = []
    cond_summaries = exp_summary.get("condition_summaries", {})
    if not isinstance(cond_summaries, dict) or not cond_summaries:
        return warnings

    # Find baseline/control condition
    baseline_name = None
    baseline_mean = None
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        name_lower = name.lower()
        if any(tag in name_lower for tag in ("baseline", "control", "vanilla", "standard")):
            metrics = data.get("metrics", {})
            # Use the first metric that has a _mean suffix or the first available
            for mk, mv in metrics.items():
                if mk.endswith("_mean"):
                    baseline_name = name
                    baseline_mean = float(mv)
                    break
            if baseline_mean is None:
                for mk, mv in metrics.items():
                    try:
                        baseline_name = name
                        baseline_mean = float(mv)
                        break
                    except (TypeError, ValueError):
                        continue
            if baseline_name:
                break

    if baseline_name is None or baseline_mean is None:
        return warnings

    # Check each ablation condition
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        name_lower = name.lower()
        if name == baseline_name:
            continue
        if not any(tag in name_lower for tag in ("ablation", "no_", "without", "reduced")):
            continue
        metrics = data.get("metrics", {})
        for mk, mv in metrics.items():
            if not mk.endswith("_mean"):
                continue
            try:
                abl_val = float(mv)
            except (TypeError, ValueError):
                continue
            if baseline_mean != 0:
                rel_diff = abs(abl_val - baseline_mean) / abs(baseline_mean)
            else:
                rel_diff = abs(abl_val - baseline_mean)
            if rel_diff < threshold:
                warnings.append(
                    f"Ablation '{name}' {mk}={abl_val:.4f} is within "
                    f"{rel_diff:.1%} of baseline '{baseline_name}' "
                    f"{mk}={baseline_mean:.4f} — ablation may be ineffective"
                )
            break  # Only check the first _mean metric per condition

    return warnings


def _detect_result_contradictions(
    exp_summary: dict[str, Any],
) -> list[str]:
    """P10: Detect contradictions in experiment results before paper writing.

    Returns a list of advisory strings to inject into paper writing prompt.
    """
    advisories: list[str] = []
    cond_summaries = exp_summary.get("condition_summaries", {})
    if not isinstance(cond_summaries, dict) or not cond_summaries:
        return advisories

    # Collect primary metric means per condition
    means: dict[str, float] = {}
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        metrics = data.get("metrics", {})
        for mk, mv in metrics.items():
            if mk.endswith("_mean"):
                try:
                    means[name] = float(mv)
                except (TypeError, ValueError):
                    pass
                break

    if len(means) < 2:
        return advisories

    # Check 1: All methods within noise margin (2% relative spread)
    vals = list(means.values())
    val_range = max(vals) - min(vals)
    val_mean = sum(vals) / len(vals)
    if val_mean != 0 and (val_range / abs(val_mean)) < 0.02:
        advisories.append(
            "NULL RESULT: All methods produce nearly identical primary metric values "
            f"(range={val_range:.4f}, mean={val_mean:.4f}). Frame this as a null result — "
            "the methods are statistically indistinguishable. Do NOT claim any method "
            "is superior. Discuss possible explanations (task too easy/hard, metric "
            "insensitive, insufficient differentiation in methods)."
        )

    # Check 2: Control/simple baseline outperforms proposed method
    baseline_val = None
    baseline_name = None
    proposed_val = None
    proposed_name = None
    for name, val in means.items():
        name_lower = name.lower()
        if any(tag in name_lower for tag in ("baseline", "control", "random", "vanilla")):
            if baseline_val is None or val > (baseline_val or 0):
                baseline_val = val
                baseline_name = name
        elif any(tag in name_lower for tag in ("proposed", "our", "novel", "method")):
            if proposed_val is None or val > (proposed_val or 0):
                proposed_val = val
                proposed_name = name

    if baseline_val is not None and proposed_val is not None:
        if baseline_val > proposed_val:
            advisories.append(
                f"NEGATIVE RESULT: Baseline '{baseline_name}' ({baseline_val:.4f}) "
                f"outperforms proposed method '{proposed_name}' ({proposed_val:.4f}). "
                "This is a NEGATIVE result. Do NOT claim the proposed method is superior. "
                "Frame as 'An Empirical Study of...' or 'When X Falls Short'. "
                "Discuss why the baseline won and what this implies for future work."
            )

    return advisories


def _execute_paper_draft(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    outline = _read_prior_artifact(run_dir, "outline.md") or ""
    preamble = _build_context_preamble(
        config,
        run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_analysis=True,
        include_experiment_data=True,  # WS-5.1: inject real experiment data
    )

    # R21-1: Read BEST experiment_summary across all stage-14 versions.
    # Refinement can regress — the final (non-versioned) stage-14 may have
    # worse data than an earlier version. Pick the richest one.
    exp_summary_text = None
    _best_metric_count = 0
    for _s14_dir in sorted(run_dir.glob("stage-14*")):
        _candidate = _s14_dir / "experiment_summary.json"
        if _candidate.is_file():
            _text = _candidate.read_text(encoding="utf-8")
            _parsed = _safe_json_loads(_text, {})
            if isinstance(_parsed, dict):
                _mcount = _parsed.get("total_metric_keys", 0) or len(
                    _parsed.get("metrics_summary", {})
                )
                _paired_count = len(_parsed.get("paired_comparisons", []))
                _score = _mcount + _paired_count * 10  # Prefer paired data
                if _score > _best_metric_count:
                    _best_metric_count = _score
                    exp_summary_text = _text
                    logger.info(
                        "R21-1: Selected %s (metric_keys=%d, paired=%d, score=%d)",
                        _s14_dir.name, _mcount, _paired_count, _score,
                    )
    # Fallback to standard artifact read
    if exp_summary_text is None:
        exp_summary_text = _read_prior_artifact(run_dir, "experiment_summary.json")
    exp_metrics_instruction = ""
    has_real_metrics = False
    if exp_summary_text:
        exp_summary = _safe_json_loads(exp_summary_text, {})
        if isinstance(exp_summary, dict) and exp_summary.get("metrics_summary"):
            has_real_metrics = True
            exp_metrics_instruction = (
                "\n\nIMPORTANT: Use the ACTUAL experiment results provided in the context. "
                "All numbers in the Results and Experiments sections MUST reference real data. "
                "Do NOT write 'no quantitative results yet' or use placeholder numbers. "
                "Cite specific metrics with their actual values.\n"
            )

    # Collect raw experiment stdout metrics as hard constraint for the paper
    raw_metrics_block, _has_parsed_metrics = _collect_raw_experiment_metrics(run_dir)
    if raw_metrics_block:
        # BUG-23: Raw stdout alone is not sufficient — require either
        # metrics_summary data, parsed metrics from run JSONs,
        # OR at least 3 condition= patterns in raw block
        _has_condition_pattern = len(re.findall(
            r"condition[=:]", raw_metrics_block, re.IGNORECASE
        )) >= 3
        if has_real_metrics or _has_parsed_metrics or _has_condition_pattern:
            has_real_metrics = True
        exp_metrics_instruction += raw_metrics_block

    # R18-1 + R19-6: Inject paired statistical comparisons AND condition summaries
    if exp_summary_text:
        exp_summary_parsed = _safe_json_loads(exp_summary_text, {})
        if isinstance(exp_summary_parsed, dict):
            # R19-6: Inject experiment scale header so LLM knows the data richness
            _total_conds = exp_summary_parsed.get("total_conditions")
            _total_mkeys = exp_summary_parsed.get("total_metric_keys")
            if _total_conds or _total_mkeys:
                scale_block = "\n\n## EXPERIMENT SCALE\n"
                if _total_conds:
                    scale_block += f"- Total conditions tested: {_total_conds}\n"
                if _total_mkeys:
                    scale_block += f"- Total metric keys collected: {_total_mkeys}\n"
                scale_block += (
                    "- This is a MULTI-SEED experiment. Report mean +/- std across seeds.\n"
                    "- Do NOT describe results as 'single run' or 'preliminary'.\n"
                )
                exp_metrics_instruction += scale_block

            # R19-6 + R33: Inject condition summaries with CIs
            cond_summaries = exp_summary_parsed.get("condition_summaries", {})
            if isinstance(cond_summaries, dict) and cond_summaries:
                cond_block = "\n\n## PER-CONDITION SUMMARY (use in Results tables)\n"
                for cname, cdata in sorted(cond_summaries.items()):
                    cond_block += f"\n### {cname}\n"
                    if not isinstance(cdata, dict):
                        continue
                    sr = cdata.get("success_rate")
                    if sr is not None:
                        cond_block += f"- Success rate: {sr:.1%}\n"
                    ns = cdata.get("n_seeds") or cdata.get("n_seed_metrics")
                    if ns:
                        cond_block += f"- Seeds: {ns}\n"
                    ci_lo = cdata.get("ci95_low")
                    ci_hi = cdata.get("ci95_high")
                    if ci_lo is not None and ci_hi is not None:
                        try:
                            cond_block += f"- Bootstrap 95% CI: [{float(ci_lo):.4f}, {float(ci_hi):.4f}]\n"
                        except (ValueError, TypeError):
                            cond_block += f"- Bootstrap 95% CI: [{ci_lo}, {ci_hi}]\n"
                    cm = cdata.get("metrics", {})
                    if cm:
                        for mk, mv in sorted(cm.items()):
                            if isinstance(mv, (int, float)):
                                cond_block += f"- {mk}: {mv:.4f}\n"
                            else:
                                cond_block += f"- {mk}: {mv}\n"
                exp_metrics_instruction += cond_block

            # R18-1: Inject paired statistical comparisons
            paired = exp_summary_parsed.get("paired_comparisons", [])
            if paired:
                paired_block = "\n\n## PAIRED STATISTICAL COMPARISONS (use these in Results)\n"
                paired_block += f"Total: {len(paired)} paired tests computed.\n"
                for pc in paired:
                    if not isinstance(pc, dict):
                        continue
                    method = pc.get("method", "?")
                    baseline = pc.get("baseline", "?")
                    regime = pc.get("regime", "all")
                    md = pc.get("mean_diff", "?")
                    sd = pc.get("std_diff", "?")
                    ts = pc.get("t_stat", "?")
                    pv = pc.get("p_value", "?")
                    ci_lo = pc.get("ci95_low")
                    ci_hi = pc.get("ci95_high")
                    ci_str = f", 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else ""
                    paired_block += (
                        f"- {method} vs {baseline} (regime={regime}): "
                        f"mean_diff={md}, std_diff={sd}, "
                        f"t={ts}, p={pv}{ci_str}\n"
                    )
                exp_metrics_instruction += paired_block

            # R24: Method naming map — translate generic condition labels
            _cond_names = list(cond_summaries.keys()) if isinstance(cond_summaries, dict) and cond_summaries else []
            if _cond_names:
                naming_block = (
                    "\n\n## METHOD NAMING (CRITICAL — do NOT use generic labels in the paper)\n"
                    "The condition labels below come from the experiment code. In the paper, "
                    "you MUST use DESCRIPTIVE algorithm names, not generic labels.\n"
                    "- If a condition name is already descriptive (e.g., 'random_search', "
                    "'bayesian_optimization', 'ppo_policy'), use it directly as a proper name.\n"
                    "- If a condition name is generic (e.g., 'baseline_1', 'method_variant_1'), "
                    "you MUST infer the algorithm from the experiment code/context and use the "
                    "real algorithm name (e.g., 'Random Search', 'Bayesian Optimization', "
                    "'PPO', 'Curiosity-Driven RL').\n"
                    "- NEVER write `baseline_1` or `method_variant_1` in the paper text.\n"
                    f"- Conditions to name: {_cond_names}\n"
                )
                exp_metrics_instruction += naming_block

            # IMP-8: Inject broken ablation warnings
            abl_warnings = exp_summary_parsed.get("ablation_warnings", [])
            if abl_warnings:
                broken_block = (
                    "\n\n## BROKEN ABLATIONS (DO NOT discuss as valid results)\n"
                    "The following ablation conditions produced IDENTICAL outputs, "
                    "indicating implementation bugs. Do NOT present their differences "
                    "as findings. Mention them ONLY in a 'Limitations' sub-section "
                    "as known implementation issues:\n"
                )
                for _aw in abl_warnings:
                    broken_block += f"- {_aw}\n"
                broken_block += (
                    "\nIf you reference these conditions, state explicitly: "
                    "'Due to an implementation defect, conditions X and Y produced "
                    "identical outputs; their comparison is therefore uninformative.'\n"
                )
                exp_metrics_instruction += broken_block

            # R25: Statistical table format requirement
            if paired:
                stat_table_block = (
                    "\n\n## STATISTICAL TABLE REQUIREMENT (MANDATORY in Results section)\n"
                    "The Results section MUST include a statistical comparison table with columns:\n"
                    "| Comparison | Mean Diff | Std Diff | t-statistic | p-value | Significance |\n"
                    "Use the PAIRED STATISTICAL COMPARISONS data above to fill this table.\n"
                    "Mark significance: *** (p<0.001), ** (p<0.01), * (p<0.05), n.s.\n"
                    "This is non-negotiable — a top-venue paper MUST have statistical tests.\n"
                )
                exp_metrics_instruction += stat_table_block

            # R26: Metric definition requirement
            exp_metrics_instruction += (
                "\n\n## METRIC DEFINITIONS (MANDATORY in Experiments section)\n"
                "The Experiments section MUST define each metric:\n"
                "- **Primary metric**: what it measures, how it is computed, range, direction "
                "(higher/lower is better), and units if applicable.\n"
                "- **Secondary metric**: same details.\n"
                "- For time-to-event metrics: explain the horizon, what constitutes success, "
                "and how failures are handled (e.g., set to max horizon).\n"
                "- These definitions MUST appear BEFORE any results tables.\n"
            )

            # R27: Multi-seed framing enforcement
            _any_seeds = any(
                (cond_summaries.get(c) or {}).get("n_seed_metrics", 0) > 1
                for c in _cond_names
            ) if _cond_names else False
            if _any_seeds:
                exp_metrics_instruction += (
                    "\n\n## MULTI-SEED EXPERIMENT FRAMING (CRITICAL)\n"
                    "This experiment uses MULTIPLE independent random seeds per condition.\n"
                    "- Report mean +/- std (or SE) for all metrics.\n"
                    "- NEVER describe this as 'a single run' or '1 benchmark-artifact run'.\n"
                    "- Frame as: 'We evaluate each method across N seeds per regime.'\n"
                    "- The seed-level data IS the evidence base — it is NOT a single observation.\n"
                    "- Include per-regime breakdowns (easy vs hard) as separate rows in tables.\n"
                )

    # BUG-003: Inject actual evaluated datasets as a hard constraint
    if exp_summary_text:
        _ds_parsed = _safe_json_loads(exp_summary_text, {})
        if isinstance(_ds_parsed, dict):
            _datasets: set[str] = set()
            # Extract from condition names (often contain dataset info)
            for _cname in (_ds_parsed.get("condition_summaries") or {}).keys():
                _datasets.add(str(_cname))
            # Extract from explicit "datasets" field if present
            for _ds in (_ds_parsed.get("datasets") or []):
                if isinstance(_ds, str):
                    _datasets.add(_ds)
            # Extract from "benchmark" or "dataset" fields
            for _key in ("benchmark", "dataset", "dataset_name"):
                _dv = _ds_parsed.get(_key)
                if isinstance(_dv, str) and _dv:
                    _datasets.add(_dv)
            if _datasets:
                exp_metrics_instruction += (
                    "\n\n## ACTUAL EVALUATED DATASETS (HARD CONSTRAINT)\n"
                    "The following datasets/conditions were ACTUALLY tested in experiments:\n"
                    + "".join(f"- {d}\n" for d in sorted(_datasets))
                    + "\nCRITICAL: Do NOT claim evaluation on any dataset not listed above.\n"
                    "Do NOT fabricate results for datasets you did not run experiments on.\n"
                    "If you reference other datasets, clearly state they are 'not evaluated "
                    "in this work' or are 'left for future work'.\n"
                )

    # P7: Ablation effectiveness check
    if exp_summary_text:
        _exp_parsed_p7 = _safe_json_loads(exp_summary_text, {})
        if isinstance(_exp_parsed_p7, dict):
            _abl_warnings = _check_ablation_effectiveness(_exp_parsed_p7)
            if _abl_warnings:
                _abl_block = (
                    "\n\n## ABLATION EFFECTIVENESS WARNINGS\n"
                    "The following ablations showed minimal effect (within 5%% of baseline). "
                    "Discuss this honestly — it may indicate the ablated component is not "
                    "important, or the ablation was not properly implemented:\n"
                )
                for _aw in _abl_warnings:
                    _abl_block += f"- {_aw}\n"
                exp_metrics_instruction += _abl_block
                logger.warning("P7: Ablation effectiveness warnings: %s", _abl_warnings)

    # P10: Contradiction detection
    if exp_summary_text:
        _exp_parsed_p10 = _safe_json_loads(exp_summary_text, {})
        if isinstance(_exp_parsed_p10, dict):
            _contradictions = _detect_result_contradictions(_exp_parsed_p10)
            if _contradictions:
                _contra_block = (
                    "\n\n## RESULT INTERPRETATION ADVISORIES (CRITICAL — read before writing)\n"
                )
                for _ca in _contradictions:
                    _contra_block += f"- {_ca}\n"
                exp_metrics_instruction += _contra_block
                logger.warning("P10: Contradiction advisories: %s", _contradictions)

    # R10: HARD BLOCK — refuse to write paper when all data is simulated
    all_simulated = True
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue
            try:
                _payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(_payload, dict) and _payload.get("status") != "simulated":
                all_simulated = False
                break
        if not all_simulated:
            break

    if all_simulated:
        logger.error(
            "BLOCKED: All experiment data is simulated (mode='simulated'). "
            "Cannot write a paper based on formulaic fake data. "
            "Switch to experiment.mode='sandbox' and re-run."
        )
        (stage_dir / "paper_draft.md").write_text(
            "# Paper Draft Blocked\n\n"
            "**Reason**: All experiment results are from simulated mode "
            "(formulaic data: `0.3 + idx * 0.03`). "
            "These are not real experimental results.\n\n"
            "**Action Required**: Set `experiment.mode: 'sandbox'` in "
            "config.arc.yaml and re-run the pipeline.",
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.PAPER_DRAFT,
            status=StageStatus.FAILED,
            artifacts=("paper_draft.md",),
            evidence_refs=(),
        )

    # R4-2: HARD BLOCK — refuse to write paper with no real data (ML/empirical domains)
    # For non-empirical domains (math proofs, theoretical economics), allow proceeding
    _domain_id, _domain_name, _domain_venues = _detect_domain(
        config.research.topic, config.research.domains
    )
    _empirical_domains = {"ml", "engineering", "biology", "chemistry"}
    if not has_real_metrics:
        if _domain_id in _empirical_domains:
            logger.error(
                "BLOCKED: Cannot write paper — experiment produced NO metrics. "
                "The pipeline will not fabricate results."
            )
            (stage_dir / "paper_draft.md").write_text(
                "# Paper Draft Blocked\n\n"
                "**Reason**: Experiment stage produced no metrics (status: failed/timeout). "
                "Cannot write a paper without real experimental data.\n\n"
                "**Action Required**: Fix experiment execution or increase time_budget_sec.",
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.PAPER_DRAFT,
                status=StageStatus.FAILED,
                artifacts=("paper_draft.md",),
                evidence_refs=(),
            )
        else:
            logger.warning(
                "No experiment metrics found, but domain '%s' may be non-empirical "
                "(theoretical/mathematical). Proceeding with paper draft.",
                _domain_name,
            )

    # R11-5: Experiment quality minimum threshold before paper writing
    # Parse analysis.md for quality rating and condition completeness
    analysis_text = _read_prior_artifact(run_dir, "analysis.md") or ""
    _quality_warnings: list[str] = []

    # Check 1: Was the analysis quality rating very low?
    import re as _re_q
    _rating_match = _re_q.search(
        r"(?:quality\s+rating|result\s+quality)[:\s]*\**(\d+)\s*/\s*10",
        analysis_text,
        _re_q.IGNORECASE,
    )
    if _rating_match:
        _analysis_rating = int(_rating_match.group(1))
        if _analysis_rating <= 3:
            _quality_warnings.append(
                f"Analysis rated experiment quality {_analysis_rating}/10"
            )
        # BUG-23: If quality rating is ≤ 2, force has_real_metrics = False
        # to prevent fabricated results even if stdout had stray numbers
        if _analysis_rating <= 2 and has_real_metrics:
            logger.warning(
                "BUG-23 guard: Analysis quality %d/10 ≤ 2 — "
                "overriding has_real_metrics to False (experiment likely failed)",
                _analysis_rating,
            )
            has_real_metrics = False

    # Check 2: Are baselines missing?
    _analysis_lower = analysis_text.lower()
    if "no" in _analysis_lower and "baseline" in _analysis_lower:
        if any(phrase in _analysis_lower for phrase in [
            "no baseline", "no bo", "no random", "baselines are missing",
            "missing baselines", "baseline coverage is missing",
        ]):
            _quality_warnings.append("Baselines appear to be missing from results")

    # Check 3: Is the metric undefined?
    if any(phrase in _analysis_lower for phrase in [
        "metric is undefined", "primary_metric is undefined",
        "undefined metric", "metric undefined",
    ]):
        _quality_warnings.append("Primary metric is undefined (direction/units/formula unknown)")

    # Check 4: Very few conditions completed
    _condition_count = len(_re_q.findall(
        r"condition[=:\s]+\w+.*?(?:mean|primary_metric)",
        raw_metrics_block or "",
        _re_q.IGNORECASE,
    ))

    if _quality_warnings:
        _warning_block = "\n".join(f"  - {w}" for w in _quality_warnings)
        logger.warning(
            "Stage 17: Experiment quality concerns detected before paper writing:\n%s",
            _warning_block,
        )
        # Inject quality warnings into the paper writing prompt so the LLM
        # writes an appropriately hedged paper
        exp_metrics_instruction += (
            "\n\n## EXPERIMENT QUALITY WARNINGS (address these honestly in the paper)\n"
            + "\n".join(f"- {w}" for w in _quality_warnings)
            + "\n\nBecause of these issues, the paper MUST:\n"
            "- Use hedged language ('preliminary', 'pilot', 'initial exploration')\n"
            "- NOT claim definitive comparisons between methods\n"
            "- Dedicate a substantial Limitations section to these gaps\n"
            "- Frame the contribution as methodology/framework, not empirical findings\n"
        )
        # Save warnings for tracking
        (stage_dir / "quality_warnings.json").write_text(
            json.dumps(_quality_warnings, indent=2), encoding="utf-8"
        )

    # R4-2: Anti-fabrication data integrity instruction
    exp_metrics_instruction += (
        "\n\n## CRITICAL: Data Integrity Rules\n"
        "- You may ONLY report numbers that appear in the experiment data above\n"
        "- If the experiment data is incomplete (fewer conditions than planned), report\n"
        "  ONLY the conditions that were actually run\n"
        "- Do NOT extrapolate, interpolate, or 'fill in' missing cells in tables\n"
        "- Do NOT invent confidence intervals, p-values, or statistical tests unless\n"
        "  the actual data supports them\n"
        "- If only N conditions completed, simply report results for those N conditions\n"
        "  without repeating apologies or disclaimers about missing conditions\n"
        "- Any table cell without real data must show '—' (not a plausible number)\n"
        "- FORBIDDEN: generating numbers that 'look right' based on your training data\n"
    )

    # IMP-6 + FA: Inject chart references into paper draft prompt
    # Prefer FigureAgent's figure_plan.json (rich descriptions) over raw file scan
    # BUG-FIX: figure_plan.json may be a list (from FigureAgent planner) or a dict
    # (from executor overwrite).  The orchestrator writes a list at planning time;
    # the executor overwrites with a dict only when figure_count > 0.  If the
    # FigureAgent renders 0 charts the list persists, and calling .get() on it
    # raises AttributeError.
    _fa_descriptions = ""
    for _s14_dir in sorted(run_dir.glob("stage-14*")):
        # Prefer the final plan (dict with figure_descriptions) if it exists
        for _fp_name in ("figure_plan_final.json", "figure_plan.json"):
            _fp_path = _s14_dir / _fp_name
            if not _fp_path.exists():
                continue
            try:
                _fp_data = json.loads(_fp_path.read_text(encoding="utf-8"))
                if isinstance(_fp_data, dict):
                    _fa_descriptions = _fp_data.get("figure_descriptions", "")
                elif isinstance(_fp_data, list) and _fp_data:
                    # List format from FigureAgent planner — synthesize descriptions
                    _desc_parts = ["## PLANNED FIGURES (from figure plan)\n"]
                    for _fig in _fp_data:
                        if isinstance(_fig, dict):
                            _fid = _fig.get("figure_id", "unnamed")
                            _ftitle = _fig.get("title", "")
                            _fcap = _fig.get("caption", "")
                            _fsec = _fig.get("section", "results")
                            _desc_parts.append(
                                f"- **{_fid}** ({_fsec}): {_ftitle}\n  {_fcap}"
                            )
                    if len(_desc_parts) > 1:
                        _fa_descriptions = "\n".join(_desc_parts)
            except (json.JSONDecodeError, OSError):
                pass
            if _fa_descriptions:
                break
        if _fa_descriptions:
            break

    if _fa_descriptions:
        exp_metrics_instruction += "\n\n" + _fa_descriptions
        logger.info("Stage 17: Injected FigureAgent figure descriptions into paper draft prompt")
    else:
        # Fallback: scan for chart files
        _chart_files: list[str] = []
        for _s14_dir in sorted(run_dir.glob("stage-14*")):
            _charts_path = _s14_dir / "charts"
            if _charts_path.is_dir():
                for _cf in sorted(_charts_path.glob("*.png")):
                    _chart_files.append(_cf.name)
        if _chart_files:
            _chart_block = (
                "\n\n## AVAILABLE FIGURES (embed in the paper)\n"
                "The following figures were generated from actual experiment data. "
                "You MUST reference at least 1-2 of these in the Results section "
                "using markdown image syntax: `![Caption](charts/filename.png)`\n\n"
            )
            for _cf_name in _chart_files:
                _label = _cf_name.replace("_", " ").replace(".png", "").title()
                _chart_block += f"- `charts/{_cf_name}` — {_label}\n"
            _chart_block += (
                "\nFor each figure referenced, write a descriptive caption and "
                "discuss what the figure shows in 2-3 sentences.\n"
            )
            exp_metrics_instruction += _chart_block
            logger.info(
                "Stage 17: Injected %d chart references into paper draft prompt",
                len(_chart_files),
            )

    # WS-5.5: Framework diagram placeholder instruction
    exp_metrics_instruction += (
        "\n\n## FRAMEWORK DIAGRAM PLACEHOLDER\n"
        "In the Method/Approach section, include a placeholder for the methodology "
        "framework overview figure. Insert this exactly:\n\n"
        "```\n"
        "![Framework Overview](charts/framework_diagram.png)\n"
        "**Figure N.** Overview of the proposed methodology. "
        "[A detailed framework diagram will be generated separately and inserted here.]\n"
        "```\n\n"
        "This figure should be referenced in the text as 'Figure N' and discussed briefly "
        "(1-2 sentences describing the overall pipeline/architecture flow). "
        "The actual image will be generated post-hoc using a text-to-image model.\n"
    )

    # P5: Extract hyperparameters from results.json for paper Method section
    _hp_table = ""
    for _s14_dir in sorted(run_dir.glob("stage-14*")):
        for _run_file in sorted(_s14_dir.glob("runs/*.json")):
            try:
                _run_data = json.loads(_run_file.read_text(encoding="utf-8"))
                if isinstance(_run_data, dict) and _run_data.get("hyperparameters"):
                    _hp = _run_data["hyperparameters"]
                    if isinstance(_hp, dict) and _hp:
                        _hp_table = "\n\n## HYPERPARAMETERS (include as a table in the Method section)\n"
                        _hp_table += "| Hyperparameter | Value |\n|---|---|\n"
                        for _hk, _hv in sorted(_hp.items()):
                            _hp_table += f"| {_hk} | {_hv} |\n"
                        _hp_table += (
                            "\nThis table MUST appear in the Method/Experiments section. "
                            "Include ALL hyperparameters used, with justification for key choices.\n"
                        )
                        break
            except (json.JSONDecodeError, OSError):
                continue
        if _hp_table:
            break
    # Also check staging dirs for results.json
    if not _hp_table:
        for _staging_dir in sorted(run_dir.glob("stage-*/runs/_docker_*")):
            _rjson = _staging_dir / "results.json"
            if _rjson.is_file():
                try:
                    _rdata = json.loads(_rjson.read_text(encoding="utf-8"))
                    if isinstance(_rdata, dict) and _rdata.get("hyperparameters"):
                        _hp = _rdata["hyperparameters"]
                        if isinstance(_hp, dict) and _hp:
                            _hp_table = "\n\n## HYPERPARAMETERS (include as a table in the Method section)\n"
                            _hp_table += "| Hyperparameter | Value |\n|---|---|\n"
                            for _hk, _hv in sorted(_hp.items()):
                                _hp_table += f"| {_hk} | {_hv} |\n"
                            _hp_table += (
                                "\nThis table MUST appear in the Method/Experiments section. "
                                "Include ALL hyperparameters used, with justification for key choices.\n"
                            )
                            break
                except (json.JSONDecodeError, OSError):
                    continue
    if _hp_table:
        exp_metrics_instruction += _hp_table

    # F2.6: Build citation list from references.bib / candidates with cite_keys
    citation_instruction = ""
    bib_text = _read_prior_artifact(run_dir, "references.bib")

    # P3: Pre-verify citations before paper draft — remove hallucinated refs
    if bib_text and bib_text.strip():
        from researchclaw.literature.verify import (
            filter_verified_bibtex,
            verify_citations as _verify_cit,
        )
        try:
            _pre_report = _verify_cit(bib_text, inter_verify_delay=0.5)
            _kept = _pre_report.verified + _pre_report.suspicious
            _removed = _pre_report.hallucinated
            if _removed > 0:
                bib_text = filter_verified_bibtex(
                    bib_text, _pre_report, include_suspicious=True
                )
                (stage_dir / "references_preverified.bib").write_text(
                    bib_text, encoding="utf-8"
                )
                logger.info(
                    "P3: Pre-verification kept %d/%d citations (removed %d hallucinated)",
                    _kept, _pre_report.total, _removed,
                )
        except Exception as exc:
            logger.warning("P3: Pre-verification failed, using original bib: %s", exc)

    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl")
    if candidates_text:
        cite_lines: list[str] = []
        for row_text in candidates_text.strip().splitlines():
            row = _safe_json_loads(row_text, {})
            if isinstance(row, dict) and row.get("cite_key"):
                authors_info = ""
                if isinstance(row.get("authors"), list) and row["authors"]:
                    first_author = row["authors"][0]
                    if isinstance(first_author, dict):
                        # BUG-38: name may be non-str (tuple/list) — force str
                        _name = first_author.get("name", "")
                        authors_info = _name if isinstance(_name, str) else str(_name)
                    elif isinstance(first_author, str):
                        authors_info = first_author
                    if len(row["authors"]) > 1:
                        authors_info += " et al."
                title = row.get("title", "")
                cite_lines.append(
                    f"- [{row['cite_key']}] → TITLE: \"{title}\" "
                    f"| {authors_info} "
                    f"({row.get('venue', '')}, {row.get('year', '')}, "
                    f"cited {row.get('citation_count', 0)} times) "
                    f"| ONLY cite this key when discussing: {title}"
                )
        if cite_lines:
            citation_instruction = (
                "\n\nAVAILABLE REFERENCES (use [cite_key] to cite in the text):\n"
                + "\n".join(cite_lines)
                + "\n\nCRITICAL CITATION RULES:\n"
                "- In the body text, cite using [cite_key] format, e.g. [smith2024transformer].\n"
                "- Do NOT write a References section — it will be auto-generated from the bibliography file.\n"
                "- Do NOT invent any references or arXiv IDs not in the above list.\n"
                "- You may cite a subset, but NEVER fabricate citations or change arXiv IDs.\n"
                "- SEMANTIC MATCHING: Before citing a reference, verify that its TITLE matches\n"
                "  the concept you are discussing. Do NOT use an unrelated cite_key just\n"
                "  because it sounds similar.\n"
                "- If no reference in the list matches the concept you want to cite,\n"
                "  write 'prior work has shown...' WITHOUT a citation, rather than using\n"
                "  a mismatched reference.\n"
                "- Each [cite_key] MUST correspond to the paper whose title is shown\n"
                "  next to that key in the list above. Cross-check before citing.\n"
                "\nCITATION QUANTITY & QUALITY CONSTRAINTS:\n"
                "- Cite 25-40 unique references in the paper body. The Related Work\n"
                "  section alone should cite at least 15 references.\n"
                "- Every citation MUST be directly relevant to the paper's topic.\n"
                "- DO NOT cite papers from unrelated domains (wireless communication, "
                "manufacturing, UAV, etc.).\n"
                "- Prefer well-known, highly-cited papers over obscure ones.\n"
                "- If unsure whether a paper exists or is relevant, DO NOT cite it.\n"
            )

    if llm is not None:
        _pm = prompts or PromptManager()
        topic_constraint = _pm.block("topic_constraint", topic=config.research.topic)

        # --- Section-by-section writing (3 calls) for conference-grade depth ---
        draft = _write_paper_sections(
            llm=llm,
            pm=_pm,
            run_dir=run_dir,
            preamble=preamble,
            topic_constraint=topic_constraint,
            exp_metrics_instruction=exp_metrics_instruction,
            citation_instruction=citation_instruction,
            outline=outline,
            model_name=config.llm.primary_model,
        )

        # R7: Strip LLM-generated References section — it often fabricates arXiv IDs.
        import re as _re_r7
        ref_pattern = _re_r7.compile(
            r'^(#{1,2}\s*References.*)', _re_r7.MULTILINE | _re_r7.DOTALL
        )
        ref_match = ref_pattern.search(draft)
        if ref_match:
            draft = draft[:ref_match.start()].rstrip()
            logger.info("Stage 17: Stripped LLM-generated References section (R7 fix)")
    else:
        # Build template with real data if available
        results_section = "Template results summary."
        if exp_summary_text:
            exp_summary = _safe_json_loads(exp_summary_text, {})
            if isinstance(exp_summary, dict) and exp_summary.get("metrics_summary"):
                lines = ["Experiment results:"]
                for mk, mv in exp_summary["metrics_summary"].items():
                    if isinstance(mv, dict):
                        lines.append(
                            f"- {mk}: mean={mv.get('mean')}, min={mv.get('min')}, "
                            f"max={mv.get('max')}, n={mv.get('count')}"
                        )
                results_section = "\n".join(lines)

        draft = f"""# Draft Title

## Abstract
Template draft abstract.

## Introduction
Template introduction for {config.research.topic}.

## Related Work
Template related work.

## Method
Template method description.

## Experiments
Template experimental setup.

## Results
{results_section}

## Limitations
Template limitations.

## Conclusion
Template conclusion.

## References
Template references.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "paper_draft.md").write_text(draft, encoding="utf-8")

    # Validate draft quality (section balance + bullet density)
    _validate_draft_quality(draft, stage_dir=stage_dir)

    return StageResult(
        stage=Stage.PAPER_DRAFT,
        status=StageStatus.DONE,
        artifacts=("paper_draft.md",),
        evidence_refs=("stage-17/paper_draft.md",),
    )


def _collect_experiment_evidence(run_dir: Path) -> str:
    """Collect actual experiment parameters and results for peer review."""
    evidence_parts: list[str] = []

    # 1. Read experiment code to find actual trial count, methods used
    exp_dir = _read_prior_artifact(run_dir, "experiment/")
    if exp_dir and Path(exp_dir).is_dir():
        main_py = Path(exp_dir) / "main.py"
        if main_py.exists():
            code = main_py.read_text(encoding="utf-8")
            evidence_parts.append(f"### Actual Experiment Code (main.py)\n```python\n{code[:3000]}\n```")

    # 2. Read sandbox run results (actual metrics, runtime, stderr)
    runs_text = _read_prior_artifact(run_dir, "runs/")
    if runs_text and Path(runs_text).is_dir():
        for run_file in sorted(Path(runs_text).glob("*.json"))[:5]:
            payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if isinstance(payload, dict):
                summary = {
                    "metrics": payload.get("metrics"),
                    "elapsed_sec": payload.get("elapsed_sec"),
                    "timed_out": payload.get("timed_out"),
                }
                stderr = payload.get("stderr", "")
                if stderr:
                    summary["stderr_excerpt"] = stderr[:500]
                evidence_parts.append(
                    f"### Run Result: {run_file.name}\n```json\n{json.dumps(summary, indent=2)}\n```"
                )

    # 3. Read refinement log for actual iteration count
    refine_log_text = _read_prior_artifact(run_dir, "refinement_log.json")
    if refine_log_text:
        try:
            rlog = json.loads(refine_log_text)
            summary = {
                "iterations_executed": len(rlog.get("iterations", [])),
                "converged": rlog.get("converged"),
                "stop_reason": rlog.get("stop_reason"),
                "best_metric": rlog.get("best_metric"),
            }
            evidence_parts.append(
                f"### Refinement Summary\n```json\n{json.dumps(summary, indent=2)}\n```"
            )
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Count actual number of experiment runs
    actual_run_count = 0
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for rf in stage_subdir.glob("*.json"):
            if rf.name != "results.json":
                actual_run_count += 1
    if actual_run_count > 0:
        evidence_parts.append(
            f"### Actual Trial Count\n"
            f"**The experiment was executed {actual_run_count} time(s).** "
            f"If the paper claims a different number of trials, this is a CRITICAL discrepancy."
        )

    if not evidence_parts:
        return ""

    return (
        "\n\n## Actual Experiment Evidence\n"
        "Use the evidence below to verify the paper's methodology claims.\n\n"
        + "\n\n".join(evidence_parts)
    )


def _execute_peer_review(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    draft = _read_prior_artifact(run_dir, "paper_draft.md") or ""
    experiment_evidence = _collect_experiment_evidence(run_dir)

    # Load draft quality warnings from Stage 17 (if available)
    _quality_suffix = ""
    _quality_json_path = _find_prior_file(run_dir, "draft_quality.json")
    if _quality_json_path and _quality_json_path.exists():
        try:
            _dq = json.loads(_quality_json_path.read_text(encoding="utf-8"))
            _dq_warnings = _dq.get("overall_warnings", [])
            if _dq_warnings:
                _quality_suffix = (
                    "\n\nAUTOMATED QUALITY ISSUES (flag these in your review):\n"
                    + "\n".join(f"- {w}" for w in _dq_warnings)
                    + "\n"
                )
        except Exception:  # noqa: BLE001
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "peer_review")
        sp = _pm.for_stage(
            "peer_review",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            draft=draft,
            experiment_evidence=experiment_evidence,
        )
        _review_user = sp.user + _quality_suffix
        resp = _chat_with_prompt(
            llm,
            sp.system,
            _review_user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        reviews = resp.content
    else:
        reviews = """# Reviews

## Reviewer A
- Strengths: Clear problem statement.
- Weaknesses: Limited ablation details.
- Actionable revisions: Add uncertainty analysis and stronger baselines.

## Reviewer B
- Strengths: Reproducibility focus.
- Weaknesses: Discussion underdeveloped.
- Actionable revisions: Expand limitations and broader impact.
"""
    (stage_dir / "reviews.md").write_text(reviews, encoding="utf-8")
    return StageResult(
        stage=Stage.PEER_REVIEW,
        status=StageStatus.DONE,
        artifacts=("reviews.md",),
        evidence_refs=("stage-18/reviews.md",),
    )


def _execute_paper_revision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    draft = _read_prior_artifact(run_dir, "paper_draft.md") or ""
    reviews = _read_prior_artifact(run_dir, "reviews.md") or ""
    draft_word_count = len(draft.split())

    # R4-2: Collect real metrics for anti-fabrication guard in revision
    # BUG-47: _collect_raw_experiment_metrics returns tuple[str, bool], must unpack
    _raw_metrics_tuple = _collect_raw_experiment_metrics(run_dir)
    raw_metrics_revision = _raw_metrics_tuple[0] if isinstance(_raw_metrics_tuple, tuple) else (_raw_metrics_tuple or "")
    data_integrity_revision = ""
    if raw_metrics_revision:
        data_integrity_revision = (
            raw_metrics_revision
            + "\nDATA INTEGRITY: Do NOT add new numbers that are not in the "
            "experiment data above. If a reviewer asks for additional results "
            "you do not have, state 'Due to computational constraints, "
            "this analysis was not conducted' instead of fabricating data.\n"
        )

    if llm is not None:
        _pm = prompts or PromptManager()
        try:
            _ws_revision = _pm.block("writing_structure")
        except (KeyError, Exception):  # noqa: BLE001
            _ws_revision = ""
        # IMP-20/25/31/24: Load style blocks for revision prompt
        _rev_blocks: dict[str, str] = {}
        for _bname in ("academic_style_guide", "narrative_writing_rules",
                        "anti_hedging_rules", "anti_repetition_rules"):
            try:
                _rev_blocks[_bname] = _pm.block(_bname)
            except (KeyError, Exception):  # noqa: BLE001
                _rev_blocks[_bname] = ""
        # Load draft quality directives from Stage 17
        _quality_prefix = ""
        _quality_json_path = _find_prior_file(run_dir, "draft_quality.json")
        if _quality_json_path and _quality_json_path.exists():
            try:
                _dq = json.loads(_quality_json_path.read_text(encoding="utf-8"))
                _dq_directives = _dq.get("revision_directives", [])
                if _dq_directives:
                    _quality_prefix = (
                        "MANDATORY QUALITY FIXES (address ALL of these):\n"
                        + "\n".join(f"- {d}" for d in _dq_directives)
                        + "\n\n"
                    )
            except Exception:  # noqa: BLE001
                pass

        _overlay = _get_evolution_overlay(run_dir, "paper_revision")
        sp = _pm.for_stage(
            "paper_revision",
            evolution_overlay=_overlay,
            topic_constraint=_pm.block("topic_constraint", topic=config.research.topic),
            writing_structure=_ws_revision,
            draft=draft,
            reviews=_quality_prefix + reviews + data_integrity_revision,
            **_rev_blocks,
        )
        # R10-Fix2: Ensure max_tokens is sufficient for full paper revision
        revision_max_tokens = sp.max_tokens
        if revision_max_tokens and draft_word_count > 0:
            # ~1.5 tokens per word, 20% headroom
            min_tokens_needed = int(draft_word_count * 1.5 * 1.2)
            if revision_max_tokens < min_tokens_needed:
                revision_max_tokens = min_tokens_needed
                logger.info(
                    "Stage 19: Increased max_tokens from %d to %d to fit full paper revision",
                    sp.max_tokens,
                    revision_max_tokens,
                )

        # R10-Fix4: Retry on timeout for paper revision (critical stage)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=revision_max_tokens,
            retries=2,
        )
        revised = resp.content
        revised_word_count = len(revised.split())
        # Length guard: if revision is shorter than 80% of draft, retry once
        if draft_word_count > 500 and revised_word_count < int(draft_word_count * 0.8):
            logger.warning(
                "Paper revision (%d words) is shorter than draft (%d words). "
                "Retrying with stronger length enforcement.",
                revised_word_count,
                draft_word_count,
            )
            retry_user = (
                f"CRITICAL LENGTH REQUIREMENT: The draft is {draft_word_count} words. "
                f"Your revision MUST be at least {draft_word_count} words — ideally longer. "
                f"Do NOT summarize or condense ANY section. Copy each section verbatim "
                f"and ONLY make targeted improvements to address reviewer comments. "
                f"If a section has no reviewer comments, include it UNCHANGED.\n\n"
                + sp.user
            )
            resp2 = _chat_with_prompt(
                llm, sp.system, retry_user,
                json_mode=sp.json_mode, max_tokens=revision_max_tokens,
            )
            revised2 = resp2.content
            revised2_word_count = len(revised2.split())
            if revised2_word_count >= int(draft_word_count * 0.8):
                revised = revised2
            elif revised2_word_count > revised_word_count:
                # Retry improved but still not enough — use the longer version
                revised = revised2
                logger.warning(
                    "Retry improved (%d → %d words) but still shorter than draft (%d).",
                    revised_word_count,
                    revised2_word_count,
                    draft_word_count,
                )
            else:
                # Both attempts produced short output — preserve full original draft
                logger.warning(
                    "Retry also produced short output (%d words). "
                    "Falling back to FULL ORIGINAL DRAFT to prevent content loss.",
                    revised2_word_count,
                )
                # Extract useful revision points as appendix
                revision_words = revised.split()
                revision_summary = (
                    " ".join(revision_words[:500]) + "\n\n*(Revision summary truncated)*"
                    if len(revision_words) > 500
                    else revised
                )
                if revision_summary.strip():
                    # Save revision notes to internal file, not paper body
                    (stage_dir / "revision_notes_internal.md").write_text(
                        revision_summary, encoding="utf-8"
                    )
                revised = draft
    else:
        revised = draft
    (stage_dir / "paper_revised.md").write_text(revised, encoding="utf-8")
    return StageResult(
        stage=Stage.PAPER_REVISION,
        status=StageStatus.DONE,
        artifacts=("paper_revised.md",),
        evidence_refs=("stage-19/paper_revised.md",),
    )


def _execute_quality_gate(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    revised = _read_prior_artifact(run_dir, "paper_revised.md") or ""
    report: dict[str, Any] | None = None

    # BUG-25: Load experiment summary for cross-checking
    _exp_summary_text = _read_prior_artifact(run_dir, "experiment_summary.json") or ""
    _exp_summary = _safe_json_loads(_exp_summary_text, {}) if _exp_summary_text else {}
    _exp_failed = False
    if isinstance(_exp_summary, dict):
        _best_run = _exp_summary.get("best_run", {})
        if isinstance(_best_run, dict):
            _exp_failed = (
                _best_run.get("status") == "failed"
                and not _best_run.get("metrics")
            )
        # Also check if metrics_summary is empty
        if not _exp_summary.get("metrics_summary"):
            _exp_failed = True

    if llm is not None:
        _pm = prompts or PromptManager()
        # IMP-33: Evaluate the full paper instead of truncating to 12K chars.
        # Split into chunks if very long, but prefer sending the full text.
        paper_for_eval = revised[:40000] if len(revised) > 40000 else revised

        # BUG-25: Inject experiment status into quality gate prompt
        _exp_context = ""
        if _exp_summary and isinstance(_exp_summary, dict):
            _exp_status_keys = {
                k: _exp_summary.get(k) for k in (
                    "total_conditions", "total_metric_keys",
                    "metrics_summary",
                ) if _exp_summary.get(k) is not None
            }
            if _best_run := _exp_summary.get("best_run"):
                _exp_status_keys["best_run_status"] = (
                    _best_run.get("status") if isinstance(_best_run, dict) else str(_best_run)
                )
            _exp_context = (
                "\n\nExperiment summary (for cross-checking reported numbers):\n"
                + json.dumps(_exp_status_keys, indent=2, default=str)[:4000]
                + "\n\nCross-check: If the experiment status is 'failed' with "
                "empty metrics, any numerical results in tables constitute "
                "fabrication. Penalize severely.\n"
            )

        _overlay = _get_evolution_overlay(run_dir, "quality_gate")
        sp = _pm.for_stage(
            "quality_gate",
            evolution_overlay=_overlay,
            quality_threshold=str(config.research.quality_threshold),
            revised=paper_for_eval + _exp_context,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        parsed = _safe_json_loads(resp.content, {})
        if isinstance(parsed, dict):
            report = parsed
    # BUG-25: If experiment failed with no metrics, cap the quality score
    if report is not None and _exp_failed:
        _orig_score = report.get("score_1_to_10", 5)
        if isinstance(_orig_score, (int, float)) and _orig_score > 3:
            report["score_1_to_10"] = min(_orig_score, 3.0)
            report.setdefault("weaknesses", []).append(
                "Experiment failed with no metrics — any reported numerical "
                "results are unsupported and likely fabricated."
            )
            logger.warning(
                "BUG-25: Experiment failed — capping quality score from %.1f to 3.0",
                _orig_score,
            )
    if report is None:
        report = _default_quality_report(config.research.quality_threshold)
    report.setdefault("generated", _utcnow_iso())
    (stage_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # T2.1: Enforce quality gate — fail if score below threshold
    score = report.get("score_1_to_10", 0)
    verdict = report.get("verdict", "proceed")
    threshold = config.research.quality_threshold or 5.0

    # --- Fabrication flag: collect real metrics for Stage 22 sanitization ---
    _fabrication_info: dict[str, Any] = {
        "experiment_failed": _exp_failed,
        "quality_score": score,
        "real_metric_values": [],
    }
    if isinstance(_exp_summary, dict):
        # Collect ALL real numeric values from experiment_summary.json
        _cond_summaries = _exp_summary.get("condition_summaries", {})
        if isinstance(_cond_summaries, dict):
            for cond_name, cond_data in _cond_summaries.items():
                if not isinstance(cond_data, dict):
                    continue
                cond_status = cond_data.get("status", "")
                if cond_status == "failed":
                    continue  # skip failed conditions
                for k, v in cond_data.items():
                    if isinstance(v, (int, float)) and k not in (
                        "seed_count", "total_steps", "training_steps",
                    ):
                        _fabrication_info["real_metric_values"].append(
                            round(float(v), 4)
                        )
        _ms = _exp_summary.get("metrics_summary", {})
        if isinstance(_ms, dict):
            for _mk, _mv in _ms.items():
                if isinstance(_mv, dict):
                    for _stat in ("mean", "min", "max"):
                        _sv = _mv.get(_stat)
                        if isinstance(_sv, (int, float)):
                            _fabrication_info["real_metric_values"].append(
                                round(float(_sv), 4)
                            )
    _fabrication_info["has_real_data"] = bool(
        _fabrication_info["real_metric_values"]
    )
    _fabrication_info["fabrication_suspected"] = (
        _exp_failed and not _fabrication_info["has_real_data"]
    )
    (stage_dir / "fabrication_flags.json").write_text(
        json.dumps(_fabrication_info, indent=2), encoding="utf-8"
    )

    if isinstance(score, (int, float)) and score < threshold:
        if config.research.graceful_degradation:
            logger.warning(
                "Quality gate DEGRADED: score %.1f < threshold %.1f — "
                "continuing with sanitization (graceful_degradation=True)",
                score, threshold,
            )
            # Write degradation signal for downstream stages
            signal = {
                "score": score,
                "threshold": threshold,
                "verdict": verdict,
                "weaknesses": report.get("weaknesses", []),
                "generated": _utcnow_iso(),
            }
            (run_dir / "degradation_signal.json").write_text(
                json.dumps(signal, indent=2), encoding="utf-8"
            )
            return StageResult(
                stage=Stage.QUALITY_GATE,
                status=StageStatus.DONE,
                artifacts=("quality_report.json",),
                evidence_refs=("stage-20/quality_report.json",),
                decision="degraded",
            )
        logger.warning(
            "Quality gate FAILED: score %.1f < threshold %.1f (verdict=%s)",
            score, threshold, verdict,
        )
        return StageResult(
            stage=Stage.QUALITY_GATE,
            status=StageStatus.FAILED,
            artifacts=("quality_report.json", "fabrication_flags.json"),
            evidence_refs=("stage-20/quality_report.json",),
            error=f"Quality score {score:.1f}/10 below threshold {threshold:.1f}. "
                  f"Paper needs revision before export.",
        )

    logger.info(
        "Quality gate PASSED: score %.1f >= threshold %.1f",
        score, threshold,
    )
    return StageResult(
        stage=Stage.QUALITY_GATE,
        status=StageStatus.DONE,
        artifacts=("quality_report.json", "fabrication_flags.json"),
        evidence_refs=("stage-20/quality_report.json",),
    )


def _execute_knowledge_archive(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    revised = _read_prior_artifact(run_dir, "paper_revised.md") or ""
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
    decision = _read_prior_artifact(run_dir, "decision.md") or ""
    preamble = _build_context_preamble(config, run_dir, include_goal=True)
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "knowledge_archive")
        sp = _pm.for_stage(
            "knowledge_archive",
            evolution_overlay=_overlay,
            preamble=preamble,
            decision=decision,
            analysis=analysis,
            revised=revised[:15000],
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        archive = resp.content
    else:
        archive = f"""# Knowledge Archive

## Lessons Learned
- Preserve strict metric reporting protocol.
- Keep refinement logs aligned with code changes.

## Reproducibility
- Include exact experiment script and schedule.
- Capture run-level JSON metrics.

## Future Work
- Extend robustness and external validity checks.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "archive.md").write_text(archive, encoding="utf-8")

    files: list[str] = []
    for stage_subdir in sorted(run_dir.glob("stage-*")):
        for artifact in sorted(stage_subdir.rglob("*")):
            if artifact.is_file() and artifact != (stage_dir / "bundle_index.json"):
                files.append(str(artifact.relative_to(run_dir)))
    index = {
        "run_id": run_dir.name,
        "generated": _utcnow_iso(),
        "artifact_count": len(files),
        "artifacts": files,
    }
    (stage_dir / "bundle_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    return StageResult(
        stage=Stage.KNOWLEDGE_ARCHIVE,
        status=StageStatus.DONE,
        artifacts=("archive.md", "bundle_index.json"),
        evidence_refs=("stage-21/archive.md", "stage-21/bundle_index.json"),
    )


def _sanitize_fabricated_data(
    paper: str,
    run_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Replace unverified numerical data in markdown tables with '---'.

    Loads experiment_summary.json as ground truth, extracts all verified
    metric values, then scans markdown tables in Results/Experiment sections.
    Numbers not matching any verified value (within 1% relative tolerance)
    are replaced with ``---``.

    Returns (sanitized_paper, sanitization_report).
    """
    import re as _re_san

    # --- 1. Build verified values set from experiment_summary.json ---
    verified_values: set[float] = set()
    exp_path = run_dir / "stage-14" / "experiment_summary.json"
    if not exp_path.exists():
        # Try other common locations
        for candidate in sorted(run_dir.glob("stage-14*/experiment_summary.json")):
            exp_path = candidate
            break

    if exp_path.exists():
        try:
            exp_data = json.loads(exp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            exp_data = {}

        def _collect_numbers(obj: Any, depth: int = 0) -> None:
            if depth > 10:
                return
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                verified_values.add(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect_numbers(v, depth + 1)
            elif isinstance(obj, list):
                for v in obj:
                    _collect_numbers(v, depth + 1)

        # Extract from well-known keys
        for key in (
            "metrics_summary", "condition_summaries", "best_run",
            "condition_metrics", "conditions", "ablation_results",
        ):
            if key in exp_data:
                _collect_numbers(exp_data[key])

    if not verified_values:
        report: dict[str, Any] = {
            "sanitized": False,
            "reason": "no verified values found in experiment_summary.json",
            "tables_processed": 0,
            "numbers_replaced": 0,
        }
        return paper, report

    def _is_verified(num: float) -> bool:
        """Check if num matches any verified value within 1% relative tolerance."""
        for v in verified_values:
            if v == 0.0:
                if abs(num) < 1e-9:
                    return True
            elif abs(num - v) / abs(v) <= 0.01:
                return True
        return False

    # --- 2. Find and sanitize markdown tables ---
    # Match markdown table blocks (header + separator + data rows)
    table_pat = _re_san.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)+"  # one or more pipe-delimited lines
        r")",
        _re_san.MULTILINE,
    )
    # Match numbers in table cells (integers, decimals, percentages, scientific)
    num_pat = _re_san.compile(
        r"(?<![a-zA-Z_])"  # not preceded by letter/underscore
        r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
        r"(%?)"  # optional percent
        r"(?![a-zA-Z_])"  # not followed by letter/underscore
    )

    numbers_replaced = 0
    numbers_kept = 0
    tables_processed = 0
    replaced_values: list[str] = []

    def _sanitize_table(match: _re_san.Match[str]) -> str:
        nonlocal numbers_replaced, numbers_kept, tables_processed
        table_text = match.group(0)
        lines = table_text.split("\n")

        # Check if this looks like a results/experiment table
        # (heuristic: has a separator row with dashes)
        has_separator = any(
            _re_san.match(r"^[ \t]*\|[\s:|-]+\|[ \t]*$", line)
            for line in lines
        )
        if not has_separator:
            return table_text

        tables_processed += 1
        sanitized_lines: list[str] = []
        for i, line in enumerate(lines):
            # Skip header row and separator row
            is_separator = bool(
                _re_san.match(r"^[ \t]*\|[\s:|-]+\|[ \t]*$", line)
            )
            is_header = i == 0  # first line is typically the header
            if is_separator or is_header:
                sanitized_lines.append(line)
                continue

            def _replace_num(m: _re_san.Match[str]) -> str:
                nonlocal numbers_replaced, numbers_kept
                num_str = m.group(1)
                pct = m.group(2)
                try:
                    val = float(num_str)
                except ValueError:
                    return m.group(0)
                if _is_verified(val):
                    numbers_kept += 1
                    return m.group(0)
                numbers_replaced += 1
                replaced_values.append(num_str + pct)
                return "---"

            sanitized_lines.append(num_pat.sub(_replace_num, line))
        return "\n".join(sanitized_lines)

    sanitized = table_pat.sub(_sanitize_table, paper)

    report = {
        "sanitized": numbers_replaced > 0,
        "tables_processed": tables_processed,
        "numbers_replaced": numbers_replaced,
        "numbers_kept": numbers_kept,
        "verified_values_count": len(verified_values),
        "replaced_samples": replaced_values[:20],
        "generated": _utcnow_iso(),
    }
    return sanitized, report


def _execute_export_publish(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    revised = _read_prior_artifact(run_dir, "paper_revised.md") or ""
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "export_publish")
        sp = _pm.for_stage("export_publish", evolution_overlay=_overlay, revised=revised)
        try:
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens,
            )
            final_paper = resp.content
            # Content guard: reject LLM output that truncates the paper
            if revised and len(final_paper) < 0.6 * len(revised):
                logger.warning(
                    "Stage 22: LLM output is %.0f%% of input length — using original",
                    100 * len(final_paper) / max(len(revised), 1),
                )
                final_paper = revised
        except Exception:  # noqa: BLE001
            # ACP session disconnect or other transient LLM failure — the paper
            # itself is complete; just use the revised draft directly rather than
            # failing the entire stage and losing all prior work.
            logger.warning(
                "Stage 22: LLM call failed — falling back to paper_revised.md"
            )
            final_paper = revised
    else:
        final_paper = revised
    if not final_paper.strip():
        final_paper = "# Final Paper\n\nNo content generated."

    # --- Graceful degradation: sanitize fabricated data ---
    _degradation_signal_path = run_dir / "degradation_signal.json"
    if _degradation_signal_path.exists():
        try:
            _deg_signal = json.loads(
                _degradation_signal_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            _deg_signal = {}

        # Back up pre-sanitized version
        (stage_dir / "paper_presanitized.md").write_text(
            final_paper, encoding="utf-8"
        )

        # Sanitize unverified data in tables
        final_paper, _san_report = _sanitize_fabricated_data(
            final_paper, run_dir
        )
        (stage_dir / "sanitization_report.json").write_text(
            json.dumps(_san_report, indent=2), encoding="utf-8"
        )

        # Insert degradation notice after abstract
        _deg_score = _deg_signal.get("score", "N/A")
        _deg_threshold = _deg_signal.get("threshold", "N/A")
        _deg_notice = (
            "\n\n> **Note:** This paper was produced in degraded mode. "
            f"Quality gate score ({_deg_score}/{_deg_threshold}) was below "
            "threshold. Unverified numerical results in tables have been "
            "replaced with `---` and require independent verification.\n\n"
        )
        # Try to insert after ## Abstract section
        _abstract_markers = ["## Abstract\n", "# Abstract\n"]
        _notice_inserted = False
        for _marker in _abstract_markers:
            if _marker in final_paper:
                _marker_end = final_paper.index(_marker) + len(_marker)
                # Find the end of the abstract paragraph
                _next_section = final_paper.find("\n## ", _marker_end)
                _next_heading = final_paper.find("\n# ", _marker_end)
                _insert_pos = min(
                    p for p in (_next_section, _next_heading)
                    if p > 0
                ) if any(p > 0 for p in (_next_section, _next_heading)) else len(final_paper)
                final_paper = (
                    final_paper[:_insert_pos]
                    + _deg_notice
                    + final_paper[_insert_pos:]
                )
                _notice_inserted = True
                break
        if not _notice_inserted:
            # Fallback: prepend to paper
            final_paper = _deg_notice + final_paper

        logger.info(
            "Stage 22: Applied degraded-mode sanitization — "
            "%d numbers replaced, %d kept",
            _san_report.get("numbers_replaced", 0),
            _san_report.get("numbers_kept", 0),
        )

    # IMP-3: Deduplicate "due to computational constraints" — keep at most 1
    import re as _re_imp3
    _CONSTRAINT_PAT = _re_imp3.compile(
        r"[Dd]ue to computational constraints", _re_imp3.IGNORECASE
    )
    _matches = list(_CONSTRAINT_PAT.finditer(final_paper))
    if len(_matches) > 1:
        # Keep only the first occurrence; remove subsequent ones by
        # deleting the enclosing sentence.
        for m in reversed(_matches[1:]):
            # Find sentence boundaries around the match
            start = final_paper.rfind(".", 0, m.start())
            start = start + 1 if start >= 0 else m.start()
            end = final_paper.find(".", m.end())
            end = end + 1 if end >= 0 else m.end()
            sentence = final_paper[start:end].strip()
            if sentence:
                final_paper = final_paper[:start] + final_paper[end:]
        final_paper = re.sub(r"[^\S\n]{2,}", " ", final_paper)
        logger.info(
            "Stage 22: Removed %d duplicate 'computational constraints' "
            "disclaimers",
            len(_matches) - 1,
        )

    # IMP-19 Layer 2: Ensure at least figures are referenced in the paper
    import re as _re_fig
    chart_files = []
    for _chart_src_dir in [stage_dir / "charts", run_dir / "stage-14" / "charts"]:
        if _chart_src_dir.is_dir():
            chart_files.extend(sorted(_chart_src_dir.glob("*.png")))
    if chart_files and "![" not in final_paper:
        # Distribute figures to relevant sections based on filename keywords
        _fig_placement: dict[str, list[str]] = {
            "method": [],       # architecture, method, model, pipeline diagrams
            "result": [],       # experiment, comparison, ablation charts
            "intro": [],        # concept, overview, illustration
        }
        _fig_counter = 0
        for cf in chart_files[:6]:
            _fig_counter += 1
            stem_lower = cf.stem.lower()
            label = cf.stem.replace("_", " ").title()
            fig_md = f"![Figure {_fig_counter}: {label}](charts/{cf.name})"
            if any(k in stem_lower for k in ("architecture", "model", "pipeline", "method", "flowchart")):
                _fig_placement["method"].append(fig_md)
            elif any(k in stem_lower for k in ("experiment", "comparison", "ablation", "result", "metric")):
                _fig_placement["result"].append(fig_md)
            elif any(k in stem_lower for k in ("concept", "overview", "illustration", "threat", "attack")):
                _fig_placement["intro"].append(fig_md)
            else:
                _fig_placement["result"].append(fig_md)  # default to results

        # Insert figures at relevant section boundaries
        _section_markers = {
            "method": ["## Method", "## Methodology", "## Approach", "## Framework",
                        "## 3. Method", "## 3 Method"],
            "result": ["## Results", "## Experiments", "## Evaluation",
                        "## 5. Results", "## 4. Experiments", "## 5 Results"],
            "intro": ["## Related Work", "## Background", "## 2. Related",
                       "## 2 Related Work"],
        }
        _total_inserted = 0
        for category, figs in _fig_placement.items():
            if not figs:
                continue
            fig_block = "\n\n" + "\n\n".join(figs) + "\n\n"
            inserted = False
            for marker in _section_markers.get(category, []):
                if marker in final_paper:
                    # Insert BEFORE the marker section (so figure appears at end of previous section)
                    final_paper = final_paper.replace(marker, fig_block + marker, 1)
                    inserted = True
                    _total_inserted += len(figs)
                    break
            if not inserted:
                # Fallback: insert before Conclusion/Limitations/Discussion
                for fallback in ["## Conclusion", "## Limitations", "## Discussion"]:
                    if fallback in final_paper:
                        final_paper = final_paper.replace(fallback, fig_block + fallback, 1)
                        inserted = True
                        _total_inserted += len(figs)
                        break
            if not inserted:
                final_paper += fig_block
                _total_inserted += len(figs)

        logger.info(
            "IMP-19: Injected %d figure references into paper_final.md (distributed across sections)",
            _total_inserted,
        )

    # IMP-24: Detect excessive number repetition
    _numbers_found = _re_fig.findall(r"\b\d+\.\d{2,}\b", final_paper)
    from collections import Counter as _Counter
    _num_counts = _Counter(_numbers_found)
    _repeated = {n: c for n, c in _num_counts.items() if c > 3}
    if _repeated:
        logger.warning(
            "IMP-24: Numbers repeated >3 times: %s",
            _repeated,
        )

    (stage_dir / "paper_final.md").write_text(final_paper, encoding="utf-8")

    # --- Fabrication sanitization: blank out unsupported numbers ---
    _fab_flags_text = _read_prior_artifact(run_dir, "fabrication_flags.json") or ""
    _fab_flags = _safe_json_loads(_fab_flags_text, {}) if _fab_flags_text else {}
    if isinstance(_fab_flags, dict) and _fab_flags.get("fabrication_suspected"):
        import re as _re_fab
        _real_vals = set()
        for rv in _fab_flags.get("real_metric_values", []):
            if isinstance(rv, (int, float)) and math.isfinite(rv):
                _real_vals.add(str(round(rv, 4)))
                _real_vals.add(str(round(rv, 2)))
                _real_vals.add(str(round(rv, 1)))
                _real_vals.add(str(int(rv)) if rv == int(rv) else "")

        def _sanitize_number(m: _re_fab.Match) -> str:  # type: ignore[name-defined]
            """Replace fabricated numbers with '--' but keep real ones."""
            num_str = m.group(0)
            # Keep the number if it matches any known real metric value
            try:
                num_val = float(num_str)
                if not math.isfinite(num_val):
                    return "--"
                rounded_strs = {
                    str(round(num_val, 4)),
                    str(round(num_val, 2)),
                    str(round(num_val, 1)),
                    str(int(num_val)) if num_val == int(num_val) else "",
                }
                if rounded_strs & _real_vals:
                    return num_str  # real value — keep it
            except (ValueError, OverflowError):
                return num_str
            return "--"

        # Only sanitize numbers in Results/Experiments/Evaluation/Ablation sections
        _result_section_pat = _re_fab.compile(
            r"(##\s*(?:\d+\.?\s*)?(?:Results|Experiments|Evaluation|Ablation"
            r"|Experimental Results|Quantitative).*?)(?=\n##\s|\Z)",
            _re_fab.DOTALL | _re_fab.IGNORECASE,
        )
        _sanitized_count = 0

        def _sanitize_section(sec_match: _re_fab.Match) -> str:  # type: ignore[name-defined]
            nonlocal _sanitized_count
            section_text = sec_match.group(0)
            # Replace decimal numbers (e.g., 73.42, 0.891) but NOT integers
            # that are likely structural (year, section number, figure number)
            def _replace_in_section(m: _re_fab.Match) -> str:  # type: ignore[name-defined]
                nonlocal _sanitized_count
                result = _sanitize_number(m)
                if result == "--":
                    _sanitized_count += 1
                return result
            return _re_fab.sub(
                r"\b\d+\.\d{1,6}\b", _replace_in_section, section_text
            )

        final_paper = _result_section_pat.sub(_sanitize_section, final_paper)

        if _sanitized_count > 0:
            logger.warning(
                "Stage 22: Fabrication sanitization — blanked %d unsupported "
                "numbers in Results sections (experiment had no real metrics)",
                _sanitized_count,
            )
            # Rewrite the sanitized paper
            (stage_dir / "paper_final.md").write_text(
                final_paper, encoding="utf-8"
            )

    # Initialize artifacts list
    artifacts = ["paper_final.md"]
    # F2.7: Post-process citations — [cite_key] → \cite{cite_key}
    # and copy final references.bib to export stage
    bib_text = _read_prior_artifact(run_dir, "references.bib")
    if bib_text:
        # Replace [cite_key] patterns in the final paper with \cite{cite_key}
        # Collect all valid cite_keys from the bib file
        import re as _re

        valid_keys = set(_re.findall(r"@\w+\{([^,]+),", bib_text))

        # R10-Fix4: Citation cross-validation
        cited_keys_in_paper = set(_re.findall(r"\[([a-z]+\d{4}[a-z]*)\]", final_paper))
        if valid_keys and cited_keys_in_paper:
            invalid_keys = cited_keys_in_paper - valid_keys
            if invalid_keys:
                logger.warning(
                    "Stage 22: Found %d citation keys in paper not in references.bib: %s",
                    len(invalid_keys),
                    ", ".join(sorted(invalid_keys)[:20]),
                )
                # IMP-29: Silently remove invalid citations instead of
                # leaving ugly [?key:NOT_IN_BIB] markers in the output.
                for bad_key in invalid_keys:
                    final_paper = final_paper.replace(f"[{bad_key}]", "")
                # Clean up whitespace artifacts from removed citations
                import re as _re_imp29
                final_paper = _re_imp29.sub(r"  +", " ", final_paper)
                final_paper = _re_imp29.sub(r" ([.,;:)])", r"\1", final_paper)
                (stage_dir / "paper_final.md").write_text(final_paper, encoding="utf-8")
                (stage_dir / "invalid_citations.json").write_text(
                    json.dumps(sorted(invalid_keys), indent=2), encoding="utf-8"
                )
                artifacts.append("invalid_citations.json")

        if valid_keys:
            _CITE_KEY_PAT = r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9]*"

            # Step 1: Convert multi-key brackets [key1, key2] → \cite{key1, key2}
            def _replace_multi_cite(m: _re.Match[str]) -> str:
                keys = [k.strip() for k in m.group(1).split(",")]
                matched = [k for k in keys if k in valid_keys]
                if matched:
                    return "\\cite{" + ", ".join(matched) + "}"
                return m.group(0)

            final_paper_latex = _re.sub(
                rf"\[({_CITE_KEY_PAT}(?:\s*,\s*{_CITE_KEY_PAT})+)\]",
                _replace_multi_cite,
                final_paper,
            )

            # Step 2: Convert single-key brackets [key] → \cite{key}
            def _replace_cite(m: _re.Match[str]) -> str:
                key = m.group(1)
                if key in valid_keys:
                    return f"\\cite{{{key}}}"
                return m.group(0)

            final_paper_latex = _re.sub(
                rf"\[({_CITE_KEY_PAT})\]", _replace_cite, final_paper_latex
            )

            # Step 3: Merge adjacent \cite{a} \cite{b} → \cite{a, b}
            def _merge_adjacent_cites(m: _re.Match[str]) -> str:
                keys = _re.findall(r"\\cite\{([^}]+)\}", m.group(0))
                return "\\cite{" + ", ".join(keys) + "}"

            final_paper_latex = _re.sub(
                r"\\cite\{[^}]+\}(?:\s*\\cite\{[^}]+\})+",
                _merge_adjacent_cites,
                final_paper_latex,
            )

            (stage_dir / "paper_final_latex.md").write_text(
                final_paper_latex, encoding="utf-8"
            )
            artifacts.append("paper_final_latex.md")
        # IMP-1: Prune uncited bibliography entries — keep only keys
        # that actually appear in the paper text (bracket or \cite form).
        if valid_keys:
            _all_cited: set[str] = set()
            # Bracket-format citations [key]
            _all_cited.update(
                _re.findall(r"\[([a-z]+\d{4}[a-z]*)\]", final_paper)
            )
            # \cite{key, key2} format (original + latex-converted)
            for _src in (
                final_paper,
                locals().get("final_paper_latex", ""),
            ):
                for _cm in _re.finditer(r"\\cite\{([^}]+)\}", _src):
                    _all_cited.update(
                        k.strip() for k in _cm.group(1).split(",")
                    )
            uncited_keys = valid_keys - _all_cited
            if uncited_keys:
                bib_text = _remove_bibtex_entries(bib_text, uncited_keys)
                logger.info(
                    "Stage 22: Pruned %d uncited bibliography entries "
                    "(kept %d)",
                    len(uncited_keys),
                    len(valid_keys) - len(uncited_keys),
                )

        # Write final references.bib
        (stage_dir / "references.bib").write_text(bib_text, encoding="utf-8")
        artifacts.append("references.bib")
        logger.info(
            "Stage 22: Exported references.bib with %d entries",
            len(valid_keys) if valid_keys else 0,
        )

    # Conference template: generate .tex file
    try:
        from researchclaw.templates import get_template, markdown_to_latex

        tpl = get_template(config.export.target_conference)
        # Use the latex-citation-processed version if available
        tex_source = locals().get("final_paper_latex", final_paper)
        # Append NeurIPS-style checklist if target is a ML conference
        if tpl.name in ("neurips_2024", "neurips_2025", "icml_2025", "icml_2026",
                         "iclr_2025", "iclr_2026"):
            _has_exp = bool(_read_prior_artifact(run_dir, "experiment_summary.json"))
            _checklist = _generate_neurips_checklist(
                has_experiments=_has_exp,
                has_code=True,
            )
            if "NeurIPS Paper Checklist" not in tex_source:
                tex_source = tex_source.rstrip() + "\n\n" + _checklist
        tex_content = markdown_to_latex(
            tex_source,
            tpl,
            title=_extract_paper_title(tex_source),
            authors=config.export.authors,
            bib_file=config.export.bib_file,
        )
        (stage_dir / "paper.tex").write_text(tex_content, encoding="utf-8")
        artifacts.append("paper.tex")
        logger.info(
            "Stage 22: Generated paper.tex for %s (%d chars)",
            tpl.display_name,
            len(tex_content),
        )
        # Copy bundled style files alongside paper.tex
        for sf in tpl.get_style_files():
            import shutil as _shutil_sty
            _shutil_sty.copy2(sf, stage_dir / sf.name)
        # Compile verification
        try:
            from researchclaw.templates.compiler import compile_latex
            _compile_result = compile_latex(stage_dir / "paper.tex", max_attempts=2)
            if _compile_result.success:
                logger.info("Stage 22: LaTeX compilation verification PASSED")
                artifacts.append("paper.pdf")
                # PDF-as-reviewer: LLM-based visual review of compiled PDF
                _pdf_path = stage_dir / "paper.pdf"
                if _pdf_path.exists() and llm is not None:
                    try:
                        _pdf_review = _review_compiled_pdf(
                            _pdf_path, llm, config.research.topic
                        )
                        if _pdf_review:
                            (stage_dir / "pdf_review.json").write_text(
                                json.dumps(_pdf_review, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            artifacts.append("pdf_review.json")
                            _pdf_score = _pdf_review.get("overall_score", 0)
                            if _pdf_score < 5:
                                logger.warning(
                                    "Stage 22: PDF visual review score %d/10 — %s",
                                    _pdf_score,
                                    _pdf_review.get("summary", ""),
                                )
                            else:
                                logger.info(
                                    "Stage 22: PDF visual review score %d/10",
                                    _pdf_score,
                                )
                    except Exception as _pdf_exc:  # noqa: BLE001
                        logger.debug("Stage 22: PDF review skipped: %s", _pdf_exc)
                # Post-compilation quality checks
                try:
                    from researchclaw.templates.compiler import check_compiled_quality
                    _qc = check_compiled_quality(stage_dir / "paper.tex")
                    if _qc.warnings_summary:
                        logger.warning(
                            "Stage 22: Quality checks: %s",
                            "; ".join(_qc.warnings_summary),
                        )
                    (stage_dir / "compilation_quality.json").write_text(
                        json.dumps({
                            "page_count": _qc.page_count,
                            "unresolved_refs": _qc.unresolved_refs,
                            "unresolved_cites": _qc.unresolved_cites,
                            "overfull_hboxes": len(_qc.overfull_hboxes),
                            "orphan_figures": _qc.orphan_figures,
                            "orphan_labels": _qc.orphan_labels,
                            "warnings": _qc.warnings_summary,
                        }, indent=2),
                        encoding="utf-8",
                    )
                    artifacts.append("compilation_quality.json")
                    # BUG-27: Warn if page count exceeds limit
                    _page_limit = 10
                    if _qc.page_count and _qc.page_count > _page_limit:
                        logger.warning(
                            "BUG-27: Paper is %d pages (limit %d). "
                            "Consider tightening content in revision.",
                            _qc.page_count, _page_limit,
                        )
                except Exception as _qc_exc:  # noqa: BLE001
                    logger.debug("Stage 22: Quality checks skipped: %s", _qc_exc)
            else:
                logger.warning("Stage 22: LaTeX compilation verification FAILED: %s", _compile_result.errors[:3])
                # Add compilation failure comment to .tex
                _tex_path = stage_dir / "paper.tex"
                if _tex_path.exists():
                    _tex_content = _tex_path.read_text(encoding="utf-8")
                    if "% WARNING: Compilation failed" not in _tex_content:
                        _tex_content = (
                            "% WARNING: Compilation failed. Errors:\n"
                            + "".join(f"% {e}\n" for e in _compile_result.errors[:5])
                            + _tex_content
                        )
                        _tex_path.write_text(_tex_content, encoding="utf-8")
        except Exception as _compile_exc:  # noqa: BLE001
            logger.debug("Stage 22: Compile verification skipped: %s", _compile_exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LaTeX generation skipped: %s", exc)

    # WS-5.4: Generate result visualizations
    # Priority: FigureAgent charts (stage-14) > fallback visualize.py charts
    try:
        chart_dir = stage_dir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        charts: list[Path] = []

        # Check if FigureAgent produced charts in stage-14 (any version)
        _fa_charts_found = False
        for _fa_dir in sorted(run_dir.glob("stage-14*/charts"), reverse=True):
            _fa_pngs = list(_fa_dir.glob("fig_*.png"))
            if _fa_pngs:
                import shutil
                for _fa_png in _fa_pngs:
                    dest = chart_dir / _fa_png.name
                    shutil.copy2(_fa_png, dest)
                    charts.append(dest)
                _fa_charts_found = True
                logger.info(
                    "Stage 22: Copied %d FigureAgent charts from %s",
                    len(_fa_pngs), _fa_dir,
                )
                break

        # Always generate structured charts from visualize.py (different names)
        from researchclaw.experiment.visualize import generate_all_charts
        _metric_dir = getattr(config.experiment, "metric_direction", "minimize")
        _viz_charts = generate_all_charts(
            run_dir,
            chart_dir,
            metric_key=config.experiment.metric_key,
            metric_direction=_metric_dir,
        )
        charts.extend(_viz_charts)

        if charts:
            artifacts.append("charts/")
            logger.info("Stage 22: Generated %d chart(s) total", len(charts))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chart generation failed: %s", exc)

    # --- Code packaging: multi-file directory or single file ---
    exp_final_dir_path = _read_prior_artifact(run_dir, "experiment_final/")
    if exp_final_dir_path and Path(exp_final_dir_path).is_dir():
        import ast

        code_dir = stage_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        all_code_combined = ""
        code_file_names: list[str] = []
        for src in sorted(Path(exp_final_dir_path).glob("*.py")):
            (code_dir / src.name).write_bytes(src.read_bytes())
            all_code_combined += src.read_text(encoding="utf-8") + "\n"
            code_file_names.append(src.name)

        # Detect dependencies from all files
        detected: set[str] = set()
        known_packages = {
            "numpy": "numpy",
            "torch": "torch",
            "tensorflow": "tensorflow",
            "sklearn": "scikit-learn",
            "scikit-learn": "scikit-learn",
            "scipy": "scipy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "transformers": "transformers",
            "datasets": "datasets",
            "jax": "jax",
        }
        try:
            tree = ast.parse(all_code_combined)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in known_packages:
                            detected.add(known_packages[top])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if top in known_packages:
                        detected.add(known_packages[top])
        except SyntaxError:
            pass

        requirements = sorted(detected)
        (code_dir / "requirements.txt").write_text(
            "\n".join(requirements) + ("\n" if requirements else ""),
            encoding="utf-8",
        )

        paper_title = _extract_paper_title(final_paper)
        file_list_md = "\n".join(f"- `{f}`" for f in code_file_names)
        readme = (
            f"# Code Package for {paper_title}\n\n"
            "## Description\n"
            "This directory contains the experiment project used for the paper.\n\n"
            "## Project Files\n"
            f"{file_list_md}\n\n"
            "## How to Run\n"
            "`python main.py`\n\n"
            "## Dependencies\n"
            "Install dependencies with `pip install -r requirements.txt` if needed.\n"
        )
        (code_dir / "README.md").write_text(readme, encoding="utf-8")
        artifacts.append("code/")
        logger.info(
            "Stage 22: Packaged multi-file code release (%d files, %d deps)",
            len(code_file_names),
            len(requirements),
        )
    else:
        # Backward compat: single-file packaging
        code_payload = _read_prior_artifact(run_dir, "experiment_final.py")
        if not code_payload:
            code_payload = _read_prior_artifact(run_dir, "experiment.py")
        if code_payload:
            import ast

            code_dir = stage_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "experiment.py").write_text(code_payload, encoding="utf-8")

            detected_single: set[str] = set()
            known_packages_single = {
                "numpy": "numpy",
                "torch": "torch",
                "tensorflow": "tensorflow",
                "sklearn": "scikit-learn",
                "scikit-learn": "scikit-learn",
                "scipy": "scipy",
                "pandas": "pandas",
                "matplotlib": "matplotlib",
                "seaborn": "seaborn",
                "transformers": "transformers",
                "datasets": "datasets",
                "jax": "jax",
            }
            try:
                tree = ast.parse(code_payload)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            if top in known_packages_single:
                                detected_single.add(known_packages_single[top])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top = node.module.split(".")[0]
                        if top in known_packages_single:
                            detected_single.add(known_packages_single[top])
            except SyntaxError:
                pass

            requirements = sorted(detected_single)
            (code_dir / "requirements.txt").write_text(
                "\n".join(requirements) + ("\n" if requirements else ""),
                encoding="utf-8",
            )
            paper_title = _extract_paper_title(final_paper)
            readme = (
                f"# Code Package for {paper_title}\n\n"
                "## Description\n"
                "This directory contains the final experiment script used for the paper.\n\n"
                "## How to Run\n"
                "`python experiment.py`\n\n"
                "## Dependencies\n"
                "Install dependencies with `pip install -r requirements.txt` if needed.\n"
            )
            (code_dir / "README.md").write_text(readme, encoding="utf-8")
            artifacts.append("code/")
            logger.info(
                "Stage 22: Packaged single-file code release with %d deps",
                len(requirements),
            )
    # WS-5.5: Generate framework diagram prompt for methodology section
    try:
        _framework_prompt = _generate_framework_diagram_prompt(
            final_paper, config, llm=llm
        )
        if _framework_prompt:
            _chart_dir = stage_dir / "charts"
            _chart_dir.mkdir(parents=True, exist_ok=True)
            (_chart_dir / "framework_diagram_prompt.md").write_text(
                _framework_prompt, encoding="utf-8"
            )
            logger.info("Stage 22: Generated framework diagram prompt → charts/framework_diagram_prompt.md")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stage 22: Framework diagram prompt generation skipped: %s", exc)

    return StageResult(
        stage=Stage.EXPORT_PUBLISH,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-22/{a}" for a in artifacts),
    )


def _check_citation_relevance(
    llm: Any,
    topic: str,
    results: list[Any],
) -> dict[str, float | None]:
    """Use LLM to assess relevance of each citation to the research topic.

    Returns a dict mapping cite_key → relevance score (0.0–1.0).
    Processes citations in batches of 30 to handle large bibliographies.
    """
    citation_lines = []
    for cr in results:
        citation_lines.append(f"- [{cr.cite_key}] \"{cr.title}\"")
    if not citation_lines:
        return {}

    all_scores: dict[str, float] = {}
    _BATCH_SIZE = 30

    for batch_start in range(0, len(citation_lines), _BATCH_SIZE):
        batch = citation_lines[batch_start:batch_start + _BATCH_SIZE]
        citations_text = "\n".join(batch)

        prompt = (
            f"Research topic: {topic}\n\n"
            f"Rate the relevance of each citation to the research topic "
            f"on a scale of 0.0 to 1.0.\n"
            f"Return ONLY a JSON object mapping cite_key to relevance score.\n"
            f"Example: {{\"smith2020\": 0.9, \"jones2019\": 0.2}}\n\n"
            f"Citations:\n{citations_text}"
        )

        try:
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                system="You assess citation relevance. Return only valid JSON.",
                json_mode=True,
            )
            parsed = _safe_json_loads(resp.content, {})
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, (int, float)):
                        all_scores[k] = max(0.0, min(1.0, float(v)))
        except Exception:  # noqa: BLE001
            logger.debug(
                "Citation relevance check failed for batch %d–%d, skipping",
                batch_start, batch_start + len(batch),
            )

    return all_scores


def _remove_bibtex_entries(bib_text: str, keys_to_remove: set[str]) -> str:
    """Remove BibTeX entries whose keys are in *keys_to_remove*."""
    kept: list[str] = []
    for m in re.finditer(r"@\w+\{([^,]+),", bib_text):
        key = m.group(1).strip()
        if key in keys_to_remove:
            continue
        # Find the full entry (from @ to the next @ or end)
        start = m.start()
        # Find balanced braces
        depth = 0
        end = start
        for i in range(start, len(bib_text)):
            if bib_text[i] == "{":
                depth += 1
            elif bib_text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        kept.append(bib_text[start:end])
    return "\n\n".join(kept) + "\n" if kept else ""


def _remove_citations_from_text(text: str, keys_to_remove: set[str]) -> str:
    """Remove \\cite{key} and [key] references for specified citation keys."""

    # Handle multi-key LaTeX cites: \cite{a,b,c} → filter keys inside braces
    def _filter_cite(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        kept = [k for k in keys if k not in keys_to_remove]
        if not kept:
            return ""
        return f"\\cite{{{','.join(kept)}}}"

    text = re.sub(r"\\cite\{([^}]+)\}", _filter_cite, text)

    # Markdown: [key]
    for key in keys_to_remove:
        text = re.sub(rf"\[{re.escape(key)}\]", "", text)
    return text


def _execute_citation_verify(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.literature.verify import (
        VerifyStatus,
        annotate_paper_hallucinations,
        filter_verified_bibtex,
        verify_citations,
    )

    bib_text = _read_prior_artifact(run_dir, "references.bib") or ""
    paper_text = _read_prior_artifact(run_dir, "paper_final.md") or ""

    if not bib_text.strip():
        report_data = {
            "summary": {
                "total": 0,
                "verified": 0,
                "suspicious": 0,
                "hallucinated": 0,
                "skipped": 0,
                "integrity_score": 1.0,
            },
            "results": [],
            "note": "No references.bib found — nothing to verify.",
        }
        (stage_dir / "verification_report.json").write_text(
            json.dumps(report_data, indent=2), encoding="utf-8"
        )
        (stage_dir / "references_verified.bib").write_text("", encoding="utf-8")
        return StageResult(
            stage=Stage.CITATION_VERIFY,
            status=StageStatus.DONE,
            artifacts=("verification_report.json", "references_verified.bib"),
            evidence_refs=(
                "stage-23/verification_report.json",
                "stage-23/references_verified.bib",
            ),
        )

    s2_api_key = getattr(config.llm, "s2_api_key", "") or ""

    from researchclaw.literature.verify import parse_bibtex_entries
    _n_entries = len(parse_bibtex_entries(bib_text))
    logger.info(
        "[citation-verify] Verifying %d references "
        "(DOI→CrossRef > OpenAlex > arXiv > S2)…",
        _n_entries,
    )
    report = verify_citations(bib_text, s2_api_key=s2_api_key)
    logger.info(
        "[citation-verify] Done: %d verified, %d suspicious, "
        "%d hallucinated, %d skipped (integrity: %.0f%%)",
        report.verified,
        report.suspicious,
        report.hallucinated,
        report.skipped,
        report.integrity_score * 100,
    )

    # --- Relevance check: assess topical relevance of verified citations ---
    if llm is not None and report.results:
        relevance_scores = _check_citation_relevance(
            llm, config.research.topic, report.results
        )
        for cr in report.results:
            score = relevance_scores.get(cr.cite_key)
            if score is not None:
                cr.relevance_score = score

    # FIX-5: Filter low-relevance citations and enforce hard cap
    RELEVANCE_THRESHOLD = 0.5
    MAX_CITATIONS = 60
    low_relevance_keys: set[str] = set()
    for cr in report.results:
        if cr.relevance_score is not None and cr.relevance_score < RELEVANCE_THRESHOLD:
            low_relevance_keys.add(cr.cite_key)

    # Hard cap: if still above MAX_CITATIONS after relevance filter, drop lowest
    # BUG-07 fix: Unscored citations (relevance_score=None) default to 0.7
    # because they passed API verification and are likely relevant.
    # Previously they defaulted to 0.0 which caused mass-deletion.
    _DEFAULT_RELEVANCE = 0.7
    remaining = [
        cr for cr in report.results
        if cr.cite_key not in low_relevance_keys
        and cr.status != VerifyStatus.HALLUCINATED
    ]
    if len(remaining) > MAX_CITATIONS:
        remaining.sort(
            key=lambda c: c.relevance_score if c.relevance_score is not None else _DEFAULT_RELEVANCE,
        )
        overflow = remaining[:len(remaining) - MAX_CITATIONS]
        for cr in overflow:
            low_relevance_keys.add(cr.cite_key)
        logger.info(
            "Stage 23: Hard cap applied, dropping %d additional low-relevance citations",
            len(overflow),
        )

    if low_relevance_keys:
        logger.info(
            "Stage 23: Filtering %d low-relevance citations (threshold=%.1f, cap=%d): %s",
            len(low_relevance_keys),
            RELEVANCE_THRESHOLD,
            MAX_CITATIONS,
            ", ".join(sorted(list(low_relevance_keys)[:20])),
        )

    (stage_dir / "verification_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    verified_bib = filter_verified_bibtex(bib_text, report, include_suspicious=True)
    # Remove low-relevance entries from BibTeX
    if low_relevance_keys:
        verified_bib = _remove_bibtex_entries(verified_bib, low_relevance_keys)

    # BUG-26: If verification stripped >50% of entries (e.g. due to rate limiting),
    # fall back to the original bib to avoid breaking the paper's references
    original_count = len(re.findall(r"@\w+\{", bib_text))
    verified_count = len(re.findall(r"@\w+\{", verified_bib))
    if original_count > 0 and verified_count < original_count * 0.5:
        logger.warning(
            "Stage 23: Verification stripped %d→%d entries (>50%% loss). "
            "Keeping original bib to avoid breaking references.",
            original_count, verified_count,
        )
        verified_bib = bib_text

    # IMP-1: Also prune uncited entries from verified bib
    if paper_text.strip():
        _vbib_keys = set(re.findall(r"@\w+\{([^,]+),", verified_bib))
        _cited_in_paper: set[str] = set()
        _cited_in_paper.update(
            re.findall(r"\[([a-z]+\d{4}[a-z]*)\]", paper_text)
        )
        for _cm in re.finditer(r"\\cite\{([^}]+)\}", paper_text):
            _cited_in_paper.update(
                k.strip() for k in _cm.group(1).split(",")
            )
        _uncited_vbib = _vbib_keys - _cited_in_paper
        if _uncited_vbib:
            verified_bib = _remove_bibtex_entries(verified_bib, _uncited_vbib)
            logger.info(
                "Stage 23: Pruned %d uncited entries from verified bib "
                "(kept %d)",
                len(_uncited_vbib),
                len(_vbib_keys) - len(_uncited_vbib),
            )

    (stage_dir / "references_verified.bib").write_text(verified_bib, encoding="utf-8")

    artifacts = ["verification_report.json", "references_verified.bib"]

    if paper_text.strip():
        annotated = annotate_paper_hallucinations(paper_text, report)
        # Remove \cite{} and [cite_key] references for low-relevance entries
        if low_relevance_keys:
            annotated = _remove_citations_from_text(annotated, low_relevance_keys)
        (stage_dir / "paper_final_verified.md").write_text(annotated, encoding="utf-8")
        artifacts.append("paper_final_verified.md")

    logger.info(
        "Stage 23 citation verify: %d total, %d verified, %d suspicious, "
        "%d hallucinated, %d skipped (integrity=%.1f%%)",
        report.total,
        report.verified,
        report.suspicious,
        report.hallucinated,
        report.skipped,
        report.integrity_score * 100,
    )

    return StageResult(
        stage=Stage.CITATION_VERIFY,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-23/{a}" for a in artifacts),
    )


_STAGE_EXECUTORS: dict[Stage, Callable[..., StageResult]] = {
    Stage.TOPIC_INIT: _execute_topic_init,
    Stage.PROBLEM_DECOMPOSE: _execute_problem_decompose,
    Stage.SEARCH_STRATEGY: _execute_search_strategy,
    Stage.LITERATURE_COLLECT: _execute_literature_collect,
    Stage.LITERATURE_SCREEN: _execute_literature_screen,
    Stage.KNOWLEDGE_EXTRACT: _execute_knowledge_extract,
    Stage.SYNTHESIS: _execute_synthesis,
    Stage.HYPOTHESIS_GEN: _execute_hypothesis_gen,
    Stage.EXPERIMENT_DESIGN: _execute_experiment_design,
    Stage.CODE_GENERATION: _execute_code_generation,
    Stage.RESOURCE_PLANNING: _execute_resource_planning,
    Stage.EXPERIMENT_RUN: _execute_experiment_run,
    Stage.ITERATIVE_REFINE: _execute_iterative_refine,
    Stage.RESULT_ANALYSIS: _execute_result_analysis,
    Stage.RESEARCH_DECISION: _execute_research_decision,
    Stage.PAPER_OUTLINE: _execute_paper_outline,
    Stage.PAPER_DRAFT: _execute_paper_draft,
    Stage.PEER_REVIEW: _execute_peer_review,
    Stage.PAPER_REVISION: _execute_paper_revision,
    Stage.QUALITY_GATE: _execute_quality_gate,
    Stage.KNOWLEDGE_ARCHIVE: _execute_knowledge_archive,
    Stage.EXPORT_PUBLISH: _execute_export_publish,
    Stage.CITATION_VERIFY: _execute_citation_verify,
}


def execute_stage(
    stage: Stage,
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    auto_approve_gates: bool = False,
) -> StageResult:
    """Execute one pipeline stage, validate outputs, and apply gate logic."""

    stage_dir = run_dir / f"stage-{int(stage):02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _t_health_start = _time.monotonic()
    contract: StageContract = CONTRACTS[stage]

    if contract.input_files:
        for input_file in contract.input_files:
            found = _read_prior_artifact(run_dir, input_file)
            if found is None:
                result = StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    artifacts=(),
                    error=f"Missing input: {input_file} (required by {stage.name})",
                    decision="retry",
                )
                _write_stage_meta(stage_dir, stage, run_id, result)
                return result

    bridge = config.openclaw_bridge
    if bridge.use_message and config.notifications.on_stage_start:
        adapters.message.notify(
            config.notifications.channel,
            f"stage-{int(stage):02d}-start",
            f"Starting {stage.name}",
        )
    if bridge.use_memory:
        adapters.memory.append("stages", f"{run_id}:{int(stage)}:running")

    llm = None
    try:
        if config.llm.provider == "acp":
            llm = create_llm_client(config)
        else:
            candidate = LLMClient.from_rc_config(config)
            if candidate.config.base_url and candidate.config.api_key:
                llm = candidate
    except Exception:  # noqa: BLE001
        llm = None

    try:
        _ = advance(stage, StageStatus.PENDING, TransitionEvent.START)
        executor = _STAGE_EXECUTORS[stage]
        prompts = PromptManager(config.prompts.custom_file or None)  # type: ignore[attr-defined]
        try:
            result = executor(
                stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
            )
        except TypeError as exc:
            if "unexpected keyword argument 'prompts'" not in str(exc):
                raise
            result = executor(stage_dir, run_dir, config, adapters, llm=llm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage %s failed", stage.name)
        result = StageResult(
            stage=stage,
            status=StageStatus.FAILED,
            artifacts=(),
            error=str(exc),
            decision="retry",
        )

    if result.status == StageStatus.DONE:
        for output_file in contract.output_files:
            if output_file.endswith("/"):
                path = stage_dir / output_file.rstrip("/")
                if not path.is_dir() or not any(path.iterdir()):
                    result = StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        artifacts=result.artifacts,
                        error=f"Missing output directory: {output_file}",
                        decision="retry",
                        evidence_refs=result.evidence_refs,
                    )
                    break
            else:
                path = stage_dir / output_file
                if not path.exists() or path.stat().st_size == 0:
                    result = StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        artifacts=result.artifacts,
                        error=f"Missing or empty output: {output_file}",
                        decision="retry",
                        evidence_refs=result.evidence_refs,
                    )
                    break

    # --- MetaClaw PRM quality gate evaluation ---
    try:
        mc_bridge = getattr(config, "metaclaw_bridge", None)
        if (
            mc_bridge
            and getattr(mc_bridge, "enabled", False)
            and result.status == StageStatus.DONE
        ):
            mc_prm = getattr(mc_bridge, "prm", None)
            if mc_prm and getattr(mc_prm, "enabled", False):
                prm_stages = getattr(mc_prm, "gate_stages", (5, 9, 15, 20))
                if int(stage) in prm_stages:
                    from researchclaw.metaclaw_bridge.prm_gate import ResearchPRMGate

                    prm_gate = ResearchPRMGate.from_bridge_config(mc_prm)
                    if prm_gate is not None:
                        # Read stage output for PRM evaluation
                        output_text = ""
                        for art in result.artifacts:
                            art_path = stage_dir / art
                            if art_path.exists() and art_path.is_file():
                                try:
                                    output_text += art_path.read_text(encoding="utf-8")[:4000]
                                except (UnicodeDecodeError, OSError):
                                    pass
                        if output_text:
                            prm_score = prm_gate.evaluate_stage(int(stage), output_text)
                            logger.info(
                                "MetaClaw PRM score for stage %d: %.1f",
                                int(stage),
                                prm_score,
                            )
                            # Write PRM score to stage health
                            import json as _prm_json

                            prm_report = {
                                "stage": int(stage),
                                "prm_score": prm_score,
                                "model": prm_gate.model,
                                "votes": prm_gate.votes,
                            }
                            (stage_dir / "prm_score.json").write_text(
                                _prm_json.dumps(prm_report, indent=2),
                                encoding="utf-8",
                            )
                            # If PRM score is -1 (fail), mark stage as failed
                            if prm_score == -1.0:
                                logger.warning(
                                    "MetaClaw PRM rejected stage %d output",
                                    int(stage),
                                )
                                result = StageResult(
                                    stage=result.stage,
                                    status=StageStatus.FAILED,
                                    artifacts=result.artifacts,
                                    error="PRM quality gate: output below quality threshold",
                                    decision="retry",
                                    evidence_refs=result.evidence_refs,
                                )
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw PRM evaluation failed (non-blocking)")

    if gate_required(stage, config.security.hitl_required_stages):
        if auto_approve_gates:
            if bridge.use_memory:
                adapters.memory.append("gates", f"{run_id}:{int(stage)}:auto-approved")
        else:
            result = StageResult(
                stage=result.stage,
                status=StageStatus.BLOCKED_APPROVAL,
                artifacts=result.artifacts,
                error=result.error,
                decision="block",
                evidence_refs=result.evidence_refs,
            )
            if bridge.use_message and config.notifications.on_gate_required:
                adapters.message.notify(
                    config.notifications.channel,
                    f"gate-{int(stage):02d}",
                    f"Approval required for {stage.name}",
                )

    if bridge.use_memory:
        adapters.memory.append("stages", f"{run_id}:{int(stage)}:{result.status.value}")

    _write_stage_meta(stage_dir, stage, run_id, result)

    _t_health_end = _time.monotonic()
    stage_health = {
        "stage_id": f"{int(stage):02d}-{stage.name.lower()}",
        "run_id": run_id,
        "duration_sec": round(_t_health_end - _t_health_start, 2),
        "status": result.status.value,
        "artifacts_count": len(result.artifacts),
        "error": result.error,
        "timestamp": _utcnow_iso(),
    }
    try:
        (stage_dir / "stage_health.json").write_text(
            json.dumps(stage_health, indent=2), encoding="utf-8"
        )
    except OSError:
        pass

    return result
