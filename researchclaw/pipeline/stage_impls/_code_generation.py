"""Stage 10: Code generation."""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.validator import (
    CodeValidation,
    format_issues_for_llm,
    validate_code,
)
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_research_repair_brief,
    _chat_with_prompt,
    detect_synthetic_proxy_signals,
    _ensure_sandbox_deps,
    _extract_code_block,
    _extract_multi_file_blocks,
    _extract_yaml_block,
    _get_evolution_overlay,
    _load_hardware_profile,
    _read_prior_artifact,
    _safe_json_loads,
    should_fail_synthetic_proxy_guard,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)

# Improvement G: Continuous-action environments that are incompatible with DQN
_CONTINUOUS_ENVS = {
    "pendulum", "halfcheetah", "hopper", "walker2d", "ant", "humanoid",
    "swimmer", "reacher", "invertedpendulum", "inverteddoublependulum",
    "mountaincarcontinuous", "lunarlander-continuous",
}

_LIKELY_LOCAL_HELPER_MODULES = {
    "backbone",
    "backbones",
    "config",
    "configs",
    "constants",
    "data_loader",
    "data_utils",
    "dataloader",
    "dataset",
    "datasets",
    "decoder",
    "decoders",
    "encoder",
    "encoders",
    "helper",
    "helpers",
    "layer",
    "layers",
    "loader",
    "loaders",
    "loss",
    "losses",
    "metric",
    "metrics",
    "model",
    "models",
    "module",
    "modules",
    "network",
    "networks",
    "postprocess",
    "postprocessing",
    "preprocess",
    "preprocessing",
    "train_utils",
    "trainer",
    "trainers",
    "transform",
    "transforms",
    "util",
    "utils",
}

_PLACEHOLDER_EXPERIMENT_PATTERNS = (
    "dummy implementation",
    "dummy implementations",
    "dummy placeholder",
    "placeholder implementation",
    "replace with actual implementation",
    "replace with actual implementations",
    "for standalone operation",
    "for demonstration",
)

_EXPERIMENT_CLASS_NAME_HINTS = (
    "ablation",
    "baseline",
    "detector",
    "fusion",
    "model",
    "reranker",
    "verifier",
)

_CORE_EXPERIMENT_METHODS = {
    "evaluate",
    "forward",
    "predict",
    "run",
    "score",
    "train_step",
}

_ABLATION_NAME_HINTS = (
    "without",
    "ablation",
    "abl_",
    "no_",
    "minus",
)

_DISTINCTNESS_CHECK_NAME_HINTS = (
    "ablation_check",
    "condition_outputs_differ",
    "distinctness",
    "outputs_differ",
    "sanity_check_condition",
    "verify_condition",
)

_CRITICAL_DEEP_KEYWORDS = (
    "unboundlocalerror",
    "unregistered",
    "does not exist",
    "empty or trivial subclass",
    "does not override",
    "import-usage mismatch",
    "nameerror",
    "was removed",
    "ptp()",
    "copy-paste",
    "identical method signatures",
    "identical ast",
    "not a real ablation",
    "shadows stdlib/pip",
    "placeholder experiment text found",
    "placeholder experiment implementation",
    "fixed-constant core method",
    "demonstration stub",
    "no ablation/condition distinctness self-check",
    "distinctness self-check",
    "does not call distinctness check",
)


def _check_rl_compatibility(code: str) -> list[str]:
    """Detect DQN + continuous-action environment mismatches.

    Returns a list of error strings if incompatible combinations are found.
    """
    errors: list[str] = []
    code_lower = code.lower()
    has_dqn = "dqn" in code_lower
    if not has_dqn:
        return errors

    for env_name in _CONTINUOUS_ENVS:
        if env_name in code_lower:
            errors.append(
                f"RL COMPATIBILITY ERROR: DQN is used with continuous-action "
                f"environment '{env_name}'. DQN only works with DISCRETE action "
                f"spaces. Use SAC, TD3, or PPO instead."
            )
    return errors


def _find_missing_local_module_imports(files: dict[str, str]) -> list[str]:
    """Detect local helper-module imports that are not present in *files*.

    This is intentionally narrower than generic import validation: we only flag
    imports that strongly indicate an intra-project Python module dependency.
    """
    known_modules = {
        fname[:-3]
        for fname in files
        if fname.endswith(".py")
    }
    issues: list[str] = []
    seen: set[tuple[str, str, int | None]] = set()

    def _record_issue(
        file_name: str,
        module_name: str,
        *,
        line: int | None,
    ) -> None:
        if (
            module_name in known_modules
            or module_name.startswith("_")
        ):
            return
        key = (file_name, module_name, line)
        if key in seen:
            return
        seen.add(key)
        line_text = f" line {line}" if line is not None else ""
        issues.append(
            f"[{file_name}] Local helper module '{module_name}.py' is imported at"
            f"{line_text} but was not generated. The experiment project must be"
            f" self-contained: either return '{module_name}.py' or inline its"
            f" code and remove the import."
        )

    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in _LIKELY_LOCAL_HELPER_MODULES:
                        _record_issue(
                            fname,
                            top,
                            line=getattr(node, "lineno", None),
                        )
            elif isinstance(node, ast.ImportFrom):
                line = getattr(node, "lineno", None)
                if node.level > 0:
                    if node.module:
                        top = node.module.split(".")[0]
                        _record_issue(fname, top, line=line)
                    else:
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            _record_issue(fname, top, line=line)
                elif node.module:
                    top = node.module.split(".")[0]
                    if top in _LIKELY_LOCAL_HELPER_MODULES:
                        _record_issue(fname, top, line=line)

    return issues


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_literal_constant(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_literal_constant(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_constant(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_literal_constant(key)) and _is_literal_constant(value)
            for key, value in zip(node.keys, node.values, strict=False)
        )
    return False


def _method_is_pass_only(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = _strip_docstring(list(node.body))
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _method_returns_fixed_constant(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    body = _strip_docstring(list(node.body))
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and _is_literal_constant(body[0].value)
    )


def _looks_like_experiment_class(node: ast.ClassDef) -> bool:
    lowered = node.name.lower()
    if any(hint in lowered for hint in _EXPERIMENT_CLASS_NAME_HINTS):
        return True
    return any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name in _CORE_EXPERIMENT_METHODS
        for item in node.body
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _extract_condition_entries(tree: ast.AST) -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []

    def _extract_label(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _extract_class_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return _extract_class_name(node.func)
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    for assign in ast.walk(tree):
        if not isinstance(assign, ast.Assign):
            continue
        if not isinstance(assign.value, (ast.List, ast.Tuple)):
            continue
        for elt in assign.value.elts:
            if not isinstance(elt, ast.Tuple) or len(elt.elts) < 2:
                continue
            label = _extract_label(elt.elts[0]) or ""
            class_name = _extract_class_name(elt.elts[1])
            if label or class_name:
                entries.append((label, class_name))
    return entries


def _looks_like_ablation_entry(label: str, class_name: str | None) -> bool:
    lowered = f"{label} {class_name or ''}".lower()
    return any(hint in lowered for hint in _ABLATION_NAME_HINTS)


def _function_has_distinctness_logic(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    body = _strip_docstring(list(node.body))
    if not body:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        if isinstance(sub, ast.Compare):
            return True
        if isinstance(sub, ast.Call):
            call_name = _call_name(sub.func).lower()
            if call_name in {"allclose", "array_equal"}:
                return True
            if "assert" in call_name or "raise" in call_name:
                return True
    return False


def _find_placeholder_experiment_issues(files: dict[str, str]) -> list[str]:
    """Detect obviously placeholder experiment implementations.

    This is stricter than generic code-complexity warnings: it looks for
    generated experiments that openly advertise themselves as demonstrations,
    or condition classes whose core methods are pass-only / fixed-constant stubs.
    """
    issues: list[str] = []

    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        lowered_code = code.lower()
        for pattern in _PLACEHOLDER_EXPERIMENT_PATTERNS:
            if pattern in lowered_code:
                issues.append(
                    f"[{fname}] Placeholder experiment text found ('{pattern}') — "
                    "generated experiment code must implement real logic, not "
                    "demonstration stubs."
                )
                break

        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not _looks_like_experiment_class(node):
                continue

            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not methods:
                continue

            pass_only_init = any(
                method.name == "__init__" and _method_is_pass_only(method)
                for method in methods
            )
            constant_core_methods = [
                method.name
                for method in methods
                if method.name in _CORE_EXPERIMENT_METHODS
                and _method_returns_fixed_constant(method)
            ]
            trivial_core_methods = [
                method.name
                for method in methods
                if method.name in _CORE_EXPERIMENT_METHODS
                and (
                    _method_is_pass_only(method)
                    or _method_returns_fixed_constant(method)
                )
            ]

            if pass_only_init and constant_core_methods:
                issues.append(
                    f"[{fname}] Class '{node.name}' looks like a placeholder "
                    "experiment implementation: __init__ is pass-only and core "
                    "method(s) "
                    + ", ".join(sorted(constant_core_methods))
                    + " use fixed-constant core method returns. Ablation/condition "
                      "classes must exercise real differentiating logic."
                )
                continue

            if trivial_core_methods and len(trivial_core_methods) == len(
                [
                    method
                    for method in methods
                    if method.name in _CORE_EXPERIMENT_METHODS
                ]
            ):
                issues.append(
                    f"[{fname}] Class '{node.name}' is a demonstration stub: all "
                    "core experiment methods ("
                    + ", ".join(sorted(trivial_core_methods))
                    + ") are pass-only or fixed-constant. Generated ablations must "
                      "implement real computation."
                )

    return issues


def _find_condition_distinctness_issues(files: dict[str, str]) -> list[str]:
    """Detect missing or non-functional ablation distinctness self-checks."""
    issues: list[str] = []
    condition_entries: list[tuple[str, str | None]] = []
    distinctness_functions: dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    called_distinctness_functions: set[str] = set()

    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        condition_entries.extend(_extract_condition_entries(tree))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lowered_name = node.name.lower()
                if any(hint in lowered_name for hint in _DISTINCTNESS_CHECK_NAME_HINTS):
                    distinctness_functions[node.name] = (fname, node)
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if any(hint in call_name.lower() for hint in _DISTINCTNESS_CHECK_NAME_HINTS):
                    called_distinctness_functions.add(call_name)

    if not condition_entries:
        return issues

    ablation_entries = [
        (label, class_name)
        for label, class_name in condition_entries
        if _looks_like_ablation_entry(label, class_name)
    ]
    if len(condition_entries) < 4 or len(ablation_entries) < 2:
        return issues

    if not distinctness_functions:
        issues.append(
            "No ablation/condition distinctness self-check found. Experiments with "
            "multiple ablation-like conditions must include a startup check that "
            "compares condition outputs on the same probe input and fails if they "
            "are identical."
        )
        return issues

    valid_function_names: set[str] = set()
    for func_name, (fname, func_node) in distinctness_functions.items():
        if _method_is_pass_only(func_node):
            issues.append(
                f"[{fname}] Distinctness self-check '{func_name}' is pass-only. It "
                "must actively compare condition outputs and fail loudly on "
                "identical behavior."
            )
            continue
        if _method_returns_fixed_constant(func_node):
            issues.append(
                f"[{fname}] Distinctness self-check '{func_name}' returns a fixed "
                "constant instead of validating ablation behavior."
            )
            continue
        if not _function_has_distinctness_logic(func_node):
            issues.append(
                f"[{fname}] Distinctness self-check '{func_name}' exists but does "
                "not contain comparison/assertion logic. It must compare outputs "
                "from multiple conditions on the same probe input."
            )
            continue
        valid_function_names.add(func_name)

    if valid_function_names and not any(
        called in valid_function_names for called in called_distinctness_functions
    ):
        issues.append(
            "Experiment defines a condition distinctness self-check but does not "
            "call it before running the main evaluation. Call the self-check at "
            "startup and fail fast on identical outputs."
        )

    return issues


def _is_critical_deep_warning(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in _CRITICAL_DEEP_KEYWORDS)


def _repair_self_contained_project(
    *,
    llm: LLMClient,
    prompt_manager: PromptManager,
    files: dict[str, str],
    issues: list[str],
    max_tokens: int,
    max_repair: int,
) -> tuple[dict[str, str], list[str]]:
    """Repair missing local helper-module files by asking for a full file set."""
    current_files = dict(files)
    current_issues = list(issues)

    for attempt in range(1, max_repair + 1):
        repair_prompt = (
            "SELF-CONTAINMENT REPAIR REQUIRED.\n\n"
            "The generated experiment project is not self-contained. Some files "
            "import local helper modules that were never returned.\n\n"
            "Missing local-module issues:\n"
            + "\n".join(f"- {issue}" for issue in current_issues)
            + "\n\nRULES:\n"
            "- If any file imports a local helper module such as models, utils, "
            "data_utils, metrics, or loaders, you MUST return that helper file "
            "too.\n"
            "- If you do not want a helper file, inline its code into an existing "
            "file and remove the import.\n"
            "- Preserve working files unless you are intentionally replacing them.\n"
            "- The final project must be runnable via `python main.py`.\n"
            "- Return ALL project files using ```filename:...``` blocks.\n\n"
            "Current files:\n"
            + "\n\n".join(
                f"```filename:{fname}\n{code}\n```"
                for fname, code in current_files.items()
            )
        )

        resp = _chat_with_prompt(
            llm,
            prompt_manager.system("code_generation"),
            repair_prompt,
            max_tokens=max_tokens,
        )
        repaired_files = _extract_multi_file_blocks(resp.content)
        if not repaired_files:
            logger.warning(
                "Stage 10: Self-containment repair attempt %d returned no files",
                attempt,
            )
            continue

        merged = dict(current_files)
        merged.update(repaired_files)
        current_files = merged
        current_issues = _find_missing_local_module_imports(current_files)
        if not current_issues:
            logger.info(
                "Stage 10: Self-containment repair succeeded on attempt %d",
                attempt,
            )
            return current_files, []

    return current_files, current_issues


def _build_real_data_guard_guidance(config: RCConfig) -> str:
    exp_cfg = config.experiment
    if not (
        getattr(exp_cfg, "require_real_data", False)
        or getattr(exp_cfg, "forbid_synthetic_proxy", False)
        or getattr(exp_cfg, "fail_on_stdout_parsed_results", False)
        or getattr(exp_cfg, "required_real_data_refs", ())
    ):
        return ""

    refs = tuple(getattr(exp_cfg, "required_real_data_refs", ()) or ())
    refs_block = ""
    if refs:
        refs_block = "Required local data references (use these, do not invent substitutes):\n"
        refs_block += "".join(f"- {ref}\n" for ref in refs)

    asset_paths_block = _build_resolved_local_asset_guidance()

    return (
        "\n\nREAL DATA ENFORCEMENT (HARD RULE):\n"
        "- This project MUST use real local project assets/caches, not an internally "
        "generated proxy benchmark.\n"
        "- If the required local assets are unavailable, FAIL FAST with a clear "
        "FileNotFoundError or RuntimeError. Do NOT silently degrade to a toy dataset.\n"
        "- FORBIDDEN fallback patterns include: helper functions such as "
        "`_build_example`, `_build_splits`, or `_sample_circle` that generate the "
        "benchmark in code; hard-coded tiny train/val/test split dictionaries; "
        "repository-local synthetic evidence repositories; or any results source that "
        "exists only as stdout metric lines.\n"
        "- main.py must write a structured `results.json`; stdout-only metrics are "
        "insufficient for this run.\n"
        "- The execution harness invokes `python main.py` directly. Do NOT require "
        "dataset/asset CLI flags just to start the experiment. Asset path flags may "
        "exist only as optional overrides; the default path resolution must come "
        "from the VECTRA_* env vars or the authoritative absolute roots below.\n"
        "- Emit machine-readable provenance where practical, including "
        "`data_manifest.json` and `protocol_manifest.json`, so later stages can verify "
        "which local assets were actually used.\n"
        "- Resolve data from the authoritative absolute roots or env vars below. "
        "Do NOT invent packaged relative directories such as "
        "`./page_minus_titleblock_train1000_local`.\n"
        + refs_block
        + asset_paths_block
    )


def _build_resolved_local_asset_guidance() -> str:
    """Expose authoritative local asset roots from the repo's experiment config."""
    specs = _load_project_dataset_specs()
    if not specs:
        return ""

    def _path_text(value: Any) -> str:
        text = " ".join(str(value).replace("\\", "/").split()).strip()
        return text

    lines = ["Authoritative local asset roots for this repository:"]

    def _append_path(dataset_name: str, label: str, env_name: str, value: Any) -> None:
        text = _path_text(value)
        if not text:
            return
        lines.append(f"- {dataset_name} {label}: {text} (env: {env_name})")

    repo_root = Path(__file__).resolve().parents[3]
    lines.append(f"- repo root: {repo_root.as_posix()} (env: VECTRA_REPO_ROOT)")

    simple_key = "engineering_primitives_simple_scenes_noslot_v1_local_20260326"
    simple_spec = specs.get(simple_key)
    if isinstance(simple_spec, dict):
        cache_roots = simple_spec.get("cache_roots")
        _append_path(simple_key, "dataset_root", "VECTRA_SIMPLE_DATASET_ROOT", simple_spec.get("dataset_root"))
        _append_path(simple_key, "dataset_root", "VECTRA_SIMPLE_ASSET_ROOT", simple_spec.get("dataset_root"))
        _append_path(simple_key, "manifest_path", "VECTRA_SIMPLE_MANIFEST_PATH", simple_spec.get("manifest_path"))
        if isinstance(cache_roots, dict):
            _append_path(simple_key, "learned_cache", "VECTRA_SIMPLE_HEATMAP_DIR", cache_roots.get("learned"))

    page_key = "page_minus_titleblock"
    page_spec = specs.get(page_key)
    if isinstance(page_spec, dict):
        dataset_root = page_spec.get("dataset_root")
        split_manifest = page_spec.get("split_manifest_path")
        _append_path(page_key, "dataset_root", "VECTRA_PAGE_DATASET_ROOT", dataset_root)
        if dataset_root:
            dataset_root_path = Path(str(dataset_root))
            _append_path(page_key, "image_dir", "VECTRA_PAGE_IMAGE_DIR", dataset_root_path / "train2017")
            _append_path(page_key, "sidecar_dir", "VECTRA_PAGE_SIDECAR_DIR", dataset_root_path / "sidecars" / "train2017")
        _append_path(page_key, "split_manifest", "VECTRA_PAGE_SPLIT_JSON", split_manifest)
        if split_manifest:
            split_manifest_path = Path(str(split_manifest))
            one_drive_png_root = split_manifest_path.parent.parent
            _append_path(page_key, "png_root", "VECTRA_ONE_DRIVE_PNG_ROOT", one_drive_png_root)
            _append_path(page_key, "gt_solid_csv", "VECTRA_PAGE_GT_SOLID_CSV", split_manifest_path.parent / "gt" / "train2017_solid.csv")
            _append_path(page_key, "gt_dashed_csv", "VECTRA_PAGE_GT_DASHED_CSV", split_manifest_path.parent / "gt" / "train2017_dashed.csv")

    probe_key = "DeepPatent2_negative_clutter_probe"
    probe_spec = specs.get(probe_key)
    if isinstance(probe_spec, dict):
        _append_path(probe_key, "dataset_root", "VECTRA_DEEPPATENT_DATASET_ROOT", probe_spec.get("dataset_root"))

    if len(lines) <= 1:
        return ""
    lines.extend(
        [
            "- Loader rule: first read the env vars above if they are set, otherwise fall back to the exact absolute paths above.",
            "- If a required asset path does not exist, raise FileNotFoundError naming the env var/path that was missing.",
        ]
    )
    return "\n" + "\n".join(lines) + "\n"


def _load_project_dataset_specs() -> dict[str, dict[str, Any]]:
    """Load the repo-root experiment dataset specs for prompt grounding."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config.py"
    if not config_path.exists():
        return {}

    try:
        spec = importlib.util.spec_from_file_location(
            "researchclaw_project_config_for_codegen",
            config_path,
        )
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build_default_config = getattr(module, "build_default_config", None)
        if not callable(build_default_config):
            return {}
        project_config = build_default_config()
        build_dataset_specs = getattr(project_config, "build_dataset_specs", None)
        if not callable(build_dataset_specs):
            return {}
        dataset_specs = build_dataset_specs()
        if not isinstance(dataset_specs, dict):
            return {}
        return dataset_specs
    except Exception:  # noqa: BLE001
        logger.debug("Resolved local asset guidance unavailable", exc_info=True)
        return {}


def _extract_named_plan_items(value: Any, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, dict):
        if "name" in value:
            candidate = " ".join(str(value.get("name", "")).split()).strip()
            if candidate:
                items.append(candidate)
        else:
            for key in value:
                candidate = " ".join(str(key).split()).strip()
                if candidate:
                    items.append(candidate)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict):
                candidate = " ".join(str(item.get("name", "")).split()).strip()
            else:
                candidate = " ".join(str(item).split()).strip()
            if candidate:
                items.append(candidate)
    else:
        candidate = " ".join(str(value).split()).strip()
        if candidate:
            items.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _build_codegen_plan_summary(exp_plan_text: str, config: RCConfig) -> str:
    """Compress the experiment plan into the subset needed for Stage 10."""
    if not exp_plan_text.strip():
        return ""

    try:
        plan_data = yaml.safe_load(exp_plan_text)
    except yaml.YAMLError:
        plan_data = None
    if not isinstance(plan_data, dict):
        excerpt = exp_plan_text[:2200].rstrip()
        suffix = "\n...\n" if len(exp_plan_text) > 2200 else "\n"
        return "PLAN EXCERPT:\n" + excerpt + suffix

    lines = ["## Experiment Plan Summary"]

    plan_topic = " ".join(str(plan_data.get("topic", "")).split()).strip()
    if plan_topic:
        if len(plan_topic) > 320:
            plan_topic = plan_topic[:320].rstrip() + "..."
        lines.append(f"- Topic anchor: {plan_topic}")

    datasets = _extract_named_plan_items(plan_data.get("datasets"), limit=6)
    if datasets:
        lines.append("- Datasets: " + ", ".join(datasets))

    baselines = _extract_named_plan_items(plan_data.get("baselines"), limit=8)
    if baselines:
        lines.append("- Baselines: " + ", ".join(baselines))

    methods = _extract_named_plan_items(plan_data.get("proposed_methods"), limit=8)
    if methods:
        lines.append("- Proposed methods: " + ", ".join(methods))

    ablations = _extract_named_plan_items(plan_data.get("ablations"), limit=8)
    if ablations:
        lines.append("- Ablations: " + ", ".join(ablations))

    metrics = plan_data.get("metrics")
    if isinstance(metrics, dict):
        primary = metrics.get("primary_metric")
        if isinstance(primary, dict):
            primary_name = " ".join(str(primary.get("name", "")).split()).strip()
            direction = " ".join(
                str(primary.get("direction", config.experiment.metric_direction)).split()
            ).strip()
            if primary_name:
                lines.append(
                    f"- Primary metric: {primary_name} ({direction or config.experiment.metric_direction})"
                )
        secondary = _extract_named_plan_items(metrics.get("secondary_metrics"), limit=8)
        if secondary:
            lines.append("- Secondary metrics: " + ", ".join(secondary))

    compute_budget = plan_data.get("compute_budget")
    if isinstance(compute_budget, dict):
        total_seconds = compute_budget.get("total_time_budget_seconds")
        seeded_conditions = compute_budget.get("seeded_condition_count")
        budget_bits: list[str] = []
        if total_seconds is not None:
            budget_bits.append(f"total_time_budget_seconds={total_seconds}")
        if seeded_conditions is not None:
            budget_bits.append(f"seeded_condition_count={seeded_conditions}")
        if budget_bits:
            lines.append("- Compute budget: " + ", ".join(budget_bits))

    refs = tuple(getattr(config.experiment, "required_real_data_refs", ()) or ())
    if refs:
        lines.append("- Required local asset refs:")
        lines.extend(f"  - {ref}" for ref in refs[:8])

    return "\n".join(lines).strip()


def _is_acp_transport_failure(exc: Exception) -> bool:
    """Return True when a Stage-10 failure came from the ACP transport layer."""
    parts = [str(exc)]
    cause = getattr(exc, "__cause__", None)
    context = getattr(exc, "__context__", None)
    if cause:
        parts.append(str(cause))
    if context:
        parts.append(str(context))
    text = " ".join(part.strip() for part in parts if part).lower()
    if not text:
        return False
    indicators = (
        "acp prompt failed",
        "acp prompt timed out after",
        "queue owner disconnected before prompt completion",
        "agent needs reconnect",
    )
    return any(indicator in text for indicator in indicators)


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
    exp_plan_prompt = _build_codegen_plan_summary(exp_plan, config)
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
                _has_existing_plan_assets = any(
                    item.get("origin") == "existing_plan"
                    for item in (_bp.selected_benchmarks + _bp.selected_baselines)
                    if isinstance(item, dict)
                )
                _bp_heading = "## BenchmarkAgent Selections (USE THESE)"
                _bp_instruction = (
                    "The following datasets, baselines, and code snippets were "
                    "automatically selected and validated by the BenchmarkAgent. "
                    "You MUST use these selections in your experiment code.\n\n"
                )
                if _has_existing_plan_assets:
                    _bp_heading = "## BenchmarkAgent Selections (PRESERVE IN-PROJECT ASSETS)"
                    _bp_instruction = (
                        "The following datasets and baselines include in-project "
                        "assets carried over from the existing experiment plan plus "
                        "BenchmarkAgent supplements. You MUST preserve the in-project "
                        "datasets/baselines and may use the extra BenchmarkAgent "
                        "selections only as supplemental additions.\n\n"
                    )
                extra_guidance += (
                    f"\n\n{_bp_heading}\n"
                    + _bp_instruction
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

    # BUG-R6-01: Add explicit implementation constraints to prevent LLM
    # from substituting unrelated DL models for lightweight algorithms.
    extra_guidance += (
        "\n\nIMPLEMENTATION CONSTRAINTS (MUST FOLLOW):\n"
        "- Implement EXACTLY the algorithm/method described in the topic.\n"
        "- Do NOT replace the stated method with a deep-learning proxy "
        "(e.g. ResNet, BERT, GPT, Gymnasium+SB3) unless the topic "
        "EXPLICITLY requires deep learning.\n"
        "- Prefer lightweight CPU-friendly libraries (numpy, scipy, "
        "sklearn, pandas) unless deep learning is inherent to the topic.\n"
        "- The experiment MUST be self-contained and runnable without GPU.\n"
        "- The returned experiment project must be self-contained at the file "
        "level. If `main.py` or any other file imports a local helper module "
        "(for example `models`, `utils`, `data_utils`, `metrics`, `loaders`), "
        "you MUST return that helper file too.\n"
        "- Never reference a local Python module that is absent from the "
        "returned file set. If in doubt, inline the helper code into an "
        "existing returned file instead of importing a missing module.\n"
    )
    repair_brief = _build_research_repair_brief(run_dir)
    if repair_brief:
        extra_guidance += "\n\n" + repair_brief
    extra_guidance += _build_real_data_guard_guidance(config)

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
                    exp_plan=exp_plan_prompt,
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

        try:
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
                exp_plan=exp_plan_prompt,
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
        except Exception as exc:
            fallback_enabled = bool(
                getattr(_ca_cfg, "fallback_to_legacy_on_acp_failure", False)
            )
            if fallback_enabled and _is_acp_transport_failure(exc):
                fallback_payload = {
                    "fallback_triggered": True,
                    "reason": "code_agent_acp_transport_failure",
                    "error": str(exc),
                    "triggered_at": _utcnow_iso(),
                }
                (stage_dir / "code_agent_fallback.json").write_text(
                    json.dumps(fallback_payload, indent=2),
                    encoding="utf-8",
                )
                logger.warning(
                    "CodeAgent ACP transport failure detected; falling back to legacy single-shot generation: %s",
                    exc,
                )
            else:
                raise

    if not _beast_mode_used and llm is not None and not _code_agent_active:
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
            exp_plan=exp_plan_prompt,
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
        # Skip non-Python files (requirements.txt, setup.py, etc.)
        if not fname.endswith(".py"):
            continue
        validation = validate_code(code)
        repair_attempt = 0
        while not validation.ok and llm is not None and repair_attempt < max_repair:
            repair_attempt += 1
            attempt += 1
            # Only send errors to the LLM — warnings don't block validation
            # and confuse the LLM into over-correcting (e.g. removing runtime imports)
            errors_only = type(validation)(
                issues=[i for i in validation.issues if i.severity == "error"]
            )
            issues_text = format_issues_for_llm(errors_only)
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
            _repaired = _extract_code_block(resp.content)
            if _repaired.strip():
                files[fname] = _repaired
            else:
                logger.warning("Repair attempt returned empty code, keeping original")
            validation = validate_code(files[fname])
        if not validation.ok:
            all_valid = False
            # BUG-14: Log remaining issues prominently
            logger.warning(
                "Code validation FAILED for %s after %d repair attempts: %s",
                fname, max_repair, validation.summary(),
            )

    # Improvement G: RL algorithm-environment compatibility check
    for fname, code in list(files.items()):
        if not fname.endswith(".py"):
            continue
        _rl_errors = _check_rl_compatibility(code)
        if _rl_errors:
            for _rl_err in _rl_errors:
                logger.error("Stage 10: %s (in %s)", _rl_err, fname)
                validation_log.append(f"RL_COMPAT: {fname}: {_rl_err}")
            all_valid = False

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

    # --- BUG-184: Cross-import validation — warn if a .py file imports a
    # local module that doesn't exist in the files dict.  This catches the
    # case where Beast Mode/CodeAgent produced an intermediate file that
    # got lost during repair iterations.
    _known_modules = {
        f.replace(".py", "") for f in files if f.endswith(".py")
    }
    _stdlib_and_common = {
        "os", "sys", "json", "math", "time", "copy", "re", "random",
        "pathlib", "argparse", "logging", "collections", "functools",
        "itertools", "abc", "typing", "dataclasses", "enum", "io",
        "csv", "pickle", "glob", "shutil", "subprocess", "datetime",
        "numpy", "np", "torch", "torchvision", "gymnasium", "gym",
        "sklearn", "scipy", "pandas", "matplotlib", "PIL", "tqdm",
        "einops", "timm", "transformers", "datasets", "peft",
        "stable_baselines3",
    }
    for fname, code in list(files.items()):
        if not fname.endswith(".py"):
            continue
        for _m in re.findall(
            r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            code, re.MULTILINE,
        ):
            if (_m not in _known_modules
                    and _m not in _stdlib_and_common
                    and not _m.startswith("_")):
                logger.warning(
                    "BUG-184: %s imports '%s' which is not in generated "
                    "files — experiment may crash on import",
                    fname, _m,
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
    placeholder_impl_issues = _find_placeholder_experiment_issues(files)
    distinctness_issues = _find_condition_distinctness_issues(files)
    deep_warnings.extend(placeholder_impl_issues)
    deep_warnings.extend(distinctness_issues)
    for w in deep_warnings:
        logger.warning("Stage 10 deep quality: %s", w)
    complexity_warnings.extend(deep_warnings)

    # --- P1.2: If critical deep issues found, attempt one repair cycle ---
    critical_deep = [w for w in deep_warnings if _is_critical_deep_warning(w)]
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
            f"- Do NOT ship dummy/placeholder/demo experiment code or comments "
            f"saying 'replace with actual implementation'\n"
            f"- Core experiment methods such as evaluate/predict/forward must "
            f"NOT return fixed constants like 0.2 or 0.5 as a stand-in for "
            f"real computation\n"
            f"- Multi-condition experiments MUST include and CALL a startup "
            f"ablation distinctness self-check that compares outputs on the "
            f"same probe input and raises/asserts if conditions are identical\n"
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
            f"student_dim, use temperature T=4 for soft targets\n"
            f"- FILENAME COLLISIONS: If a file like config.py shadows a pip/stdlib "
            f"package, rename it (e.g., config.py → experiment_config.py) and update "
            f"ALL imports referencing it\n\n"
            f"Current code:\n{all_code_ctx}\n"
        )
        try:
            repair_resp = _chat_with_prompt(
                llm,
                _pm.system("code_generation"),
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
                deep_warnings_after.extend(_find_placeholder_experiment_issues(files))
                deep_warnings_after.extend(_find_condition_distinctness_issues(files))
                fixed = len(critical_deep) - len([
                    w for w in deep_warnings_after
                    if _is_critical_deep_warning(w)
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
        (stage_dir / "code_complexity.json").write_text(
            json.dumps(health, indent=2), encoding="utf-8"
        )

    # --- Hard gate: reject placeholder/dummy experiment implementations ---
    unresolved_placeholder_issues = _find_placeholder_experiment_issues(files)
    if unresolved_placeholder_issues:
        for issue in unresolved_placeholder_issues:
            logger.warning("Stage 10 placeholder gate: %s", issue)
            validation_log.append(f"PLACEHOLDER_IMPL: {issue}")
        (stage_dir / "validation_report.md").write_text(
            "# Code Validation Report\n\n"
            "**Status**: BLOCKED — generated experiment code still contains "
            "placeholder or demonstration-only implementations\n\n"
            + "\n".join(f"- {issue}" for issue in unresolved_placeholder_issues),
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=("validation_report.md",),
            evidence_refs=(),
        )

    # --- Hard gate: require active condition-distinctness self-checks ---
    unresolved_distinctness_issues = _find_condition_distinctness_issues(files)
    if unresolved_distinctness_issues:
        for issue in unresolved_distinctness_issues:
            logger.warning("Stage 10 distinctness gate: %s", issue)
            validation_log.append(f"DISTINCTNESS_IMPL: {issue}")
        (stage_dir / "validation_report.md").write_text(
            "# Code Validation Report\n\n"
            "**Status**: BLOCKED — generated experiment does not prove condition "
            "wiring is distinct\n\n"
            + "\n".join(f"- {issue}" for issue in unresolved_distinctness_issues),
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=("validation_report.md",),
            evidence_refs=(),
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
            f"EXPERIMENT PLAN:\n{exp_plan_prompt[:3000]}\n\n"
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
                            _pm.system("code_generation"),
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
    # BUG-171: Previous 8000-char truncation caused false-positive misalignment
    # for multi-file experiments (30-90K chars). LLM saw "[truncated]" and
    # concluded code was incomplete. Fix: build a structured summary that
    # includes file inventory + full main.py + per-file function/class headers.
    alignment_ok = True
    alignment_note = ""
    if llm is not None:
        # Build structured code summary for alignment check
        _file_inventory = []
        for _fn, _cd in files.items():
            _lines = _cd.count("\n") + 1
            _file_inventory.append(f"  {_fn}: {_lines} lines, {len(_cd)} chars")
        _inventory_block = "FILES GENERATED:\n" + "\n".join(_file_inventory)

        # BUG-179: Beast Mode may use a different entry point (e.g.
        # run_experiment.py).  Detect the actual entry point by scanning
        # for ``if __name__ == "__main__"`` in all files, preferring main.py.
        _entry_file = "main.py"
        if "main.py" not in files or not files.get("main.py", "").strip():
            for _fn, _cd in files.items():
                if 'if __name__' in _cd and '__main__' in _cd:
                    _entry_file = _fn
                    break
        elif files.get("main.py", ""):
            # main.py exists but may be a stub — if another file has the
            # real orchestration (more lines + __main__ guard), prefer it
            _main_lines = files["main.py"].count("\n")
            for _fn, _cd in files.items():
                if _fn == "main.py":
                    continue
                if ('if __name__' in _cd and '__main__' in _cd
                        and _cd.count("\n") > _main_lines * 1.5):
                    _entry_file = _fn
                    break

        _main_code = files.get(_entry_file, files.get("main.py", ""))
        _main_block = f"# --- {_entry_file} (FULL — entry point) ---\n{_main_code}"
        # Cap main.py at 12000 chars to stay within token budget
        if len(_main_block) > 12000:
            _main_block = _main_block[:12000] + "\n... [main.py truncated at 12000 chars]"

        # For other files, include imports + function/class signatures
        _other_summaries = []
        for _fn, _cd in files.items():
            if _fn == _entry_file:
                continue
            _sig_lines = []
            for _line in _cd.split("\n"):
                _stripped = _line.strip()
                if (_stripped.startswith("def ") or _stripped.startswith("class ")
                        or _stripped.startswith("async def ")
                        # BUG-209: Include import lines — they reveal which
                        # techniques/libraries are used (e.g. CosineAnnealingLR)
                        or _stripped.startswith("import ")
                        or _stripped.startswith("from ")):
                    _sig_lines.append(_line)
            if _sig_lines:
                _other_summaries.append(
                    f"# --- {_fn} (imports + signatures) ---\n"
                    + "\n".join(_sig_lines)
                )
            else:
                # Small file — include first 800 chars
                _preview = _cd[:800]
                if len(_cd) > 800:
                    _preview += f"\n... [{len(_cd) - 800} more chars]"
                _other_summaries.append(f"# --- {_fn} (preview) ---\n{_preview}")
        _other_block = "\n\n".join(_other_summaries)
        # Cap other summaries
        if len(_other_block) > 6000:
            _other_block = _other_block[:6000] + "\n... [other files truncated]"

        all_code_for_check = (
            f"{_inventory_block}\n\n{_main_block}\n\n{_other_block}"
        )
        align_prompt = (
            f"Research topic: {config.research.topic}\n\n"
            f"Experiment code:\n```python\n{all_code_for_check}\n```\n\n"
            "TASK: Evaluate whether this experiment code actually tests the "
            "stated research topic. Answer with JSON:\n"
            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n\n'
            "IMPORTANT: The code spans MULTIPLE files. The file inventory above "
            "shows ALL generated files. Only main.py is shown in full; other "
            "files show function/class signatures. Do NOT mark as misaligned "
            "just because helper files are summarized — they contain full "
            "implementations.\n\n"
            "Check specifically:\n"
            "- Does main.py orchestrate an experiment matching the topic?\n"
            "- Do the helper file signatures indicate relevant models/methods?\n"
            "- If the topic mentions a specific technique, is there evidence of "
            "its implementation (function names, class names, imports)?\n"
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
                # BUG-R6-01: Allow up to 2 regeneration attempts with re-check.
                _max_regen = 2
                for _regen_attempt in range(1, _max_regen + 1):
                    logger.info(
                        "Stage 10: Alignment regen attempt %d/%d",
                        _regen_attempt, _max_regen,
                    )
                    regen_prompt = (
                        f"The experiment code you previously generated does NOT align "
                        f"with the research topic.\n\n"
                        f"TOPIC: {config.research.topic}\n"
                        f"MISALIGNMENT: {alignment_note}\n"
                        f"SUGGESTIONS: {suggestions}\n\n"
                        f"REGENERATE the experiment code to DIRECTLY test the stated "
                        f"topic. The code MUST implement the core technique described "
                        f"in the topic, not a generic proxy.\n\n"
                        f"CRITICAL CONSTRAINTS:\n"
                        f"- You MUST implement the EXACT algorithm/method from the topic.\n"
                        f"- Do NOT substitute a deep-learning proxy (ResNet, BERT, etc.) "
                        f"when the topic describes a tabular, bandit, or game-theoretic method.\n"
                        f"- Use ONLY lightweight CPU-friendly libraries (numpy, scipy, "
                        f"sklearn) unless the topic EXPLICITLY requires deep learning.\n"
                        f"- The experiment must be self-contained and runnable without GPU.\n"
                        f"- If any file imports a local helper module, return that helper "
                        f"file too. Do not leave unresolved imports like `from models import ...` "
                        f"without a generated `models.py`.\n\n"
                        f"{pkg_hint}\n{compute_budget}\n"
                        f"PLAN:\n{exp_plan_prompt}\n\n"
                        f"Return multiple files using ```filename:xxx.py format."
                    )
                    regen_resp = _chat_with_prompt(
                        llm,
                        system=_pm.system("code_generation"),
                        user=regen_prompt,
                        max_tokens=_code_max_tokens,
                    )
                    regen_files = _extract_multi_file_blocks(regen_resp.content)
                    if not regen_files or "main.py" not in regen_files:
                        logger.warning(
                            "Stage 10: Regen attempt %d produced no main.py",
                            _regen_attempt,
                        )
                        continue
                    files = regen_files
                    for fname, code in files.items():
                        (exp_dir / fname).write_text(code, encoding="utf-8")
                    # Re-check alignment on regenerated code (BUG-171 fix)
                    _rc_inv = []
                    for _fn, _cd in files.items():
                        _rc_inv.append(f"  {_fn}: {_cd.count(chr(10))+1} lines")
                    _rc_main = files.get("main.py", "")
                    if len(_rc_main) > 12000:
                        _rc_main = _rc_main[:12000] + "\n... [truncated]"
                    _rc_sigs = []
                    for _fn, _cd in files.items():
                        if _fn == "main.py":
                            continue
                        # BUG-209: Include imports alongside signatures
                        _slines = [l for l in _cd.split("\n")
                                   if l.strip().startswith((
                                       "def ", "class ", "async def ",
                                       "import ", "from ",
                                   ))]
                        if _slines:
                            _rc_sigs.append(f"# {_fn} imports+signatures:\n" + "\n".join(_slines))
                    recheck_code = (
                        "FILES:\n" + "\n".join(_rc_inv) + "\n\n"
                        f"# main.py (FULL):\n{_rc_main}\n\n"
                        + "\n".join(_rc_sigs)
                    )
                    recheck_resp = llm.chat(
                        [{"role": "user", "content": (
                            f"Research topic: {config.research.topic}\n\n"
                            f"Experiment code:\n```python\n{recheck_code}\n```\n\n"
                            "TASK: Evaluate whether this experiment code actually tests "
                            "the stated research topic. Only main.py is shown in full; "
                            "other files show signatures only. Answer with JSON:\n"
                            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n'
                        )}],
                        system="You are a scientific code reviewer checking topic-experiment alignment.",
                        max_tokens=1024,
                    )
                    recheck_data = _safe_json_loads(recheck_resp.content, {})
                    if isinstance(recheck_data, dict) and recheck_data.get("aligned", False):
                        alignment_ok = True
                        alignment_note = f"Regenerated after alignment check (attempt {_regen_attempt})"
                        logger.info(
                            "Stage 10: Code aligned after regen attempt %d",
                            _regen_attempt,
                        )
                        break
                    else:
                        alignment_note = recheck_data.get("reason", alignment_note)
                        suggestions = recheck_data.get("suggestions", suggestions)
                        logger.warning(
                            "Stage 10: Regen attempt %d still misaligned: %s",
                            _regen_attempt, alignment_note,
                        )
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
                        _pm.system("code_generation"),
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
        except Exception as exc:
            logger.debug("Ablation validation skipped: %s", exc)

    # --- Self-contained project gate ---
    unresolved_local_imports = _find_missing_local_module_imports(files)
    if unresolved_local_imports:
        for issue in unresolved_local_imports:
            logger.warning("Stage 10 self-containment: %s", issue)
            validation_log.append(f"SELF_CONTAINED: {issue}")
        if llm is not None:
            files, unresolved_local_imports = _repair_self_contained_project(
                llm=llm,
                prompt_manager=_pm,
                files=files,
                issues=unresolved_local_imports,
                max_tokens=_code_max_tokens,
                max_repair=max_repair,
            )
            for fname, code in files.items():
                (exp_dir / fname).write_text(code, encoding="utf-8")
        if unresolved_local_imports:
            (stage_dir / "validation_report.md").write_text(
                "# Code Validation Report\n\n"
                "**Status**: BLOCKED — generated experiment project is not self-contained\n\n"
                + "\n".join(f"- {issue}" for issue in unresolved_local_imports),
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.FAILED,
                artifacts=("validation_report.md",),
                evidence_refs=(),
            )

    # --- Write spec ---
    if getattr(config.experiment, "forbid_synthetic_proxy", False):
        _proxy_signals = detect_synthetic_proxy_signals(
            {fname: code for fname, code in files.items() if fname.endswith(".py")}
        )
        if should_fail_synthetic_proxy_guard(_proxy_signals):
            guard_payload = {
                "status": "failed",
                "reason": "synthetic_proxy_detected",
                "signals": _proxy_signals,
                "timestamp": _utcnow_iso(),
            }
            (stage_dir / "real_data_guard.json").write_text(
                json.dumps(guard_payload, indent=2), encoding="utf-8"
            )
            logger.error(
                "Stage 10: Real-data guard blocked generated experiment code: %s",
                "; ".join(_proxy_signals),
            )
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.FAILED,
                artifacts=("experiment/", "real_data_guard.json"),
                evidence_refs=("stage-10/experiment/", "stage-10/real_data_guard.json"),
                error="Real-data guard blocked synthetic/proxy fallback code generation.",
            )

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

    # BUG-R6-01: Fail stage if alignment check detected persistent mismatch
    # after all regen attempts, instead of silently proceeding.
    if not alignment_ok:
        logger.error(
            "Stage 10: Persistent topic-experiment misalignment after all "
            "regen attempts. Failing stage. Reason: %s",
            alignment_note,
        )
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
            error=f"Topic-experiment misalignment: {alignment_note}",
        )

    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
    )
