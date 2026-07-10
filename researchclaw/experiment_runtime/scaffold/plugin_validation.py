"""Static validation for bounded detector plugins."""

from __future__ import annotations

import ast


PLUGIN_BANNED_IMPORTS = frozenset(
    {
        "aiohttp",
        "cffi",
        "ctypes",
        "http",
        "httpx",
        "importlib",
        "multiprocessing",
        "os",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "threading",
        "urllib",
    }
)

PLUGIN_BANNED_CALLS = {
    ("io", "open"),
    ("os", "system"),
    ("os", "popen"),
    ("shutil", "rmtree"),
}


class PluginValidationError(ValueError):
    """Raised when a generated detector plugin violates the scaffold API."""


def validate_detector_plugin(code: str) -> None:
    """Fail closed unless `code` defines a bounded DetectorPlugin class."""
    if not code.strip():
        raise PluginValidationError("detector_plugin.py is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PluginValidationError(f"detector_plugin.py syntax error: {exc}") from exc

    _check_banned_imports(tree)
    _check_banned_calls(tree)
    _check_file_writes(tree)

    plugin = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DetectorPlugin"
        ),
        None,
    )
    if plugin is None:
        raise PluginValidationError("DetectorPlugin class missing")

    methods = {
        node.name
        for node in plugin.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = {"fit", "predict", "describe"} - methods
    if missing:
        raise PluginValidationError(
            "DetectorPlugin missing method(s): " + ", ".join(sorted(missing))
        )
    if any(isinstance(node, ast.AsyncFunctionDef) for node in plugin.body):
        raise PluginValidationError("DetectorPlugin methods must be synchronous")


def _check_banned_imports(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in PLUGIN_BANNED_IMPORTS:
                    raise PluginValidationError(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in PLUGIN_BANNED_IMPORTS:
                raise PluginValidationError(f"banned import: {module}")


def _check_banned_calls(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if (func.value.id, func.attr) in PLUGIN_BANNED_CALLS:
                raise PluginValidationError(
                    f"banned call: {func.value.id}.{func.attr}"
                )
        elif isinstance(func, ast.Name) and func.id in {"eval", "exec", "__import__"}:
            raise PluginValidationError(f"banned call: {func.id}")


def _check_file_writes(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_open = isinstance(func, ast.Name) and func.id == "open"
        is_path_write = isinstance(func, ast.Attribute) and func.attr in {
            "write_text",
            "write_bytes",
            "open",
        }
        if not (is_open or is_path_write):
            continue
        mode = ""
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = str(kw.value.value)
        if is_path_write or any(flag in mode for flag in ("w", "a", "x", "+")):
            raise PluginValidationError("plugin may not write files")
