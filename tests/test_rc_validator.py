# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import pytest

from researchclaw.experiment.validator import (
    BANNED_MODULES,
    DANGEROUS_BUILTINS,
    DANGEROUS_CALLS,
    CodeValidation,
    ValidationIssue,
    check_filename_collisions,
    extract_imports,
    format_issues_for_llm,
    validate_code,
    validate_imports,
    validate_security,
    validate_syntax,
)


def _call_source(name: str) -> str:
    top = name.split(".")[0]
    lines: list[str] = []
    if top in {"os", "subprocess", "shutil"}:
        lines.append(f"import {top}")
    lines.append(f"{name}()")
    return "\n".join(lines)


def test_validate_syntax_accepts_valid_code():
    result = validate_syntax("x = 1\nif x > 0:\n    x += 1")

    assert result.ok is True
    assert result.issues == []


def test_validate_syntax_reports_syntax_error_with_location():
    result = validate_syntax("def bad(:\n    pass")

    assert result.ok is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "syntax"
    assert issue.line == 1
    assert issue.col is not None
    assert issue.message


@pytest.mark.parametrize("code", ["", "   \n\t  ", "# comment only\n# still comment"])
def test_validate_syntax_accepts_empty_whitespace_and_comment_only(code: str):
    result = validate_syntax(code)

    assert result.ok is True
    assert result.issues == []


def test_validate_security_accepts_safe_code():
    code = 'import os\nvalue = os.path.join("a", "b")\nprint(value)'
    result = validate_security(code)

    assert result.ok is True
    assert result.issues == []


def test_validate_security_skips_when_code_has_syntax_error():
    result = validate_security("def broken(:\n    pass")

    assert result.ok is True
    assert result.issues == []


@pytest.mark.parametrize("builtin_name", sorted(DANGEROUS_BUILTINS))
def test_validate_security_flags_every_dangerous_builtin_call(builtin_name: str):
    if builtin_name == "__import__":
        code = '__import__("os")'
    elif builtin_name == "compile":
        code = 'compile("x = 1", "<string>", "exec")'
    else:
        code = f'{builtin_name}("print(1)")'

    result = validate_security(code)

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "security"
    assert issue.message == f"Dangerous built-in call: {builtin_name}()"


@pytest.mark.parametrize("call_name", sorted(DANGEROUS_CALLS))
def test_validate_security_flags_every_dangerous_call(call_name: str):
    result = validate_security(_call_source(call_name))

    messages = [issue.message for issue in result.issues]
    assert f"Dangerous call: {call_name}()" in messages
    assert all(issue.severity == "error" for issue in result.issues)
    assert all(issue.category == "security" for issue in result.issues)


@pytest.mark.parametrize("module_name", sorted(BANNED_MODULES))
def test_validate_security_flags_every_banned_import(module_name: str):
    result = validate_security(f"import {module_name}")

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "security"
    assert issue.message == f"Banned module import: {module_name}"


@pytest.mark.parametrize("module_name", sorted(BANNED_MODULES))
def test_validate_security_flags_every_banned_from_import(module_name: str):
    result = validate_security(f"from {module_name} import x")

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "security"
    assert issue.message == f"Banned module import: from {module_name}"


def test_validate_imports_recognizes_stdlib_modules_by_default():
    result = validate_imports("import json\nfrom math import sqrt")

    assert result.ok is True
    assert result.warnings == []


def test_validate_imports_warns_for_unavailable_package():
    result = validate_imports("import totally_missing_pkg")

    assert result.ok is True
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.severity == "warning"
    assert warning.category == "import"
    assert (
        warning.message
        == "Module 'totally_missing_pkg' may not be available in sandbox"
    )


def test_validate_imports_respects_custom_available_set():
    result = validate_imports(
        "import alpha\nimport beta\nimport gamma",
        available={"alpha", "gamma"},
    )

    assert [w.message for w in result.warnings] == [
        "Module 'beta' may not be available in sandbox",
    ]


def test_validate_imports_returns_no_warnings_for_syntax_error_input():
    result = validate_imports("def bad(:\n    pass", available=set())

    assert result.ok is True
    assert result.warnings == []


@pytest.mark.parametrize("code", ["", "   \n\t  ", "# comment only"])
def test_validate_imports_handles_empty_like_inputs(code: str):
    result = validate_imports(code, available=set())

    assert result.ok is True
    assert result.warnings == []


def test_validate_code_combines_security_and_import_issues_in_order():
    code = 'import os\nos.system("echo hi")\nimport unknown_mod'
    result = validate_code(code, available_packages={"os"})

    assert result.ok is False
    assert [i.category for i in result.issues] == ["security", "import"]
    assert result.issues[0].message == "Dangerous call: os.system()"
    assert (
        result.issues[1].message
        == "Module 'unknown_mod' may not be available in sandbox"
    )


def test_validate_code_short_circuits_after_syntax_error():
    result = validate_code("def bad(:\n    pass")

    assert len(result.issues) == 1
    assert result.issues[0].category == "syntax"


def test_validate_code_skip_security_excludes_security_issues():
    code = 'import os\nos.system("echo hi")\nimport unknown_mod'
    result = validate_code(code, available_packages={"os"}, skip_security=True)

    assert [i.category for i in result.issues] == ["import"]


def test_validate_code_skip_imports_excludes_import_warnings():
    code = 'import os\nos.system("echo hi")\nimport unknown_mod'
    result = validate_code(code, available_packages={"os"}, skip_imports=True)

    assert all(issue.category == "security" for issue in result.issues)
    assert len(result.issues) == 1


def test_validate_code_skip_both_returns_clean_for_safe_code():
    result = validate_code("x = 1", skip_security=True, skip_imports=True)

    assert result.ok is True
    assert result.issues == []


def test_validate_code_uses_available_packages_for_import_validation():
    code = "import alpha\nimport beta"
    result = validate_code(code, available_packages={"alpha"})

    assert [i.message for i in result.issues] == [
        "Module 'beta' may not be available in sandbox",
    ]


def test_extract_imports_supports_import_and_from_import_styles():
    code = (
        "import os\nimport numpy as np\nfrom pandas import DataFrame\nfrom x.y import z"
    )

    assert extract_imports(code) == {"os", "numpy", "pandas", "x"}


def test_extract_imports_supports_multiple_aliases_and_dedupes():
    code = "import os.path, os, json as js\nfrom json import loads"

    assert extract_imports(code) == {"os", "json"}


def test_extract_imports_ignores_relative_import_without_module_name():
    assert extract_imports("from . import local_mod") == set()


def test_extract_imports_includes_relative_import_with_module_name():
    assert extract_imports("from ..pkg.sub import thing") == {"pkg"}


def test_extract_imports_returns_empty_set_for_syntax_error():
    assert extract_imports("def bad(:\n    pass") == set()


@pytest.mark.parametrize("code", ["", "   \n\t", "# comment only"])
def test_extract_imports_handles_empty_like_inputs(code: str):
    assert extract_imports(code) == set()


def test_format_issues_for_llm_returns_no_issues_message_when_clean():
    assert format_issues_for_llm(CodeValidation()) == "No issues found."


def test_format_issues_for_llm_formats_issues_with_and_without_line():
    validation = CodeValidation(
        issues=[
            ValidationIssue(
                severity="error",
                category="syntax",
                message="invalid syntax",
                line=3,
            ),
            ValidationIssue(
                severity="warning",
                category="import",
                message="Module 'x' may be missing",
                line=None,
            ),
        ]
    )

    formatted = format_issues_for_llm(validation)

    assert "- [ERROR] (syntax) invalid syntax @ line 3" in formatted
    assert (
        "- [WARNING] (import) Module 'x' may be missing @ unknown location" in formatted
    )


def test_format_issues_for_llm_preserves_issue_order():
    validation = CodeValidation(
        issues=[
            ValidationIssue(severity="warning", category="import", message="first"),
            ValidationIssue(
                severity="error", category="security", message="second", line=9
            ),
        ]
    )

    formatted = format_issues_for_llm(validation).splitlines()

    assert formatted[0] == "- [WARNING] (import) first @ unknown location"
    assert formatted[1] == "- [ERROR] (security) second @ line 9"


def test_code_validation_ok_true_when_no_errors_present():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="warning", category="import", message="warn")]
    )

    assert validation.ok is True


def test_code_validation_ok_false_when_error_present():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="error", category="syntax", message="bad")]
    )

    assert validation.ok is False


def test_code_validation_errors_and_warnings_filter_correctly():
    err = ValidationIssue(severity="error", category="security", message="danger")
    warn = ValidationIssue(
        severity="warning", category="import", message="maybe missing"
    )
    validation = CodeValidation(issues=[err, warn])

    assert validation.errors == [err]
    assert validation.warnings == [warn]


def test_code_validation_summary_for_no_issues():
    assert CodeValidation().summary() == "Code validation passed."


def test_code_validation_summary_for_errors_only():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="error", category="syntax", message="bad")]
    )

    assert validation.summary() == "Code validation: 1 error(s)"


def test_code_validation_summary_for_warnings_only():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="warning", category="import", message="warn")]
    )

    assert validation.summary() == "Code validation: 1 warning(s)"


def test_code_validation_summary_for_errors_and_warnings():
    validation = CodeValidation(
        issues=[
            ValidationIssue(severity="error", category="syntax", message="bad"),
            ValidationIssue(severity="warning", category="import", message="warn"),
        ]
    )

    assert validation.summary() == "Code validation: 1 error(s), 1 warning(s)"


# ---------------------------------------------------------------------------
# check_filename_collisions (BUG-202)
# ---------------------------------------------------------------------------


def test_filename_collision_detects_config_py():
    """BUG-202: config.py shadows pip 'config' package."""
    warnings = check_filename_collisions({"config.py": "x = 1", "main.py": "print(1)"})
    assert len(warnings) == 1
    assert "shadows stdlib/pip" in warnings[0]
    assert "config" in warnings[0]


def test_filename_collision_detects_stdlib_shadows():
    """Filenames shadowing stdlib modules should be flagged."""
    warnings = check_filename_collisions({"json.py": "x = 1"})
    assert len(warnings) == 1
    assert "json" in warnings[0]


def test_filename_collision_allows_safe_names():
    """Normal experiment filenames should not trigger warnings."""
    files = {
        "main.py": "print(1)",
        "models.py": "class M: pass",
        "training.py": "def train(): pass",
        "data_loader.py": "def load(): pass",
        "experiment_config.py": "LR = 0.01",
        "requirements.txt": "torch",
    }
    warnings = check_filename_collisions(files)
    assert warnings == []


def test_filename_collision_multiple_shadows():
    """Multiple shadowing files should each produce a warning."""
    files = {"config.py": "", "logging.py": "", "main.py": ""}
    warnings = check_filename_collisions(files)
    assert len(warnings) == 2
