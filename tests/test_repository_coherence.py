import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".rst", ".txt", ".yml", ".yaml"}
ACTIVE_DIRS = ("src", "tests", "docs", "examples", "benchmarks", "tools")


def _active_text_files():
    for dirname in ACTIVE_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path


def test_active_tree_has_no_external_cas_brand_references():
    names = ("Mathe" + "matica", "Wolf" + "ram")
    banned = re.compile(r"\b(?:" + "|".join(names) + r")\b", re.IGNORECASE)
    hits = []
    for path in _active_text_files():
        if banned.search(path.read_text(encoding="utf-8")):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_active_filenames_describe_subject_not_development_stage():
    stages = ("new_" + "capabilities", "pha" + "se", "mile" + "stone")
    stage_names = re.compile(r"(?:^|_)(?:" + "|".join(stages) + r")[0-9]*(?:_|\.|$)")
    hits = []
    for dirname in ACTIVE_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and stage_names.search(path.name):
                hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_production_code_has_no_broad_exception_handlers_or_assert_control_flow():
    import ast

    violations = []
    for path in (ROOT / "src" / "asymptotic").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
            ):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: broad Exception handler"
                )
            if isinstance(node, ast.Assert):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: production assert")
    assert violations == []


def test_tests_do_not_use_monkeypatch_fixture():
    import ast

    hits = []
    fixture_name = "monkey" + "patch"
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                if any(arg.arg == fixture_name for arg in args):
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert hits == []


def test_production_scopes_have_no_overridden_function_definitions():
    import ast

    duplicates = []

    def visit_scope(path, body, scope):
        seen = set()
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen:
                    duplicates.append(f"{path}:{node.lineno}:{scope}.{node.name}")
                seen.add(node.name)
                visit_scope(path, node.body, f"{scope}.{node.name}")

    for path in (ROOT / "src" / "asymptotic").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visit_scope(path.relative_to(ROOT), tree.body, path.stem)
    assert duplicates == []


def test_local_names_and_parameters_remain_readable_and_compact():
    import ast

    too_long = []
    for path in (ROOT / "src" / "asymptotic").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                names.extend(arg.arg for arg in args)
                if node.args.vararg:
                    names.append(node.args.vararg.arg)
                if node.args.kwarg:
                    names.append(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append(node.id)
            for name in names:
                if len(name) > 24:
                    too_long.append(f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}:{name}")
    assert too_long == []


def test_active_text_has_no_trailing_whitespace():
    hits = []
    for path in _active_text_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.rstrip() != line:
                hits.append(f"{path.relative_to(ROOT)}:{number}")
    assert hits == []


def test_project_identity_and_license_are_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    maintained = []
    for base in (
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "examples",
        ROOT / "src",
        ROOT / "tests",
    ):
        paths = [base] if base.is_file() else list(base.rglob("*.py")) + list(base.rglob("*.md"))
        for path in paths:
            maintained.append(path.read_text(encoding="utf-8"))
    text = "\n".join(maintained)
    assert 'license = "GPL-3.0-only"' in pyproject
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3" in license_text
    import re

    prohibited = ("Mathe" + "matica", "Wolf" + "ram")
    assert not any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in prohibited)


def test_sum_method_registry_is_shared_with_statistics():
    from asymptotic.probability import _STATISTICAL_METHODS
    from asymptotic.sums import DISCRETE_STAT_METHODS, SUM_METHODS

    assert SUM_METHODS <= DISCRETE_STAT_METHODS
    assert DISCRETE_STAT_METHODS <= _STATISTICAL_METHODS
    assert DISCRETE_STAT_METHODS - SUM_METHODS == {"pmf", "sum"}
