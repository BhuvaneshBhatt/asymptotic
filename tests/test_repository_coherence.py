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


def test_repository_root_contains_no_scratch_validation_artifacts():
    forbidden = {
        "matrix-integration.json",
        "matrix-tracks.json",
        "mma_multivar_excerpt.txt",
        "timeout-recheck-15s.json",
        "unsafe-audit-after.json",
    }
    present = {path.name for path in ROOT.iterdir() if path.is_file()}
    assert present.isdisjoint(forbidden)


def test_active_filenames_describe_subject_not_development_stage():
    stages = ("new_" + "capabilities", "pha" + "se", "mile" + "stone")
    stage_names = re.compile(r"(?:^|_)(?:" + "|".join(stages) + r")[0-9]*(?:_|\.|$)")
    # These files concern simultaneous modular and trigonometric angles.
    mathematical_phase_files = {
        "src/asymptotic/joint_phase_germs.py",
        "tests/limits/univariate/test_joint_phase_germs.py",
    }
    hits = []
    for dirname in ACTIVE_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (
                path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and stage_names.search(path.name)
                and str(path.relative_to(ROOT)) not in mathematical_phase_files
            ):
                hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_production_proof_boundaries():
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
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: production assert"
                )
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
                    too_long.append(
                        f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}:{name}"
                    )
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
    assert 'license = "GPL-3.0-only"' in pyproject
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3" in license_text


def test_sum_method_registry_is_shared_with_statistics():
    from asymptotic.probability import _STATISTICAL_METHODS
    from asymptotic.sums import DISCRETE_STAT_METHODS, SUM_METHODS

    assert SUM_METHODS <= DISCRETE_STAT_METHODS
    assert DISCRETE_STAT_METHODS <= _STATISTICAL_METHODS
    assert DISCRETE_STAT_METHODS - SUM_METHODS == {"pmf", "sum"}


def test_pyproject_prerelease_extras_and_pytest_config_are_valid():
    import tomllib

    raw = (ROOT / "pyproject.toml").read_bytes()
    project = tomllib.loads(raw.decode("utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert {"test", "docs", "release"} <= extras.keys()
    assert "ini_options" in project["tool"]["pytest"]


def test_active_tree_uses_timeless_capability_language():
    banned = (
        "future " + "work",
        "for a " + "later",
        "previous private " + "inconsistency",
        "keep their previous " + "meaning",
        "currently " + "requires",
    )
    hits = []
    for path in _active_text_files():
        text = path.read_text(encoding="utf-8").lower()
        if any(phrase in text for phrase in banned):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_test_names_are_compact():
    import ast

    long_names = []
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                and len(node.name) > 75
            ):
                long_names.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert long_names == []


def test_active_prose_avoids_development_history_language():
    banned = (
        "backward compatibility",
        "backwards compatibility",
        "compatibility shim",
        "legacy api",
        "old api",
        "new api",
        "implementation phase",
        "development milestone",
    )
    hits = []
    prose = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    for path in prose:
        text = path.read_text(encoding="utf-8").lower()
        if any(phrase in text for phrase in banned):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_documented_primary_api_count_matches_manifest():
    from asymptotic._api_manifest import PRIMARY_API

    expected = f"{len(PRIMARY_API)} primary entry points"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api_guide = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    classification = (ROOT / "docs" / "api-classification.md").read_text(
        encoding="utf-8"
    )
    assert expected in readme
    assert expected in api_guide
    assert f"**{len(PRIMARY_API)}** primary entry points" in classification


def test_class_field_names():
    import ast

    duplicates = []
    for path in (ROOT / "src/asymptotic").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                names = [
                    member.target.id
                    for member in node.body
                    if isinstance(member, ast.AnnAssign)
                    and isinstance(member.target, ast.Name)
                ]
                if len(names) != len(set(names)):
                    duplicates.append(f"{path.name}:{node.name}")
    assert duplicates == []
