"""Repository-wide guard for the deliberately small root namespace."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import asymptotic

_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_SCAN_DIRS = (Path("tests"), Path("examples"), Path("benchmarks"))
_DOCS = (Path("README.md"), *sorted(Path("docs").glob("*.md")))


def _retired_root_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    public = set(asymptotic.__all__)
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "asymptotic"
        for alias in node.names
        if alias.name not in public
    }


def test_repository_python_uses_only_primary_root_imports():
    violations: dict[str, list[str]] = {}
    for directory in _SCAN_DIRS:
        for path in directory.rglob("*.py"):
            retired = sorted(_retired_root_imports(path.read_text()))
            if retired:
                violations[str(path)] = retired
    assert violations == {}


def test_documentation_python_fences_use_only_primary_root_imports():
    violations: dict[str, list[str]] = {}
    for path in _DOCS:
        retired: set[str] = set()
        for block in _PYTHON_FENCE.findall(path.read_text()):
            try:
                retired.update(_retired_root_imports(block))
            except SyntaxError:
                # Some documentation fragments are intentionally incomplete.
                continue
        if retired:
            violations[str(path)] = sorted(retired)
    assert violations == {}
