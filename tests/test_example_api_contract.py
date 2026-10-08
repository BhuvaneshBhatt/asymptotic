"""Keep user-facing examples on the documented public API."""

from __future__ import annotations

import ast
from pathlib import Path

from asymptotic._api_manifest import PRIMARY_API

EXAMPLES = Path(__file__).parents[1] / "examples"


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "asymptotic":
            names.update(alias.name for alias in node.names)
    return names


def test_public_examples_use_primary_api() -> None:
    for path in EXAMPLES.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imported = _top_level_imports(path)
        assert imported <= PRIMARY_API, (path.name, sorted(imported - PRIMARY_API))


def test_expert_examples_are_separated() -> None:
    expert = {path.name for path in (EXAMPLES / "expert").glob("*.py")}
    assert {
        "certified_green.py",
        "computational_complexity.py",
        "singular_implicit.py",
    } <= expert
