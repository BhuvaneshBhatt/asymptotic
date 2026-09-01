from __future__ import annotations

import ast
from pathlib import Path


def _direct_sympy_calls(path: Path, names: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sp"
            and node.func.attr in names
        ):
            found.add(node.func.attr)
    return found


def test_general_recurrence_solver_is_isolated_in_symbolic_policy():
    root = Path(__file__).resolve().parents[1] / "src" / "asymptotic"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "_symbolic_policy.py":
            continue
        if "rsolve" in _direct_sympy_calls(path, {"rsolve"}):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_core_solve_and_probability_routes_do_not_call_generic_solve_or_limit_directly():
    root = Path(__file__).resolve().parents[1] / "src" / "asymptotic"
    assert "solve" not in _direct_sympy_calls(root / "solve.py", {"solve"})
    assert "limit" not in _direct_sympy_calls(root / "probability.py", {"limit"})


def test_remaining_direct_generic_sympy_calls_are_explicitly_audited():
    root = Path(__file__).resolve().parents[1] / "src" / "asymptotic"
    expensive = {"rsolve", "solve", "limit", "integrate", "ask"}
    observed: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        if path.name == "_symbolic_policy.py":
            continue
        calls = _direct_sympy_calls(path, expensive)
        if calls:
            observed[path.relative_to(root).as_posix()] = calls
    # _power_simplify performs a bounded branch-safety assumption query and is
    # deliberately below _symbolic_policy to avoid a dependency cycle.
    # multiseries/nested integrations are explicit user-requested calculus
    # operations, not hidden certification/search fallbacks.
    assert observed == {
        "_power_simplify.py": {"ask"},
        "multiseries.py": {"integrate"},
        "nested.py": {"integrate"},
    }
