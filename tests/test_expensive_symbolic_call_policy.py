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


def test_core_routes_avoid_unbounded_sympy_calls():
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
    # These are explicit mathematical computations requested by their APIs,
    # not hidden theorem-applicability or certification decisions.
    assert observed == {
        "multiseries.py": {"integrate"},
        "nested.py": {"integrate"},
        "remainder.py": {"integrate"},
        "olver_coefficients.py": {"integrate"},
    }
