import ast
from dataclasses import fields
from pathlib import Path

from asymptotic.instrumentation import SymbolicMetrics


def test_every_recorded_symbolic_event_has_a_declared_counter():
    declared = {field.name for field in fields(SymbolicMetrics)}
    used = set()
    source = Path(__file__).parents[1] / "src" / "asymptotic"
    for path in source.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "record_symbolic_event":
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                used.add(node.args[0].value)
    assert used <= declared
