import ast
from pathlib import Path


def test_only_formal_power_helper_forces_power_identities():
    source = Path(__file__).parents[1] / "src" / "asymptotic"
    offenders = []
    for path in sorted(source.glob("*.py")):
        if path.name == "_power_simplify.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else None
            )
            if name not in {"powsimp", "expand_power_base"}:
                continue
            if any(
                keyword.arg == "force"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                offenders.append(path.name)
                break
    assert offenders == []


def test_mixed_power_simplification_preserves_coefficient_branches():
    import sympy as sp

    from asymptotic._power_simplify import mixed_powsimp

    a, b, x = sp.symbols("a b x")
    coefficient = sp.sqrt(a) * sp.sqrt(b)
    result = mixed_powsimp(coefficient, x ** sp.Rational(1, 2) * x ** sp.Rational(1, 2))
    assert result.has(sp.sqrt(a))
    assert result.has(sp.sqrt(b))
    assert not result.has(sp.sqrt(a * b))
    assert sp.simplify(result / (coefficient * x)) == 1


def test_recursive_logexp_parser_does_power_identity():
    import sympy as sp

    from asymptotic.logexp_transseries import canonical_recursive_logexp_monomial

    a, b, x = sp.symbols("a b x", nonzero=True)
    coefficient, monomial = canonical_recursive_logexp_monomial(
        sp.sqrt(a) * sp.sqrt(b) / x, x
    )
    assert coefficient == sp.sqrt(a) * sp.sqrt(b)
    assert monomial.expression == 1 / x
