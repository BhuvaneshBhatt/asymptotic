import sympy as sp

from asymptotic.limits import LimitStatus, limit


def test_infinite_target_dispatch_proves_projective_radial_limit():
    x, y = sp.symbols("x y", real=True)
    r = limit(
        (x * x + y * y) / (x * x + y * y + 1),
        (x, y),
        (sp.oo, sp.oo),
        return_result=True,
    )
    assert r.status is LimitStatus.PROVED and r.value == 1


def test_infinite_target_dispatch_preserves_direction_conflict():
    x, y = sp.symbols("x y", real=True)
    r = limit(x / y, (x, y), (sp.oo, sp.oo), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_projective_witnesses_use_original_coordinates():
    x, y = sp.symbols("x y", real=True)
    expression = x / y
    result = limit(expression, (x, y), (sp.oo, sp.oo), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert len(result.evidence) == 2
    for witness in result.evidence:
        paths = dict(witness.substitutions)
        assert set(paths) == {x, y}
        (parameter,) = set().union(*(path.free_symbols for path in paths.values()))
        assert parameter.is_positive is True
        assert all(
            sp.limit(path, parameter, 0, dir="+") == sp.oo for path in paths.values()
        )
        assert sp.cancel(expression.subs(paths, simultaneous=True)) == witness.value
    assert result.evidence[0].value != result.evidence[1].value
