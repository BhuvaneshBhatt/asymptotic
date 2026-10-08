import sympy as sp

from asymptotic.limits import limit

x, y = sp.symbols("x y", real=True)


def test_mixed_infinite_finite_target_uses_joint_projective_germ():
    result = limit(sp.exp(-x) + y**2, (x, y), (sp.oo, 0), return_result=True)
    assert result.status.name == "PROVED"
    assert result.value == 0
