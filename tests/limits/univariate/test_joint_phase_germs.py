import pytest
import sympy as sp
from flint import arb, ctx

from asymptotic import limit
from asymptotic.integral_ray_germs import (
    gaussian_root_side_certificate,
    vertical_expint_certificate,
)
from asymptotic.joint_phase_germs import (
    PhaseHittingIndex,
    _phase_inverse,
    joint_phase_certificate,
    phase_hitting_index,
)
from asymptotic.limit_models import LimitStatus


@pytest.mark.parametrize(
    "kind,phase,expected",
    [
        ("mod", "x", (0, 1)),
        ("sum", "x", (3, sp.Rational(9, 2))),
        ("product", "x", (sp.Rational(1, 4), sp.Rational(3, 2))),
        ("min", "x", (sp.Rational(1, 2), sp.Rational(2, 3))),
        ("max", "x", (1, sp.Rational(3, 2))),
        ("min", "x**2+1", (sp.Rational(1, 2), sp.Rational(2, 3))),
        ("sum", "log(x)", (3, sp.Rational(9, 2))),
        ("product", "x**3-x", (sp.Rational(1, 4), sp.Rational(3, 2))),
        ("offset_product", "x**3-x", (sp.Rational(5, 4), sp.Rational(5, 2))),
    ],
)
def test_joint_values(kind, phase, expected):
    x = sp.Symbol("x", real=True)
    p = sp.sympify(phase, locals={"x": x})
    a = sp.cos(sp.expand_mul(sp.sqrt(2) * p)) / 2 + 1
    b = sp.Mod(p, 2) / 2 + sp.Rational(1, 2)
    c = sp.cos(sp.sin(p) ** 2)
    ratio = (x + 1) / (x - 1)
    expressions = {
        "mod": sp.Mod(p, 2),
        "sum": ratio + a + b + c,
        "product": ratio * a * b * c,
        "offset_product": ratio * a * b * c + (sp.log(x) + 1) / sp.log(x),
        "min": sp.Min(2 * ratio / 3, a, b, c),
        "max": sp.Max(2 * ratio / 3, a, b, c),
    }
    result = limit(expressions[kind], x, sp.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    evidence = [
        e for e in result.evidence if e.method == "attained_joint_modular_phase"
    ]
    assert tuple(e.value for e in evidence) == expected
    for e in evidence:
        assert e.substitutions[0][0] == x
        seq = e.substitutions[0][1]
        assert seq.free_symbols
        if kind != "mod":
            assert len(seq.atoms(PhaseHittingIndex)) == 1


@pytest.mark.parametrize("j", [2, 4, 8, 16])
@pytest.mark.parametrize("residue,cosine", [(0, -1), (1, 1)])
def test_first_hit(j, residue, cosine):
    m = phase_hitting_index(j, residue, cosine)
    assert m >= j
    with ctx.workprec(160):
        tolerance = arb(1) / j
        for k in range(j, m + 1):
            p = arb(2 * k + residue)
            errors = (abs(p.sin()), abs((arb(2).sqrt() * p).cos() - cosine))
            if k == m:
                assert all(e < tolerance for e in errors)
            else:
                assert any(e >= tolerance for e in errors)
    assert (2 * m + residue) % 2 == residue


@pytest.mark.parametrize(
    "phase", ["x", "2*x+3", "x**2+1", "log(x)", "log(2*x+3)", "x**3-x"]
)
def test_inverse_sequences(phase):
    x = sp.Symbol("x", real=True)
    p = sp.sympify(phase, locals={"x": x})
    inverse = _phase_inverse(p, x)
    for residue, cosine in ((0, -1), (1, 1)):
        m = phase_hitting_index(8, residue, cosine)
        value = sp.sympify(inverse(2 * m + residue))
        assert value.is_positive is True
        assert abs((p.subs(x, value) - (2 * m + residue)).evalf(50)) < sp.Float("1e-40")
        assert abs((value - 1).evalf(40)) > sp.Float("1e-20")
        assert abs(sp.log(value).evalf(40)) > sp.Float("1e-20")


@pytest.mark.parametrize(
    "expression",
    [
        "x*sin(x)+Mod(x,2)",
        "x*Mod(x,2)",
        "Mod(x,2)/(cos(sqrt(2)*x)+1)",
        "Mod(x,2)+cos(sqrt(3)*x)",
        "Mod(-x,2)",
        "Mod(x,3)",
        "Mod(x,2)/x",
        "Mod(x,2)+sin(x**2)",
    ],
)
def test_joint_rejections(expression):
    x = sp.Symbol("x", real=True)
    e = sp.sympify(expression, locals={"x": x})
    assert joint_phase_certificate(e, x, sp.oo, sp.S.true, sp.S.true) is None


def test_sampling_guards():
    x = sp.Symbol("x", real=True)
    e = sp.Mod(x, 2)
    assert joint_phase_certificate(e, x, sp.oo, x > 3, sp.S.true) is None
    assert joint_phase_certificate(e, x, sp.oo, sp.S.true, x > 3) is None
    n = sp.Symbol("n", integer=True)
    assert joint_phase_certificate(sp.Mod(n, 2), n, sp.oo, sp.S.true, sp.S.true) is None


@pytest.mark.parametrize("args", [(0, 0, -1), (2, 2, -1), (2, 0, 0), (2, 0, -1, 0)])
def test_search_validation(args):
    with pytest.raises(ValueError):
        phase_hitting_index(*args)


def test_search_budget():
    with pytest.raises(ValueError, match="max_checks"):
        phase_hitting_index(16, 0, -1, max_checks=1)


@pytest.mark.parametrize(
    "coefficient,values", [(1, (1, -sp.oo)), (-2, (1, -sp.oo)), (sp.I, (-sp.oo, 1))]
)
def test_gaussian_sides(coefficient, values):
    x = sp.Symbol("x", real=True)
    z = coefficient / sp.sqrt(x)
    e = sp.sqrt(sp.pi) * z * sp.exp(-z * z) * sp.erfi(z)
    result = gaussian_root_side_certificate(e, x, sp.S.Zero, sp.S.true, sp.S.true)
    assert result[:2] == (LimitStatus.DOES_NOT_EXIST, None)
    assert tuple(item.value for item in result[2]) == values
    assert all(item.substitutions for item in result[2])
    right = gaussian_root_side_certificate(e, x, sp.S.Zero, x > 0, sp.S.true)
    assert right[:2] == (LimitStatus.PROVED, values[0])


def test_gaussian_rejections():
    x = sp.Symbol("x", real=True)
    for coefficient in (1 + sp.I, 0):
        z = coefficient / sp.sqrt(x)
        e = sp.sqrt(sp.pi) * z * sp.exp(-z * z) * sp.erfi(z)
        assert (
            gaussian_root_side_certificate(e, x, sp.S.Zero, sp.S.true, sp.S.true)
            is None
        )
    z = 1 / sp.sqrt(x)
    e = 2 * sp.sqrt(sp.pi) * z * sp.exp(-z * z) * sp.erfi(z)
    assert gaussian_root_side_certificate(e, x, sp.S.Zero, sp.S.true, sp.S.true) is None


@pytest.mark.parametrize("order", [1, 2, 10, 16])
@pytest.mark.parametrize("slope", [sp.I, -2 * sp.I])
def test_vertical_expint(order, slope):
    x = sp.Symbol("x", real=True)
    e = sp.expint(order, 3 + sp.I + slope * x)
    result = vertical_expint_certificate(e, x, sp.oo, sp.S.true, sp.S.true)
    assert result[:2] == (LimitStatus.PROVED, sp.S.Zero)
    assert result[2].value == 0
    assert result[2].substitutions[0][0] == x


def test_expint_parameter():
    x, v = sp.symbols("x v", real=True)
    e = sp.expint(1, -sp.I * x * v - v)
    result = vertical_expint_certificate(e, x, sp.oo, sp.S.true, v > 0)
    assert result[:2] == (LimitStatus.PROVED, sp.S.Zero)
    assert vertical_expint_certificate(e, x, sp.oo, sp.S.true, sp.S.true) is None
    for bad in (
        sp.expint(1, sp.I * x * 0 - 1),
        sp.expint(v, sp.I * x),
        sp.expint(1, x),
        sp.expint(1, sp.I * x * x),
    ):
        assert vertical_expint_certificate(bad, x, sp.oo, sp.S.true, sp.S.true) is None
