"""Native contracts preserve approach, branch, condition and attainment semantics."""

import pytest
import sympy as s

from asymptotic import (
    Circle,
    ExactClusterResult,
    cluster_set,
    complex_limit,
    complex_ray_limit,
    limit,
)
from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_contracts import conditional_reference, parse_reference
from asymptotic.reference_normalization import scalar_reference_namespace


@pytest.mark.parametrize(
    "expression,expected",
    [
        (lambda z: z * z, 0),
        (lambda z: 1 / z, s.zoo),
        (lambda z: s.sin(z) / z, 1),
        (lambda z: s.Abs(z), 0),
    ],
)
def test_whole_plane(expression, expected):
    z = s.Symbol("z")
    assert complex_limit(expression(z), z, 0) == expected


def test_complex_pole_conventions():
    z = s.Symbol("z")
    assert complex_limit(s.tan(z), z, s.pi / 2) == s.zoo
    assert complex_ray_limit(1 / z, z, 0, ray=s.I) == DirectionalInfinity(-s.I)
    assert complex_ray_limit(1 / z, z, 0, ray=-s.I) == DirectionalInfinity(s.I)
    with pytest.raises(ValueError):
        DirectionalInfinity(0)


def test_whole_plane_attained_paths():
    z = s.Symbol("z")
    result = complex_limit(
        s.re(z) * s.im(z) / (s.re(z) ** 2 + s.im(z) ** 2), z, 0, return_result=True
    )
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [e for e in result.evidence if e.substitutions]
    assert len({e.value for e in witnesses}) >= 2
    for e in witnesses:
        sequence = e.substitutions[0][1]
        t = next(iter(sequence.free_symbols))
        endpoint = s.oo if t.is_integer is True else s.S.Zero
        assert s.limit(sequence, t, endpoint, dir="+") == 0
        assert (
            s.simplify(result.expression.subs(z, sequence).expand(complex=True))
            == e.value
        )


@pytest.mark.parametrize(
    "point,direction,frequency,radius",
    [(0, "+", 1, 1), (0, "-", -8, s.exp(8 * s.pi)), (s.oo, None, -2, 1)],
)
def test_attained_circle(point, direction, frequency, radius):
    x = s.Symbol("x")
    result = cluster_set(
        x ** (s.I * frequency), x, point, direction=direction, return_result=True
    )
    assert result.cluster_set == Circle(0, radius)
    evidence = result.evidence[0]
    sequence = evidence.substitutions[0][1]
    n = next(a for a in sequence.free_symbols if a.is_integer)
    theta = next(a for a in sequence.free_symbols if a != n)
    for phase in (s.S.Zero, s.pi / 3, s.pi):
        chart = sequence.subs(theta, phase)
        expected_point = point if direction != "-" else 0
        assert s.limit(chart, n, s.oo) == expected_point
        actual = s.exp(s.I * frequency * s.expand_complex(s.log(chart)))
        assert (
            s.simplify(s.expand_complex(actual - evidence.value.subs(theta, phase)))
            == 0
        )


def test_attained_trigonometric_interval():
    x = s.Symbol("x")
    result = cluster_set(
        3 * s.sin(2 * x + 1) + 4 * s.cos(2 * x + 1) + 2, x, s.oo, return_result=True
    )
    assert result.cluster_set == s.Interval(-3, 7)
    witness = result.evidence[0]
    assert (
        s.trigsimp(
            result.expression.subs(x, witness.substitutions[0][1]) - witness.value
        )
        == 0
    )


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: s.sin(x) / s.sin(2 * x),
        lambda x: s.sin(x * x),
        lambda x: x ** (s.I * s.Symbol("b")),
        lambda x: x ** (x**s.I),
    ],
)
def test_cluster_declines_missing_proof(expression):
    x = s.Symbol("x")
    result = cluster_set(expression(x), x, s.oo, return_result=True)
    assert result.certified is False
    assert result.cluster_set is None


def test_piecewise_priority_and_default():
    namespace = scalar_reference_namespace()
    x = namespace["x"] = s.Symbol("x")
    parsed = parse_reference("Piecewise(((1,x>0),(2,x>1)),3)", namespace)
    assert parsed.subs(x, 2) == 1
    assert parsed.subs(x, 0) == 3
    missing = parse_reference("Piecewise(((x,x>0),))", namespace)
    assert missing.subs(x, -1) is s.nan
    spike = parse_reference("Piecewise(((45,Eq(x,2)),),3*x-1)", namespace)
    assert limit(spike, x, 2) == 5


def test_real_root_and_condition():
    namespace = scalar_reference_namespace()
    assert parse_reference("CubeRoot(-8)", namespace) == -2
    value, condition = conditional_reference(
        parse_reference("conditional_value(0,Contains(1/a,Reals))", namespace)
    )
    a = s.Symbol("a")
    assert value == 0
    assert condition.has(s.Q.real(a), s.Ne(a, 0))


def test_malformed_constants():
    with pytest.raises(TypeError):
        parse_reference("sqrt(2(pi))", scalar_reference_namespace())


def test_whole_plane_rejects_real_variable():
    x = s.Symbol("x", real=True)
    with pytest.raises(ValueError):
        complex_limit(x, x, 0)


def test_parameter_growth_cells():
    x = s.Symbol("x")
    a = s.Symbol("a", real=True)
    result = limit(x**a, x, s.oo, return_result=True)
    assert result.exhaustive is True
    assert {stratum.result.value for stratum in result.strata} == {0, 1, s.oo}
    value = limit(x**a, x, s.oo)
    assert value.subs(a, -2) == 0
    assert value.subs(a, 0) == 1
    assert value.subs(a, 2) == s.oo
    b = s.Symbol("b")
    partial = limit(s.exp(b * x), x, s.oo, return_result=True)
    unknown = [
        cell for cell in partial.strata if cell.result.status is LimitStatus.UNKNOWN
    ]
    assert len(unknown) == 1
    assert unknown[0].condition == ~s.Q.real(b)


def test_complex_exponential_direction():
    x = s.Symbol("x")
    assert limit(s.exp(x + 5 * s.I), x, s.oo) == DirectionalInfinity(s.exp(5 * s.I))


@pytest.mark.parametrize("function,expected", [(s.asec, s.I), (s.acsc, -s.I)])
def test_nested_logarithmic_direction(function, expected):
    x = s.Symbol("x")
    expression = function(s.acos(1 / s.log(x)) - s.pi / 2)
    result = limit(expression, x, -s.oo, return_result=True)
    assert result.value == DirectionalInfinity(expected)
    witness = result.evidence[0]
    assert witness.substitutions[0][0] == x
    sequence = witness.substitutions[0][1]
    n = next(iter(sequence.free_symbols))
    assert s.limit(sequence, n, s.oo) == -s.oo


@pytest.mark.parametrize(
    "function,point,ray,expected",
    [
        (s.acosh, 0, s.I, s.I * s.pi / 2),
        (s.acosh, 0, -s.I, -s.I * s.pi / 2),
        (s.asech, 2, s.I, -s.I * s.pi / 3),
        (s.asec, 0, -s.I, DirectionalInfinity(-s.I)),
        (s.loggamma, -s.Rational(1, 2), s.I, s.log(2 * s.sqrt(s.pi)) - s.I * s.pi),
    ],
)
def test_inverse_and_gamma_cut(function, point, ray, expected):
    z = s.Symbol("z")
    assert complex_ray_limit(function(z), z, point, ray=ray) == expected


def test_lattice_pole_directions():
    from asymptotic import analytic_limit

    x = s.Symbol("x")
    k = s.Symbol("k", integer=True)
    assert analytic_limit(s.csc(s.pi * x), x, k, direction="+") == DirectionalInfinity(
        (-1) ** k
    )
    result = analytic_limit(s.csc(s.pi * x), x, k, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert len(result.evidence) == 2
    for witness in result.evidence:
        sequence = witness.substitutions[0][1]
        n = next(a for a in sequence.free_symbols if a != k)
        assert s.simplify(s.sin(s.pi * sequence.subs(n, 10))) != 0


@pytest.mark.parametrize(
    "expression,point,expected",
    [
        (
            lambda x: x * (-s.Ci(x) + s.Si(x) - s.pi / 2),
            s.oo,
            s.Interval(-s.sqrt(2), s.sqrt(2)),
        ),
        (
            lambda x: x * (-1 + s.fresnels(x) / s.fresnelc(x)),
            s.oo,
            s.Interval(-2 * s.sqrt(2) / s.pi, 2 * s.sqrt(2) / s.pi),
        ),
        (
            lambda x: (
                s.exp(-(x + s.sin(2 * x) / 2) * s.sin(x)) / (x + s.sin(x) * s.cos(x))
            ),
            s.oo,
            s.Interval(0, s.oo),
        ),
        (lambda x: x * (1 + s.sin(x)), s.oo, s.Interval(0, s.oo)),
        (lambda x: s.log(x) * s.sin(1 / x) * s.csc(x), 0, s.S.Reals),
        (lambda x: x * (s.sin(x * x - x) + s.sin(x * x + x)) / 2, s.oo, s.S.Reals),
        (lambda x: s.sin(x) ** (1 / x), s.oo, s.Interval(0, 1)),
    ],
)
def test_tail_cluster_attainment(expression, point, expected):
    x = s.Symbol("x")
    result = cluster_set(expression(x), x, point, return_result=True)
    assert result.cluster_set == expected
    witness = result.evidence[0]
    assert witness.substitutions[0][0] == x
    sequence = witness.substitutions[0][1]
    n = next(a for a in sequence.free_symbols if a.is_integer)
    parameters = sequence.free_symbols - {n}
    fixed = {p: s.Rational(1, 2) if "value" in str(p) else 0 for p in parameters}
    chart = sequence.subs(fixed)
    attained = witness.value.subs(fixed)
    index = 8 if expression(x).is_Pow and expression(x).base.func is s.sin else 1000
    value = s.N(expression(chart.subs(n, index)), 80)
    assert abs(complex(value) - complex(s.N(attained, 30))) < 0.02


def test_decaying_trigonometric_cluster():
    x = s.Symbol("x")
    expression = -s.I * s.exp(2 * x) * s.sin(x) + s.exp(2 * x) * s.cos(x) + s.sin(x)
    assert cluster_set(expression, x, -s.oo) == s.Interval(-1, 1)


def test_piecewise_parameter_branch_is_not_discarded():
    x, a = s.symbols("x a")
    expression = s.Piecewise((s.Function("f")(x), a > 0), (2, True))
    result = limit(expression, x, 0, return_result=True)
    assert result.status is LimitStatus.UNKNOWN


def test_real_root_rejects_complex_tail():
    from asymptotic.function_normalization import RealRoot

    x = s.Symbol("x")
    result = limit(RealRoot(x + s.I, 4), x, s.oo, return_result=True)
    assert result.status is LimitStatus.UNKNOWN


def test_piecewise_parameter_values():
    x, a = s.symbols("x a")
    expression = s.Piecewise((x + 1, a > 0), (x * x + 2, True))
    assert limit(expression, x, 0) == s.Piecewise((1, a > 0), (2, True))


def test_empty_assumptions_are_not_certificates():
    z = s.Symbol("z")
    assert (
        complex_limit(z, z, 0, assumptions=False, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    result = cluster_set(
        z**s.I, z, 0, direction="+", assumptions=False, return_result=True
    )
    assert isinstance(result, ExactClusterResult)
    assert result.certified is False
    assert result.assumptions is s.S.false


@pytest.mark.parametrize(
    "variable,point,direction",
    [
        (s.Symbol("x", positive=True), 0, "-"),
        (s.Symbol("x", negative=True), s.oo, None),
        (s.Symbol("x", integer=False), s.oo, None),
        (s.Symbol("x", real=False), 0, "+"),
    ],
)
def test_cluster_chart_types(variable, point, direction):
    with pytest.raises(ValueError):
        cluster_set(variable**s.I, variable, point, direction=direction)


def test_constant_and_positive_chart_clusters():
    x = s.Symbol("x", positive=True)
    assert cluster_set(3, x, s.oo) == s.FiniteSet(3)
    assert cluster_set(x**s.I, x, 0) == Circle(0, 1)


def test_whole_plane_nonreal_type_is_restricted():
    z = s.Symbol("z", real=False)
    with pytest.raises(ValueError):
        complex_limit(z, z, 0)


@pytest.mark.parametrize("point,expected", [(-1, -1), (0, 0), (8, 2)])
def test_odd_real_root_charts(point, expected):
    from asymptotic.function_normalization import RealRoot

    x = s.Symbol("x")
    assert limit(RealRoot(x, 3), x, point) == expected


def test_real_coordinate_nonreal_target():
    x = s.Symbol("x", real=True)
    assert limit(x, x, s.I, return_result=True).status is LimitStatus.UNKNOWN


def test_positive_coordinate_negative_target():
    x = s.Symbol("x", positive=True)
    assert (
        limit(x ** s.Rational(1, 3), x, -1, return_result=True).status
        is LimitStatus.UNKNOWN
    )


def test_translated_parameter_pole():
    z = s.Symbol("z")
    a = s.Symbol("a")
    assert complex_limit(1 / (z - a), z, a, assumptions=s.Q.finite(a)) is s.zoo
    c = s.Symbol("c", finite=True, zero=False)
    assert complex_limit(c / z, z, 0) is s.zoo
    b = s.Symbol("b", real=True)
    assert complex_limit(b / z, z, 0, return_result=True).status is LimitStatus.UNKNOWN


def test_whole_plane_zero_coefficient_cell():
    z = s.Symbol("z")
    a = s.Symbol("a", real=True)
    expression = a * s.re(z) * s.im(z) / (s.re(z) ** 2 + s.im(z) ** 2)
    assert (
        complex_limit(expression, z, 0, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert complex_limit(expression.subs(a, 0), z, 0) == 0


def test_unused_complex_coordinate_is_attained():
    z = s.Symbol("z")
    result = complex_limit(s.sign(s.re(z)), z, 0, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [item for item in result.evidence if item.substitutions]
    assert {item.value for item in witnesses} == {-1, 1}
    for item in witnesses:
        sequence = item.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        assert n.is_integer is True
        assert s.limit(sequence, n, s.oo) == 0


def test_real_root_tangent_tail():
    from asymptotic.function_normalization import RealRoot

    x = s.Symbol("x")
    expression = s.tan(RealRoot(s.pi**4 / 16 - 1 / x, 4))
    assert limit(expression, x, s.oo) == s.oo
