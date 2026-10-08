import pytest
import sympy as s

from asymptotic import limit
from asymptotic.attained_oscillatory_germs import attained_local_phase_certificate
from asymptotic.limit_models import LimitStatus

x = s.Symbol("x")


@pytest.mark.parametrize(
    "expr,p",
    [
        (s.cos(s.log(s.Abs(x))) + s.sin(x), 0),
        (s.sin(s.tan(x)) + x - s.pi / 2, s.pi / 2),
    ],
)
def test_attained_local_witnesses_have_avoid_poles(expr, p):
    result = limit(expr, x, p, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    evidence = [e for e in result.evidence if e.method.startswith("attained_inverse")]
    assert len(evidence) == 2 and evidence[0].value != evidence[1].value
    for e in evidence:
        _, sequence = e.substitutions[0]
        n = next(iter(sequence.free_symbols))
        point = sequence.subs(n, 100)
        assert abs(complex(s.N(point - p, 30))) < 0.02
        assert abs(complex(s.N(expr.subs(x, point) - e.value, 30))) < 0.02
        if expr.has(s.tan):
            assert abs(complex(s.N(s.cos(point), 30))) > 0
        else:
            assert point.is_positive is True


def test_exponential_tangent_witnesses_avoid_both_denominators():
    expr = -s.I * (s.exp(2 * s.I * x) - 1) / (x * (s.exp(2 * s.I * x) + 1))
    result = limit(expr, x, s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [
        e for e in result.evidence if e.method == "attained_exponential_tangent_tail"
    ]
    assert {e.value for e in witnesses} == {0, 1}
    for e in witnesses:
        _, seq = e.substitutions[0]
        n = next(iter(seq.free_symbols))
        point = seq.subs(n, 100)
        assert abs(complex(s.N(s.exp(2 * s.I * point) + 1, 40))) > 0
        assert abs(complex(s.N(expr.subs(x, point) - e.value, 40))) < 0.01


def test_phase_certificate_declines_unsupported_singular_amplitude():
    p = s.pi / 2
    expr = s.sin(s.tan(x))
    assert attained_local_phase_certificate(expr, x, p, x < p, s.S.true) is None
    assert (
        attained_local_phase_certificate(expr / (x - p), x, p, s.S.true, s.S.true)
        is None
    )
    z = s.Symbol("z", real=False)
    assert (
        attained_local_phase_certificate(
            s.cos(s.log(z)), z, s.S.Zero, s.S.true, s.S.true
        )
        is None
    )
