from types import SimpleNamespace

import pytest
import sympy as sp

from asymptotic.branch_monodromy import ExtendedClusterKind, classify_monodromy_cluster
from asymptotic.complex_domain import ComplexSector
from asymptotic.multivariate_expansion import UniformRemainderCertificate
from asymptotic.ode_adapter import FormalODEAdapterError, from_odeanalysis_formal_data
from asymptotic.parameter_cells import satisfiable_parameter_condition
from asymptotic.remainder_theorems import _root_half_plane


def test_uniform_remainder_rejects_unproved_or_invalid_bounds():
    a = sp.Symbol("a", real=True)
    with pytest.raises(ValueError):
        UniformRemainderCertificate(2, -1, 1, "bad", "bad")
    with pytest.raises(ValueError):
        UniformRemainderCertificate(2, a, 1, "bad", "bad")
    with pytest.raises(ValueError):
        UniformRemainderCertificate(2, 1, a, "bad", "bad")


def test_unresolved_conditionset_is_not_satisfiable_proof():
    x = sp.Symbol("x", real=True)
    condition = sp.Eq(sp.sin(x), x / 2)
    # This transcendental real solution set may remain unresolved.
    result = condition.as_set()
    if isinstance(result, sp.ConditionSet):
        assert satisfiable_parameter_condition(condition, (x,)) is None


def test_green_half_plane_never_uses_floating_sign_as_certificate():
    x = sp.Symbol("x")
    unresolved = sp.RootOf(x**5 - x + 1, 0)
    if sp.re(unresolved).is_positive is None and sp.re(unresolved).is_negative is None:
        assert _root_half_plane(unresolved, sp.oo) is None


def test_complex_sector_honors_excluded_rays_modulo_two_pi():
    sector = ComplexSector(0, sp.pi, excluded_rays=(0,))
    assert sector.contains_angle(0) is False
    assert sector.contains_angle(2 * sp.pi) is False


def test_unit_modulus_alone_does_not_certify_dense_monodromy():
    class Op:
        exp = sp.Symbol("alpha", real=True)

    divisor = SimpleNamespace(kind="power", operation=Op())
    # A synthetic divisor with an unresolved real exponent must remain unknown;
    # unit modulus by itself is not an irrationality proof.
    result = classify_monodromy_cluster(sp.Symbol("z") ** Op.exp, (divisor,))
    assert result.kind is not ExtendedClusterKind.DENSE_PHASE


def _formal_data(*, recurrence_ok=True, residual_ok=True):
    t = sp.Symbol("t")
    h = sp.Symbol("h")
    vector = SimpleNamespace(
        local_parameter=t,
        amplitude_parameter=sp.Integer(1),
        source_exponent=0,
        exponent=0,
        ramified_exponent=0,
        logarithmic_degree=0,
        expression=sp.Integer(1),
    )
    block = SimpleNamespace(
        index=0,
        local_parameter=t,
        local_coordinate=h,
        local_exponential_polynomial=0,
        ramification_index=1,
        basis_vectors=(vector,),
        semisimple_exponent=sp.zeros(1),
        nilpotent_exponent=sp.zeros(1),
        formal_exponent_matrix=sp.zeros(1),
        cover_monodromy=sp.eye(1),
        has_logarithms=False,
    )
    recurrence = SimpleNamespace(verify=lambda: recurrence_ok, values=(1, 2, 3))
    residual = SimpleNamespace(verify=lambda: residual_ok, order=4)
    scalar = SimpleNamespace(recurrence=recurrence, residual_certificate=residual)
    domain = SimpleNamespace(
        index=0,
        cover_start_angle=0,
        cover_end_angle=sp.pi,
        local_start_angle=0,
        local_end_angle=sp.pi,
        original_start_angle=0,
        original_end_angle=sp.pi,
        ramification_index=1,
        sheet=0,
        left_open=True,
        right_open=True,
        dominance_levels=((0,),),
    )
    return SimpleNamespace(
        schema_version=1,
        point=sp.oo,
        ramification_index=1,
        blocks=(block,),
        cover_monodromy=sp.eye(1),
        local_monodromy=sp.eye(1),
        stokes=None,
        complete=True,
        limitation=None,
        scalar_series=(scalar,),
        domains=(domain,),
    )


def test_modern_odeanalysis_contract_validates_and_domains():
    x = sp.Symbol("x")
    converted = from_odeanalysis_formal_data(_formal_data(), x)
    assert converted.scalar_series[0].recurrence_values == (1, 2, 3)
    assert len(converted.validated_residuals()) == 1
    assert converted.dominance_at(sp.pi / 2) == ((0,),)
    assert converted.domain_for_angle(0) is None  # open boundary


def test_modern_odeanalysis_contract_rejects_and_residual():
    x = sp.Symbol("x")
    with pytest.raises(FormalODEAdapterError, match="recurrence"):
        from_odeanalysis_formal_data(_formal_data(recurrence_ok=False), x)
    with pytest.raises(FormalODEAdapterError, match="residual"):
        from_odeanalysis_formal_data(_formal_data(residual_ok=False), x)


def test_ode_contract_rejects_tampered_residual_valuation():
    x = sp.Symbol("x")
    data = _formal_data()
    t = sp.Symbol("t")
    bad_residual = SimpleNamespace(
        verify=lambda: True,
        residual=t**3,
        local_parameter=t,
        certified_valuation=4,
        exact_zero=False,
    )
    scalar = SimpleNamespace(
        recurrence=data.scalar_series[0].recurrence,
        residual_certificate=bad_residual,
    )
    data = SimpleNamespace(**{**data.__dict__, "scalar_series": (scalar,)})
    with pytest.raises(FormalODEAdapterError, match="valuation"):
        from_odeanalysis_formal_data(data, x)


def test_native_ode_record():
    provider = pytest.importorskip("odeanalysis")
    x = sp.Symbol("x", positive=True)
    u = sp.Function("u")
    data = provider.formal_ode_data(
        sp.diff(u(x), x) - u(x),
        u(x),
        x,
        point=sp.oo,
        terms=3,
        include_stokes=False,
    )
    result = from_odeanalysis_formal_data(data, x)
    assert len(result.solutions) == 1
    solution = result.solutions[0].truncate()
    assert sp.simplify(solution / sp.exp(x)) == 1
    assert sp.simplify(sp.diff(solution, x) - solution) == 0


@pytest.mark.parametrize("record_name", ["FormalODEData", "FormalODEGreenOperatorData"])
def test_native_ode_identity(record_name):
    from asymptotic.ode_adapter import certify_green_operator_data

    record = type(record_name, (), {"__module__": "odeanalysis.interchange"})()
    x = sp.Symbol("x")
    with pytest.raises(FormalODEAdapterError, match="schema version"):
        if record_name == "FormalODEData":
            from_odeanalysis_formal_data(record, x)
        else:
            certify_green_operator_data(record, sp.exp(-x))


def test_native_green_record():
    provider = pytest.importorskip("odeanalysis")
    from odeanalysis.interchange import green_operator_data

    from asymptotic.ode_adapter import certify_green_operator_data

    x = sp.Symbol("x", positive=True)
    operator = provider.LinearDifferentialOperator(
        x, sp.Function("u"), (sp.Integer(2), sp.Integer(-3), sp.Integer(1))
    )
    data = green_operator_data(operator, point=sp.oo)
    theorem, green = certify_green_operator_data(data, sp.exp(-x))
    assert theorem.certified
    assert green.replay(x) is True
    assert sp.simplify(green.particular + sp.exp(-x) / 6) == 0
