import sympy as sp

from asymptotic.function_properties import (
    AssumptionProperties,
    DomainProperties,
    FunctionProperties,
    RealUnivariateProperties,
    SingularityProperties,
)
from asymptotic.function_properties.query import (
    domain_properties,
    function_properties,
    register_function_properties,
    singularity_properties,
)
from asymptotic.function_properties.registry import FunctionPropertyRegistry


def test_log_properties_include_real_domain_cut_and_branch_point():
    x = sp.symbols("x", real=True)
    props = function_properties(sp.log(x))
    assert props is not None
    assert props.branch_cuts
    assert props.branch_points
    assert sp.ask(props.real_domain.subs(x, 2)) is True
    assert props.domain is not None
    assert props.singularities is not None


def test_sqrt_properties_are_expression_head_aware():
    x = sp.symbols("x", real=True)
    props = function_properties(sp.sqrt(x))
    assert props is not None
    assert props.real_range == sp.Interval(0, sp.oo)
    assert props.locally_analytic is False
    assert props.real_univariate is not None


def test_entire_special_function_has_no_registered_singular_loci():
    x = sp.symbols("x")
    props = function_properties(sp.airyai(x))
    assert props is not None
    assert props.locally_analytic is True
    assert props.branch_cuts == ()
    assert props.poles == ()


def test_registry_is_extensible_and_can_be_isolated():
    f = sp.Function("f")
    x = sp.symbols("x")
    registry = FunctionPropertyRegistry()

    def build(expr):
        return FunctionProperties(
            expression=expr,
            arguments=expr.args,
            domain=DomainProperties(expr.args, real_domain=sp.S.true),
        )

    register_function_properties(f, build, registry=registry)
    props = function_properties(f(x), registry=registry)
    assert props is not None
    assert props.real_domain is sp.S.true
    assert function_properties(f(x)) is None


def test_component_queries_do_not_expose_asymptotic_internals():
    x = sp.symbols("x")
    domain = domain_properties(sp.log(x))
    singularities = singularity_properties(sp.log(x))
    assert isinstance(domain, DomainProperties)
    assert isinstance(singularities, SingularityProperties)
    assert singularities.branch_points


def test_composite_property_model_keeps_independent_components():
    x = sp.symbols("x")
    props = FunctionProperties(
        expression=sp.exp(x),
        arguments=(x,),
        assumptions=AssumptionProperties(real=sp.Q.real(x)),
        domain=DomainProperties((x,), complex_domain=sp.S.true),
        singularities=SingularityProperties(locally_analytic=True, poles=()),
        real_univariate=RealUnivariateProperties(variable=x, range=sp.Interval.open(0, sp.oo)),
    )
    assert props.real_valued == sp.Q.real(x)
    assert props.complex_domain is sp.S.true
    assert props.poles == ()
    assert props.real_range == sp.Interval.open(0, sp.oo)
