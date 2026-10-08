import sympy as sp
from sympy.stats import MultivariateNormal, Normal

from asymptotic import (
    expectation,
    probability,
)


def test_probability_accepts_symbol_binding():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x")
    X = Normal("X_probability_binding", 0, 1 / sp.sqrt(n))
    result = probability(x > 0, parameter=n, bindings={x: X}, return_result=True)
    assert result.expression == sp.Rational(1, 2)
    assert result.status == "EXACT"


def test_probability_accepts_raw_distribution_binding():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x")
    X = Normal("X_probability_distribution_binding", 0, 1 / sp.sqrt(n))
    result = probability(
        x > 0, parameter=n, bindings={x: X.pspace.distribution}, return_result=True
    )
    assert result.expression == sp.Rational(1, 2)
    assert result.status == "EXACT"


def test_probability_accepts_conditional_event():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x")
    X = Normal("X_probability_condition", 0, 1 / sp.sqrt(n))
    result = probability(
        x > 1,
        parameter=n,
        bindings={x: X},
        condition=x > 0,
        method="exact",
        return_result=True,
    )
    expected = sp.simplify(sp.erfc(sp.sqrt(n / 2)))
    assert sp.simplify(result.expression - expected) == 0


def test_probability_tries_exact_joint_rv_fallback():
    n = sp.symbols("n", positive=True)
    x, y = sp.symbols("x y")
    X = Normal("X_joint_probability", 0, 1 / sp.sqrt(n))
    Y = Normal("Y_joint_probability", 0, 2 / sp.sqrt(n))
    result = probability(
        x + y > 0, parameter=n, bindings={x: X, y: Y}, return_result=True
    )
    assert result.expression == sp.Rational(1, 2)
    assert result.method == "exact-probability"


def test_expectation_accepts_multivariate_joint_binding():
    n = sp.symbols("n", positive=True)
    x, y = sp.symbols("x y")
    Z = MultivariateNormal(
        "Z_joint_binding",
        [0, 0],
        [[1 / n, 0], [0, 4 / n]],
    )
    result = expectation(
        x**2 + y**2, parameter=n, bindings={(x, y): Z}, return_result=True
    )
    assert result.expression == 5 / n
    assert result.status == "EXACT"


def test_probability_expectation_indicator_identity():
    n = sp.symbols("n", positive=True)
    X = Normal("X_probability_indicator", 0, 1 / sp.sqrt(n))
    probability_result = probability(X > 0, parameter=n, return_result=True)
    indicator = sp.Piecewise((1, X > 0), (0, True))
    expectation_result = expectation(indicator, parameter=n, return_result=True)
    assert (
        sp.simplify(probability_result.expression - expectation_result.expression) == 0
    )


def test_binding_validation_is_shared_and_expectation():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x")
    for function, expression in (
        (probability, x > 0),
        (expectation, x),
    ):
        try:
            function(expression, parameter=n, bindings={x: sp.Symbol("not_random")})
        except TypeError as exc:
            assert "RandomSymbols or SymPy distributions" in str(exc)
        else:
            raise AssertionError("invalid binding should fail")


def test_continuous_probability_rejects_discrete_only_sum_methods():
    n = sp.symbols("n", positive=True)
    X = Normal("X_discrete_method_rejection", 0, 1 / sp.sqrt(n))

    for method in ("zeilberger", "poisson", "oscillatory"):
        try:
            probability(X > 0, parameter=n, method=method, return_result=True)
        except TypeError as exc:
            assert "requires a discrete random variable" in str(exc)
        else:
            raise AssertionError(
                f"{method} should be rejected for continuous variables"
            )
