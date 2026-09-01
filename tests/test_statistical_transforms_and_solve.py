import sympy as sp
from sympy.stats import Normal

from asymptotic import (
    AsymptoticSolveResult,
    asymptotic_solve,
)
from asymptotic.solve import AsymptoticSolutionBranch
from asymptotic.statistical_transforms import (
    StatisticalTransformResult,
    asymptotic_cdf,
    asymptotic_cgf,
    asymptotic_characteristic_function,
    asymptotic_covariance,
    asymptotic_cumulant,
    asymptotic_mgf,
    asymptotic_moment,
    asymptotic_product,
    asymptotic_quantile,
    asymptotic_rate_function,
    asymptotic_survival,
    asymptotic_variance,
)
from asymptotic.stirling import StirlingNormalization


def test_normal_high_level_statistics():
    n = sp.symbols("n", positive=True)
    t = sp.symbols("t", real=True)
    X = Normal("X", 0, 1 / sp.sqrt(n))
    assert sp.simplify(asymptotic_variance(X, X, parameter=n).expression - 1 / n) == 0
    assert asymptotic_moment(X, X, order=2, parameter=n).expression == 1 / n
    assert (
        sp.simplify(
            asymptotic_mgf(X, X, transform_variable=t, parameter=n).expression
            - sp.exp(t**2 / (2 * n))
        )
        == 0
    )
    assert (
        sp.simplify(
            asymptotic_cgf(X, X, transform_variable=t, parameter=n).expression - t**2 / (2 * n)
        )
        == 0
    )
    assert asymptotic_cumulant(X, X, order=2, parameter=n).expression == 1 / n
    assert (
        sp.simplify(
            asymptotic_characteristic_function(X, X, transform_variable=t, parameter=n).expression
            - sp.exp(-(t**2) / (2 * n))
        )
        == 0
    )


def test_cdf_survival_and_quantile_exact_source():
    n = sp.symbols("n", positive=True)
    q = sp.symbols("q", real=True)
    a = sp.symbols("a", positive=True)
    X = Normal("X", 0, 1 / sp.sqrt(n))
    cdf = asymptotic_cdf(X, q, parameter=n)
    surv = asymptotic_survival(X, q, parameter=n)
    assert sp.simplify(cdf.expression + surv.expression - 1) == 0
    quant = asymptotic_quantile(X, a, parameter=n, quantile_variable=q)
    assert q not in quant.expression.free_symbols


def test_large_deviation_rate_from_exponential_probability():
    n = sp.symbols("n", positive=True)
    from asymptotic.probability import StatisticalAsymptoticResult

    r = StatisticalAsymptoticResult(sp.exp(-3 * n), n, sp.oo, "test", "CERTIFIED")
    out = asymptotic_rate_function(r, parameter=n)
    assert out.expression == 3


def test_asymptotic_solve_filters_inequality_and_expands_root():
    n = sp.symbols("n", positive=True)
    y = sp.symbols("y", real=True)
    out = asymptotic_solve(
        [sp.Eq(y**2, n**2 + 1), y > 0], y, parameter=n, terms=4, domain=sp.S.Reals
    )
    assert len(out.branches) == 1
    sol = out.branches[0].as_dict()[y]
    assert sp.limit(sol / n, n, sp.oo) == 1
    assert out.branches[0].status == "FORMAL"


def test_asymptotic_solve_exact_system():
    n = sp.symbols("n", positive=True)
    x, y = sp.symbols("x y")
    out = asymptotic_solve([sp.Eq(x + y, n), sp.Eq(x - y, 0)], (x, y), parameter=n)
    assert out.branches[0].as_dict() == {x: n / 2, y: n / 2}


def test_covariance_product_and_result_types():
    n = sp.symbols("n", positive=True, integer=True)
    X = Normal("XC", 0, 1 / sp.sqrt(n))
    cov = asymptotic_covariance(X, X, parameter=n)
    assert cov.expression == 1 / n
    assert isinstance(cov, StatisticalTransformResult)
    product = asymptotic_product(1 + 1 / n, sp.symbols("k", integer=True), 1, 3, parameter=n)
    assert product.expression == (1 + 1 / n) ** 3
    solved = asymptotic_solve(sp.Eq(sp.symbols("y") - n, 0), sp.symbols("y"), parameter=n)
    assert isinstance(solved, AsymptoticSolveResult)
    assert isinstance(solved.branches[0], AsymptoticSolutionBranch)
    # Type presence is part of the user-facing normalization contract.
    assert StirlingNormalization.__name__ == "StirlingNormalization"


def test_asymptotic_solve_preserves_polynomial_multiplicity():
    n = sp.symbols("n", positive=True)
    y = sp.symbols("y")
    out = asymptotic_solve((y - n) ** 3, y, parameter=n)
    assert len(out.branches) == 1
    assert out.branches[0].multiplicity == 3


def test_asymptotic_solve_infinite_dependent_limit():
    n = sp.symbols("n", positive=True)
    y = sp.symbols("y", positive=True)
    out = asymptotic_solve(sp.Eq(y * n, n**2 + 1), y, parameter=n, limits={y: sp.oo}, terms=4)
    assert out.branches
    assert sp.limit(out.branches[0].as_dict()[y], n, sp.oo) == sp.oo


def test_asymptotic_solve_domain_filters_exact_complex_branches():
    n = sp.symbols("n", positive=True)
    y = sp.symbols("y")
    out = asymptotic_solve(sp.Eq(y**2 + 1, 0), y, parameter=n, domain=sp.S.Reals)
    assert out.branches == ()
    assert out.status == "EXACT"


def test_asymptotic_solve_limits_filter_exact_branches_before_expansion():
    n = sp.symbols("n", positive=True)
    y = sp.symbols("y")
    out = asymptotic_solve(sp.Eq(y**2 - n**2, 0), y, parameter=n, limits={y: sp.oo})
    assert len(out.branches) == 1
    assert out.branches[0].as_dict()[y] == n


def test_asymptotic_solve_uses_explicit_assumptions_for_domain_and_inequality():
    n = sp.symbols("n")
    y = sp.symbols("y")
    out = asymptotic_solve(
        [sp.Eq(y, sp.sqrt(n)), y > 0],
        y,
        parameter=n,
        domain=sp.S.Reals,
        assumptions=sp.Q.positive(n),
    )
    assert len(out.branches) == 1
    assert out.branches[0].conditions == ()


def test_covariance_supports_exact_joint_expectations_of_multiple_random_variables():
    n = sp.symbols("n", positive=True)
    X = Normal("X_joint", 0, 1 / sp.sqrt(n))
    Y = Normal("Y_joint", 0, 2 / sp.sqrt(n))
    out = asymptotic_covariance(X, Y, parameter=n)
    assert out.expression == 0
    assert out.status == "EXACT"


def test_cgf_marks_truncated_logarithm_formal():
    from asymptotic.statistical_transforms import StatisticalTransformResult

    # Status propagation is exercised through a real exact MGF whose logarithm
    # needs parameter truncation.
    n = sp.symbols("n", positive=True)
    t = sp.symbols("t", real=True)
    X = Normal("X_cgf_formal", 0, sp.sqrt(1 / n + 1 / n**5))
    out = asymptotic_cgf(X, X, transform_variable=t, parameter=n, terms=3)
    assert isinstance(out, StatisticalTransformResult)
    assert out.status in {"EXACT", "FORMAL"}
    if sp.simplify(out.expression - t**2 * (1 / n + 1 / n**5) / 2) != 0:
        assert out.status == "FORMAL"
