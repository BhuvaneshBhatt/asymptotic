"""Native provider registration without runtime function replacement."""

import sympy as sp

from asymptotic.local_expansion import (
    LocalExpansion,
    local_series,
    register_local_expansion,
)


def test_custom_provider_and_builtins():
    variable = sp.Symbol("x", positive=True)
    function = sp.Function("registered_linear_germ")

    def provider(expr, x, point, depth):
        return LocalExpansion(
            expr, x, point, 1 + x, x * x, depth, "registered_linear_germ"
        )

    register_local_expansion(function, provider)
    result = local_series(function(variable), variable, 0, depth=3)
    assert result.prefix == 1 + variable and result.order == variable**2
    assert result.expression == function(variable) and result.depth == 3
    bessel = local_series(sp.bessely(1, variable), variable, sp.oo, depth=3)
    assert bessel.expression == sp.bessely(
        1, variable
    ) and bessel.order == variable ** (-sp.Rational(7, 2))
