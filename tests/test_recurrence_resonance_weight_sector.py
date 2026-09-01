from __future__ import annotations

import sympy as sp

from asymptotic.complex_domain import (
    ComplexBranchMetadata,
    ComplexSector,
)
from asymptotic.discrete_scale import (
    inhomogeneous_particular_solution,
    linear_recurrence_data,
)
from asymptotic.multivariate import multivariate_dominant_balance_candidates
from asymptotic.rsolve import asymptotic_rsolve
from asymptotic.transseries import transseries_from_expression


def test_native_first_order_constant_forcing_particular_is_exact():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    result = asymptotic_rsolve(a(n + 1) - 2 * a(n) - 1, a(n), n, method="native")
    assert result.particular_expression == -1
    assert result.particular_residual == 0
    assert sp.simplify(result.residual(a(n + 1) - 2 * a(n) - 1)) == 0


def test_native_hypergeometric_forcing_builds_particular_solution():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - 2 * a(n) - 3**n
    data = linear_recurrence_data(recurrence, a(n), n)
    particular = inhomogeneous_particular_solution(data, terms=4)
    assert particular is not None
    expression, residual = particular
    assert sp.simplify(expression - 3**n) == 0
    assert residual == 0


def test_simple_first_order_resonance_produces_logarithmic_correction():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - a(n) - 1 / n
    result = asymptotic_rsolve(recurrence, a(n), n, method="native", terms=4)
    particular = result.particular_expression
    assert particular is not None
    assert particular.has(sp.log(n))
    # The finite Euler--Maclaurin-like antidifference has residual smaller than n^-4.
    residual = sp.simplify(result.particular_residual)
    assert sp.limit(n**4 * residual, n, sp.oo) == 0


def test_automatic_weight_cones_feed_public_dominant_balance_api():
    x, y, u = sp.symbols("x y u", positive=True)
    equation = u**2 + x * u + y**2
    automatic = multivariate_dominant_balance_candidates(
        equation, u, (x, y), weights=None, stratify_parameters=False
    )
    assert automatic
    assert all(candidate.path.weights for candidate in automatic)


def test_transseries_records_sector_and_branch_metadata():
    z = sp.symbols("z")
    sector = ComplexSector(center_angle=0, opening=sp.pi / 2, excluded_rays=(sp.pi / 4,))
    branch = ComplexBranchMetadata(
        logarithm_branch=1,
        power_branch=-1,
        branch_cuts=(sp.pi,),
        stokes_rays=(sp.pi / 2,),
        continuation="counterclockwise from positive real axis",
        principal=False,
    )
    expansion = transseries_from_expression(
        1 / z + 1 / z**2, z, point=sp.oo, sector=sector, branch=branch
    )
    assert expansion.metadata["complex_sector"] == sector
    assert expansion.metadata["complex_branch"] == branch
    assert (-expansion).metadata["complex_sector"] == sector
