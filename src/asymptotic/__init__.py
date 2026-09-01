"""Primary public API for symbolic asymptotic analysis.

The root namespace intentionally contains only common workflow entry points.
Specialized theorem, property, statistics, and representation APIs live in their
own submodules; see ``docs/api-classification.md``.
"""

from __future__ import annotations

__version__ = "0.53.2"

from .algebra import (
    AsymptoticAlgebra,
    AsymptoticElement,
    asymptotic_element,
)
from .calculus import (
    differentiate,
    integrate,
)
from .context import (
    AsymptoticContext,
    GrowthComparison,
)
from .dominant import (
    dominant_balance_candidates,
)
from .dsolve import (
    AsymptoticDSolveResult,
    asymptotic_dsolve,
)
from .general_ops import (
    asymptotic_integrate,
    compose_transseries,
)
from .implicit import (
    implicit_asymptotic,
)
from .mrv import (
    mrv_decomposition,
)
from .multiseries import (
    Multiseries,
    multiseries,
)
from .multivariate import (
    multivariate_dominant_balance_candidates,
)
from .multivariate_implicit import (
    multivariate_implicit_asymptotics,
)
from .nested import (
    NestedExpansion,
    nested_expansion,
)
from .optimization import (
    AsymptoticOptimizationResult,
    asymptotic_argmax,
    asymptotic_argmin,
    asymptotic_maximize,
    asymptotic_minimize,
)
from .probability import (
    StatisticalAsymptoticResult,
    airy_uniform_saddle_asymptotic,
    asymptotic_expectation,
    asymptotic_probability,
    coalescing_saddle_asymptotic,
    laplace_asymptotic_integral,
)
from .puiseux import (
    puiseux_series,
)
from .relations import (
    AsymptoticRelationResult,
    asymptotic_big_o,
    asymptotic_equivalent,
    asymptotic_little_o,
    asymptotic_relation,
)
from .remainder import (
    AsymptoticRemainder,
    AsymptoticTruncation,
    RemainderKind,
)
from .reversion import (
    inverse_asymptotic,
    series_reversion,
)
from .roots import (
    asymptotic_root,
)
from .rsolve import (
    AsymptoticRSolveResult,
    asymptotic_rsolve,
)
from .scale import (
    AsymptoticScale,
    discover_scale,
)
from .solve import (
    AsymptoticSolveResult,
    asymptotic_solve,
)
from .sums import (
    AsymptoticSumResult,
    asymptotic_sum,
)
from .transseries import (
    TransseriesExpansion,
    transseries_from_expression,
)

__all__ = [
    "AsymptoticAlgebra",
    "AsymptoticContext",
    "AsymptoticDSolveResult",
    "AsymptoticElement",
    "AsymptoticOptimizationResult",
    "AsymptoticRSolveResult",
    "AsymptoticRelationResult",
    "AsymptoticRemainder",
    "AsymptoticScale",
    "AsymptoticSolveResult",
    "AsymptoticSumResult",
    "AsymptoticTruncation",
    "GrowthComparison",
    "Multiseries",
    "NestedExpansion",
    "RemainderKind",
    "StatisticalAsymptoticResult",
    "TransseriesExpansion",
    "__version__",
    "airy_uniform_saddle_asymptotic",
    "asymptotic_argmax",
    "asymptotic_argmin",
    "asymptotic_big_o",
    "asymptotic_dsolve",
    "asymptotic_element",
    "asymptotic_equivalent",
    "asymptotic_expectation",
    "asymptotic_integrate",
    "asymptotic_little_o",
    "asymptotic_maximize",
    "asymptotic_minimize",
    "asymptotic_probability",
    "asymptotic_relation",
    "asymptotic_root",
    "asymptotic_rsolve",
    "asymptotic_solve",
    "asymptotic_sum",
    "coalescing_saddle_asymptotic",
    "compose_transseries",
    "differentiate",
    "discover_scale",
    "dominant_balance_candidates",
    "implicit_asymptotic",
    "integrate",
    "inverse_asymptotic",
    "laplace_asymptotic_integral",
    "mrv_decomposition",
    "multiseries",
    "multivariate_dominant_balance_candidates",
    "multivariate_implicit_asymptotics",
    "nested_expansion",
    "puiseux_series",
    "series_reversion",
    "transseries_from_expression",
]


def __dir__() -> list[str]:
    """Expose the intentionally small primary namespace to interactive discovery."""
    return sorted(set(__all__) | {"__all__", "__doc__", "__name__", "__package__"})
