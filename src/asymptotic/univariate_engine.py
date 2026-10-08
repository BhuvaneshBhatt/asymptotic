"""Permanent univariate asymptotic pipeline built from package-native components."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .generalized_series import GeneralizedSeries, generalized_series
from .gruntz_normal_form import GruntzNormalForm, gruntz_normal_form
from .oscillatory_normal_form import (
    OscillatoryClusterInterval,
    oscillatory_cluster_interval,
)
from .special_function_infinity import InfinityAsymptotic, special_function_infinity


@dataclass(frozen=True)
class UnivariateAsymptoticAnalysis:
    normal_form: GruntzNormalForm
    series: GeneralizedSeries | None = None
    infinity_asymptotic: InfinityAsymptotic | None = None
    oscillatory_cluster: OscillatoryClusterInterval | None = None
    provider: str = "univariate_engine"


def analyze_univariate(expr, variable, *, point=sp.oo, direction="+", terms=6):
    from .instrumentation import record_symbolic_event

    record_symbolic_event("univariate_calls")
    nf = gruntz_normal_form(expr, variable, point=point, direction=direction)
    special = (
        special_function_infinity(sp.sympify(expr), variable)
        if point is sp.oo
        else None
    )
    osc = (
        oscillatory_cluster_interval(expr, variable)
        if point in (sp.oo, -sp.oo)
        else None
    )
    series = (
        None
        if (special or osc)
        else generalized_series(expr, variable, point=point, terms=terms)
    )
    return UnivariateAsymptoticAnalysis(nf, series, special, osc)
