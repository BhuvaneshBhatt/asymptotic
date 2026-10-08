"""Certificate families used by the multivariate limit engine."""

from __future__ import annotations

from ._multivariate_analytic import (
    analytic_equivalence_rewrite_certificate,
    analytic_jet_path_conflict_certificate,
    analytic_leading_ratio_certificate,
    analytic_meromorphic_taylor_reduction,
    angular_cluster_path_certificate,
    certified_analytic_taylor_reduction,
    complex_hankel_projection_certificate,
    complex_sech_phase_certificate,
    discontinuity_branch_certificate,
    divided_difference_certificate,
    exponential_small_power_certificate,
    generalized_divided_difference_certificate,
    generalized_log_product_order_certificate,
    growth_scale_limit,
    inverse_branch_limit_certificate,
    logarithmic_germ_order_certificate,
    oscillatory_phase_sequence_certificate,
    parameter_weighted_order_certificate,
    radial_norm_canonicalization_certificate,
    radical_rationalization_certificate,
    real_projection_simplification_certificate,
    removable_transcendental_normalization_certificate,
    singular_polylog_special_germ_certificate,
    special_function_local_germ_certificate,
    variable_power_branch_certificate,
)
from ._multivariate_germ import (
    GermAnalysis,
    ParityCertificate,
    candidate_rational_limit_certificate,
    isolated_zero_certificate,
    local_sign_certificate,
    rational_power_order_certificate,
    reciprocal_pole_certificate,
    sertoz_rational_limit,
)
from ._multivariate_optimization import (
    LojasiewiczExponentResult,
    complete_lagrange_critical_certificate,
    extremal_limit,
    general_isolated_zero_certificate,
    general_lojasiewicz_exponent,
    lagrange_critical_path_certificate,
    lojasiewicz_exponent_bound,
)
from ._multivariate_parameters import (
    parameter_stratified_limit,
    recursive_parameter_stratified_limit,
)


def fast_multivariate_certificate(expr, variables, target, *, domain=True):
    """Run cheap exact certificate families before CAD/Newton certification."""
    import sympy as sp

    domain = sp.sympify(domain)
    context = GermAnalysis.create(expr, variables, target)

    # Radical conjugation is meaningful on relative principal-root domains, so
    # it must run before the unrestricted-domain guard below.
    radical = radical_rationalization_certificate(
        expr, variables, target, domain=domain, _context=context
    )
    if radical.certified:
        return radical
    if domain is not sp.S.true:
        return None

    for fn in (
        reciprocal_pole_certificate,
        oscillatory_phase_sequence_certificate,
        divided_difference_certificate,
        generalized_divided_difference_certificate,
        parameter_weighted_order_certificate,
        variable_power_branch_certificate,
        generalized_log_product_order_certificate,
        exponential_small_power_certificate,
        inverse_branch_limit_certificate,
        discontinuity_branch_certificate,
        complex_hankel_projection_certificate,
        complex_sech_phase_certificate,
        singular_polylog_special_germ_certificate,
        angular_cluster_path_certificate,
        analytic_jet_path_conflict_certificate,
        removable_transcendental_normalization_certificate,
        logarithmic_germ_order_certificate,
        real_projection_simplification_certificate,
        special_function_local_germ_certificate,
        growth_scale_limit,
        rational_power_order_certificate,
        radial_norm_canonicalization_certificate,
        analytic_leading_ratio_certificate,
        sertoz_rational_limit,
        candidate_rational_limit_certificate,
        analytic_equivalence_rewrite_certificate,
    ):
        certificate = (
            fn(expr, variables, target, domain=domain, _context=context)
            if fn is removable_transcendental_normalization_certificate
            else fn(expr, variables, target, _context=context)
        )
        if certificate.certified:
            return certificate
    return None


__all__ = [
    "LojasiewiczExponentResult",
    "ParityCertificate",
    "analytic_equivalence_rewrite_certificate",
    "analytic_jet_path_conflict_certificate",
    "analytic_leading_ratio_certificate",
    "analytic_meromorphic_taylor_reduction",
    "candidate_rational_limit_certificate",
    "certified_analytic_taylor_reduction",
    "complete_lagrange_critical_certificate",
    "complex_hankel_projection_certificate",
    "complex_sech_phase_certificate",
    "divided_difference_certificate",
    "exponential_small_power_certificate",
    "extremal_limit",
    "fast_multivariate_certificate",
    "general_isolated_zero_certificate",
    "general_lojasiewicz_exponent",
    "generalized_divided_difference_certificate",
    "growth_scale_limit",
    "isolated_zero_certificate",
    "lagrange_critical_path_certificate",
    "local_sign_certificate",
    "logarithmic_germ_order_certificate",
    "lojasiewicz_exponent_bound",
    "oscillatory_phase_sequence_certificate",
    "parameter_stratified_limit",
    "radial_norm_canonicalization_certificate",
    "radical_rationalization_certificate",
    "rational_power_order_certificate",
    "real_projection_simplification_certificate",
    "reciprocal_pole_certificate",
    "recursive_parameter_stratified_limit",
    "removable_transcendental_normalization_certificate",
    "sertoz_rational_limit",
    "special_function_local_germ_certificate",
    "variable_power_branch_certificate",
]
