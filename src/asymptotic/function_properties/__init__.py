"""Structured mathematical function-property registry.

The data model and registry are independent of asymptotic expansion algorithms,
so property queries remain reusable and easy to test in isolation.
"""

from .data import register_defaults
from .model import (
    ArgumentDomain,
    ArgumentSignature,
    ArgumentSpec,
    AssumptionProperties,
    Discontinuity,
    DomainEndpoint,
    DomainInterval,
    DomainProperties,
    FunctionProperties,
    GlobalExtremum,
    RealUnivariateProperties,
    SingularityLocus,
    SingularityProperties,
)
from .query import (
    analytic_at,
    analytic_at_decision,
    branch_safe_substitution_decision,
    domain_contains_decision,
    domain_properties,
    expression_head,
    function_properties,
    function_property_rules,
    nested_branch_safe_substitution_decision,
    nested_branch_safety_decisions,
    register_function_properties,
    singularity_properties,
)
from .registry import DEFAULT_REGISTRY, FunctionPropertyRegistry, PropertyBuilder
from .semantics import (
    PropertyDecision,
    PropertyEnforcementError,
    PropertyKnowledge,
    PropertyProvenance,
    PropertyRule,
    applicable_rule,
    decide,
    entails,
    require_decision,
)

register_defaults(DEFAULT_REGISTRY)

__all__ = [
    "DEFAULT_REGISTRY",
    "ArgumentDomain",
    "ArgumentSignature",
    "ArgumentSpec",
    "AssumptionProperties",
    "Discontinuity",
    "DomainEndpoint",
    "DomainInterval",
    "DomainProperties",
    "FunctionProperties",
    "FunctionPropertyRegistry",
    "GlobalExtremum",
    "PropertyBuilder",
    "PropertyDecision",
    "PropertyEnforcementError",
    "PropertyKnowledge",
    "PropertyProvenance",
    "PropertyRule",
    "RealUnivariateProperties",
    "SingularityLocus",
    "SingularityProperties",
    "analytic_at",
    "analytic_at_decision",
    "applicable_rule",
    "branch_safe_substitution_decision",
    "decide",
    "domain_contains_decision",
    "domain_properties",
    "entails",
    "expression_head",
    "function_properties",
    "function_property_rules",
    "nested_branch_safe_substitution_decision",
    "nested_branch_safety_decisions",
    "register_function_properties",
    "require_decision",
    "singularity_properties",
]
