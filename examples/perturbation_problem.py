"""Unified perturbation workflow with automatic dispatch and validation."""

import sympy as sp

from asymptotic.perturbation_problem import PerturbationProblem

x, eps = sp.symbols("x eps")
problem = PerturbationProblem(x - 1 - eps * x, x, eps)

recommendation = problem.recommend_method()
assert recommendation.method == "regular"

result = problem.solve(method="auto", order=2)
assert sp.expand(result.approximation[0] - (1 + eps + eps**2)) == 0

validation = result.validate()
assert validation.residual_orders == (sp.Integer(3),)
assert validation.evidence.proved
