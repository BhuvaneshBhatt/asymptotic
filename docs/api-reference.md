# Generated primary API reference

This file is generated from the live root API signatures and docstrings. Edit the public docstrings, then run `python tools/generate_api_reference.py`.

## `AsymptoticContext`

```python
AsymptoticContext(variable: 'sp.Symbol', point: 'sp.Expr' = oo, direction: 'str' = '+', simplify_results: 'bool' = True, zero_confidence: 'str' = 'certified', use_sympy_zero_fallback: 'bool' = True, zero_oracle: 'Callable[..., bool | None] | None' = None, sector: 'ComplexSector | None' = None, branch: 'ComplexBranchMetadata | None' = None, assumptions: 'sp.Expr' = True) -> None
```

Shared exact-asymptotic services.

The implementation keeps zero tests and limit/growth queries
centralized because they are the expensive operations in Shackell-style
algorithms.  The optional ``exprtest`` zero oracle is preferred for nontrivial identity
tests, with bounded SymPy fallbacks for limits, signs, and growth.

## `Circle`

```python
Circle(center, radius)
```

Complex numbers at a fixed finite nonnegative distance from a center.

## `DSolveResult`

```python
DSolveResult(solutions: 'tuple[sp.Expr, ...]', function: 'sp.FunctionClass', variable: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', branches: 'tuple[Any, ...]', complete: 'bool', limitation: 'str | None' = None, interchange: 'Any | None' = None) -> None
```

Structured result returned by :func:`dsolve`.

``solutions`` contains ordinary SymPy expressions for the finite prefixes.
``branches`` retains the richer transseries or nonlinear-lifting objects so
callers can inspect certificates, residuals, monodromy metadata, and other
structural information without forcing everything into one expression.

## `DirectionalInfinity`

```python
DirectionalInfinity(d)
```

Infinite modulus with a certified nonzero normalized complex direction.

## `Evidence`

```python
Evidence(status: 'EvidenceStatus', statement: 'str', assumptions: 'tuple[str, ...]' = (), obligations: 'tuple[str, ...]' = (), details: 'tuple[tuple[str, object], ...]' = ()) -> None
```

Auditable evidence for a public asymptotic result or validation check.

## `EvidenceStatus`

```python
EvidenceStatus(value)
```

Strength of the evidence supporting an asymptotic claim.

## `ExactClusterResult`

```python
ExactClusterResult(expression: sympy.core.expr.Expr, variable: sympy.core.symbol.Symbol, point: sympy.core.expr.Expr, cluster_set: sympy.sets.sets.Set | None = None, evidence: tuple[asymptotic.limit_models.LimitEvidence, ...] = (), assumptions: sympy.core.expr.Expr = True) -> None
```

An exact set and its containment/attainment evidence, or an unresolved result.

## `Multiseries`

```python
Multiseries(expr: 'sp.Expr', scale: 'Scale', *, level: 'int | None' = None, context: 'AsymptoticContext | None' = None, default_terms: 'int' = 6, knowledge: 'AsymptoticKnowledge | None' = None, allow_series_fallback: 'bool' = True) -> 'None'
```

Demand-driven recursive multiseries.

At level ``k`` the expression is expanded in ``scale[k]``; coefficients
remain exact expressions and can themselves be expanded on demand in lower
scale elements.  Elementary analytic composition is handled by the package's
own heap frontier; SymPy's univariate series engine is retained as a fallback
for unsupported formal expressions.

## `OptimizationResult`

```python
OptimizationResult(optimum_value: 'sp.Expr', optimizers: 'tuple[sp.Expr, ...]', variable: 'sp.Symbol', parameter: 'sp.Symbol', point: 'sp.Expr', sense: "Literal['min', 'max']", status: 'str', method: 'str', conditions: 'tuple[sp.Expr, ...]' = (), certificate: 'object | None' = None, approached_boundaries: 'tuple[sp.Expr, ...]' = ()) -> None
```

Result of a univariate asymptotic optimization problem.

## `ProductResult`

```python
ProductResult(expression: 'sp.Expr', method: 'str', certified: 'bool', logarithmic_sum: 'SumResult | None' = None) -> None
```

Asymptotic description of a parameter-dependent discrete product.

## `PublicResult`

```python
PublicResult(*args, **kwargs)
```

Protocol for results that provide their own user-facing explanation.

## `RSolveResult`

```python
RSolveResult(expression: 'sp.Expr', sequence: 'sp.Expr', index: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', series: 'TransseriesExpansion | None' = None, limitation: 'str | None' = None, branches: 'tuple[DiscreteAsymptoticBranch, ...]' = (), particular_expression: 'sp.Expr | None' = None, particular_residual: 'sp.Expr | None' = None) -> None
```

Result of asymptotically solving a scalar recurrence.

Exact recurrence solving is preferred when available. Otherwise polynomial-
coefficient linear recurrences can be analyzed by discrete Newton polygons
and factorial-scale Birkhoff–Trjitzinsky lifting.

## `Remainder`

```python
Remainder(variable: 'sp.Symbol', point: 'sp.Expr', kind: 'RemainderKind', scale: 'sp.Expr | None' = None, exact_expression: 'sp.Expr | None' = None, provenance: 'tuple[RemainderProvenance, ...]' = ()) -> None
```

Certified or explicitly unknown remainder attached to a finite prefix.

``kind`` describes the mathematical statement about the error ``R``:

* ``EXACT``: ``R == 0``;
* ``LITTLE_O``: ``R = o(scale)``;
* ``BIG_O``: ``R = O(scale)``;
* ``UNKNOWN``: no asymptotic bound has been certified.

``exact_expression`` may additionally store the exact represented error.
This is useful when truncating a finite exact expression: the exact omitted
tail is known even though its compact asymptotic description is normally
only ``O(first_omitted_monomial)``.

## `RemainderKind`

```python
RemainderKind(value)
```

Semantic strength of an asymptotic remainder statement.

## `Scale`

```python
Scale(variable: 'sp.Symbol', elements: 'tuple[ScaleElement, ...]', point: 'sp.Expr' = oo) -> None
```

A Shackell-style scale ordered from slowest to fastest vanishing.

## `SolveResult`

```python
SolveResult(branches: 'tuple[AsymptoticSolutionBranch, ...]', parameter: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', certificate: 'object | None' = None) -> None
```

SolveResult(branches: 'tuple[AsymptoticSolutionBranch, ...]', parameter: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', certificate: 'object | None' = None)

## `SquareWave`

```python
SquareWave(*args)
```

Odd period-one wave, with zero values at its jumps.

## `StatisticalResult`

```python
StatisticalResult(expression: 'sp.Expr', parameter: 'sp.Symbol', point: 'sp.Expr', method: 'str', status: 'StatisticalStatus', series: 'TransseriesExpansion | None' = None, reduction: 'sp.Expr | None' = None, integration_variable: 'sp.Symbol | None' = None, domain: 'sp.Set | None' = None, transformation: 'tuple[sp.Symbol, sp.Expr] | None' = None, conditions: 'tuple[sp.Expr, ...]' = (), remainder: 'Remainder | None' = None, certificate: 'object | None' = None, normalization: 'object | None' = None) -> None
```

Result of an asymptotic probability or expectation computation.

``expression`` is the finite asymptotic expression returned by the chosen
route.  ``series`` is present when that expression belongs to the package's
finite transseries algebra.  ``reduction`` records the exact density/PMF
problem, while ``transformation`` records a moving-domain substitution.

## `SumResult`

```python
SumResult(expression: 'sp.Expr', variable: 'sp.Symbol | tuple[sp.Symbol, ...]', lower: 'sp.Expr | tuple[sp.Expr, ...]', upper: 'sp.Expr | tuple[sp.Expr, ...]', parameter: 'sp.Symbol', point: 'sp.Expr', method: 'str', status: 'SumStatus', series: 'TransseriesExpansion | None' = None, remainder: 'Remainder | None' = None, reduction: 'sp.Sum | None' = None, transformation: 'tuple[sp.Symbol, sp.Expr] | None' = None, certificate: 'object | None' = None) -> None
```

Finite asymptotic description of a parameter-dependent discrete sum.

## `TransseriesExpansion`

```python
TransseriesExpansion(expression: 'sp.Expr', variable: 'sp.Symbol', point: 'sp.Expr', terms: 'tuple[TransseriesTerm, ...]', center: 'sp.Expr' = 0, complete: 'bool' = False, metadata: 'dict[str, object]' = <factory>, remainder: 'Remainder | None' = None) -> None
```

Finite exact prefix of a generalized transseries branch.

Terms are stored in descending asymptotic magnitude. Structural monomials
support exact finite-prefix arithmetic alongside expression-backed
construction.

## `Truncation`

```python
Truncation(prefix: 'sp.Expr', remainder: 'Remainder', terms_kept: 'int', total_known_terms: 'int') -> None
```

A finite prefix together with its explicit remainder semantics.

## `__version__`

Current package version string.

## `analytic_limit`

```python
analytic_limit(expr, x, point, *, direction=None, analytic_functions=(), assumptions=True, return_result=False)
```

Evaluate a limit with a call-scoped analytic contract for named functions.

Finite image points and the variable approach domain must be admissible.
Unspecified functions receive no continuity assumption. Value mode is the
default; return_result=True exposes status, hypotheses and evidence.

## `argmax`

```python
argmax(objective: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Reals, assumptions: 'sp.Expr | bool' = True) -> 'tuple[sp.Expr, ...]'
```

Return the asymptotic maximizers of a scalar objective.

## `argmin`

```python
argmin(objective: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Reals, assumptions: 'sp.Expr | bool' = True) -> 'tuple[sp.Expr, ...]'
```

Return the asymptotic minimizers of a scalar objective.

## `as_element`

```python
as_element(obj, variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr | None' = None, context: 'AsymptoticContext | None' = None) -> 'AsymptoticElement'
```

Adapt a supported native representation to the common field protocol.

## `big_o`

```python
big_o(left, right, variable, point=oo, *, assumptions=True, **kwargs) -> 'bool | ConditionalExpression | None'
```

Return whether ``left`` is big-O of ``right`` at the requested germ.

## `cluster_set`

```python
cluster_set(expression, variable, point, *, direction=None, assumptions=True, return_result=False)
```

Return a certified exact finite cluster set on a continuous real approach.

Linear sine/cosine phases and imaginary fixed powers have bounded direct
certificates. Other expressions return an unresolved ``ExactClusterResult``.
Infinite values are excluded from these finite cluster sets.

## `complex_limit`

```python
complex_limit(expression, variable, point=0, *, assumptions=True, return_result=False)
```

Take a limit over every complex approach to a finite point or infinity.

The chart ``z = point + u + I*v`` ranges over the punctured plane. A
spherical pole is returned as ``zoo``; it carries no preferred direction.
Unresolved branch or domain questions remain structured unknown results.

## `complex_ray_limit`

```python
complex_ray_limit(expr, variable, point, *, ray, assumptions=True, return_result=False)
```

Finite-target limit on z=point+ray*t, t->0+, for a fixed nonzero ray.

The ray points outward from the target: ray=I approaches from above.
This certifies one path, not a limit across all complex directions.

## `compose`

```python
compose(outer, inner, *, argument: 'sp.Symbol | None' = None, terms: 'int' = 6, assumptions: 'sp.Expr | bool' = True, allow_unknown_properties: 'bool' = False, return_result=False)
```

Return the asymptotic composition ``outer(inner)``.

``outer`` may be a symbolic expression, callable, or supported asymptotic
representation. ``inner`` supplies the asymptotic germ substituted into it.
The argument order follows the standard convention ``(f \circ g)(x) =
f(g(x))``.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `differentiate`

```python
differentiate(obj, order=1, *, variable=None, return_result=False)
```

Differentiate an expression or an asymptotic representation.

A symbolic expression with one free symbol supplies its differentiation
variable. Specify ``variable`` when the expression contains several symbols.
``return_result=True`` retains a native representation when the input has one.

## `discover_scale`

```python
discover_scale(expr: 'sp.Expr', x: 'sp.Symbol', point: 'sp.Expr' = oo, *, assumptions: 'sp.Expr' = True) -> 'Scale'
```

Discover a dependency-driven exp-log scale.

For callers that need the comparison obligations and cached knowledge,
instantiate :class:`ScaleDiscovery` directly and call ``discover()``.

## `discrete_limit`

```python
discrete_limit(expr: 'sp.Expr', variable: 'sp.Symbol', point: 'sp.Expr' = oo, *, assumptions=True, return_result: 'bool' = False)
```

Compute a certified limit along integer values of ``variable``.

The direct route is used when the ordinary real limit is proved.  For
quotients at positive infinity, a Stolz--Cesaro reduction is attempted
when the denominator is eventually increasing and unbounded.  Failure to
prove those hypotheses leaves the result unknown.

## `dsolve`

```python
dsolve(equation: 'sp.Expr | sp.Equality', function: 'sp.FunctionClass', variable: 'sp.Symbol', *, point: 'sp.Expr' = oo, terms: 'int' = 6, assumptions: 'sp.Expr | bool' = True, method: 'str' = 'auto', return_result=False)
```

Solve an ODE asymptotically near a finite point or infinity.

In ``auto`` mode linear equations are first sent through the stable
:mod:`odeanalysis` formal-data interface, which can expose Frobenius,
ramification, exponential blocks, monodromy, and Stokes metadata.  If that
route does not apply, differential-polynomial nonlinear equations are
handled by recursive Newton/transseries lifting. Unsupported equations raise
``NotImplementedError`` when no supported method can justify a complete
asymptotic solution.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `equivalent`

```python
equivalent(left, right, variable, point=oo, *, assumptions=True, **kwargs) -> 'bool | ConditionalExpression | None'
```

Return whether two expressions are asymptotically equivalent at a germ.

## `expectation`

```python
expectation(expr: 'sp.Expr', random_symbol: 'RandomSymbol | None' = None, *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, method: "Literal['auto', 'exact', 'density', 'pmf', 'laplace', 'sum', 'series', 'summation-by-parts', 'saddle', 'euler-maclaurin', 'mellin', 'riemann', 'zeilberger', 'poisson', 'oscillatory']" = 'auto', bindings: 'dict[object, object] | None' = None, condition: 'sp.Expr | None' = None, assumptions: 'sp.Expr' = True, return_result=False)
```

Compute the asymptotic expectation of ``expr``.

The first argument is always the expression being averaged and may depend
on one or more SymPy random variables. ``bindings`` may map ordinary SymPy
symbols in that expression to ``RandomSymbol`` objects; ``condition`` forms
a conditional expectation through SymPy's probability-space machinery.
``auto`` first asks SymPy for the exact joint expectation. If that route does
not settle the problem, the package provides density/PMF and
Laplace/saddle fallbacks for a single random variable; ``random_symbol`` can
disambiguate that fallback. ``laplace`` skips exact integration after density
reduction so concentration asymptotics can be inspected directly.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `explain`

```python
explain(result: 'object') -> 'ResultExplanation'
```

Explain any public result through one stable presentation boundary.

Result types may implement ``PublicResult.explain`` directly. For other
public records, common status, method, certification, and limitation fields
are summarized without coupling theorem code to presentation logic.

## `hyperasymptotic_series`

```python
hyperasymptotic_series(poincare_prefix: 'sp.Expr', coefficient_provider: 'Callable[[int, int], Sequence[sp.Expr]]', scales: 'Sequence[ExponentialScale]', *, levels: 'int' = 1, reexpansion_terms: 'int' = 2, max_levels: 'int' = 3, return_result=False)
```

Recursively re-expand exponentially small remainders to bounded depth.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `implicit`

```python
implicit(equation: 'sp.Expr', dependent: 'sp.Symbol', variable: 'sp.Symbol', *, point: 'sp.Expr' = 0, dependent_limit: 'sp.Expr' = 0, terms: 'int' = 6, context: 'AsymptoticContext | None' = None, taylor_degree: 'int' = 8, max_depth: 'int' = 32, corrections_must_vanish: 'bool' = True, assumptions: 'sp.Expr | bool' = True, stratify_parameters: 'bool' = True, max_parameter_splits: 'int' = 6, return_result=False)
```

Construct Puiseux or generalized-transseries implicit branches.

Dominant balance uses the same exp-log monomial hierarchy as multiseries.
Pure rational-power branches use ``PuiseuxSeries``; logarithmic or
exponential corrections use ``TransseriesExpansion``.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `integrate`

```python
integrate(obj, variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr' = oo, constant: 'sp.Expr' = 0, terms: 'int | None' = None, assumptions: 'sp.Expr | bool' = True, allow_unknown_properties: 'bool' = False, return_result=False)
```

Integrate an asymptotic representation or asymptotically integrate an expression.

``integrate(representation)`` preserves the representation's native calculus
protocol. ``integrate(expr, variable, ...)`` constructs a scale-aware finite
asymptotic primitive of a symbolic expression.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `inverse`

```python
inverse(expr, variable: 'sp.Symbol | None' = None, inverse_variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr' = oo, terms: 'int' = 6, branch: 'int | None' = 0, context: 'AsymptoticContext | None' = None, assumptions: 'sp.Expr | bool' = True, allow_unknown_properties: 'bool' = False, return_result=False)
```

Asymptotically invert ``y=f(x)`` at a finite point or infinity.

Infinite inversion is reduced exactly to local reversion by reciprocal
coordinates. If ``f(x)->oo`` we revert ``1/f(1/u)`` against ``z=1/y``;
if ``f(x)->0`` we revert ``f(1/u)`` against ``y``. The returned expression
is mapped back to the original inverse variable.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `leading_term`

```python
leading_term(obj, *, return_result=False)
```

Return the mathematical leading term of an asymptotic representation.

``return_result=True`` retains a stored coefficient/monomial record when
the representation supplies one. An exhausted expansion has leading term 0.

## `limit`

```python
limit(expr, variables, target, *, domain=True, assumptions=True, return_result=False)
```

Compute a proof-aware limit within a shared call-scoped analysis context.

## `lindstedt_poincare`

```python
lindstedt_poincare(equation: 'EquationLike', unknown: 'sp.Expr', parameter: 'sp.Symbol', *, order: 'int' = 1, conditions: 'EquationLike | Sequence[EquationLike]' = (), base_frequency: 'sp.Expr | None' = None, fast_variable: 'sp.Symbol | None' = None, assumptions: 'sp.Expr' = True, return_result=False)
```

Construct a Lindstedt–Poincaré expansion for a scalar oscillator.

The unperturbed equation must be an autonomous, undamped, constant-
coefficient second-order oscillator.  Its natural frequency is inferred
unless ``base_frequency`` is supplied.  At each later order, periodic
Fredholm projections against ``cos(tau)`` and ``sin(tau)`` determine the
frequency correction before the bounded coefficient ODE is solved.

Unsupported or unresolved orders are returned as a solved prefix with an
explicit ``unresolved_order`` instead of accepting secular growth.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `little_o`

```python
little_o(left, right, variable, point=oo, *, assumptions=True, **kwargs) -> 'bool | ConditionalExpression | None'
```

Return whether ``left`` is little-o of ``right`` at the requested germ.

## `local_series`

```python
local_series(expr: 'sp.Expr', variable: 'sp.Symbol', point: 'sp.Expr' = 0, *, depth: 'int' = 4, return_result=False)
```

Expand a scalar local germ after exact algebraic cancellation.

Primitive providers are consulted when the expression itself is a registered
function. Composite expressions use the common series backend so products
and sums are expanded as a whole and equal singular monomials cancel before
valuation.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `maximize`

```python
maximize(objective: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Reals, assumptions: 'sp.Expr | bool' = True, return_result=False)
```

Asymptotically maximize a univariate parameter-dependent objective.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `mellin`

```python
mellin(expr: 'sp.Expr', variable: 'sp.Symbol', transform_variable: 'sp.Symbol', *, toward_zero: 'bool' = True, max_poles: 'int' = 4, return_result=False)
```

Extract inverse-Mellin asymptotic terms from explicit transform poles.

Only transforms returned explicitly by SymPy with a fundamental strip are
accepted. Pole enumeration is restricted to finite polynomial/rational
denominators; unresolved Gamma lattices and contour-growth conditions are
reported as obligations instead of being guessed.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `minimize`

```python
minimize(objective: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Reals, assumptions: 'sp.Expr | bool' = True, return_result=False)
```

Asymptotically minimize a univariate parameter-dependent objective.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `multiseries`

```python
multiseries(expr: 'sp.Expr', variable: 'sp.Symbol', *, scale: 'Scale | Iterable[sp.Expr] | None' = None, point: 'sp.Expr' = oo, terms: 'int' = 6, allow_series_fallback: 'bool' = True, assumptions: 'sp.Expr' = True, context: 'AsymptoticContext | None' = None, return_result=False)
```

Create a lazy multiseries, discovering an asymptotic scale when omitted.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `nested_series`

```python
nested_series(expr: 'sp.Expr', variable: 'sp.Symbol', *, depth: 'int' = 4, point: 'sp.Expr' = oo, max_exp_depth: 'int | None' = None, max_log_depth: 'int | None' = None, assumptions: 'sp.Expr' = True, return_result=False)
```

Build a resumable nested expansion, eagerly refining up to *depth* levels.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `one_sided_limit`

```python
one_sided_limit(expr, variable, point, *, direction: 'str', assumptions=True, return_result=False)
```

Compute a certified univariate limit from one real side.

``direction='+'`` approaches from larger real values and ``direction='-'``
approaches from smaller real values.  The request is reduced exactly to a
positive local parameter before invoking :func:`asymptotic.limit`.

## `path_limit`

```python
path_limit(expr, variable, point, *, path: 'ParametricContour', assumptions=True, return_result=False)
```

Compute a certified limit along the terminal germ of a parametric contour.

## `probability`

```python
probability(event: 'sp.Expr', random_symbol: 'RandomSymbol | None' = None, *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, method: "Literal['auto', 'exact', 'density', 'pmf', 'laplace', 'sum', 'series', 'summation-by-parts', 'saddle', 'euler-maclaurin', 'mellin', 'riemann', 'zeilberger', 'poisson', 'oscillatory']" = 'auto', bindings: 'dict[object, object] | None' = None, condition: 'sp.Expr | None' = None, assumptions: 'sp.Expr' = True, return_result=False)
```

Compute an asymptotic probability for an event.

``event`` may contain SymPy ``RandomSymbol`` objects directly, or ordinary
symbols may be mapped to random symbols with ``bindings``.  Exact SymPy
probability is attempted before a one-random-variable fallback, so exact
joint events are supported when SymPy can evaluate them.  ``condition``
requests conditional probability.  Structural density/PMF and Laplace
fallbacks remain one-dimensional.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `product`

```python
product(factor: 'sp.Expr', variable: 'sp.Symbol', lower: 'sp.Expr', upper: 'sp.Expr', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, assumptions: 'sp.Expr' = True, return_result=False)
```

Expand a positive real product through ``exp(sum(log(factor)))``.

Certification is inherited from the asymptotic-sum proof and additionally
requires positivity of the factor under the supplied assumptions. Products
with unresolved sign or branch behavior are returned as uncertified.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `puiseux_series`

```python
puiseux_series(expr: 'sp.Expr', variable: 'sp.Symbol', *, point: 'sp.Expr' = 0, terms: 'int' = 6, branch: 'BranchChoice | None' = None, assumptions: 'sp.Expr' = True, return_result=False)
```

Construct a rational-exponent local series with explicit ramification.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `regular_perturbation`

```python
regular_perturbation(equations: 'EquationLike | Sequence[EquationLike]', unknowns: 'sp.Expr | Sequence[sp.Expr]', parameter: 'sp.Symbol', *, order: 'int | None' = None, gauges: 'Sequence[sp.Expr] | None' = None, conditions: 'EquationLike | Sequence[EquationLike]' = (), assumptions: 'sp.Expr' = True, return_result=False)
```

Solve a regular perturbation problem recursively order by order.

The function first constructs the common :func:`perturbation_hierarchy`,
then solves each coefficient problem using bounded algebraic solving or
ordinary differential-equation solving as appropriate.  Algebraic branch
multiplicity is preserved.  If a later order cannot be solved, the solved
prefix is returned as a partial branch instead of being discarded or
guessed.

Initial, boundary, and normalization conditions are expanded at the same
perturbation orders as the governing equations.  ODE integration constants
are fixed from the conditions whenever they are uniquely determined.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `relation`

```python
relation(left: 'sp.Expr', right: 'sp.Expr', variables: 'sp.Symbol | tuple[sp.Symbol, ...] | list[sp.Symbol]', points: 'sp.Expr | tuple[sp.Expr, ...] | list[sp.Expr]', *, relation: 'RelationKind' = 'equivalent', directions: "Literal['real', 'complex']" = 'real', ray_samples: 'int' = 8, assumptions: 'sp.Expr | bool' = True, return_result=False)
```

Decide or conservatively test an asymptotic relation.

For one variable at a finite real point, ``directions="real"`` checks both
one-sided germs and certifies ``True`` only when both sides agree.  For
several variables, coordinate and deterministic pseudo-random rays can
*disprove* a relation, but finite ray sampling never certifies a positive
multivariate statement.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `root`

```python
root(expression: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Complexes, assumptions: 'sp.Expr | bool' = True, limit: 'sp.Expr | None' = None, branch: 'int | None' = None, return_result=False)
```

Find roots of ``expression == 0`` asymptotically in ``parameter``.

With ``branch=None`` the complete :class:`SolveResult` is
returned.  Otherwise the requested branch expression is returned directly.
``limit`` may select roots tending to a prescribed value.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `rsolve`

```python
rsolve(recurrence: 'sp.Expr | sp.Equality', sequence: 'sp.Expr', index: 'sp.Symbol', *, point: 'sp.Expr' = oo, terms: 'int' = 6, initial_conditions: 'dict | None' = None, method: 'str' = 'auto', assumptions: 'sp.Expr' = True, return_result=False)
```

Solve a scalar recurrence and expand the resulting solution asymptotically.

``auto`` prefers an exact recurrence solution and otherwise applies native
discrete Newton analysis. Simple roots use ordinary Birkhoff–Trjitzinsky
lifting, repeated constant-coefficient roots use exact polynomial Jordan
chains, and supported repeated variable-coefficient roots use secondary
Newton phases with ramified inverse-power lattices.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `series`

```python
series(expr: 'sp.Expr', variable: 'sp.Symbol', *, point: 'sp.Expr' = oo, terms: 'int' = 6, assumptions: 'sp.Expr' = True, method: 'ExpansionMethod' = 'auto', return_result=False)
```

Expand *expr* asymptotically using one task-oriented entry point.

``method="auto"`` uses a local Puiseux expansion at finite points and the
general multiseries engine at infinity. Specialized representations remain
directly selectable when the user knows the desired asymptotic structure.
The dispatcher contains no independent asymptotic algorithm.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `solve`

```python
solve(system, variables, *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 6, limits: 'dict[sp.Symbol, sp.Expr] | None' = None, domain=Complexes, assumptions: 'sp.Expr' = True, return_result=False)
```

Solve algebraic equations/inequalities asymptotically in ``parameter``.

For a univariate polynomial with transcendental Hardy/log-exp
coefficients, a Newton–MRV backend is tried before exact algebraic solving.
It values coefficient scales, lifts smaller Newton corrections recursively,
and uses an asymptotic Sturm sequence to certify completeness of real roots.
Rational/algebraic systems retain the exact-solve route.  A supplied branch
limit still enables the implicit/Puiseux backend when neither route settles
the problem. Undecidable predicates remain symbolic branch conditions.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `stratified_series`

```python
stratified_series(expr, variables, *, target=None, order=3, assumptions=True)
```

Discover a finite certified power/log-exp/oscillatory stratification.

The multivariate transseries engine supports two competing finite-target power/log-exp scales and the critical
symbolic-exponent denominator family. Unsupported geometry returns a
structured obligation rather than an uncertified formal expansion.

## `sum`

```python
sum(summand: 'sp.Expr', variable: 'sp.Symbol | tuple[sp.Symbol, ...]', lower: 'sp.Expr | tuple[sp.Expr, ...]', upper: 'sp.Expr | tuple[sp.Expr, ...]', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, method: 'SumMethod' = 'auto', assumptions: 'sp.Expr' = True, return_result=False)
```

Expand a parameter-dependent discrete sum asymptotically.

``auto`` tries exact summation, certified/formal termwise expansion, Abel
transforms, creative telescoping into :func:`rsolve`, Poisson or
finite oscillatory reduction, Euler–Maclaurin, certified Mellin shifts,
scaled Riemann sums, and lattice saddles. Tuple-valued variables and bounds
support separable multidimensional sums and fixed finite boxes.

A route is marked ``CERTIFIED`` only when its replayable theorem obligations
are proved; unsupported contour, uniformity, or lattice hypotheses remain
``FORMAL`` or ``UNKNOWN`` rather than being guessed.

The root API returns the mathematical value. Use ``return_result=True`` for the full representation and evidence record. Unresolved or unproved formal calculations retain their records.

## `truncate`

```python
truncate(obj, terms: 'int | None' = None) -> 'sp.Expr'
```

Return the finite expression obtained by truncating a representation.

