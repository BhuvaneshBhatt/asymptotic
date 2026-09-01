# Generated primary API reference

This file is generated from the live root API signatures and docstrings. Edit the public docstrings, then run `python tools/generate_api_reference.py`.

## `AsymptoticAlgebra`

```python
AsymptoticAlgebra(variable: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 6, context: 'AsymptoticContext | None' = None) -> None
```

Coordinate-aware common algebra for heterogeneous asymptotic objects.

The algebra owns coercion and the finite transseries normal form used for
cross-representation operations.  Native algorithms remain preferred for
unary operations, but binary arithmetic, comparisons, and remainder
propagation all pass through this single boundary.

## `AsymptoticContext`

```python
AsymptoticContext(variable: 'sp.Symbol', point: 'sp.Expr' = oo, direction: 'str' = '+', simplify_results: 'bool' = True, zero_confidence: 'str' = 'certified', use_sympy_zero_fallback: 'bool' = True, zero_oracle: 'Callable[..., bool | None] | None' = None, sector: 'ComplexSector | None' = None, branch: 'ComplexBranchMetadata | None' = None) -> None
```

Shared exact-asymptotic services.

The implementation deliberately keeps zero tests and limit/growth queries
centralized because they are the expensive operations in Shackell-style
algorithms.  The optional ``exprtest`` zero oracle is preferred for nontrivial identity
tests, with conservative SymPy fallbacks for limits, signs, and growth.

## `AsymptoticDSolveResult`

```python
AsymptoticDSolveResult(solutions: 'tuple[sp.Expr, ...]', function: 'sp.FunctionClass', variable: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', branches: 'tuple[Any, ...]', complete: 'bool', limitation: 'str | None' = None) -> None
```

Structured result returned by :func:`asymptotic_dsolve`.

``solutions`` contains ordinary SymPy expressions for the finite prefixes.
``branches`` retains the richer transseries or nonlinear-lifting objects so
callers can inspect certificates, residuals, monodromy metadata, and other
structural information without forcing everything into one expression.

## `AsymptoticElement`

```python
AsymptoticElement(native: 'object', variable: 'sp.Symbol', point: 'sp.Expr' = oo, context: 'AsymptoticContext | None' = None) -> None
```

Uniform algebraic view of any supported asymptotic representation.

## `AsymptoticOptimizationResult`

```python
AsymptoticOptimizationResult(optimum_value: 'sp.Expr', optimizers: 'tuple[sp.Expr, ...]', variable: 'sp.Symbol', parameter: 'sp.Symbol', point: 'sp.Expr', sense: "Literal['min', 'max']", status: 'str', method: 'str', conditions: 'tuple[sp.Expr, ...]' = (), certificate: 'object | None' = None, approached_boundaries: 'tuple[sp.Expr, ...]' = ()) -> None
```

Result of a univariate asymptotic optimization problem.

## `AsymptoticRSolveResult`

```python
AsymptoticRSolveResult(expression: 'sp.Expr', sequence: 'sp.Expr', index: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', series: 'TransseriesExpansion | None' = None, limitation: 'str | None' = None, branches: 'tuple[DiscreteAsymptoticBranch, ...]' = (), particular_expression: 'sp.Expr | None' = None, particular_residual: 'sp.Expr | None' = None) -> None
```

Result of asymptotically solving a scalar recurrence.

Exact recurrence solving is preferred when available. Otherwise polynomial-
coefficient linear recurrences can be analyzed by discrete Newton polygons
and factorial-scale Birkhoff--Trjitzinsky lifting.

## `AsymptoticRelationResult`

```python
AsymptoticRelationResult(relation: 'RelationKind', left: 'sp.Expr', right: 'sp.Expr', variables: 'tuple[sp.Symbol, ...]', points: 'tuple[sp.Expr, ...]', value: 'bool | None', certified: 'bool', evidence: 'tuple[DirectedRelationEvidence, ...]' = (), reason: 'str' = '') -> None
```

AsymptoticRelationResult(relation: 'RelationKind', left: 'sp.Expr', right: 'sp.Expr', variables: 'tuple[sp.Symbol, ...]', points: 'tuple[sp.Expr, ...]', value: 'bool | None', certified: 'bool', evidence: 'tuple[DirectedRelationEvidence, ...]' = (), reason: 'str' = '')

## `AsymptoticRemainder`

```python
AsymptoticRemainder(variable: 'sp.Symbol', point: 'sp.Expr', kind: 'RemainderKind', scale: 'sp.Expr | None' = None, exact_expression: 'sp.Expr | None' = None, provenance: 'tuple[RemainderProvenance, ...]' = ()) -> None
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

## `AsymptoticScale`

```python
AsymptoticScale(variable: 'sp.Symbol', elements: 'tuple[ScaleElement, ...]', point: 'sp.Expr' = oo) -> None
```

A Shackell-style scale ordered from slowest to fastest vanishing.

## `AsymptoticSolveResult`

```python
AsymptoticSolveResult(branches: 'tuple[AsymptoticSolutionBranch, ...]', parameter: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', certificate: 'object | None' = None) -> None
```

AsymptoticSolveResult(branches: 'tuple[AsymptoticSolutionBranch, ...]', parameter: 'sp.Symbol', point: 'sp.Expr', status: 'str', method: 'str', certificate: 'object | None' = None)

## `AsymptoticSumResult`

```python
AsymptoticSumResult(expression: 'sp.Expr', variable: 'sp.Symbol | tuple[sp.Symbol, ...]', lower: 'sp.Expr | tuple[sp.Expr, ...]', upper: 'sp.Expr | tuple[sp.Expr, ...]', parameter: 'sp.Symbol', point: 'sp.Expr', method: 'str', status: 'SumStatus', series: 'TransseriesExpansion | None' = None, remainder: 'AsymptoticRemainder | None' = None, reduction: 'sp.Sum | None' = None, transformation: 'tuple[sp.Symbol, sp.Expr] | None' = None, certificate: 'object | None' = None) -> None
```

Finite asymptotic description of a parameter-dependent discrete sum.

## `AsymptoticTruncation`

```python
AsymptoticTruncation(prefix: 'sp.Expr', remainder: 'AsymptoticRemainder', terms_kept: 'int', total_known_terms: 'int') -> None
```

A finite prefix together with its explicit remainder semantics.

## `GrowthComparison`

```python
GrowthComparison(value)
```

No public docstring.

## `Multiseries`

```python
Multiseries(expr: 'sp.Expr', scale: 'AsymptoticScale', *, level: 'int | None' = None, context: 'AsymptoticContext | None' = None, default_terms: 'int' = 6, knowledge: 'AsymptoticKnowledge | None' = None, allow_series_fallback: 'bool' = True) -> 'None'
```

Demand-driven recursive multiseries.

At level ``k`` the expression is expanded in ``scale[k]``; coefficients
remain exact expressions and can themselves be expanded on demand in lower
scale elements.  Elementary analytic composition is handled by the package's
own heap frontier; SymPy's univariate series engine is retained as a fallback
for unsupported formal expressions.

## `NestedExpansion`

```python
NestedExpansion(expr: 'sp.Expr', variable: 'sp.Symbol', *, point: 'sp.Expr' = oo, context: 'AsymptoticContext | None' = None, seed: 'tuple[NestedForm, ...]' = ()) -> 'None'
```

Resumable exact nested expansion.

Successive forms are generated from the exact terminal residual only when
requested.  The object therefore has no fixed expansion depth. Arithmetic
creates another lazy ``NestedExpansion`` from exact operands; cancellation
is checked by the shared ``AsymptoticContext``/``exprtest`` zero oracle
before another residual form is requested.

## `RemainderKind`

```python
RemainderKind(value)
```

Semantic strength of an asymptotic remainder statement.

## `StatisticalAsymptoticResult`

```python
StatisticalAsymptoticResult(expression: 'sp.Expr', parameter: 'sp.Symbol', point: 'sp.Expr', method: 'str', status: 'StatisticalStatus', series: 'TransseriesExpansion | None' = None, reduction: 'sp.Expr | None' = None, integration_variable: 'sp.Symbol | None' = None, domain: 'sp.Set | None' = None, transformation: 'tuple[sp.Symbol, sp.Expr] | None' = None, conditions: 'tuple[sp.Expr, ...]' = (), remainder: 'AsymptoticRemainder | None' = None, certificate: 'object | None' = None, normalization: 'object | None' = None) -> None
```

Result of an asymptotic probability or expectation computation.

``expression`` is the finite asymptotic expression returned by the chosen
route.  ``series`` is present when that expression belongs to the package's
finite transseries algebra.  ``reduction`` records the exact density/PMF
problem, while ``transformation`` records a moving-domain substitution.

## `TransseriesExpansion`

```python
TransseriesExpansion(expression: 'sp.Expr', variable: 'sp.Symbol', point: 'sp.Expr', terms: 'tuple[TransseriesTerm, ...]', center: 'sp.Expr' = 0, complete: 'bool' = False, metadata: 'dict[str, object]' = <factory>, remainder: 'AsymptoticRemainder | None' = None) -> None
```

Finite exact prefix of a generalized transseries branch.

Terms are stored in descending asymptotic magnitude. Structural monomials
support exact finite-prefix arithmetic alongside expression-backed
construction.

## `__version__`

Current package version string.

## `airy_uniform_saddle_asymptotic`

```python
airy_uniform_saddle_asymptotic(integrand: 'sp.Expr', variable: 'sp.Symbol', domain: 'sp.Interval | tuple[sp.Expr, sp.Expr]', *, parameter: 'sp.Symbol', control_parameter: 'sp.Symbol', coalescence_value: 'sp.Expr' = 0, location: 'sp.Expr' = 0, terms: 'int' = 1) -> 'StatisticalAsymptoticResult'
```

Leading uniform Airy approximation at a simple cubic turning point.

This routine covers the canonical oscillatory coalescence in which an
integral contains ``exp(I*parameter*phase(x, mu))`` and, at ``(x0, mu0)``,
``phase_x = phase_xx = 0`` while ``phase_xxx`` and ``phase_xmu`` are real
and nonzero.  With ``mu-mu0 = O(parameter**(-2/3))`` two simple stationary
points coalesce and the local integral is uniformly represented by an Airy
function.  The implementation intentionally returns only the leading CFU
term; higher ``terms`` are reserved for later Airy/Airy-prime transport.

The full real line is required because the normalization uses the standard
oscillatory identity

    integral exp(I*(t**3/3 + z*t)) dt = 2*pi*Ai(z).

The result is FORMAL: local cubic reduction alone does not certify contour
deformation, remote saddle dominance, or tail cancellation.

## `asymptotic_argmax`

```python
asymptotic_argmax(*args, **kwargs) -> 'tuple[sp.Expr, ...]'
```

Return the asymptotic maximizers of a scalar objective.

## `asymptotic_argmin`

```python
asymptotic_argmin(*args, **kwargs) -> 'tuple[sp.Expr, ...]'
```

Return the asymptotic minimizers of a scalar objective.

## `asymptotic_big_o`

```python
asymptotic_big_o(left, right, variable, point=oo, **kwargs) -> 'bool | None'
```

Return whether ``left`` is big-O of ``right`` at the requested germ.

## `asymptotic_dsolve`

```python
asymptotic_dsolve(equation: 'sp.Expr | sp.Equality', function: 'sp.FunctionClass', variable: 'sp.Symbol', *, point: 'sp.Expr' = oo, terms: 'int' = 6, assumptions: 'sp.Expr | bool' = True, method: 'str' = 'auto') -> 'AsymptoticDSolveResult'
```

Solve an ODE asymptotically near a finite point or infinity.

In ``auto`` mode linear equations are first sent through the stable
:mod:`odeanalysis` formal-data interface, which can expose Frobenius,
ramification, exponential blocks, monodromy, and Stokes metadata.  If that
route does not apply, differential-polynomial nonlinear equations are
handled by recursive Newton/transseries lifting.  The function is
conservative: unsupported equations raise ``NotImplementedError`` rather
than being mislabeled as complete asymptotic solutions.

## `asymptotic_element`

```python
asymptotic_element(obj, variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr | None' = None, context: 'AsymptoticContext | None' = None) -> 'AsymptoticElement'
```

Adapt a supported native representation to the common field protocol.

## `asymptotic_equivalent`

```python
asymptotic_equivalent(left, right, variable, point=oo, **kwargs) -> 'bool | None'
```

Return whether two expressions are asymptotically equivalent at a germ.

## `asymptotic_expectation`

```python
asymptotic_expectation(expr: 'sp.Expr', random_symbol: 'RandomSymbol | None' = None, *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, method: "Literal['auto', 'exact', 'density', 'pmf', 'laplace', 'sum', 'series', 'summation-by-parts', 'saddle', 'euler-maclaurin', 'mellin', 'riemann', 'zeilberger', 'poisson', 'oscillatory']" = 'auto', bindings: 'dict[object, object] | None' = None, condition: 'sp.Expr | None' = None) -> 'StatisticalAsymptoticResult'
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

## `asymptotic_integrate`

```python
asymptotic_integrate(obj: 'object', variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr' = oo, constant: 'sp.Expr' = 0, terms: 'int' = 6, assumptions: 'sp.Expr | bool' = True, allow_unknown_properties: 'bool' = False) -> 'TransseriesExpansion'
```

Compute a scale-aware finite asymptotic primitive.

Any common-protocol representation may be supplied directly. Exact primitives are preferred. Otherwise exponential scales use repeated
integration by parts, and power/log scales use the Hardy/Shackell leading
integral forms with recursive lower-log corrections.

## `asymptotic_little_o`

```python
asymptotic_little_o(left, right, variable, point=oo, **kwargs) -> 'bool | None'
```

Return whether ``left`` is little-o of ``right`` at the requested germ.

## `asymptotic_maximize`

```python
asymptotic_maximize(objective: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Reals, assumptions: 'sp.Expr | bool' = True) -> 'AsymptoticOptimizationResult'
```

Asymptotically maximize a univariate parameter-dependent objective.

## `asymptotic_minimize`

```python
asymptotic_minimize(objective: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Reals, assumptions: 'sp.Expr | bool' = True) -> 'AsymptoticOptimizationResult'
```

Asymptotically minimize a univariate parameter-dependent objective.

## `asymptotic_probability`

```python
asymptotic_probability(event: 'sp.Expr', random_symbol: 'RandomSymbol | None' = None, *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, method: "Literal['auto', 'exact', 'density', 'pmf', 'laplace', 'sum', 'series', 'summation-by-parts', 'saddle', 'euler-maclaurin', 'mellin', 'riemann', 'zeilberger', 'poisson', 'oscillatory']" = 'auto', bindings: 'dict[object, object] | None' = None, condition: 'sp.Expr | None' = None) -> 'StatisticalAsymptoticResult'
```

Compute an asymptotic probability for an event.

``event`` may contain SymPy ``RandomSymbol`` objects directly, or ordinary
symbols may be mapped to random symbols with ``bindings``.  Exact SymPy
probability is attempted before a one-random-variable fallback, so exact
joint events are supported when SymPy can evaluate them.  ``condition``
requests conditional probability.  Structural density/PMF and Laplace
fallbacks remain one-dimensional.

## `asymptotic_relation`

```python
asymptotic_relation(left: 'sp.Expr', right: 'sp.Expr', variables: 'sp.Symbol | tuple[sp.Symbol, ...] | list[sp.Symbol]', points: 'sp.Expr | tuple[sp.Expr, ...] | list[sp.Expr]', *, relation: 'RelationKind' = 'equivalent', directions: "Literal['real', 'complex']" = 'real', ray_samples: 'int' = 8, assumptions: 'sp.Expr | bool' = True) -> 'AsymptoticRelationResult'
```

Decide or conservatively test an asymptotic relation.

For one variable at a finite real point, ``directions="real"`` checks both
one-sided germs and certifies ``True`` only when both sides agree.  For
several variables, coordinate and deterministic pseudo-random rays can
*disprove* a relation, but finite ray sampling never certifies a positive
multivariate statement.

## `asymptotic_root`

```python
asymptotic_root(expression: 'sp.Expr', variable: 'sp.Symbol', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, domain: 'sp.Set' = Complexes, assumptions: 'sp.Expr | bool' = True, limit: 'sp.Expr | None' = None, branch: 'int | None' = None) -> 'AsymptoticSolveResult | sp.Expr'
```

Find roots of ``expression == 0`` asymptotically in ``parameter``.

With ``branch=None`` the complete :class:`AsymptoticSolveResult` is
returned.  Otherwise the requested branch expression is returned directly.
``limit`` may select roots tending to a prescribed value.

## `asymptotic_rsolve`

```python
asymptotic_rsolve(recurrence: 'sp.Expr | sp.Equality', sequence: 'sp.Expr', index: 'sp.Symbol', *, point: 'sp.Expr' = oo, terms: 'int' = 6, initial_conditions: 'dict | None' = None, method: 'str' = 'auto') -> 'AsymptoticRSolveResult'
```

Solve a scalar recurrence and expand the resulting solution asymptotically.

``auto`` prefers an exact recurrence solution and otherwise applies native
discrete Newton analysis. Simple roots use ordinary Birkhoff--Trjitzinsky
lifting, repeated constant-coefficient roots use exact polynomial Jordan
chains, and supported repeated variable-coefficient roots use secondary
Newton phases with ramified inverse-power lattices.

## `asymptotic_solve`

```python
asymptotic_solve(system, variables, *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 6, limits: 'dict[sp.Symbol, sp.Expr] | None' = None, domain=Complexes, assumptions: 'sp.Expr' = True) -> 'AsymptoticSolveResult'
```

Solve algebraic equations/inequalities asymptotically in ``parameter``.

For a univariate polynomial with transcendental Hardy/log-exp
coefficients, a Newton--MRV backend is tried before exact algebraic solving.
It values coefficient scales, lifts smaller Newton corrections recursively,
and uses an asymptotic Sturm sequence to certify completeness of real roots.
Rational/algebraic systems retain the exact-solve route.  A supplied branch
limit still enables the implicit/Puiseux backend when neither route settles
the problem. Undecidable predicates remain symbolic branch conditions.

## `asymptotic_sum`

```python
asymptotic_sum(summand: 'sp.Expr', variable: 'sp.Symbol | tuple[sp.Symbol, ...]', lower: 'sp.Expr | tuple[sp.Expr, ...]', upper: 'sp.Expr | tuple[sp.Expr, ...]', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, method: 'SumMethod' = 'auto') -> 'AsymptoticSumResult'
```

Expand a parameter-dependent discrete sum asymptotically.

``auto`` tries exact summation, certified/formal termwise expansion, Abel
transforms, creative telescoping into :func:`asymptotic_rsolve`, Poisson or
finite oscillatory reduction, Euler--Maclaurin, certified Mellin shifts,
scaled Riemann sums, and lattice saddles. Tuple-valued variables and bounds
support separable multidimensional sums and fixed finite boxes.

A route is marked ``CERTIFIED`` only when its replayable theorem obligations
are proved; unsupported contour, uniformity, or lattice hypotheses remain
``FORMAL`` or ``UNKNOWN`` rather than being guessed.

## `coalescing_saddle_asymptotic`

```python
coalescing_saddle_asymptotic(integrand: 'sp.Expr', variable: 'sp.Symbol', domain: 'sp.Interval | tuple[sp.Expr, sp.Expr]', *, parameter: 'sp.Symbol', control_parameter: 'sp.Symbol', coalescence_value: 'sp.Expr' = 0, location: 'sp.Expr' = 0, transition_symbol: 'sp.Symbol | None' = None, terms: 'int' = 2) -> 'StatisticalAsymptoticResult'
```

Uniform quartic transition for a symmetric pair of coalescing minima.

The method targets the real Laplace normal form in which, at the
coalescence, the first three x-derivatives vanish, the fourth derivative is
positive, and the control parameter enters the quadratic term.  It uses
``x=x0+p^-1/4*u`` and ``mu=mu0+tau*p^-1/2`` and retains the resulting
canonical quartic profile integrals.  This is uniform for bounded ``tau``.

## `compose_transseries`

```python
compose_transseries(outer: 'sp.Expr | TransseriesExpansion', inner: 'object', *, argument: 'sp.Symbol | None' = None, terms: 'int' = 6, assumptions: 'sp.Expr | bool' = True, allow_unknown_properties: 'bool' = False) -> 'TransseriesExpansion'
```

Compose a finite LE expression/transseries with an asymptotic argument.

Protocol-compatible representations are converted to their finite
transseries view. Arithmetic, exp, log and constant powers are evaluated recursively in the
native transseries algebra. Other functions are admitted only through a
certified finite-center analytic expansion.

## `differentiate`

```python
differentiate(obj, order: 'int' = 1)
```

Differentiate any supported asymptotic object through the common protocol.

Native inputs return their native representation; passing an
:class:`AsymptoticElement` keeps the unified wrapper.

## `discover_scale`

```python
discover_scale(expr: 'sp.Expr', x: 'sp.Symbol', point: 'sp.Expr' = oo) -> 'AsymptoticScale'
```

Discover a dependency-driven exp-log scale.

For callers that need the comparison obligations and cached knowledge,
instantiate :class:`ScaleDiscovery` directly and call ``discover()``.

## `dominant_balance_candidates`

```python
dominant_balance_candidates(equation: 'sp.Expr', dependent: 'sp.Symbol', variable: 'sp.Symbol', *, context: 'AsymptoticContext | None' = None, valuation: 'Callable[[sp.Expr, sp.Symbol], tuple[sp.Rational, sp.Expr] | None]' = <function rational_valuation>, taylor_degree: 'int' = 8, minimum_exponent: 'sp.Rational | None' = None, assumptions: 'sp.Expr | bool' = True, stratify_parameters: 'bool' = True, max_parameter_splits: 'int' = 6) -> 'tuple[DominantBalanceCandidate, ...] | AsymptoticStratification[tuple[DominantBalanceCandidate, ...]]'
```

Find exact Newton-style dominant balances.

Polynomial equations use their exact coefficient set.  For genuinely
transcendental dependence on ``dependent``, an exact local Taylor jet in
the correction variable is used.  Candidate exponents are still accepted
only after the tied terms are verified to attain the global minimum.

## `implicit_asymptotic`

```python
implicit_asymptotic(equation: 'sp.Expr', dependent: 'sp.Symbol', variable: 'sp.Symbol', *, point: 'sp.Expr' = 0, dependent_limit: 'sp.Expr' = 0, terms: 'int' = 6, context: 'AsymptoticContext | None' = None, taylor_degree: 'int' = 8, max_depth: 'int' = 32, corrections_must_vanish: 'bool' = True, assumptions: 'sp.Expr | bool' = True, stratify_parameters: 'bool' = True, max_parameter_splits: 'int' = 6) -> 'tuple[ImplicitAsymptoticBranch, ...] | AsymptoticStratification[tuple[ImplicitAsymptoticBranch, ...]]'
```

Construct Puiseux or generalized-transseries implicit branches.

Dominant balance uses the same exp-log monomial hierarchy as multiseries.
Pure rational-power branches use ``PuiseuxSeries``; logarithmic or
exponential corrections use ``TransseriesExpansion``.

## `integrate`

```python
integrate(obj, *, constant: 'sp.Expr' = 0, terms: 'int | None' = None)
```

Integrate any supported asymptotic object through the common protocol.

## `inverse_asymptotic`

```python
inverse_asymptotic(expr: 'sp.Expr', variable: 'sp.Symbol', inverse_variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr' = oo, terms: 'int' = 6, branch: 'int | None' = 0, context: 'AsymptoticContext | None' = None, assumptions: 'sp.Expr | bool' = True, allow_unknown_properties: 'bool' = False) -> 'ReversionBranch | tuple[ReversionBranch, ...]'
```

Asymptotically invert ``y=f(x)`` at a finite point or infinity.

Infinite inversion is reduced exactly to local reversion by reciprocal
coordinates. If ``f(x)->oo`` we revert ``1/f(1/u)`` against ``z=1/y``;
if ``f(x)->0`` we revert ``f(1/u)`` against ``y``. The returned expression
is mapped back to the original inverse variable.

## `laplace_asymptotic_integral`

```python
laplace_asymptotic_integral(integrand: 'sp.Expr', variable: 'sp.Symbol', domain: 'sp.Interval | tuple[sp.Expr, sp.Expr]', *, parameter: 'sp.Symbol', point: 'sp.Expr' = oo, terms: 'int' = 4, certify: 'bool' = True, _extracted_form: 'tuple[sp.Expr, sp.Expr, sp.Expr, bool] | None' = None) -> 'StatisticalAsymptoticResult'
```

Expand a one-dimensional Laplace integral at ``parameter -> +oo``.

Supported geometries include nondegenerate and even-order degenerate
interior minima plus monotone/stationary finite endpoints.  When ``certify``
is true, real polynomial phases are checked against a global Laplace
theorem class; otherwise the local expansion remains formal.

## `mrv_decomposition`

```python
mrv_decomposition(expr: 'sp.Expr', variable: 'sp.Symbol', point: 'sp.Expr' = oo, *, context: 'AsymptoticContext | None' = None, structural: 'StructuralDecomposition | None' = None) -> 'MRVDecomposition'
```

Compute explicit MRV comparability classes for an expression.

Candidates come from the dependency-ordered exp/log tower plus the
independent variable.  They are grouped by logarithmic growth and the
class with the largest variation measure is selected.  Unknown comparisons
remain separate rather than being silently equated.

## `multiseries`

```python
multiseries(expr: 'sp.Expr', variable: 'sp.Symbol', *, scale: 'AsymptoticScale | Iterable[sp.Expr] | None' = None, point: 'sp.Expr' = oo, terms: 'int' = 6, allow_series_fallback: 'bool' = True) -> 'Multiseries'
```

Create a lazy multiseries, discovering an asymptotic scale when omitted.

## `multivariate_dominant_balance_candidates`

```python
multivariate_dominant_balance_candidates(equation: 'sp.Expr', dependent: 'sp.Symbol', variables: 'tuple[sp.Symbol, ...] | list[sp.Symbol]', weights: 'tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None' = None, *, parameter: 'sp.Symbol | None' = None, centers: 'tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None' = None, amplitudes: 'tuple[sp.Expr, ...] | list[sp.Expr] | Mapping[sp.Symbol, sp.Expr] | None' = None, assumptions: 'sp.Expr | bool' = True, stratify_parameters: 'bool' = True, max_parameter_splits: 'int' = 6) -> 'tuple[MultivariateDominantBalanceCandidate, ...] | AsymptoticStratification[tuple[MultivariateDominantBalanceCandidate, ...]]'
```

Compute dominant balances on an explicit or automatically discovered path.

When ``weights`` is omitted, all admissible automatic Newton weight cones
are discovered and the balances from their rational representative paths
are returned.  Supplying ``weights`` preserves the historical single-path
behavior.

## `multivariate_implicit_asymptotics`

```python
multivariate_implicit_asymptotics(equations: 'tuple[sp.Expr, ...] | list[sp.Expr]', dependents: 'tuple[sp.Symbol, ...] | list[sp.Symbol]', variables: 'tuple[sp.Symbol, ...] | list[sp.Symbol]', *, terms: 'int' = 4, assumptions: 'sp.Expr | bool' = True, stratify_parameters: 'bool' = True, max_parameter_splits: 'int' = 6) -> 'tuple[MultivariateImplicitRegime, ...] | AsymptoticStratification[tuple[MultivariateImplicitRegime, ...]]'
```

Discover and jointly lift multivariate implicit-system asymptotics.

Unlike one-dimensional path reduction, dependent weights are solved
simultaneously from one active Newton face in every equation.  The leading
coefficient system and subsequent ramified corrections are then solved as
a coupled system.

## `nested_expansion`

```python
nested_expansion(expr: 'sp.Expr', variable: 'sp.Symbol', *, depth: 'int' = 4, point: 'sp.Expr' = oo, max_exp_depth: 'int | None' = None, max_log_depth: 'int | None' = None) -> 'NestedExpansion'
```

Build a resumable nested expansion, eagerly refining up to *depth* levels.

## `puiseux_series`

```python
puiseux_series(expr: 'sp.Expr', variable: 'sp.Symbol', *, point: 'sp.Expr' = 0, terms: 'int' = 6, branch: 'BranchChoice | None' = None) -> 'PuiseuxSeries'
```

Construct a rational-exponent local series with explicit ramification.

## `series_reversion`

```python
series_reversion(expr: 'sp.Expr', variable: 'sp.Symbol', inverse_variable: 'sp.Symbol | None' = None, *, point: 'sp.Expr' = 0, terms: 'int' = 6, branch: 'int | None' = None, context: 'AsymptoticContext | None' = None) -> 'tuple[ReversionBranch, ...] | ReversionBranch'
```

Revert a local (possibly Puiseux) series ``y=f(x)``.

The algorithm ramifies the source and target coordinates so a rational
leading exponent becomes integral, chooses every algebraic leading inverse
coefficient unless ``branch`` is supplied, and then lifts coefficients
recursively. Exact zero decisions are delegated to ``AsymptoticContext``.

## `transseries_from_expression`

```python
transseries_from_expression(expr: 'sp.Expr', variable: 'sp.Symbol', *, point: 'sp.Expr' = 0, complete: 'bool' = False, metadata: 'dict[str, object] | None' = None, remainder: 'AsymptoticRemainder | None' = None, sector: 'ComplexSector | None' = None, branch: 'ComplexBranchMetadata | None' = None) -> 'TransseriesExpansion'
```

Convert a finite additive exp/power/log expression to native terms.

Constants are stored in ``center``.  Every variable-dependent summand must
belong to the canonical multiplicative monomial group; unsupported terms
are rejected rather than silently wrapped as opaque expressions.

