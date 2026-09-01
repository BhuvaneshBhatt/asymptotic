# Choosing an API by problem

Start from the mathematical question rather than the internal representation.

```text
I have f(x) and want an expansion
├─ one dominant small scale or a few ordered scales
│  └─ multiseries()
├─ nested exp/log structure is itself important
│  └─ nested_expansion() or nested_form()
└─ I already have a finite generalized transseries expression
   └─ transseries_from_expression()

I need to solve for an unknown
├─ algebraic P(x,y)=0
│  └─ algebraic_branches() / puiseux_series()
├─ implicit F(x,y)=0
│  └─ implicit_asymptotic()
├─ invert y=f(x)
│  ├─ local power/Puiseux problem -> series_reversion()
│  └─ general asymptotic inverse -> inverse_asymptotic()
└─ nonlinear differential equation
   └─ nonlinear_differential_transseries()

I have a probability or expectation depending on a large parameter
├─ expectation of one random variable -> asymptotic_expectation()
├─ probability of a one-variable event -> asymptotic_probability()
└─ already have A(x,p) exp(-p phi(x)) -> laplace_asymptotic_integral()

I have several asymptotic variables
├─ inspect scaling chambers -> multivariate_scaling_regimes()
├─ test one weighted path -> multivariate_dominant_balance_candidates()
└─ solve a coupled implicit system -> multivariate_implicit_asymptotics()

I need rigor rather than only a formal prefix
├─ inspect result.remainder / certificate.certified
├─ use operation-specific certify_*_remainder() theorem
└─ if UNKNOWN, inspect certificate.hypotheses before adding assumptions
```

## Common representation algebra

Use `AsymptoticAlgebra` when several native representations must interact. Native unary algorithms are retained; heterogeneous arithmetic goes through one finite certified transseries normal form. This is preferable to manually converting each representation at every call site.

## When to use the structural views

`decompose_expression()`, `mrv_decomposition()`, `discover_scale()`, `nested_form()`, and `implicit_singularity_profile()` are diagnostic APIs. They are useful when you need to understand *why* a solver selected a scale, branch, or blow-up rather than merely obtaining the final prefix.

## See also

- [User guide](user-guide.md)
- [Probability and expectation asymptotics](probability-asymptotics.md)
- [Capabilities and limitations](capabilities.md)
- [Why results are UNKNOWN](unknown-results.md)
- [Worked algorithm traces](algorithm-traces.md)
- [Generated API reference](api-reference.md)
