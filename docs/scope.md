# Mathematical scope and non-goals

`asymptotic` is principally a directed-real symbolic asymptotics engine. Its preferred behavior is conservative: produce a formal finite object when possible, certify it when hypotheses are proved, and otherwise preserve `UNKNOWN`.

## In scope

- finite power/log/exp and nested asymptotic scales;
- lazy multiseries and finite generalized transseries prefixes;
- Puiseux branches, local reversion, and singular implicit blow-ups;
- multivariate weighted scaling regimes and automatic Newton weight-cone discovery in public balance APIs;
- nonlinear differential dominant balance and recursive corrections;
- operation-specific `O/o` remainder theorems;
- constant and asymptotically constant scalar Green/Frechet estimates;
- finite asymptotic differential-field/shadow constructions;
- reviewed real-domain, singularity, and branch facts for a finite function registry;
- explicit complex-sector/branch metadata and optional ODE Stokes-sector interchange (metadata, not general sectorial certification).

## Outside the supported scope

- general sectorial complex asymptotics;
- general Stokes connection/continuation theory, resurgence, alien calculus, or Borel summation (optional `odeanalysis` can supply formal Stokes ray/sector geometry);
- a complete Hahn/log-exp transseries field with arbitrary well-based supports;
- a complete Hardy-field zero/sign/comparability decision procedure;
- general resolution of higher-dimensional singular implicit systems;
- arbitrary variable-coefficient Green operators without a hyperbolic asymptotic limit;
- complete branch-sheet/monodromy tracking for all special functions;
- guaranteed termination of explicitly user-requested general SymPy antiderivatives.

## Formal, certified, unknown

A **formal** result records a finite asymptotic construction. A **certified** result additionally carries a proved remainder theorem or replayable mathematical certificate. **Unknown** means a necessary hypothesis could not be proved; it must not be interpreted as false.
