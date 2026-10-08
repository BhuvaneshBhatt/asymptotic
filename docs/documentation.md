# Documentation and example checks

Documentation should answer a mathematical question before describing an internal object. Introduce the approach domain and scale, show a runnable calculation, then explain what its status and remainder do and do not establish.

## What is verified

The README Python examples execute in order in a shared namespace. Introductory limit, branch and tail scripts run as subprocesses with explicit budgets. Markdown links to local files are checked, and README documentation links point to absolute repository URLs so they work on the package index.

## Writing examples

Give every script one question. State the symbol assumptions, check the exact value or mathematical status, and include a nearby unsupported or domain-sensitive case when it explains a proof boundary. Numerical samples can verify a branch convention but cannot certify a limit on all approaches.

Keep optional integrations in examples that name their dependencies. Avoid a tutorial that requires an expensive symbolic calculation before its first useful result.

## Improving coverage

The strongest next additions are executable tutorial sequences for multiscale expansions and remainder propagation, parameter-stratified implicit solutions, and recurrence/ODE workflows. Each needs an independently checked expected result and a limitation example. Larger numerical and symbolic examples belong in a separately budgeted integration suite.

The capability matrix should link a supported family to an example, its hypotheses and its limitations. A family-level support statement must not imply that every expression involving that function is decidable.

## Formula rendering

Use `$...$` for inline mathematics and `$$` on separate lines for display
mathematics. Keep display blocks outside lists and code fences when possible,
with a blank line before and after the block and no blank paragraph inside it.
Escape underscores in text commands, for example `\text{residual\_order}`.
Markdown editors need math rendering enabled; these delimiters work with GitHub
and common MathJax or KaTeX integrations.

`python tools/check_documentation_math.py` checks delimiters outside literal code
and can export every formula with `--output formulas.json` for renderer testing.
The documentation formulas are also checked with Markdown-it and KaTeX using
strict error reporting.
