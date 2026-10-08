# Imported multivariate limit corpus (development validation)

This tree contains a development-only converter and frozen imported differential corpora. The original external-system syntax, source names, and identifiers are retained only for traceability during validation. They are not runtime package data and must be removed or renamed to neutral provenance before release.

The imported source corpus contains 169 and 999 limit cases in its two source groups. The converter preserves every parsed case.

Translation is triaged as `supported`, `partially_supported`, or `unsupported`. Unsupported/partial cases are never counted as implementation failures. The runner separately records agreement/disagreement, UNKNOWN, exception, and timeout.

The initial broad stress runs use short per-case budgets because this environment cannot spend several seconds on ~1,200 imported cases in one command. Those timeout counts are therefore triage data, not release-performance claims. Frozen case IDs/results should be rerun with larger budgets by family before solver changes.
