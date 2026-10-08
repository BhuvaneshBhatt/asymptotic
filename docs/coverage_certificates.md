# Coverage and completeness certificates

`CoverageCertificate` is the common proof object for statements that a finite
family of charts, parameter cells, domain components, branch sectors or
valuation regimes exhausts all admissible approaches.

A certificate is COMPLETE, PARTIAL or UNKNOWN. Completeness additionally
requires every explicit `CoverageObligation` to be discharged and no missing
regimes. Certificates compose without upgrading partial coverage.

Limit and cluster
algorithms should certify a result only from complete coverage; discovered
paths or analyzed subsets remain partial evidence.
