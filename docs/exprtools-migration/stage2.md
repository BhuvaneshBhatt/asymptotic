# Stage 2 — extraction

The new independently built distribution lives in `exprtools/`; the root
`exprtest` distribution is a compatibility package depending on it. All original
predicate modules are moved intact, retaining relative imports. Compatibility
submodules alias the new modules, preserving class identity, caches, monkeypatch
behavior, and old pickle import paths. The old public export list is unchanged.

The recurrence engine and family definitions are copied with their existing
mathematics. At this intermediate stage their legacy dependencies are explicit;
stage 3 removes them. `asymptotic.context` uses the new predicate package.

Validation after extraction: unchanged exprtest tests 321 passed, 10 skipped;
unchanged recurrence/integration selection 39 passed. Three additional
compatibility tests passed. These runs did not enable coverage and should not
be timed against the instrumented baseline test duration.
