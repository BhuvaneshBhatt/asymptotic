# Stage 3 — dependency separation

`exprtools.recurrence` imports only SymPy and exprtools modules. It returns
`TransformationEvidence` and accepts an optional `record_event(name)` callback.
`RecurrenceFamilies` binds custom function classes by identity, per invocation.
The default provider exposes native SymPy families; no host package is imported
or globally registered. Providers may subclass the interface for more families.

The asymptotic adapter supplies its exact existing function classes, translates
transformation evidence back to `LimitEvidence`, and forwards instrumentation.
Its public function signature and result dataclass fields remain unchanged.
The limit-only `uncertified_germ_reason` policy remains in asymptotic.

Validation: unchanged recurrence/integration selection 39 passed; five new
standalone/adapter checks passed. These include a fresh subprocess that rejects
all asymptotic imports, per-call custom-head isolation, principal-power and
budget guards, evidence type compatibility, and exactly one instrumented call.
