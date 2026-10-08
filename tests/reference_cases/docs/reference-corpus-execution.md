# Exhaustive reference-corpus execution

The 1,168-case corpus is a release/soundness gate. `tools/reference_corpus.py` is the durable coordinator.

Each mathematical case runs in a **fresh Python process**. The solver has its own alarm, and the coordinator also imposes a hard wall-clock timeout. The coordinator itself never imports `asymptotic`, so a corrupted SymPy cache, recursion state, signal handler, or worker crash cannot contaminate subsequent cases.

Results use append-only JSONL and `fsync` after every case. A killed shard therefore resumes without discarding completed work. Sixty shards are deterministic (`index % 60`) and may run independently on separate machines.

```text
python tools/reference_corpus.py run --shard 0 --shards 60 --timeout 2 \
  --output reference-results/shard_00.jsonl
...
python tools/reference_corpus.py aggregate --directory reference-results \
  --output reference-results/aggregate.json --strict
```

Use `--pythonpath` repeatedly for sibling working trees such as `semialg/src` and `funcprops/src` when they are not installed. Aggregation is complete only when `total_completed == total_expected == 1168`, `missing_count == 0`, and no conflicting duplicate indices are present. Strict mode additionally exits nonzero for `WRONG`, `ERROR`, or `TIMEOUT` outcomes.

Triage order is **WRONG → ERROR → persistent TIMEOUT → UNKNOWN**. A TIMEOUT is rerun diagnostically with a larger budget only after every case has an initial result. A worker infrastructure failure is an ERROR and is not reclassified as mathematical UNKNOWN.
