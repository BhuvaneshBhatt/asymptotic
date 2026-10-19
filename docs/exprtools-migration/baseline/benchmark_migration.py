"""Reproducible baseline/post-migration workloads; run without coverage."""
import importlib.util
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path
import sympy as sp

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('benchmark_zerotest', root / 'exprtest/benchmarks/benchmark_zerotest.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
from asymptotic.recurrence_simplification import simplify_recurrences
from exprtest._cache import cache_clear


def measure(fn, data):
    times, outputs = [], []
    for term in data:
        start = time.perf_counter()
        result = fn(term)
        times.append(time.perf_counter() - start)
        outputs.append(result)
    ordered = sorted(times)
    return dict(n=len(times), total_seconds=sum(times), median_seconds=statistics.median(times),
                p95_seconds=ordered[int(.95*(len(times)-1))]), outputs

x = sp.Symbol('x', positive=True)
v = sp.Symbol('v')
a = sp.Symbol('a')
recurrences = [
    sp.gamma(x+3)/sp.gamma(x),
    sp.polygamma(1,x+1)-sp.polygamma(1,x),
    sp.besselj(v,x)+sp.besselj(v+2,x)-2*(v+1)*sp.besselj(v+1,x)/x,
    sp.uppergamma(a+1,x)-a*sp.uppergamma(a,x),
    sp.zeta(a,x+1)-sp.zeta(a,x),
    sp.gamma(x+1000)/sp.gamma(x),
]
# Explicitly warm interpreter/SymPy caches for comparable measurements.
for e in recurrences:
    simplify_recurrences(e, budget_seconds=1)
data = bench.cases()
cache_clear()
cold, cv = measure(lambda e: bench.zerotest(e, use_cache=False), data)
cache_clear()
warm, wv = measure(lambda e: bench.zerotest(e, use_cache=True), data[:125]*40)
rr, rv = measure(lambda e: simplify_recurrences(e, budget_seconds=1), recurrences*100)
for result, values in ((cold,cv), (warm,wv)):
    result['verdicts']={str(k):values.count(k) for k in (True,False,None)}
rr['outputs']=[sp.srepr(e) for e in rv[:6]]
rr['consistent']=all(rv[i]==rv[i%6] for i in range(len(rv)))
report=dict(python=sys.version, platform=platform.platform(), packages={k:importlib.metadata.version(k) for k in ('sympy','mpmath','python-flint','pytest','coverage','funcprops','semialg')}, zero_cold=cold,zero_repeated=warm,recurrence=rr)
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
