"""Run the multivariate O/o/~ certification corpus and report actual providers."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from asymptotic.multivariate_expansion import (
    multivariate_big_o,
    multivariate_equivalent,
    multivariate_little_o,
)


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    relation: str
    lhs: sp.Expr
    rhs: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    domain: sp.Expr = sp.S.true
    order: int = 2


def cases():
    x, y, z, w = sp.symbols("x y z w", real=True)
    xy, xyz, xyzw = (x, y), (x, y, z), (x, y, z, w)
    o2, o3, o4 = (0, 0), (0, 0, 0), (0, 0, 0, 0)
    r2 = x**2 + y**2
    r3 = x**2 + y**2 + z**2
    r4 = x**2 + y**2 + z**2 + w**2
    C = Case
    return [
        # Baseline two-variable cases retained verbatim in spirit.
        C("rational_little_o", "rational", "o", x**2 * y**2, r2, xy, o2),
        C("rational_equivalent", "rational", "~", r2 + x**4, r2, xy, o2),
        C("rational_big_o_angular", "rational", "O", x**2 * y**2, r2, xy, o2),
        C(
            "restricted_halfspace_little_o",
            "restricted-domain",
            "o",
            x**2 * y**2,
            r2,
            xy,
            o2,
            sp.Ge(x, 0),
        ),
        C(
            "restricted_wedge_equivalent",
            "restricted-domain",
            "~",
            r2 + x**4,
            r2,
            xy,
            o2,
            sp.And(sp.Ge(x, 0), sp.Ge(y, 0)),
        ),
        C(
            "multi_newton_identity",
            "multi-Newton-chart",
            "O",
            x**2 + y**4,
            x**2 + y**4,
            xy,
            o2,
        ),
        C(
            "multi_newton_decay",
            "multi-Newton-chart",
            "o",
            x**4 + y**8,
            x**2 + y**4,
            xy,
            o2,
        ),
        C(
            "difficult_angular_quadratic",
            "difficult-angular-extremum",
            "O",
            x**2 + 3 * x * y + 2 * y**2,
            r2,
            xy,
            o2,
        ),
        C(
            "difficult_angular_quartic",
            "difficult-angular-extremum",
            "O",
            x**4 + 5 * x**2 * y**2 + y**4,
            r2**2,
            xy,
            o2,
        ),
        C("algebraic_sqrt_equivalent", "algebraic", "~", sp.sqrt(1 + r2), 1, xy, o2),
        C(
            "algebraic_radical_little_o",
            "algebraic",
            "o",
            sp.sqrt(1 + r2) - 1,
            sp.sqrt(r2),
            xy,
            o2,
        ),
        C(
            "transcendental_exp_equivalent",
            "transcendental-composition",
            "~",
            sp.exp(r2),
            1,
            xy,
            o2,
        ),
        C(
            "transcendental_sinc_equivalent",
            "transcendental-composition",
            "~",
            sp.sin(r2),
            r2,
            xy,
            o2,
        ),
        # 3+ variable Newton fans / anisotropic blow-ups.
        C(
            "newton3_identity",
            "3plus-Newton-fan",
            "O",
            x**2 + y**4 + z**6,
            x**2 + y**4 + z**6,
            xyz,
            o3,
        ),
        C(
            "newton3_decay",
            "3plus-Newton-fan",
            "o",
            x**4 + y**8 + z**12,
            x**2 + y**4 + z**6,
            xyz,
            o3,
        ),
        C(
            "newton3_equivalent",
            "3plus-Newton-fan",
            "~",
            x**2 + y**4 + z**6 + x**4 * y**4,
            x**2 + y**4 + z**6,
            xyz,
            o3,
        ),
        C(
            "newton3_mixed_O",
            "3plus-Newton-fan",
            "O",
            x * y**2 * z**3,
            x**2 + y**4 + z**6,
            xyz,
            o3,
        ),
        C(
            "newton3_competing_faces",
            "3plus-Newton-fan",
            "O",
            x**4 + y**6 + z**12 + x**2 * y**2 * z**2,
            x**4 + y**6 + z**12,
            xyz,
            o3,
        ),
        C(
            "newton4_decay",
            "3plus-Newton-fan",
            "o",
            x**4 + y**8 + z**12 + w**16,
            x**2 + y**4 + z**6 + w**8,
            xyzw,
            o4,
        ),
        C(
            "newton4_equivalent",
            "3plus-Newton-fan",
            "~",
            r4 + x**4 + y**4 + z**4 + w**4,
            r4,
            xyzw,
            o4,
        ),
        # Nonradial algebraic compositions.
        C(
            "alg_nonradial_sqrt_equiv",
            "nonradial-algebraic",
            "~",
            sp.sqrt(1 + x**2 + y**4),
            1,
            xy,
            o2,
        ),
        C(
            "alg_nonradial_sqrt_decay",
            "nonradial-algebraic",
            "o",
            sp.sqrt(1 + x**2 + y**4) - 1,
            sp.sqrt(x**2 + y**4),
            xy,
            o2,
        ),
        C(
            "alg_nonradial_nested_equiv",
            "nonradial-algebraic",
            "~",
            sp.sqrt(1 + x**2 + sp.sqrt(1 + y**4) - 1),
            1,
            xy,
            o2,
        ),
        C(
            "alg_nonradial_ratio_O",
            "nonradial-algebraic",
            "O",
            sp.sqrt(x**2 + y**4),
            sp.Abs(x) + y**2,
            xy,
            o2,
        ),
        C(
            "alg3_sqrt_equiv",
            "nonradial-algebraic",
            "~",
            sp.sqrt(1 + x**2 + y**4 + z**6),
            1,
            xyz,
            o3,
        ),
        C(
            "alg3_decay",
            "nonradial-algebraic",
            "o",
            sp.sqrt(1 + x**2 + y**4 + z**6) - 1,
            sp.sqrt(x**2 + y**4 + z**6),
            xyz,
            o3,
        ),
        # Genuinely angular transcendental coefficients (not purely radial compositions).
        C(
            "trans_angular_exp_equiv",
            "angular-transcendental",
            "~",
            sp.exp(x * y),
            1,
            xy,
            o2,
        ),
        C(
            "trans_angular_sin_o",
            "angular-transcendental",
            "o",
            sp.sin(x * y),
            sp.sqrt(r2),
            xy,
            o2,
        ),
        C(
            "trans_angular_cos_equiv",
            "angular-transcendental",
            "~",
            sp.cos(x * y),
            1,
            xy,
            o2,
        ),
        C(
            "trans_angular_exp3_equiv",
            "angular-transcendental",
            "~",
            sp.exp(x * y + y * z + z * x),
            1,
            xyz,
            o3,
        ),
        C(
            "trans_angular_sin3_o",
            "angular-transcendental",
            "o",
            sp.sin(x * y * z),
            r3,
            xyz,
            o3,
        ),
        C(
            "trans_anisotropic_sinc",
            "angular-transcendental",
            "~",
            sp.sin(x**2 + y**4),
            x**2 + y**4,
            xy,
            o2,
        ),
        # Mixed semialgebraic approach domains.
        C(
            "domain_parabolic_wedge_o",
            "mixed-restricted-domain",
            "o",
            x**4 + y**4,
            r2,
            xy,
            o2,
            sp.And(sp.Ge(y, x**2), sp.Ge(x, 0)),
        ),
        C(
            "domain_cusp_equiv",
            "mixed-restricted-domain",
            "~",
            y + x**4,
            y,
            xy,
            o2,
            sp.And(sp.Ge(y, x**2), sp.Ge(x, 0)),
        ),
        C(
            "domain_annular_cusp_O",
            "mixed-restricted-domain",
            "O",
            x**2 * y,
            r2,
            xy,
            o2,
            sp.And(sp.Ge(y, 0), sp.Le(y, x**2), sp.Ge(x, 0)),
        ),
        C(
            "domain3_orthant_o",
            "mixed-restricted-domain",
            "o",
            x**2 * y**2 * z**2,
            r3,
            xyz,
            o3,
            sp.And(sp.Ge(x, 0), sp.Ge(y, 0), sp.Ge(z, 0)),
        ),
        C(
            "domain3_ordered_equiv",
            "mixed-restricted-domain",
            "~",
            r3 + x**4,
            r3,
            xyz,
            o3,
            sp.And(sp.Ge(x, y), sp.Ge(y, z), sp.Ge(z, 0)),
        ),
        C(
            "domain3_cone_O",
            "mixed-restricted-domain",
            "O",
            x * y * z,
            r3 ** sp.Rational(3, 2),
            xyz,
            o3,
            sp.And(sp.Ge(z, 0), sp.Ge(x**2 + y**2, z**2)),
        ),
        # Ratios whose denominator has zeros/poles on sector boundaries.
        C(
            "boundary_zero_cancelled_equiv",
            "sector-boundary-zero-pole",
            "~",
            (x - y) * (1 + r2),
            x - y,
            xy,
            o2,
            sp.Ne(x, y),
        ),
        C(
            "boundary_zero_cancelled_O",
            "sector-boundary-zero-pole",
            "O",
            (x - y) ** 2,
            x - y,
            xy,
            o2,
            sp.Ne(x, y),
        ),
        C(
            "boundary_axis_cancelled_equiv",
            "sector-boundary-zero-pole",
            "~",
            x * (1 + y**2),
            x,
            xy,
            o2,
            sp.Ne(x, 0),
        ),
        C(
            "boundary_axis_cancelled_o",
            "sector-boundary-zero-pole",
            "o",
            x * y,
            x,
            xy,
            o2,
            sp.Ne(x, 0),
        ),
        C(
            "boundary_cone_cancelled_equiv",
            "sector-boundary-zero-pole",
            "~",
            (x**2 - y**2) * (1 + r2),
            x**2 - y**2,
            xy,
            o2,
            sp.Ne(x**2, y**2),
        ),
        C(
            "boundary3_plane_cancelled_O",
            "sector-boundary-zero-pole",
            "O",
            (x - y) * z,
            x - y,
            xyz,
            o3,
            sp.Ne(x, y),
        ),
    ]


def run_case(case):
    fn = {
        "O": multivariate_big_o,
        "o": multivariate_little_o,
        "~": multivariate_equivalent,
    }[case.relation]
    result = fn(
        case.lhs,
        case.rhs,
        case.variables,
        case.target,
        domain=case.domain,
        order=case.order,
    )
    atlas = result.atlas
    remainder = sorted(
        {
            c.expansion.remainder_certificate.provider
            for c in atlas.charts
            if c.expansion.remainder_certificate
        }
        if atlas
        else set()
    )
    return {
        "name": case.name,
        "family": case.family,
        "relation": case.relation,
        "certified": result.certified,
        "relation_provider": result.provider,
        "certificate_backends": list(result.certificate_backends),
        "coverage_provider": atlas.coverage_certificate.provider if atlas else "none",
        "remainder_providers": remainder,
        "charts": len(atlas.charts) if atlas else 0,
        "statement": result.statement,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    ap.add_argument("--timeout", type=float, default=8.0)
    ns = ap.parse_args()
    corpus = cases()
    rows = []
    for case in corpus:
        q = mp.Queue()

        def worker(case=case, q=q):
            try:
                q.put(run_case(case))
            except Exception as exc:  # noqa: BLE001 - corpus records backend failures
                q.put(
                    {
                        "name": case.name,
                        "family": case.family,
                        "relation": case.relation,
                        "certified": False,
                        "relation_provider": "none",
                        "certificate_backends": [],
                        "coverage_provider": "none",
                        "remainder_providers": [],
                        "charts": 0,
                        "statement": f"{type(exc).__name__}: {exc}",
                    }
                )

        proc = mp.Process(target=worker)
        proc.start()
        proc.join(ns.timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            rows.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "relation": case.relation,
                    "certified": False,
                    "relation_provider": "none",
                    "certificate_backends": [],
                    "coverage_provider": "none",
                    "remainder_providers": [],
                    "charts": 0,
                    "statement": f"timed out after {ns.timeout:g}s",
                }
            )
        else:
            rows.append(
                q.get()
                if not q.empty()
                else {
                    "name": case.name,
                    "family": case.family,
                    "relation": case.relation,
                    "certified": False,
                    "relation_provider": "none",
                    "certificate_backends": [],
                    "coverage_provider": "none",
                    "remainder_providers": [],
                    "charts": 0,
                    "statement": "worker exited without result",
                }
            )
    availability = {}
    for name in ("semialg", "symbopt"):
        try:
            __import__(name)
            availability[name] = True
        except ImportError:
            availability[name] = False
    payload = {"backend_availability": availability, "rows": rows}
    if ns.json:
        Path(ns.json).write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# Relation-certification corpus",
        "",
        f"Backend availability: `semialg={availability['semialg']}`, `symbopt={availability['symbopt']}`.",
        "",
        "| Case | Family | Relation | Certified | Actual certificate backends | Charts |",
        "|---|---|:---:|:---:|---|---:|",
    ]
    for r in rows:
        providers = ", ".join(r["certificate_backends"]) or "none"
        md.append(
            f"| `{r['name']}` | {r['family']} | {r['relation']} | {'yes' if r['certified'] else 'no'} | {providers} | {r['charts']} |"
        )
    certified = [r for r in rows if r["certified"]]
    sym = [r for r in certified if "symbopt" in r["certificate_backends"]]
    md += [
        "",
        f"Certified: **{len(certified)}/{len(rows)}**. Certified cases requiring symbopt: **{len(sym)}/{len(certified)}**.",
        "",
        "An UNKNOWN result is not evidence that symbopt would certify the case. If symbopt is unavailable, this run can establish semialg/exact coverage and expose optimization gaps, but cannot measure symbopt rescue rate.",
    ]
    text = "\n".join(md) + "\n"
    if ns.markdown:
        Path(ns.markdown).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
