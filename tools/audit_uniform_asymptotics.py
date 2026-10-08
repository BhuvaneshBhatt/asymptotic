"""Source-coherence audit for uniform and beyond-all-orders conventions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from asymptotic.olver_coefficients import (
    airy_u,
    airy_v,
    debye_u,
    debye_v,
    olver_coefficient,
    olver_coefficient_block,
)


@dataclass(frozen=True)
class AuditFinding:
    check: str
    passed: bool
    detail: str


def mathematical_convention_audit() -> tuple[AuditFinding, ...]:
    p = sp.Symbol("p")
    findings = []
    findings.append(
        AuditFinding(
            "debye-u1",
            sp.expand(debye_u(1) - (3 * p - 5 * p**3) / 24) == 0,
            "U1=(3p-5p^3)/24",
        )
    )
    findings.append(
        AuditFinding(
            "debye-v1",
            sp.expand(debye_v(1) - (-9 * p + 7 * p**3) / 24) == 0,
            "V1=(-9p+7p^3)/24",
        )
    )
    findings.append(
        AuditFinding(
            "airy-derivative-indexing",
            sp.simplify(airy_v(1) + sp.Rational(7, 5) * airy_u(1)) == 0,
            "v1=-(7/5)u1",
        )
    )
    findings.append(
        AuditFinding(
            "turning-A0-D0",
            olver_coefficient("A", 0, 1) == 1 and olver_coefficient("D", 0, 1) == 1,
            "A0(0)=D0(0)=1",
        )
    )
    findings.append(
        AuditFinding(
            "turning-B0",
            sp.simplify(olver_coefficient("B", 0, 1) - sp.real_root(2, 3) / 70) == 0,
            "B0(0)=2^(1/3)/70",
        )
    )
    block = olver_coefficient_block(3, sp.S.One)
    findings.append(
        AuditFinding(
            "matched-olver-block",
            set(block) == {"A", "B", "C", "D"}
            and all(len(values) == 3 for values in block.values()),
            "A/B/C/D share one indexed coefficient engine",
        )
    )
    return tuple(findings)


def source_convention_audit(root: Path) -> tuple[AuditFinding, ...]:
    source = root / "src/asymptotic"
    docs = root / "docs"
    findings = list(mathematical_convention_audit())
    uniform = (source / "uniform_special_expansions.py").read_text()
    hyper = (source / "hyperasymptotics.py").read_text()
    olver = (source / "olver_coefficients.py").read_text()
    findings.extend(
        (
            AuditFinding(
                "negative-axis-cut",
                "negative real axis" in uniform
                or "negative real axis" in (source / "complex_domains.py").read_text(),
                "principal Bessel cut documented in source",
            ),
            AuditFinding(
                "terminant-normalization",
                "sp.gamma(p) * sp.uppergamma(1 - p, z) / (2 * sp.pi)" in hyper,
                "G_p uses Gamma(p)Gamma(1-p,z)/(2pi) before rescaling",
            ),
            AuditFinding(
                "olver-indexing",
                'family not in {"A", "B", "C", "D"}' in olver,
                "single A/B/C/D implementation",
            ),
            AuditFinding(
                "docs-regime-selection",
                (docs / "asymptotic-regime-selection.md").exists(),
                "dispatcher conventions documented",
            ),
        )
    )
    for filename in (
        "regime_selection.py",
        "uniform_integration.py",
        "uniform_special_expansions.py",
        "hyperasymptotics.py",
        "olver_coefficients.py",
        "saddle_geometry.py",
        "stokes_geometry.py",
    ):
        tree = ast.parse((source / filename).read_text())
        broad = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ExceptHandler)
            and (
                n.type is None
                or (isinstance(n.type, ast.Name) and n.type.id == "Exception")
            )
        ]
        findings.append(
            AuditFinding(
                f"no-broad-except:{filename}",
                not broad,
                "no broad production exception handling",
            )
        )
    return tuple(findings)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = source_convention_audit(root)
    for item in findings:
        print(("PASS" if item.passed else "FAIL"), item.check, "-", item.detail)
    return 0 if all(item.passed for item in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
