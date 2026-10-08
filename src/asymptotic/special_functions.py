"""Special functions absent from SymPy's core function namespace."""

from __future__ import annotations

import sympy as sp
from sympy.core.function import ArgumentIndexError


class StruveH(sp.Function):
    """Symbolic classical Struve function H_nu(z) for asymptotic analysis."""

    nargs = 2


class StruveL(sp.Function):
    """Symbolic modified Struve function L_nu(z) for asymptotic analysis."""

    nargs = 2


class ParabolicCylinderD(sp.Function):
    """Standard entire D_nu(z)=U(-nu-1/2,z), DLMF 12.2.5.

    This head supports exact order recurrences; general germs remain separate.
    """

    nargs = 2


class HypergeometricU(sp.Function):
    """Principal Kummer U(a,b,z), DLMF 13.2; cut on negative real z."""

    nargs = 3


class WhittakerM(sp.Function):
    """M(k,m,z)=exp(-z/2)*z**(m+1/2)*1F1(m-k+1/2;2m+1;z)."""

    nargs = 3


class WhittakerW(sp.Function):
    """W(k,m,z)=exp(-z/2)*z**(m+1/2)*U(m-k+1/2,2m+1,z)."""

    nargs = 3


class RegularizedHypergeometric1F1(sp.Function):
    """Entire-parameter 1F1(a;b;z)/Gamma(b), continued at gamma poles."""

    nargs = 3


class RegularizedHypergeometric2F1(sp.Function):
    """Principal 2F1(a,b;c;z)/Gamma(c), continued at gamma poles."""

    nargs = 4


class KelvinBer(sp.Function):
    """Real component of J_nu(x*exp(3*pi*I/4)), real nu, x>0."""

    nargs = 2


class KelvinBei(sp.Function):
    """Imaginary component of J_nu(x*exp(3*pi*I/4)), real nu, x>0."""

    nargs = 2


class KelvinKer(sp.Function):
    """Real component of exp(-nu*pi*I/2)*K_nu(x*exp(pi*I/4))."""

    nargs = 2


class KelvinKei(sp.Function):
    """Imaginary component of exp(-nu*pi*I/2)*K_nu(x*exp(pi*I/4))."""

    nargs = 2


class LegendreQ(sp.Function):
    """Ferrers Q_nu^m(x) for -1<x<1, integer m>=0 (DLMF 14.3).

    Explicitly three arguments; complex/type-3 source conventions stay opaque.
    """

    nargs = 3


class AppellF1(sp.Function):
    """Appell F1(a,b1,b2;c;u,v), represented by SymPy's native appellf1."""

    nargs = 6

    @classmethod
    def eval(cls, a, b1, b2, c, u, v):
        return sp.appellf1(a, b1, b2, c, u, v)


class FresnelF(sp.Function):
    """Fresnel auxiliary f, defined by DLMF 7.5.3 and 7.5.4."""

    nargs = 1

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        phase = sp.pi * z**2 / 2
        return (sp.S.Half - sp.fresnels(z)) * sp.cos(phase) + (
            sp.fresnelc(z) - sp.S.Half
        ) * sp.sin(phase)


class FresnelG(sp.Function):
    """Fresnel auxiliary g, defined by DLMF 7.5.3 and 7.5.4."""

    nargs = 1

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        phase = sp.pi * z**2 / 2
        return (sp.S.Half - sp.fresnelc(z)) * sp.cos(phase) + (
            sp.S.Half - sp.fresnels(z)
        ) * sp.sin(phase)


class EllipticNome(sp.Function):
    """Principal parameter nome exp(-pi*K(1-m)/K(m))."""

    nargs = 1

    def _eval_rewrite_as_exp(self, m, **kwargs):
        return sp.exp(-sp.pi * sp.elliptic_k(1 - m) / sp.elliptic_k(m))


class InverseEllipticNome(sp.Function):
    """Local inverse parameter m(q)=16*q-128*q**2+O(q**3) at q=0."""

    nargs = 1


class OwenT(sp.Function):
    """Principal Owen T(h,a), continued from a=0 off cuts from +/-I."""

    nargs = 2

    def _eval_rewrite_as_Integral(self, h, a, **kwargs):
        t = sp.Dummy("integration_parameter", real=True)
        return sp.Integral(
            sp.exp(-h * h * (1 + t * t) / 2) / (1 + t * t), (t, 0, a)
        ) / (2 * sp.pi)


class ScorerGi(sp.Function):
    """Entire Scorer Gi, with Gi''(z)-z*Gi(z)=-1/pi (DLMF 9.12)."""

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        return ScorerGiPrime(self.args[0])


class ScorerGiPrime(sp.Function):
    """Derivative of the entire Scorer Gi function."""

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        z = self.args[0]
        return z * ScorerGi(z) - 1 / sp.pi


class ScorerHi(sp.Function):
    """Entire Scorer Hi, with Hi''(z)-z*Hi(z)=1/pi (DLMF 9.12)."""

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        return ScorerHiPrime(self.args[0])


class ScorerHiPrime(sp.Function):
    """Derivative of the entire Scorer Hi function."""

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        z = self.args[0]
        return z * ScorerHi(z) + 1 / sp.pi


class JacobiAmplitude(sp.Function):
    """Principal amplitude germ at zero, inverse of F(phi|m)."""

    nargs = 2
