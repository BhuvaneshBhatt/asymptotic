"""Resolve the two sides of a logarithmic cut and a fixed imaginary-ray pole."""

import sympy as sp

from asymptotic import DirectionalInfinity, complex_ray_limit


def main():
    z = sp.Symbol("z")
    assert complex_ray_limit(sp.log(z), z, -1, ray=sp.I) == sp.I * sp.pi
    assert complex_ray_limit(sp.log(z), z, -1, ray=-sp.I) == -sp.I * sp.pi
    assert complex_ray_limit(1 / z, z, 0, ray=sp.I) == DirectionalInfinity(-sp.I)
    # Off-cut samples independently check which principal logarithm is selected.
    for side in (1, -1):
        sample = sp.log(-1 + side * sp.I / sp.Integer(10) ** 8)
        assert abs(complex((sample - side * sp.I * sp.pi).evalf(30))) < 1e-7
    print("Upper and lower logarithmic cuts and an imaginary pole direction verified.")


if __name__ == "__main__":
    main()
