# Limit contracts

Use `complex_ray_limit` for a fixed outward ray, `complex_limit` for every complex approach to a finite point, and `cluster_set` when an oscillatory expression has an exact set of finite accumulation values. These are different mathematical questions.

```python
import sympy as s
from asymptotic import cluster_set, complex_limit, complex_ray_limit

z = s.Symbol("z")
complex_limit(1 / z, z, 0)  # zoo: spherical infinity
complex_ray_limit(1 / z, z, 0, ray=s.I)  # DirectionalInfinity(-I)
cluster_set(z**s.I, z, 0, direction="+")  # Circle(0, 1)
```

Whole-plane limits use independent real and imaginary displacements. Quotients of regular analytic germs also have a bounded Taylor-valuation certificate, which distinguishes a removable value from a spherical pole. The implementation does not infer a whole-plane limit from a collection of agreeing rays. At spherical complex infinity, bounded polynomial powers have a uniform modulus-growth certificate. General angular behavior and unrestricted branch-cut germs need additional certificates.

An exact cluster certificate proves containment and attainment. Structured results retain symbolic sequences in the original variable and the necessary late-tail domain or pole-avoidance argument. Cluster sets contain finite values: `[0, infinity)` includes arbitrarily large finite values, without treating infinity as an element. Unsupported families return an unresolved result, without replacing an exact set by a range bound. Integer sampling is a different contract and does not use continuous phase certificates.

## Reference conditions and normalization

Conditional reference values are compared under their declared parameter conditions, conjoined with recorded assumptions. Audit records retain `comparison_condition`; a pass proves that cell's claim and says nothing about its complement. Zero coefficients, poles and missing conditions stay visible. Existing native parameter-stratified results remain available through the ordinary limit API.

Grouped Piecewise branches are flattened with their order preserved. An explicit default becomes a final true branch; a missing default retains the undefined branch. `CubeRoot` is the real third-root definition, and real-root normalization preserves even-order domains. Unknown user functions and malformed integer calls do not receive guessed definitions. Semantic certificates require the registered function type: a user function with the same printed name remains unspecified. Step factorials use their own principal gamma continuation; the branch-dependent SymPy falling factorial does not acquire its Taylor certificate at a zero base.

Source direction metadata uses an approach velocity. The outward ray is its negative; each adapted row retains its original direction and the orientation conversion. Directional infinity compares normalized nonzero directions and stays distinct from spherical infinity.

## Independent exact-cluster reference corrections

For case 0724, the sine and cosine integral remainder bounds give
`x*(-Ci(x)+Si(x)-pi/2) = -sin(x)-cos(x)+O(1/x)`.
Every phase is attained along `x=2*pi*n+theta`; the exact interval is
`[-sqrt(2),sqrt(2)]`, not `[-2,2]`.
See the [integral asymptotics and remainder formulas](https://dlmf.nist.gov/6.12).

For 0741, the Fresnel remainder bounds give
`x*(-1+S(x)/C(x)) = -2*(sin(theta)+cos(theta))/pi+O(1/x)`,
with `theta=pi*x**2/2`. The denominator tends to one half and stays away from zero. Square-root phase sequences attain the exact interval
`[-2*sqrt(2)/pi,2*sqrt(2)/pi]`, not `[-4/pi,4/pi]`.
See the [Fresnel asymptotics and remainder formulas](https://dlmf.nist.gov/7.12).

For 3423 put `D=x+sin(x)*cos(x)`. On the positive tail `D>=x-1/2` and the expression is positive. For any fixed `y>0`, choose
`L=2*pi*n` and `x=L-(log(L)+log(y))/L`.
Then `D/L->1` and `-D*sin(x)=log(L)+log(y)+o(1)`, so `exp(-D*sin(x))/D->y`.
For zero, `x=L` gives `1/L->0`. Consequently its exact finite cluster set is `[0,infinity)`, not `[exp(-1),E]`.

The old expectations remain in each row's `original_contract`. These are corrections to an exact-cluster interpretation, not confirmations of historical coarse enclosures.

## Branch-side cancellation planning

Cases 0734 and 0735 use `1/(acos(1/log(x))-pi/2)=-log(x)+O(1/log(x))` on the negative real tail. Its imaginary part tends to `-pi`, keeping the reciprocal strictly below the cut. The principal logarithmic inverse-function definitions then give positive imaginary divergence for inverse secant and negative imaginary divergence for inverse cosecant, of size `log(log(abs(x)))`. The sign is proved before a general series is requested. See the [principal inverse-function definitions](https://dlmf.nist.gov/4.23).

## Additional special-function certificates

Scorer Gi and its derivative have algebraic tails on the positive real axis; Hi and its derivative have algebraic tails on the negative real axis. Rational argument substitutions and rational weights are supported when the controlled remainder still vanishes. Cancellations requiring further terms, opposite tails and general complex sectors remain unresolved. See the [Scorer definitions and expansions](https://dlmf.nist.gov/9.12).

The parameter nome is `exp(-pi*K(1-m)/K(m))`. Its signed boundary at m=2 is certified from the nonzero K denominator. The inverse parameter germ has derivative 16 at q=0. The local amplitude identity `am(F(phi|m)|m)=phi` is used only as phi tends to zero for a fixed finite parameter; it is not a global inverse identity.

Owen T uses its exact integral definition. At a=+/-I, the integrand residue is `1/(4*pi*a)`, giving a logarithmic primitive and the corresponding imaginary directional infinity. Transverse rays and inward tangent rays avoid the cuts; outward tangent rays require another branch convention. The parameter h must be fixed and finite.

For the polygamma cluster example 0295, put `z=x/(2*pi)` and use reflection to replace the negative-argument first and second polygamma functions by their positive-argument counterparts at `1-z`. The csc/cot singular terms cancel exactly, leaving

`A(x)*sin(x)+B(x)*(cos(x)-1)` with

`A=(x**2*psi1(1-z)+2*pi*(x+2*pi))/(2*pi**2)` and
`B=(x**3*psi2(1-z)+4*pi**2*(x+4*pi))/(4*pi**3*x)`.

The positive-tail expansions give A=1+O(1/abs(x)) and B=O(1/abs(x)), uniformly even near the original poles. The attained sequences `x=-2*pi*n+theta`, with `0<theta<2*pi`, avoid all those poles and fill the exact set [-1,1]. Zero is attained using theta=pi. See the [reflection and polygamma expansions](https://dlmf.nist.gov/5.15).

Circular Piecewise ray certificates preserve branch priority. The rational sign of `abs(argument)**2-radius**2` selects the eventual branch, including quadratic tangent rays; finite rational branch limits retain denominator avoidance.

## Nested oscillations with accumulating poles

For `sec(tan(csc(phi)))`, choose a finite outer angle A and set `csc(phi)=pi+atan(A)`. Then `tan(csc(phi))=A` exactly. The inner sine is a nonzero reciprocal, the middle cosine is `-1/sqrt(1+A**2)`, and the outer cosine is prescribed to be nonzero. Positive inverse maps of affine/quadratic phases and logarithmic tails give attained sequences. A bounded sine inside phi is inverted only on its proved local arcsine range.

Angles 0 and pi attain +1 and -1. For an added sin(x) with quadratic phase, rounding the phase at x=2*pi*n changes x by O(1/n), so that extra sine tends to zero. For an added x, prescribe the secant value -2*L, with L the inverse quadratic phase near the pole; the attained x/L tends to one, proving negative divergence beside a positive-divergence sequence.

For `(2+sin(x))*sec(tan(csc(log(x))))+exp(x)`, prescribe the secant value `-exp(2*L)`. The exact inverse angles give `x=L+O(L*exp(-2*L))`; thus `exp(x)/exp(2*L)->0`, while 2+sin(x) stays in [1,3]. A second sequence with secant=1 diverges positively. These witnesses approach the poles without hitting any original denominator zero. Integer sampling and additional approach constraints need separate constructions.

The quotient `(x+sec(x))/(x**2*sec(x)+1)` instead tends to zero. On its original domain, clearing the reciprocal cosine gives `(x*cos(x)+1)/(x**2+cos(x))`, bounded by `(x+1)/(x**2-1)` for x>1. This bound is uniform even near secant poles; x=2*pi*n supplies an attained admissible tail.

## Definition and cancellation certificates

`Integrate(f, (t, a, b))` in the reference adapter denotes the exact definite `Integral(f, (t, a, b))`. It does not request an antiderivative. For a local integral from x to its finite target, the solver determines the multiplier's pole order first. It integrates only enough Taylor terms of a checked analytic integrand to make the multiplied remainder vanish. The supported entire compositions use polynomial algebra, sin, cos and exp; rational integrand factors require nonzero local denominators. Singular integrands, absolute-value cusps and cancellation orders above six need another certificate.

Even real roots preserve their real domain. For nonnegative integer order and positive real argument, the defining Bessel-I series has positive coefficients. This proves eventual positivity of the checked Bessel ratios before their real square root is normalized. Unknown function values cannot supply that positivity.

On a punctured imaginary-axis germ, a nonzero leading imaginary coefficient fixes the argument at plus or minus pi/2. Floor then has the corresponding constant value. Floor and ceiling also preserve signed rational divergence because their error relative to a real argument is bounded by one. Distinct left and right infinities carry attained sequences and prove two-sided nonexistence.

The local real-axis Ei expansion separates `log(x)-Ei(log(1-x))` into values `-EulerGamma` and `-EulerGamma+I*pi`. Both are attained on the original logarithm branches. For `-x**I/log(x)+I*Ei(I*log(x))`, the bounded oscillatory numerator divided by log(x) vanishes; the lower vertical Ei tail gives pi on both real sides. See the [Ei expansions and continuations](https://dlmf.nist.gov/6.12).

The Gaussian log-moment cancellation extends to `erfinv(1-delta)` when delta is a positive rational germ tending to zero. The real inverse tends to positive infinity and the exact inverse identity transfers the same convergent moment. An argument approaching from outside the real interval remains an explicit complex-branch capability gap; the solver stops before asking for an unsupported general expansion.

The Stirling remainder certificates retain the first nonzero term after subtraction. Subtracting `1+1/(12*x)` from the normalized gamma tail leaves `1/(288*x**2)+O(x**-3)`. Subtracting the full gamma tail from `exp(1/(12*x))` leaves `1/(360*x**3)+O(x**-4)`. A derivative quotient uses the compatible digamma expansion to control the differentiated remainder. Differentiating an unspecified big-O term is insufficient. See the [gamma and digamma asymptotic expansions](https://dlmf.nist.gov/5.11).

## Joint modular phase certificates

For a shared phase P, the solver handles Mod(P,2), sin(P), cos(sqrt(2)*P) and cos(sin(P)**2) in finite sums, products, nonnegative integer powers and Min/Max. Nonoscillatory factors must have finite real limits before any phase limits are substituted. This condition prevents a zero sine limit from erasing an unbounded multiplier. Accepted phase inverses cover positive affine and quadratic tails, logarithms of positive affine tails, and the positive tail of x**3-x.

For j>=1, residue r in {0,1} and cosine target c in {-1,1}, define m_j as the least integer m>=j satisfying
`abs(sin(2*m+r))<1/j` and `abs(cos(sqrt(2)*(2*m+r))-c)<1/j`.
The numbers 1, 1/pi and sqrt(2)/pi are rationally independent: an integer relation either makes pi algebraic or gives a nontrivial rational relation between 1 and sqrt(2). The discrete Kronecker density theorem guarantees a qualifying integer on every tail. Thus m_j exists and tends to infinity. This is an exact attained sequence definition, not a sampled interval enclosure. The symbolic `PhaseHittingIndex` records this definition without an unbounded search. Its bounded numeric evaluator uses real-ball arithmetic to certify every comparison and the first qualifying integer.

Set x_j to the positive inverse phase at 2*m_j+r. The modular residue is exactly r, the sine tends to zero, the sqrt(2) cosine tends to c, and the folded cosine tends to one. Choosing (r,c)=(0,-1) and (1,1) yields two distinct limits in the checked reference family. Finite-tail algebra and continuity of Min/Max transfer these limits to the original expression. Rational denominator poles are finite in number, logarithmic denominators are valid on the eventual positive tail, and each inverse is taken on its proved real branch. Oscillatory denominators, integer sampling and additional domain restrictions require separate certificates.

For the cubic inverse, y>=2 gives q=(y/2+sqrt(y**2/4-1/27))**(1/3)>0 and x=q+1/(3*q)>1. The identity x**3-x=q**3+1/(27*q**3)=y proves the original phase exactly. The method establishes nonexistence from two attained limits; it does not assert the complete cluster set. See the [publisher's Kronecker theorem reference](https://www.ams.org/books/chel/364/chel364-endmatter.pdf) and the [research discussion of oscillatory limsups](https://www.texmacs.org/joris/limsup/limsup-corrected.html).

## Gaussian sides and vertical integral tails

For z=a/sqrt(x) with fixed nonzero real or purely imaginary a, `sqrt(pi)*z*exp(-z**2)*erfi(z)` has different real-side behavior. On a real z tail it tends to one. On an imaginary z=i*t tail, the exact identity erfi(i*t)=i*erf(t) turns it into `-sqrt(pi)*t*exp(t**2)*erf(t)`, which tends to negative infinity. The principal square root selects the side, and x=+/-1/n**2 gives attained punctured-domain witnesses. General complex coefficients need another sector analysis. See the [error-function definitions](https://dlmf.nist.gov/7.2) and [asymptotic expansions](https://dlmf.nist.gov/7.12).

For fixed integer order 1 through 16 and z=b+i*a*x with fixed finite b and nonzero real a, the principal exponential integral has `E_p(z)=exp(-z)/z*(1+O(1/z))` on a closed sector around the vertical ray. Its modulus is O(1/x), since Re(z)=Re(b) is fixed, so its limit is zero. Explicit positive parameter assumptions can certify a nonzero slope. Variable order, a possibly zero slope and general approach directions remain outside this certificate. See the [exponential-integral sector expansions](https://dlmf.nist.gov/8.20).

## Polynomial domain witnesses and coercivity

A closure check first tries attained rational rays through a finite real target. After substituting `u=t*v`, the first nonzero coefficient of each polynomial relation fixes its sign for sufficiently small positive `t`. A successful finite conjunction supplies an actual punctured approach. The search supports up to three coordinates, eight relations, degree sixteen and a fixed set of at most twenty-six nonzero directions. It handles equalities, strict and weak inequalities, and denominator exclusions. Unknown coefficient signs and unsupported relations fall through to real elimination.

A failed ray search does not prove non-closure. For example, `x>0, y=x**2` accumulates at the origin along a curved path even though no tested straight ray belongs to it. Closure certifies attainability of the domain; it does not certify a simultaneous function limit.

Polynomial isolated-zero checks reuse definite quadratic and weighted pure-even principal forms. Strictly higher weighted-order sign-indefinite terms are absorbed by the principal lower bound. An attained ray of zeros certifies non-isolation. Remaining cases use exact elimination with positive radius and lower-bound constants represented as formula constraints, so a zero-radius ball cannot satisfy the query.

## Proportional large-order Bessel tails

For positive rational a, integer m from one through eight, and fixed rational lambda with 1<lambda<=16, the limit certificate admits J_(lambda*z)(z) or Y_(lambda*z)(z), where z=a*x**m and x tends to positive real infinity. Put mu=sqrt(lambda**2-1) and rho=lambda*log(lambda+mu)-mu. The leading terms are exp(-rho*z)/sqrt(2*pi*mu*z) for J and -2*exp(rho*z)/sqrt(2*pi*mu*z) for Y, each with relative error O(1/order). See [Debye's large-order expansions](https://dlmf.nist.gov/10.19.E3).

A single Bessel factor may be multiplied by an exact real constant, a rational power of x, and exponentials with polynomial phases of degree at most eight. Exact cancellation of exponential scales precedes comparison of the remaining polynomial scale and power. Constant signs use exact arithmetic or outward-rounded real balls. This covers both balanced finite limits and growth or decay under competing exponential weights.

The relative remainder controls these multiplicative weights. Subtracting leading terms, reciprocals, variable ratios, turning points, complex phases, floating inputs, free parameters and restricted approach domains require other certificates. An undecided sign declines the rule. Structural size bounds keep unsupported inputs from triggering expansion of large powers.

## Complex tails and parameter cells

`I*oo` and `-I*oo` normalize to fixed directional infinities. On those axes, cosh has attained sequences `+/-I*2*pi*j` and `+/-I*(2*j+1)*pi`, giving distinct values one and minus one without poles. Fixed finite complex-order Bessel I tends to zero there: the [rotation identity](https://dlmf.nist.gov/10.27.E6) and [fixed-order Bessel expansion](https://dlmf.nist.gov/10.17.E3) give an O(r**(-1/2)) bound. A varying order requires another theorem.

For li at a real center between zero and one, the sign of a rational imaginary germ selects `li(center)+/-I*pi`, using li(z)=Ei(Log(z)) and the [Ei continuation](https://dlmf.nist.gov/6.4). Centered atan germs whose argument vanishes are analytic locally. Bounded Taylor valuations then certify removable quotients such as atan((z**2+1)**2)/sin(z**2+1)**2 at z=I. A complex pole is spherical infinity; an explicitly selected ray can instead have positive or negative real infinity. Unresolved parameter denominators retain their regularity obligation before a whole-plane chart is used.

A negative integer step factorial at step zero separates nonzero base, where the reciprocal finite product is continuous, from zero base, where its pole has explicit attained directions. Noninteger orders use the principal gamma quotient. The regular order-zero cell currently checks nonnegative real base and positive step. At base zero the argument-limit cell checks nonzero step, and reciprocal gamma handles integer-order zeros. Complementary zero-step and pole/branch cells remain UNKNOWN. Finite Taylor truncation near base zero commutes with specialization to a nonnegative integer order because gamma's numerator is regular there and reciprocal gamma is entire.
