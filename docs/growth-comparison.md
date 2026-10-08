# Growth-comparison proof integration

A certified `GrowthComparisonProof` can discharge a positive-side scalar
quotient limit directly. The lift is narrow: target zero,
one active variable, ambient or exact positive-side domain, and a certified
numerator/denominator scale comparison.

`LESS` certifies quotient limit zero. `GREATER` certifies positive infinity
only when positivity of the local Hardy germ is independently established.
The implementation therefore does not infer a signed infinity from magnitude
comparison alone.

The growth-comparison integration cases are permanent regression tests and must return `PROVED` instead of `UNKNOWN`.
