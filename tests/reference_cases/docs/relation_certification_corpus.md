# Relation-certification corpus

Backend availability: `semialg=True`, `symbopt=False`.

| Case | Family | Relation | Certified | Actual certificate backends | Charts |
|---|---|:---:|:---:|---|---:|
| `rational_little_o` | rational | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `rational_equivalent` | rational | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `rational_big_o_angular` | rational | O | yes | exact_zero_remainder, sector_simplex_semialg, semialg+weighted_blowup_geometry | 2 |
| `restricted_halfspace_little_o` | restricted-domain | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `restricted_wedge_equivalent` | restricted-domain | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `multi_newton_identity` | multi-Newton-chart | O | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `multi_newton_decay` | multi-Newton-chart | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `difficult_angular_quadratic` | difficult-angular-extremum | O | yes | angular_coefficient_norm, exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `difficult_angular_quartic` | difficult-angular-extremum | O | yes | exact_zero_remainder, sector_simplex_semialg, semialg+weighted_blowup_geometry | 2 |
| `algebraic_sqrt_equivalent` | algebraic | ~ | yes | positive_radical_denominator, semialg+weighted_blowup_geometry | 2 |
| `algebraic_radical_little_o` | algebraic | o | yes | positive_radical_denominator, semialg+weighted_blowup_geometry | 2 |
| `transcendental_exp_equivalent` | transcendental-composition | ~ | yes | semialg+weighted_blowup_geometry, univariate_taylor_theorem | 2 |
| `transcendental_sinc_equivalent` | transcendental-composition | ~ | yes | alternating_sine_remainder, semialg+weighted_blowup_geometry | 2 |
| `newton3_identity` | 3plus-Newton-fan | O | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 3 |
| `newton3_decay` | 3plus-Newton-fan | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 3 |
| `newton3_equivalent` | 3plus-Newton-fan | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 3 |
| `newton3_mixed_O` | 3plus-Newton-fan | O | yes | exact_zero_remainder, sector_dominant_coordinate_bound, semialg+weighted_blowup_geometry | 3 |
| `newton3_competing_faces` | 3plus-Newton-fan | O | yes | exact_zero_remainder, sector_dominant_coordinate_bound, semialg+weighted_blowup_geometry | 3 |
| `newton4_decay` | 3plus-Newton-fan | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 4 |
| `newton4_equivalent` | 3plus-Newton-fan | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 4 |
| `alg_nonradial_sqrt_equiv` | nonradial-algebraic | ~ | yes | exact_zero_remainder, positive_square_root_identity, semialg+weighted_blowup_geometry | 2 |
| `alg_nonradial_sqrt_decay` | nonradial-algebraic | o | yes | exact_zero_remainder, positive_power_decay, positive_square_root_identity, semialg+weighted_blowup_geometry | 2 |
| `alg_nonradial_nested_equiv` | nonradial-algebraic | ~ | no | none | 0 |
| `alg_nonradial_ratio_O` | nonradial-algebraic | O | yes | exact_norm_inequality | 0 |
| `alg3_sqrt_equiv` | nonradial-algebraic | ~ | yes | exact_zero_remainder, positive_square_root_identity, semialg+weighted_blowup_geometry | 3 |
| `alg3_decay` | nonradial-algebraic | o | yes | exact_zero_remainder, positive_power_decay, positive_square_root_identity, semialg+weighted_blowup_geometry | 3 |
| `trans_angular_exp_equiv` | angular-transcendental | ~ | yes | exact_zero_remainder, exp_continuity, semialg+weighted_blowup_geometry | 2 |
| `trans_angular_sin_o` | angular-transcendental | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry, sine_lipschitz | 2 |
| `trans_angular_cos_equiv` | angular-transcendental | ~ | yes | cos_continuity, exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `trans_angular_exp3_equiv` | angular-transcendental | ~ | yes | exact_zero_remainder, exp_continuity, semialg+weighted_blowup_geometry | 3 |
| `trans_angular_sin3_o` | angular-transcendental | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry, sine_lipschitz | 3 |
| `trans_anisotropic_sinc` | angular-transcendental | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry, sinc_continuity | 2 |
| `domain_parabolic_wedge_o` | mixed-restricted-domain | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `domain_cusp_equiv` | mixed-restricted-domain | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `domain_annular_cusp_O` | mixed-restricted-domain | O | yes | exact_zero_remainder, sector_simplex_semialg, semialg+weighted_blowup_geometry | 2 |
| `domain3_orthant_o` | mixed-restricted-domain | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 3 |
| `domain3_ordered_equiv` | mixed-restricted-domain | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 3 |
| `domain3_cone_O` | mixed-restricted-domain | O | yes | exact_norm_inequality | 0 |
| `boundary_zero_cancelled_equiv` | sector-boundary-zero-pole | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `boundary_zero_cancelled_O` | sector-boundary-zero-pole | O | yes | angular_coefficient_norm, exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `boundary_axis_cancelled_equiv` | sector-boundary-zero-pole | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `boundary_axis_cancelled_o` | sector-boundary-zero-pole | o | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `boundary_cone_cancelled_equiv` | sector-boundary-zero-pole | ~ | yes | exact_zero_remainder, semialg+weighted_blowup_geometry | 2 |
| `boundary3_plane_cancelled_O` | sector-boundary-zero-pole | O | yes | exact_zero_remainder, sector_simplex_semialg, semialg+weighted_blowup_geometry | 3 |

Certified: **43/44**. Certified cases requiring symbopt: **0/43**.

An UNKNOWN result is not evidence that symbopt would certify the case. If symbopt is unavailable, this run can establish semialg/exact coverage and expose optimization gaps, but cannot measure symbopt rescue rate.
