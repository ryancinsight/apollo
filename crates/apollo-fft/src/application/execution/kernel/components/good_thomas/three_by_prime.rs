apollo_fft_macros::generate_three_by_prime_dispatch! {
    primes: [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
    // Kept for API consistency; actual dead-code exclusion uses the
    // `THREE_BY_PRIME_PRIMES` const below, merged into fixed::FIXED_EXCLUDE_PRIMES.
    direct_pair_primes: [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
}

/// Primes handled by three_by_prime dispatch (producing canonical
/// 3×prime pairs).  Referenced by `fixed::FIXED_EXCLUDE_PRIMES`.
pub(crate) const THREE_BY_PRIME_PRIMES: &[usize] =
    &[5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

#[cfg(test)]
#[derive(Clone, Copy)]
struct ThreeByPrimePlan<const P: usize> {
    input: [[usize; P]; 3],
    output: [[usize; P]; 3],
}

#[cfg(test)]
impl<const P: usize> ThreeByPrimePlan<P> {
    const PLAN: Self = Self::generate();

    const fn generate() -> Self {
        assert!(P % 3 != 0, "3-by-prime route requires gcd(3, P) = 1");
        let n = 3 * P;
        let k1_stride = P * inverse_mod(P % 3, 3);
        let k2_stride = 3 * inverse_mod(3, P);
        let mut input = [[0usize; P]; 3];
        let mut output = [[0usize; P]; 3];

        let mut row = 0usize;
        while row < 3 {
            let mut col = 0usize;
            while col < P {
                input[row][col] = (row * P + 3 * col) % n;
                output[row][col] = (row * k1_stride + col * k2_stride) % n;
                col += 1;
            }
            row += 1;
        }

        Self { input, output }
    }
}

#[cfg(test)]
#[inline]
const fn inverse_mod(a: usize, modulus: usize) -> usize {
    let mut x = 1usize;
    while x < modulus {
        if (a * x) % modulus == 1 {
            return x;
        }
        x += 1;
    }
    panic!("coprime modulus must have an inverse");
}

#[cfg(test)]
mod tests {
    use super::{supports, ThreeByPrimePlan};

    #[test]
    fn support_is_limited_to_three_by_short_prime() {
        for p in [5usize, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] {
            assert!(supports(p, 3), "3-by-{p} must use the fused route");
        }
        for pair in [(11usize, 2usize), (9, 3), (7, 6)] {
            assert!(
                !supports(pair.0, pair.1),
                "{pair:?} is outside the fused 3-by-prime contract"
            );
        }
    }

    #[test]
    fn plan_matches_good_thomas_crt_maps() {
        for p in [5usize, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] {
            check_plan(p);
        }
    }

    fn check_plan(p: usize) {
        match p {
            5 => check_plan_const::<5>(),
            7 => check_plan_const::<7>(),
            11 => check_plan_const::<11>(),
            13 => check_plan_const::<13>(),
            17 => check_plan_const::<17>(),
            19 => check_plan_const::<19>(),
            23 => check_plan_const::<23>(),
            29 => check_plan_const::<29>(),
            31 => check_plan_const::<31>(),
            37 => check_plan_const::<37>(),
            41 => check_plan_const::<41>(),
            43 => check_plan_const::<43>(),
            47 => check_plan_const::<47>(),
            53 => check_plan_const::<53>(),
            _ => unreachable!("test only covers supported short primes"),
        }
    }

    fn check_plan_const<const P: usize>() {
        let plan = ThreeByPrimePlan::<P>::PLAN;
        let n = 3 * P;
        let inv_p_mod_3 = if P % 3 == 1 { 1 } else { 2 };
        let inv_3_mod_p = (1..P)
            .find(|x| (3 * x) % P == 1)
            .expect("supported prime must be coprime to 3");
        for row in 0..3usize {
            for col in 0..P {
                assert_eq!(
                    plan.input[row][col],
                    (row * P + 3 * col) % n,
                    "input CRT map mismatch for P={P}, row={row}, col={col}"
                );
                assert_eq!(
                    plan.output[row][col],
                    (row * P * inv_p_mod_3 + col * 3 * inv_3_mod_p) % n,
                    "output CRT map mismatch for P={P}, row={row}, col={col}"
                );
            }
        }
    }
}
