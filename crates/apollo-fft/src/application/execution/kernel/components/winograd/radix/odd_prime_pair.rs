use super::super::traits::WinogradScalar;

/// Canonical catalog of odd-prime Winograd-pair (N, H) sizes.
/// Each `(N, H)` pair satisfies `N = 2*H + 1`. The `PrimePairTable<N, H>`
/// trait impls and `impl_prime_pair_table!` calls must mirror this array.
#[allow(dead_code)] // referenced by tests; also serves as documentation
pub(crate) const ODD_PRIME_PAIR_SIZES: &[(usize, usize)] = &[
    (11, 5),
    (13, 6),
    (17, 8),
    (19, 9),
    (23, 11),
    (29, 14),
    (31, 15),
    (37, 18),
    (41, 20),
    (43, 21),
    (47, 23),
    (53, 26),
];

pub trait PrimePairTable<const N: usize, const H: usize>: 'static + Send + Sync + Copy {
    fn cos_table() -> &'static [[Self; H]; H];
    fn sin_table() -> &'static [[Self; H]; H];
}

pub trait PrimePairTables:
    PrimePairTable<11, 5>
    + PrimePairTable<13, 6>
    + PrimePairTable<17, 8>
    + PrimePairTable<19, 9>
    + PrimePairTable<23, 11>
    + PrimePairTable<29, 14>
    + PrimePairTable<31, 15>
    + PrimePairTable<37, 18>
    + PrimePairTable<41, 20>
    + PrimePairTable<43, 21>
    + PrimePairTable<47, 23>
    + PrimePairTable<53, 26>
{
}

impl<T> PrimePairTables for T where
    T: PrimePairTable<11, 5>
        + PrimePairTable<13, 6>
        + PrimePairTable<17, 8>
        + PrimePairTable<19, 9>
        + PrimePairTable<23, 11>
        + PrimePairTable<29, 14>
        + PrimePairTable<31, 15>
        + PrimePairTable<37, 18>
        + PrimePairTable<41, 20>
        + PrimePairTable<43, 21>
        + PrimePairTable<47, 23>
        + PrimePairTable<53, 26>
{
}

#[inline(always)]
pub(crate) fn dft_pair_impl<
    F: WinogradScalar,
    const N: usize,
    const H: usize,
    const INVERSE: bool,
>(
    data: &mut [num_complex::Complex<F>; N],
    cos: &[[F; H]; H],
    sin: &[[F; H]; H],
) {
    debug_assert_eq!(N, 2 * H + 1);
    let zero = F::zero();
    let x0 = data[0];
    let mut sums = [num_complex::Complex::new(zero, zero); H];
    let mut idiffs = [num_complex::Complex::new(zero, zero); H];

    let sign = if INVERSE { F::one() } else { -F::one() };

    for m in 0..H {
        let a = unsafe { *data.get_unchecked(m + 1) };
        let b = unsafe { *data.get_unchecked(N - 1 - m) };
        unsafe {
            *sums.get_unchecked_mut(m) = num_complex::Complex::new(a.re + b.re, a.im + b.im);
            let diff_re = a.re - b.re;
            let diff_im = a.im - b.im;
            *idiffs.get_unchecked_mut(m) =
                num_complex::Complex::new(-diff_im * sign, diff_re * sign);
        }
    }

    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    for s in &sums {
        y0_re = y0_re + s.re;
        y0_im = y0_im + s.im;
    }
    data[0] = num_complex::Complex::new(y0_re, y0_im);

    for k in 0..H {
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        let cos_row = unsafe { cos.get_unchecked(k) };
        let sin_row = unsafe { sin.get_unchecked(k) };

        for m in 0..H {
            let s_m = unsafe { *sums.get_unchecked(m) };
            let id_m = unsafe { *idiffs.get_unchecked(m) };
            let c = unsafe { *cos_row.get_unchecked(m) };
            let s = unsafe { *sin_row.get_unchecked(m) };
            base_re = base_re + s_m.re * c;
            base_im = base_im + s_m.im * c;
            delta_re = delta_re + id_m.re * s;
            delta_im = delta_im + id_m.im * s;
        }
        unsafe {
            *data.get_unchecked_mut(k + 1) =
                num_complex::Complex::new(base_re + delta_re, base_im + delta_im);
            *data.get_unchecked_mut(N - 1 - k) =
                num_complex::Complex::new(base_re - delta_re, base_im - delta_im);
        }
    }
}

/// Forward Winograd-pair DFT with fused kernel-spectrum pointwise multiplication.
///
/// Identical to `dft_pair_impl::<..., INVERSE=false>` but each output bin is
/// immediately multiplied by the corresponding `kernel_spectrum[k]` before being
/// stored back, eliminating the separate `pointwise_mul` pass used in Rader
/// convolution (FullCyclic and HalfCyclic Nussbaumer paths).
///
/// `N` must equal `2*H + 1` (odd prime). `kernel_spectrum` must have length `N`.
#[inline(always)]
pub(crate) fn dft_pair_forward_with_pointwise<
    F: WinogradScalar,
    const N: usize,
    const H: usize,
>(
    data: &mut [num_complex::Complex<F>; N],
    kernel_spectrum: &[num_complex::Complex<F>; N],
    cos: &[[F; H]; H],
    sin: &[[F; H]; H],
) {
    debug_assert_eq!(N, 2 * H + 1);
    let zero = F::zero();
    let one = F::one();
    let neg_one = -one;
    let sign = neg_one; // forward: sign = -1

    let x0 = data[0];
    let mut sums = [num_complex::Complex::new(zero, zero); H];
    let mut idiffs = [num_complex::Complex::new(zero, zero); H];

    for m in 0..H {
        let a = unsafe { *data.get_unchecked(m + 1) };
        let b = unsafe { *data.get_unchecked(N - 1 - m) };
        unsafe {
            *sums.get_unchecked_mut(m) = num_complex::Complex::new(a.re + b.re, a.im + b.im);
            let diff_re = a.re - b.re;
            let diff_im = a.im - b.im;
            *idiffs.get_unchecked_mut(m) =
                num_complex::Complex::new(-diff_im * sign, diff_re * sign);
        }
    }

    // DC bin: sum of all inputs × kernel_spectrum[0]
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    for s in &sums {
        y0_re = y0_re + s.re;
        y0_im = y0_im + s.im;
    }
    let ks0 = unsafe { *kernel_spectrum.get_unchecked(0) };
    data[0] = num_complex::Complex::new(y0_re, y0_im) * ks0;

    // Remaining bins: compute Winograd output then multiply by kernel_spectrum[k]
    for k in 0..H {
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        let cos_row = unsafe { cos.get_unchecked(k) };
        let sin_row = unsafe { sin.get_unchecked(k) };

        for m in 0..H {
            let s_m = unsafe { *sums.get_unchecked(m) };
            let id_m = unsafe { *idiffs.get_unchecked(m) };
            let c = unsafe { *cos_row.get_unchecked(m) };
            let s = unsafe { *sin_row.get_unchecked(m) };
            base_re = base_re + s_m.re * c;
            base_im = base_im + s_m.im * c;
            delta_re = delta_re + id_m.re * s;
            delta_im = delta_im + id_m.im * s;
        }

        let ks_kp1 = unsafe { *kernel_spectrum.get_unchecked(k + 1) };
        let ks_nmk = unsafe { *kernel_spectrum.get_unchecked(N - 1 - k) };

        unsafe {
            *data.get_unchecked_mut(k + 1) =
                num_complex::Complex::new(base_re + delta_re, base_im + delta_im) * ks_kp1;
            *data.get_unchecked_mut(N - 1 - k) =
                num_complex::Complex::new(base_re - delta_re, base_im - delta_im) * ks_nmk;
        }
    }
}

// Target of generated two-by-prime dispatch. Some build surfaces do not
// instantiate the promoted-prime route set, so `dead_code` can fire.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn two_by_prime_impl<
    F: WinogradScalar,
    const P: usize,
    const H: usize,
    const INVERSE: bool,
>(
    data: &mut [num_complex::Complex<F>],
    twiddles: &[num_complex::Complex<F>],
    cos: &[[F; H]; H],
    sin: &[[F; H]; H],
) {
    debug_assert!(data.len() >= 2 * P);
    debug_assert!(twiddles.len() >= P);
    debug_assert_eq!(P, 2 * H + 1);

    let zero = F::zero();
    let input = data.as_ptr();
    let twiddle_ptr = twiddles.as_ptr();

    // The input contract is an interleaved two-by-prime layout:
    // data[2*j] is the even half and data[2*j + 1] is the odd half for j < P.
    let (even_x0, odd_x0) = unsafe { (*input, *input.add(1)) };
    let mut even_sums = [num_complex::Complex::new(zero, zero); H];
    let mut even_idiffs = [num_complex::Complex::new(zero, zero); H];
    let mut odd_sums = [num_complex::Complex::new(zero, zero); H];
    let mut odd_idiffs = [num_complex::Complex::new(zero, zero); H];

    let sign = if INVERSE {
        F::from_precise(1.0)
    } else {
        F::from_precise(-1.0)
    };

    for m in 0..H {
        let lo = 2 * (m + 1);
        let hi = 2 * (P - 1 - m);
        // lo/hi are derived from m < H and P = 2H + 1, so both pairs are inside 2P.
        let (even_a, odd_a, even_b, odd_b) = unsafe {
            (
                *input.add(lo),
                *input.add(lo + 1),
                *input.add(hi),
                *input.add(hi + 1),
            )
        };
        even_sums[m] = num_complex::Complex::new(even_a.re + even_b.re, even_a.im + even_b.im);
        let even_diff_re = even_a.re - even_b.re;
        let even_diff_im = even_a.im - even_b.im;
        even_idiffs[m] = num_complex::Complex::new(-even_diff_im * sign, even_diff_re * sign);

        odd_sums[m] = num_complex::Complex::new(odd_a.re + odd_b.re, odd_a.im + odd_b.im);
        let odd_diff_re = odd_a.re - odd_b.re;
        let odd_diff_im = odd_a.im - odd_b.im;
        odd_idiffs[m] = num_complex::Complex::new(-odd_diff_im * sign, odd_diff_re * sign);
    }

    let mut even_y0_re = even_x0.re;
    let mut even_y0_im = even_x0.im;
    let mut odd_y0_re = odd_x0.re;
    let mut odd_y0_im = odd_x0.im;
    for m in 0..H {
        even_y0_re = even_y0_re + even_sums[m].re;
        even_y0_im = even_y0_im + even_sums[m].im;
        odd_y0_re = odd_y0_re + odd_sums[m].re;
        odd_y0_im = odd_y0_im + odd_sums[m].im;
    }
    let even_y0 = num_complex::Complex::new(even_y0_re, even_y0_im);
    let odd_y0 = num_complex::Complex::new(odd_y0_re, odd_y0_im);
    let wb0 = unsafe { *twiddle_ptr } * odd_y0;
    data[0] = even_y0 + wb0;
    data[P] = even_y0 - wb0;

    for k in 0..H {
        let cos_row = &cos[k];
        let sin_row = &sin[k];
        let mut even_base_re = even_x0.re;
        let mut even_base_im = even_x0.im;
        let mut even_delta_re = zero;
        let mut even_delta_im = zero;
        let mut odd_base_re = odd_x0.re;
        let mut odd_base_im = odd_x0.im;
        let mut odd_delta_re = zero;
        let mut odd_delta_im = zero;
        for m in 0..H {
            let c = cos_row[m];
            let s = sin_row[m];
            let e_s = even_sums[m];
            let e_id = even_idiffs[m];
            let o_s = odd_sums[m];
            let o_id = odd_idiffs[m];
            even_base_re = even_base_re + e_s.re * c;
            even_base_im = even_base_im + e_s.im * c;
            even_delta_re = even_delta_re + e_id.re * s;
            even_delta_im = even_delta_im + e_id.im * s;
            odd_base_re = odd_base_re + o_s.re * c;
            odd_base_im = odd_base_im + o_s.im * c;
            odd_delta_re = odd_delta_re + o_id.re * s;
            odd_delta_im = odd_delta_im + o_id.im * s;
        }

        let lo = k + 1;
        let hi = P - 1 - k;
        let even_lo =
            num_complex::Complex::new(even_base_re + even_delta_re, even_base_im + even_delta_im);
        let odd_lo =
            num_complex::Complex::new(odd_base_re + odd_delta_re, odd_base_im + odd_delta_im);
        let wb_lo = unsafe { *twiddle_ptr.add(lo) } * odd_lo;
        data[lo] = even_lo + wb_lo;
        data[lo + P] = even_lo - wb_lo;

        let even_hi =
            num_complex::Complex::new(even_base_re - even_delta_re, even_base_im - even_delta_im);
        let odd_hi =
            num_complex::Complex::new(odd_base_re - odd_delta_re, odd_base_im - odd_delta_im);
        let wb_hi = unsafe { *twiddle_ptr.add(hi) } * odd_hi;
        data[hi] = even_hi + wb_hi;
        data[hi + P] = even_hi - wb_hi;
    }
}

// Compile-time cos/sin table generation via proc-macro.
// Replaces 48 `OnceLock` statics (4 per prime × 12 primes) and their
// deferred runtime `cos`/`sin` computations with literal-embedded statics
// emitted during Rust compilation (proc-macro expansion time).
apollo_fft_macros::generate_prime_pair_tables![
    (11, 5),
    (13, 6),
    (17, 8),
    (19, 9),
    (23, 11),
    (29, 14),
    (31, 15),
    (37, 18),
    (41, 20),
    (43, 21),
    (47, 23),
    (53, 26)
];

#[cfg(test)]
mod tests {
    use super::ODD_PRIME_PAIR_SIZES;

    #[test]
    fn odd_prime_pair_sizes_satisfy_n_equals_2h_plus_1() {
        for &(n, h) in ODD_PRIME_PAIR_SIZES {
            assert_eq!(
                n,
                2 * h + 1,
                "ODD_PRIME_PAIR_SIZES pair (n={n}, h={h}) must satisfy n = 2*h + 1"
            );
        }
    }

    #[test]
    fn odd_prime_pair_primes_match_direct_pair_primes() {
        // The primes in ODD_PRIME_PAIR_SIZES must exactly match
        // `two_by_prime::DIRECT_PAIR_PRIMES`.
        let mut pair_primes: Vec<usize> = ODD_PRIME_PAIR_SIZES.iter().map(|&(n, _)| n).collect();
        pair_primes.sort_unstable();

        // Access DIRECT_PAIR_PRIMES through the good_thomas module path,
        // which re-exports two_by_prime publicly enough to be reachable
        // from this sibling test.
        let mut direct: Vec<usize> =
            crate::application::execution::kernel::components::good_thomas::two_by_prime::DIRECT_PAIR_PRIMES
                .to_vec();
        direct.sort_unstable();

        assert_eq!(
            pair_primes, direct,
            "ODD_PRIME_PAIR_SIZES primes must match two_by_prime::DIRECT_PAIR_PRIMES"
        );
    }
}
