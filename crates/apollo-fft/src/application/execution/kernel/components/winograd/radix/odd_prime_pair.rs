use super::super::traits::WinogradScalar;

/// Canonical catalog of odd-prime Winograd-pair (N, H) sizes.
/// Each `(N, H)` pair satisfies `N = 2*H + 1`. The `PrimePairTable<N, H>`
/// trait impls and `impl_prime_pair_table!` calls must mirror this array.
/// Canonical inventory of supported odd-prime Winograd-pair sizes; the
/// constant is consumed by the in-file invariant tests and serves as the
/// single authoritative pair table.
#[cfg(test)]
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

#[inline]
pub(crate) fn dft_pair_impl<
    F: WinogradScalar,
    const N: usize,
    const H: usize,
    const INVERSE: bool,
>(
    data: &mut [eunomia::Complex<F>; N],
    cos: &[[F; H]; H],
    sin: &[[F; H]; H],
) {
    debug_assert_eq!(N, 2 * H + 1);
    let zero = <F as eunomia::NumericElement>::ZERO;
    let x0 = data[0];
    let mut sums = [eunomia::Complex::new(zero, zero); H];
    let mut idiffs = [eunomia::Complex::new(zero, zero); H];
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;

    let sign = if INVERSE {
        <F as eunomia::NumericElement>::ONE
    } else {
        -<F as eunomia::NumericElement>::ONE
    };

    for m in 0..H {
        let a = data[m + 1];
        let b = data[N - 1 - m];
        let sum_re = a.re + b.re;
        let sum_im = a.im + b.im;
        y0_re += sum_re;
        y0_im += sum_im;
        sums[m] = eunomia::Complex::new(sum_re, sum_im);
        let diff_re = a.re - b.re;
        let diff_im = a.im - b.im;
        idiffs[m] = eunomia::Complex::new(-diff_im * sign, diff_re * sign);
    }

    data[0] = eunomia::Complex::new(y0_re, y0_im);

    for k in 0..H {
        let cos_row = &cos[k];
        let sin_row = &sin[k];

        // Two-pass accumulation: compute base and delta contributions separately.
        // This improves instruction-level parallelism and CPU pipelining,
        // enabling better utilization of dual FMA units on modern out-of-order CPUs.
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        for m in 0..H {
            let s_m = sums[m];
            let c = cos_row[m];
            base_re += s_m.re * c;
            base_im += s_m.im * c;
        }
        for m in 0..H {
            let id_m = idiffs[m];
            let s = sin_row[m];
            delta_re += id_m.re * s;
            delta_im += id_m.im * s;
        }

        data[k + 1] = eunomia::Complex::new(base_re + delta_re, base_im + delta_im);
        data[N - 1 - k] = eunomia::Complex::new(base_re - delta_re, base_im - delta_im);
    }
}

/// Wide form of [`dft_pair_impl`]: two output bins per k-iteration.
///
/// Arithmetically identical to [`dft_pair_impl`] — same inputs, same outputs,
/// and the same per-output summation order (m ascending, base pass then delta
/// pass). The only difference is scheduling: where [`dft_pair_impl`] carries
/// four accumulators per iteration (base re/im, delta re/im — two sequential
/// FMA chains per lane), this form carries eight, covering the bin pair
/// `k` and `k + 1` side by side. At f32 the leaf probe measured the narrow
/// form losing 1.37x to 1.44x against f64 on the odd half-sizes (H = 5, 9,
/// 11, 14, 15) while winning 0.57x at H = 8 — a latency-bound reduction
/// signature, not a lane-width one, and this form shortens that critical
/// path instead of relying on the vectorizer to repair it.
///
/// ## Why only `f32` routes here
///
/// The narrow form's f64 codegen schedules the same reductions fine (the
/// probe's f64 numbers are healthy at every H), and re-plumbing f64 would
/// churn twelve monomorphizations for no measured gain. The route lives in
/// `ShortWinogradScalar for f32`, which selects the sizes; this function
/// stays scalar-generic so the in-file equivalence tests can compare both
/// forms at either width.
#[inline]
pub(crate) fn dft_pair_impl_wide<
    F: WinogradScalar,
    const N: usize,
    const H: usize,
    const INVERSE: bool,
>(
    data: &mut [eunomia::Complex<F>; N],
    cos: &[[F; H]; H],
    sin: &[[F; H]; H],
) {
    debug_assert_eq!(N, 2 * H + 1);
    let zero = <F as eunomia::NumericElement>::ZERO;
    let x0 = data[0];
    let mut sums = [eunomia::Complex::new(zero, zero); H];
    let mut idiffs = [eunomia::Complex::new(zero, zero); H];
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;

    let sign = if INVERSE {
        <F as eunomia::NumericElement>::ONE
    } else {
        -<F as eunomia::NumericElement>::ONE
    };

    for m in 0..H {
        let a = data[m + 1];
        let b = data[N - 1 - m];
        let sum_re = a.re + b.re;
        let sum_im = a.im + b.im;
        y0_re += sum_re;
        y0_im += sum_im;
        sums[m] = eunomia::Complex::new(sum_re, sum_im);
        let diff_re = a.re - b.re;
        let diff_im = a.im - b.im;
        idiffs[m] = eunomia::Complex::new(-diff_im * sign, diff_re * sign);
    }

    data[0] = eunomia::Complex::new(y0_re, y0_im);

    let mut k = 0usize;
    while k + 2 <= H {
        let cos_k = &cos[k];
        let sin_k = &sin[k];
        let cos_k1 = &cos[k + 1];
        let sin_k1 = &sin[k + 1];

        let mut base0_re = x0.re;
        let mut base0_im = x0.im;
        let mut delta0_re = zero;
        let mut delta0_im = zero;
        let mut base1_re = x0.re;
        let mut base1_im = x0.im;
        let mut delta1_re = zero;
        let mut delta1_im = zero;

        for m in 0..H {
            let s = sums[m];
            let id = idiffs[m];
            let c0 = cos_k[m];
            let sn0 = sin_k[m];
            let c1 = cos_k1[m];
            let sn1 = sin_k1[m];
            base0_re += s.re * c0;
            base0_im += s.im * c0;
            delta0_re += id.re * sn0;
            delta0_im += id.im * sn0;
            base1_re += s.re * c1;
            base1_im += s.im * c1;
            delta1_re += id.re * sn1;
            delta1_im += id.im * sn1;
        }

        data[k + 1] = eunomia::Complex::new(base0_re + delta0_re, base0_im + delta0_im);
        data[N - 1 - k] = eunomia::Complex::new(base0_re - delta0_re, base0_im - delta0_im);
        data[k + 2] = eunomia::Complex::new(base1_re + delta1_re, base1_im + delta1_im);
        data[N - 2 - k] = eunomia::Complex::new(base1_re - delta1_re, base1_im - delta1_im);
        k += 2;
    }
    if k < H {
        let cos_row = &cos[k];
        let sin_row = &sin[k];
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;
        for m in 0..H {
            let s_m = sums[m];
            let id_m = idiffs[m];
            let c = cos_row[m];
            let s = sin_row[m];
            base_re += s_m.re * c;
            base_im += s_m.im * c;
            delta_re += id_m.re * s;
            delta_im += id_m.im * s;
        }
        data[k + 1] = eunomia::Complex::new(base_re + delta_re, base_im + delta_im);
        data[N - 1 - k] = eunomia::Complex::new(base_re - delta_re, base_im - delta_im);
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
#[inline]
pub(crate) fn dft_pair_forward_with_pointwise<F: WinogradScalar, const N: usize, const H: usize>(
    data: &mut [eunomia::Complex<F>; N],
    kernel_spectrum: &[eunomia::Complex<F>; N],
    cos: &[[F; H]; H],
    sin: &[[F; H]; H],
) {
    debug_assert_eq!(N, 2 * H + 1);
    let zero = <F as eunomia::NumericElement>::ZERO;
    let one = <F as eunomia::NumericElement>::ONE;
    let neg_one = -one;
    let sign = neg_one; // forward: sign = -1

    let x0 = data[0];
    let mut sums = [eunomia::Complex::new(zero, zero); H];
    let mut idiffs = [eunomia::Complex::new(zero, zero); H];

    for m in 0..H {
        let a = data[m + 1];
        let b = data[N - 1 - m];
        sums[m] = eunomia::Complex::new(a.re + b.re, a.im + b.im);
        let diff_re = a.re - b.re;
        let diff_im = a.im - b.im;
        idiffs[m] = eunomia::Complex::new(-diff_im * sign, diff_re * sign);
    }

    // DC bin: sum of all inputs × kernel_spectrum[0]
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    for s in &sums {
        y0_re += s.re;
        y0_im += s.im;
    }
    let ks0 = kernel_spectrum[0];
    data[0] = eunomia::Complex::new(y0_re, y0_im) * ks0;

    // Remaining bins: compute Winograd output then multiply by kernel_spectrum[k]
    for k in 0..H {
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        let cos_row = &cos[k];
        let sin_row = &sin[k];

        // Two-pass accumulation: compute base and delta contributions separately.
        // This improves instruction-level parallelism and CPU pipelining,
        // enabling better utilization of dual FMA units on modern out-of-order CPUs.
        for m in 0..H {
            let s_m = sums[m];
            let c = cos_row[m];
            base_re += s_m.re * c;
            base_im += s_m.im * c;
        }
        for m in 0..H {
            let id_m = idiffs[m];
            let s = sin_row[m];
            delta_re += id_m.re * s;
            delta_im += id_m.im * s;
        }

        let ks_kp1 = kernel_spectrum[k + 1];
        let ks_nmk = kernel_spectrum[N - 1 - k];

        data[k + 1] = eunomia::Complex::new(base_re + delta_re, base_im + delta_im) * ks_kp1;
        data[N - 1 - k] = eunomia::Complex::new(base_re - delta_re, base_im - delta_im) * ks_nmk;
    }
}

// Generate `PrimePairTable<N, H>` implementations for the canonical
// Winograd-pair inventory. These tables are active inputs to the short
// Winograd and two-by-prime paths.
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
    use super::{dft_pair_impl, dft_pair_impl_wide, PrimePairTable};
    use crate::application::execution::kernel::components::winograd::traits::WinogradScalar;

    /// Each output retains its summation order when neighboring bins are paired.
    fn check_wide_equivalence<F, const N: usize, const H: usize, const INVERSE: bool>()
    where
        F: WinogradScalar + PrimePairTable<N, H>,
    {
        let mut narrow = std::array::from_fn(|i| {
            let x = f64::from(u32::try_from(i).unwrap()) * 0.173;
            eunomia::Complex::new(
                F::from_precise(x.sin() * 3.0 - 0.5),
                F::from_precise(x.cos() * 1.5),
            )
        });
        let mut wide = narrow;
        dft_pair_impl::<F, N, H, INVERSE>(&mut narrow, F::cos_table(), F::sin_table());
        dft_pair_impl_wide::<F, N, H, INVERSE>(&mut wide, F::cos_table(), F::sin_table());
        assert_eq!(
            narrow, wide,
            "wide pair form diverged from narrow at n={N}, inverse={INVERSE}"
        );
    }

    fn check_wide_sizes<F: WinogradScalar>() {
        fn direction<F: WinogradScalar, const INVERSE: bool>() {
            check_wide_equivalence::<F, 11, 5, INVERSE>();
            check_wide_equivalence::<F, 19, 9, INVERSE>();
            check_wide_equivalence::<F, 23, 11, INVERSE>();
            check_wide_equivalence::<F, 29, 14, INVERSE>();
            check_wide_equivalence::<F, 31, 15, INVERSE>();
        }
        direction::<F, false>();
        direction::<F, true>();
    }

    #[test]
    fn wide_pair_form_matches_narrow_exactly() {
        // The wide form is a scheduling change only: per output bin the
        // summation order (m ascending, base then delta) is the narrow
        // form's, so finite results must compare exactly without a tolerance.
        check_wide_sizes::<f32>();
        check_wide_sizes::<f64>();
    }

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
