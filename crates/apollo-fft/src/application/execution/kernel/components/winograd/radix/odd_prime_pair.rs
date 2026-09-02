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

/// Split-array (structure-of-arrays) form of [`dft_pair_impl`].
///
/// Arithmetically identical to [`dft_pair_impl`] — same operations in the same
/// order, so both carry the same error bound. The sole difference is layout:
/// the sum and i-difference staging arrays are split into separate real and
/// imaginary arrays instead of one interleaved `[Complex<F>; H]`, which changes
/// what the vectorizer can do with the two accumulation loops.
///
/// ## Why only `f32` at N=31 uses this
///
/// Layout choice here is empirical, not analytical, so it is narrowed to the
/// one instantiation where it measured as a win. A broader SoA route covering
/// `f32` N=29/37/41/53 was measured and rejected: it gave no stable
/// improvement and regressed the larger short-prime rows. At N=31 the SoA form
/// measured 87.31 ns against 107.39 ns for the interleaved probe, so
/// `ShortWinogradScalar for f32` routes `dft31` here and every other `f32`
/// odd prime — and every `f64` odd prime — stays on [`dft_pair_impl`].
///
/// Two consequences worth knowing before touching this:
///
/// - The `_reduced` suffix names the *reduced-precision scalar* this route was
///   introduced for, not a reduced error bound and not reduced register
///   pressure.
/// - `benchmark_kernels::winograd_pair_prime_forward` calls [`dft_pair_impl`]
///   for whatever `F` it is handed, so it does **not** exercise this route.
///   Benchmarking `f32` at N=31 through that entry point measures the
///   interleaved path this special case exists to avoid; a comparison meant to
///   re-evaluate the narrowing has to call both kernels explicitly.
#[inline]
pub(crate) fn dft_pair_impl_reduced<
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
    let mut sums_re = [zero; H];
    let mut sums_im = [zero; H];
    let mut idiffs_re = [zero; H];
    let mut idiffs_im = [zero; H];

    let sign = if INVERSE {
        <F as eunomia::NumericElement>::ONE
    } else {
        -<F as eunomia::NumericElement>::ONE
    };
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;

    for m in 0..H {
        let a = data[m + 1];
        let b = data[N - 1 - m];
        let sum_re = a.re + b.re;
        let sum_im = a.im + b.im;
        y0_re += sum_re;
        y0_im += sum_im;
        sums_re[m] = sum_re;
        sums_im[m] = sum_im;
        let diff_re = a.re - b.re;
        let diff_im = a.im - b.im;
        idiffs_re[m] = -diff_im * sign;
        idiffs_im[m] = diff_re * sign;
    }

    data[0] = eunomia::Complex::new(y0_re, y0_im);

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
            let sr = sums_re[m];
            let si = sums_im[m];
            let c = cos_row[m];
            base_re += sr * c;
            base_im += si * c;
        }
        for m in 0..H {
            let ir = idiffs_re[m];
            let ii = idiffs_im[m];
            let s = sin_row[m];
            delta_re += ir * s;
            delta_im += ii * s;
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
