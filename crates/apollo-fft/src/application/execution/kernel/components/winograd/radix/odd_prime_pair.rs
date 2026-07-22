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

    let sign = if INVERSE {
        <F as eunomia::NumericElement>::ONE
    } else {
        -<F as eunomia::NumericElement>::ONE
    };

    for m in 0..H {
        let a = unsafe { *data.get_unchecked(m + 1) };
        let b = unsafe { *data.get_unchecked(N - 1 - m) };
        unsafe {
            *sums.get_unchecked_mut(m) = eunomia::Complex::new(a.re + b.re, a.im + b.im);
            let diff_re = a.re - b.re;
            let diff_im = a.im - b.im;
            *idiffs.get_unchecked_mut(m) = eunomia::Complex::new(-diff_im * sign, diff_re * sign);
        }
    }

    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    for s in &sums {
        y0_re += s.re;
        y0_im += s.im;
    }
    data[0] = eunomia::Complex::new(y0_re, y0_im);

    if H
        <= <F as crate::application::execution::kernel::components::winograd::traits::private::Sealed>::COMPACT_PAIR_MAX_HALF_LENGTH
    {
        // SAFETY: `sums` and `idiffs` have length H, both coefficient tables
        // have shape H x H, and `data` has length N + 1 = 2 * H + 1.
        unsafe {
            accumulate_pair_outputs(
                data,
                x0,
                &sums,
                &idiffs,
                cos.as_flattened(),
                sin.as_flattened(),
            );
        }
        return;
    }

    for k in 0..H {
        let cos_row = unsafe { cos.get_unchecked(k) };
        let sin_row = unsafe { sin.get_unchecked(k) };

        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        for m in 0..H {
            let sum = unsafe { *sums.get_unchecked(m) };
            let cosine = unsafe { *cos_row.get_unchecked(m) };
            base_re += sum.re * cosine;
            base_im += sum.im * cosine;
        }
        for m in 0..H {
            let idiff = unsafe { *idiffs.get_unchecked(m) };
            let sine = unsafe { *sin_row.get_unchecked(m) };
            delta_re += idiff.re * sine;
            delta_im += idiff.im * sine;
        }

        unsafe {
            *data.get_unchecked_mut(k + 1) =
                eunomia::Complex::new(base_re + delta_re, base_im + delta_im);
            *data.get_unchecked_mut(N - 1 - k) =
                eunomia::Complex::new(base_re - delta_re, base_im - delta_im);
        }
    }
}

/// Accumulates paired outputs without specializing the quadratic loop for each
/// prime length.
///
/// The const-generic boundary validates the matrix shapes before this call.
/// Keeping the quadratic body in a non-const inner function prevents LLVM from
/// fully unrolling the measured profitable small-prime specializations into
/// multi-kilobyte kernels. Four output rows remain the fixed SIMD/ILP unit,
/// while `h` stays a runtime loop bound to keep instruction and stack footprints
/// independent of `N`.
///
/// # Safety
///
/// `idiffs` must have the same length `h` as `sums`; `cos` and `sin` must each
/// have length `h * h`; and `data` must have length `2 * h + 1`.
#[inline(never)]
unsafe fn accumulate_pair_outputs<F: WinogradScalar>(
    data: &mut [eunomia::Complex<F>],
    x0: eunomia::Complex<F>,
    sums: &[eunomia::Complex<F>],
    idiffs: &[eunomia::Complex<F>],
    cos: &[F],
    sin: &[F],
) {
    const LANES: usize = 4;

    let h = sums.len();
    debug_assert_eq!(idiffs.len(), h);
    debug_assert_eq!(cos.len(), h * h);
    debug_assert_eq!(sin.len(), h * h);
    debug_assert_eq!(data.len(), 2 * h + 1);

    let zero = <F as eunomia::NumericElement>::ZERO;
    let mut k = 0;
    while k + LANES <= h {
        let mut base_re = [x0.re; LANES];
        let mut base_im = [x0.im; LANES];
        let mut delta_re = [zero; LANES];
        let mut delta_im = [zero; LANES];

        let unrolled = h - h % LANES;
        let mut m = 0;
        while m < unrolled {
            // SAFETY: `m < h`; all four rows start below `h`; the shape
            // assertions establish `cos.len() == sin.len() == h * h`.
            unsafe {
                for element in 0..LANES {
                    let input = m + element;
                    let sum = *sums.get_unchecked(input);
                    let idiff = *idiffs.get_unchecked(input);
                    for lane in 0..LANES {
                        let offset = (k + lane) * h + input;
                        let cosine = *cos.get_unchecked(offset);
                        let sine = *sin.get_unchecked(offset);
                        *base_re.get_unchecked_mut(lane) += sum.re * cosine;
                        *base_im.get_unchecked_mut(lane) += sum.im * cosine;
                        *delta_re.get_unchecked_mut(lane) += idiff.re * sine;
                        *delta_im.get_unchecked_mut(lane) += idiff.im * sine;
                    }
                }
            }
            m += LANES;
        }
        while m < h {
            // SAFETY: `m < h`; all four rows start below `h`; the shape
            // assertions establish `cos.len() == sin.len() == h * h`.
            unsafe {
                let sum = *sums.get_unchecked(m);
                let idiff = *idiffs.get_unchecked(m);
                for lane in 0..LANES {
                    let offset = (k + lane) * h + m;
                    let cosine = *cos.get_unchecked(offset);
                    let sine = *sin.get_unchecked(offset);
                    *base_re.get_unchecked_mut(lane) += sum.re * cosine;
                    *base_im.get_unchecked_mut(lane) += sum.im * cosine;
                    *delta_re.get_unchecked_mut(lane) += idiff.re * sine;
                    *delta_im.get_unchecked_mut(lane) += idiff.im * sine;
                }
            }
            m += 1;
        }

        for lane in 0..LANES {
            let output = k + lane;
            // SAFETY: `output < h` and `data.len() == 2 * h + 1`, so both
            // paired output indices are distinct and in bounds.
            unsafe {
                let base_re = *base_re.get_unchecked(lane);
                let base_im = *base_im.get_unchecked(lane);
                let delta_re = *delta_re.get_unchecked(lane);
                let delta_im = *delta_im.get_unchecked(lane);
                *data.get_unchecked_mut(output + 1) =
                    eunomia::Complex::new(base_re + delta_re, base_im + delta_im);
                *data.get_unchecked_mut(2 * h - output) =
                    eunomia::Complex::new(base_re - delta_re, base_im - delta_im);
            }
        }
        k += LANES;
    }

    while k < h {
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;
        for m in 0..h {
            // SAFETY: `k < h`, `m < h`, and the shape assertions establish
            // every indexed slice bound.
            unsafe {
                let sum = *sums.get_unchecked(m);
                let idiff = *idiffs.get_unchecked(m);
                let offset = k * h + m;
                let cosine = *cos.get_unchecked(offset);
                let sine = *sin.get_unchecked(offset);
                base_re += sum.re * cosine;
                base_im += sum.im * cosine;
                delta_re += idiff.re * sine;
                delta_im += idiff.im * sine;
            }
        }
        // SAFETY: `k < h` and `data.len() == 2 * h + 1` establish both
        // paired output indices.
        unsafe {
            *data.get_unchecked_mut(k + 1) =
                eunomia::Complex::new(base_re + delta_re, base_im + delta_im);
            *data.get_unchecked_mut(2 * h - k) =
                eunomia::Complex::new(base_re - delta_re, base_im - delta_im);
        }
        k += 1;
    }
}

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
        let a = unsafe { *data.get_unchecked(m + 1) };
        let b = unsafe { *data.get_unchecked(N - 1 - m) };
        let sum_re = a.re + b.re;
        let sum_im = a.im + b.im;
        y0_re += sum_re;
        y0_im += sum_im;
        unsafe {
            *sums_re.get_unchecked_mut(m) = sum_re;
            *sums_im.get_unchecked_mut(m) = sum_im;
            let diff_re = a.re - b.re;
            let diff_im = a.im - b.im;
            *idiffs_re.get_unchecked_mut(m) = -diff_im * sign;
            *idiffs_im.get_unchecked_mut(m) = diff_re * sign;
        }
    }

    data[0] = eunomia::Complex::new(y0_re, y0_im);

    for k in 0..H {
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        let cos_row = unsafe { cos.get_unchecked(k) };
        let sin_row = unsafe { sin.get_unchecked(k) };

        // Two-pass accumulation: compute base and delta contributions separately.
        // This improves instruction-level parallelism and CPU pipelining,
        // enabling better utilization of dual FMA units on modern out-of-order CPUs.
        for m in 0..H {
            let sr = unsafe { *sums_re.get_unchecked(m) };
            let si = unsafe { *sums_im.get_unchecked(m) };
            let c = unsafe { *cos_row.get_unchecked(m) };
            base_re += sr * c;
            base_im += si * c;
        }
        for m in 0..H {
            let ir = unsafe { *idiffs_re.get_unchecked(m) };
            let ii = unsafe { *idiffs_im.get_unchecked(m) };
            let s = unsafe { *sin_row.get_unchecked(m) };
            delta_re += ir * s;
            delta_im += ii * s;
        }

        unsafe {
            *data.get_unchecked_mut(k + 1) =
                eunomia::Complex::new(base_re + delta_re, base_im + delta_im);
            *data.get_unchecked_mut(N - 1 - k) =
                eunomia::Complex::new(base_re - delta_re, base_im - delta_im);
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
        let a = unsafe { *data.get_unchecked(m + 1) };
        let b = unsafe { *data.get_unchecked(N - 1 - m) };
        unsafe {
            *sums.get_unchecked_mut(m) = eunomia::Complex::new(a.re + b.re, a.im + b.im);
            let diff_re = a.re - b.re;
            let diff_im = a.im - b.im;
            *idiffs.get_unchecked_mut(m) = eunomia::Complex::new(-diff_im * sign, diff_re * sign);
        }
    }

    // DC bin: sum of all inputs × kernel_spectrum[0]
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    for s in &sums {
        y0_re += s.re;
        y0_im += s.im;
    }
    let ks0 = unsafe { *kernel_spectrum.get_unchecked(0) };
    data[0] = eunomia::Complex::new(y0_re, y0_im) * ks0;

    // Remaining bins: compute Winograd output then multiply by kernel_spectrum[k]
    for k in 0..H {
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        let cos_row = unsafe { cos.get_unchecked(k) };
        let sin_row = unsafe { sin.get_unchecked(k) };

        // Two-pass accumulation: compute base and delta contributions separately.
        // This improves instruction-level parallelism and CPU pipelining,
        // enabling better utilization of dual FMA units on modern out-of-order CPUs.
        for m in 0..H {
            let s_m = unsafe { *sums.get_unchecked(m) };
            let c = unsafe { *cos_row.get_unchecked(m) };
            base_re += s_m.re * c;
            base_im += s_m.im * c;
        }
        for m in 0..H {
            let id_m = unsafe { *idiffs.get_unchecked(m) };
            let s = unsafe { *sin_row.get_unchecked(m) };
            delta_re += id_m.re * s;
            delta_im += id_m.im * s;
        }

        let ks_kp1 = unsafe { *kernel_spectrum.get_unchecked(k + 1) };
        let ks_nmk = unsafe { *kernel_spectrum.get_unchecked(N - 1 - k) };

        unsafe {
            *data.get_unchecked_mut(k + 1) =
                eunomia::Complex::new(base_re + delta_re, base_im + delta_im) * ks_kp1;
            *data.get_unchecked_mut(N - 1 - k) =
                eunomia::Complex::new(base_re - delta_re, base_im - delta_im) * ks_nmk;
        }
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
