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
        let cos_row = unsafe { cos.get_unchecked(k) };
        let sin_row = unsafe { sin.get_unchecked(k) };

        // Two-pass accumulation: compute base and delta contributions separately.
        // This improves instruction-level parallelism and CPU pipelining,
        // enabling better utilization of dual FMA units on modern out-of-order CPUs.
        let mut base_re = x0.re;
        let mut base_im = x0.im;
        let mut delta_re = zero;
        let mut delta_im = zero;

        for m in 0..H {
            let s_m = unsafe { *sums.get_unchecked(m) };
            let c = unsafe { *cos_row.get_unchecked(m) };
            base_re = base_re + s_m.re * c;
            base_im = base_im + s_m.im * c;
        }
        for m in 0..H {
            let id_m = unsafe { *idiffs.get_unchecked(m) };
            let s = unsafe { *sin_row.get_unchecked(m) };
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

#[inline(always)]
pub(crate) fn dft_pair_impl_reduced<
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
    let mut sums_re = [zero; H];
    let mut sums_im = [zero; H];
    let mut idiffs_re = [zero; H];
    let mut idiffs_im = [zero; H];

    let sign = if INVERSE { F::one() } else { -F::one() };
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

    data[0] = num_complex::Complex::new(y0_re, y0_im);

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
pub(crate) fn dft_pair_forward_with_pointwise<F: WinogradScalar, const N: usize, const H: usize>(
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

        // Two-pass accumulation: compute base and delta contributions separately.
        // This improves instruction-level parallelism and CPU pipelining,
        // enabling better utilization of dual FMA units on modern out-of-order CPUs.
        for m in 0..H {
            let s_m = unsafe { *sums.get_unchecked(m) };
            let c = unsafe { *cos_row.get_unchecked(m) };
            base_re = base_re + s_m.re * c;
            base_im = base_im + s_m.im * c;
        }
        for m in 0..H {
            let id_m = unsafe { *idiffs.get_unchecked(m) };
            let s = unsafe { *sin_row.get_unchecked(m) };
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

        // Final accumulator values that will be written to output.
        // Declared in outer scope so they're accessible after the if/else block.
        let (even_base_re, even_base_im, odd_base_re, odd_base_im,
             even_delta_re, even_delta_im, odd_delta_re, odd_delta_im) =
            if H >= 18 {
                // 4-way loop unrolling for H >= 18 (N >= 37) to maximize ILP.
                // With 4 independent accumulator chains (_a, _b, _c, _d) each
                // processing one quarter of the m-values, enabling better CPU
                // utilization across dual-FMA units on modern out-of-order CPUs.
                //
                // _a: m=0,4,8,...  _b: m=1,5,9,...  _c: m=2,6,10,...  _d: m=3,7,11,...
                let mut even_base_re_a = even_x0.re;
                let mut even_base_re_b = zero;
                let mut even_base_re_c = zero;
                let mut even_base_re_d = zero;
                let mut even_base_im_a = even_x0.im;
                let mut even_base_im_b = zero;
                let mut even_base_im_c = zero;
                let mut even_base_im_d = zero;
                let mut odd_base_re_a = odd_x0.re;
                let mut odd_base_re_b = zero;
                let mut odd_base_re_c = zero;
                let mut odd_base_re_d = zero;
                let mut odd_base_im_a = odd_x0.im;
                let mut odd_base_im_b = zero;
                let mut odd_base_im_c = zero;
                let mut odd_base_im_d = zero;

                // Base pass: 4-way unrolled
                let mut m = 0usize;
                while m + 4 <= H {
                    let c0 = cos_row[m];
                    let c1 = cos_row[m + 1];
                    let c2 = cos_row[m + 2];
                    let c3 = cos_row[m + 3];
                    let e_s0 = even_sums[m];
                    let e_s1 = even_sums[m + 1];
                    let e_s2 = even_sums[m + 2];
                    let e_s3 = even_sums[m + 3];
                    let o_s0 = odd_sums[m];
                    let o_s1 = odd_sums[m + 1];
                    let o_s2 = odd_sums[m + 2];
                    let o_s3 = odd_sums[m + 3];

                    even_base_re_a = even_base_re_a + e_s0.re * c0;
                    even_base_re_b = even_base_re_b + e_s1.re * c1;
                    even_base_re_c = even_base_re_c + e_s2.re * c2;
                    even_base_re_d = even_base_re_d + e_s3.re * c3;
                    even_base_im_a = even_base_im_a + e_s0.im * c0;
                    even_base_im_b = even_base_im_b + e_s1.im * c1;
                    even_base_im_c = even_base_im_c + e_s2.im * c2;
                    even_base_im_d = even_base_im_d + e_s3.im * c3;
                    odd_base_re_a = odd_base_re_a + o_s0.re * c0;
                    odd_base_re_b = odd_base_re_b + o_s1.re * c1;
                    odd_base_re_c = odd_base_re_c + o_s2.re * c2;
                    odd_base_re_d = odd_base_re_d + o_s3.re * c3;
                    odd_base_im_a = odd_base_im_a + o_s0.im * c0;
                    odd_base_im_b = odd_base_im_b + o_s1.im * c1;
                    odd_base_im_c = odd_base_im_c + o_s2.im * c2;
                    odd_base_im_d = odd_base_im_d + o_s3.im * c3;

                    m += 4;
                }
                // Handle remaining m values (0, 1, 2, or 3 iterations)
                if m < H {
                    let c = cos_row[m];
                    let e_s = even_sums[m];
                    let o_s = odd_sums[m];
                    even_base_re_a = even_base_re_a + e_s.re * c;
                    even_base_im_a = even_base_im_a + e_s.im * c;
                    odd_base_re_a = odd_base_re_a + o_s.re * c;
                    odd_base_im_a = odd_base_im_a + o_s.im * c;
                }
                m += 1;
                if m < H {
                    let c = cos_row[m];
                    let e_s = even_sums[m];
                    let o_s = odd_sums[m];
                    even_base_re_b = even_base_re_b + e_s.re * c;
                    even_base_im_b = even_base_im_b + e_s.im * c;
                    odd_base_re_b = odd_base_re_b + o_s.re * c;
                    odd_base_im_b = odd_base_im_b + o_s.im * c;
                }
                m += 1;
                if m < H {
                    let c = cos_row[m];
                    let e_s = even_sums[m];
                    let o_s = odd_sums[m];
                    even_base_re_c = even_base_re_c + e_s.re * c;
                    even_base_im_c = even_base_im_c + e_s.im * c;
                    odd_base_re_c = odd_base_re_c + o_s.re * c;
                    odd_base_im_c = odd_base_im_c + o_s.im * c;
                }

                // Delta pass: 4-way unrolled
                let mut even_delta_re_a = zero;
                let mut even_delta_re_b = zero;
                let mut even_delta_re_c = zero;
                let mut even_delta_re_d = zero;
                let mut even_delta_im_a = zero;
                let mut even_delta_im_b = zero;
                let mut even_delta_im_c = zero;
                let mut even_delta_im_d = zero;
                let mut odd_delta_re_a = zero;
                let mut odd_delta_re_b = zero;
                let mut odd_delta_re_c = zero;
                let mut odd_delta_re_d = zero;
                let mut odd_delta_im_a = zero;
                let mut odd_delta_im_b = zero;
                let mut odd_delta_im_c = zero;
                let mut odd_delta_im_d = zero;

                m = 0usize;
                while m + 4 <= H {
                    let s0 = sin_row[m];
                    let s1 = sin_row[m + 1];
                    let s2 = sin_row[m + 2];
                    let s3 = sin_row[m + 3];
                    let e_id0 = even_idiffs[m];
                    let e_id1 = even_idiffs[m + 1];
                    let e_id2 = even_idiffs[m + 2];
                    let e_id3 = even_idiffs[m + 3];
                    let o_id0 = odd_idiffs[m];
                    let o_id1 = odd_idiffs[m + 1];
                    let o_id2 = odd_idiffs[m + 2];
                    let o_id3 = odd_idiffs[m + 3];

                    even_delta_re_a = even_delta_re_a + e_id0.re * s0;
                    even_delta_re_b = even_delta_re_b + e_id1.re * s1;
                    even_delta_re_c = even_delta_re_c + e_id2.re * s2;
                    even_delta_re_d = even_delta_re_d + e_id3.re * s3;
                    even_delta_im_a = even_delta_im_a + e_id0.im * s0;
                    even_delta_im_b = even_delta_im_b + e_id1.im * s1;
                    even_delta_im_c = even_delta_im_c + e_id2.im * s2;
                    even_delta_im_d = even_delta_im_d + e_id3.im * s3;
                    odd_delta_re_a = odd_delta_re_a + o_id0.re * s0;
                    odd_delta_re_b = odd_delta_re_b + o_id1.re * s1;
                    odd_delta_re_c = odd_delta_re_c + o_id2.re * s2;
                    odd_delta_re_d = odd_delta_re_d + o_id3.re * s3;
                    odd_delta_im_a = odd_delta_im_a + o_id0.im * s0;
                    odd_delta_im_b = odd_delta_im_b + o_id1.im * s1;
                    odd_delta_im_c = odd_delta_im_c + o_id2.im * s2;
                    odd_delta_im_d = odd_delta_im_d + o_id3.im * s3;

                    m += 4;
                }
                if m < H {
                    let s = sin_row[m];
                    let e_id = even_idiffs[m];
                    let o_id = odd_idiffs[m];
                    even_delta_re_a = even_delta_re_a + e_id.re * s;
                    even_delta_im_a = even_delta_im_a + e_id.im * s;
                    odd_delta_re_a = odd_delta_re_a + o_id.re * s;
                    odd_delta_im_a = odd_delta_im_a + o_id.im * s;
                }
                m += 1;
                if m < H {
                    let s = sin_row[m];
                    let e_id = even_idiffs[m];
                    let o_id = odd_idiffs[m];
                    even_delta_re_b = even_delta_re_b + e_id.re * s;
                    even_delta_im_b = even_delta_im_b + e_id.im * s;
                    odd_delta_re_b = odd_delta_re_b + o_id.re * s;
                    odd_delta_im_b = odd_delta_im_b + o_id.im * s;
                }
                m += 1;
                if m < H {
                    let s = sin_row[m];
                    let e_id = even_idiffs[m];
                    let o_id = odd_idiffs[m];
                    even_delta_re_c = even_delta_re_c + e_id.re * s;
                    even_delta_im_c = even_delta_im_c + e_id.im * s;
                    odd_delta_re_c = odd_delta_re_c + o_id.re * s;
                    odd_delta_im_c = odd_delta_im_c + o_id.im * s;
                }

                // Combine 4 accumulator chains
                (
                    even_base_re_a + even_base_re_b + even_base_re_c + even_base_re_d,
                    even_base_im_a + even_base_im_b + even_base_im_c + even_base_im_d,
                    odd_base_re_a + odd_base_re_b + odd_base_re_c + odd_base_re_d,
                    odd_base_im_a + odd_base_im_b + odd_base_im_c + odd_base_im_d,
                    even_delta_re_a + even_delta_re_b + even_delta_re_c + even_delta_re_d,
                    even_delta_im_a + even_delta_im_b + even_delta_im_c + even_delta_im_d,
                    odd_delta_re_a + odd_delta_re_b + odd_delta_re_c + odd_delta_re_d,
                    odd_delta_im_a + odd_delta_im_b + odd_delta_im_c + odd_delta_im_d,
                )
            } else {
                // 2-way loop unrolling for smaller H (< 18) to avoid register pressure.
                // _a: m=0,2,4,...  _b: m=1,3,5,...
                let mut even_base_re_a = even_x0.re;
                let mut even_base_re_b = zero;
                let mut even_base_im_a = even_x0.im;
                let mut even_base_im_b = zero;
                let mut odd_base_re_a = odd_x0.re;
                let mut odd_base_re_b = zero;
                let mut odd_base_im_a = odd_x0.im;
                let mut odd_base_im_b = zero;

                // Base pass: 2-way unrolled
                let mut m = 0usize;
                while m + 2 <= H {
                    let c0 = cos_row[m];
                    let c1 = cos_row[m + 1];
                    let e_s0 = even_sums[m];
                    let e_s1 = even_sums[m + 1];
                    let o_s0 = odd_sums[m];
                    let o_s1 = odd_sums[m + 1];

                    even_base_re_a = even_base_re_a + e_s0.re * c0;
                    even_base_re_b = even_base_re_b + e_s1.re * c1;
                    even_base_im_a = even_base_im_a + e_s0.im * c0;
                    even_base_im_b = even_base_im_b + e_s1.im * c1;
                    odd_base_re_a = odd_base_re_a + o_s0.re * c0;
                    odd_base_re_b = odd_base_re_b + o_s1.re * c1;
                    odd_base_im_a = odd_base_im_a + o_s0.im * c0;
                    odd_base_im_b = odd_base_im_b + o_s1.im * c1;

                    m += 2;
                }
                // Handle remaining m if H is odd
                if m < H {
                    let c = cos_row[m];
                    let e_s = even_sums[m];
                    let o_s = odd_sums[m];
                    even_base_re_a = even_base_re_a + e_s.re * c;
                    even_base_im_a = even_base_im_a + e_s.im * c;
                    odd_base_re_a = odd_base_re_a + o_s.re * c;
                    odd_base_im_a = odd_base_im_a + o_s.im * c;
                }

                // Delta pass: 2-way unrolled
                let mut even_delta_re_a = zero;
                let mut even_delta_re_b = zero;
                let mut even_delta_im_a = zero;
                let mut even_delta_im_b = zero;
                let mut odd_delta_re_a = zero;
                let mut odd_delta_re_b = zero;
                let mut odd_delta_im_a = zero;
                let mut odd_delta_im_b = zero;

                m = 0usize;
                while m + 2 <= H {
                    let s0 = sin_row[m];
                    let s1 = sin_row[m + 1];
                    let e_id0 = even_idiffs[m];
                    let e_id1 = even_idiffs[m + 1];
                    let o_id0 = odd_idiffs[m];
                    let o_id1 = odd_idiffs[m + 1];

                    even_delta_re_a = even_delta_re_a + e_id0.re * s0;
                    even_delta_re_b = even_delta_re_b + e_id1.re * s1;
                    even_delta_im_a = even_delta_im_a + e_id0.im * s0;
                    even_delta_im_b = even_delta_im_b + e_id1.im * s1;
                    odd_delta_re_a = odd_delta_re_a + o_id0.re * s0;
                    odd_delta_re_b = odd_delta_re_b + o_id1.re * s1;
                    odd_delta_im_a = odd_delta_im_a + o_id0.im * s0;
                    odd_delta_im_b = odd_delta_im_b + o_id1.im * s1;

                    m += 2;
                }
                if m < H {
                    let s = sin_row[m];
                    let e_id = even_idiffs[m];
                    let o_id = odd_idiffs[m];
                    even_delta_re_a = even_delta_re_a + e_id.re * s;
                    even_delta_im_a = even_delta_im_a + e_id.im * s;
                    odd_delta_re_a = odd_delta_re_a + o_id.re * s;
                    odd_delta_im_a = odd_delta_im_a + o_id.im * s;
                }

                // Combine 2 accumulator chains
                (
                    even_base_re_a + even_base_re_b,
                    even_base_im_a + even_base_im_b,
                    odd_base_re_a + odd_base_re_b,
                    odd_base_im_a + odd_base_im_b,
                    even_delta_re_a + even_delta_re_b,
                    even_delta_im_a + even_delta_im_b,
                    odd_delta_re_a + odd_delta_re_b,
                    odd_delta_im_a + odd_delta_im_b,
                )
            };

        let lo = k + 1;
        let hi = P - 1 - k;

        // Batch-load both twiddles for this iteration and prefetch next iteration's twiddles.
        // This reduces pointer arithmetic overhead and enables better memory parallelism.
        //
        // For iteration k:
        //   - Load twiddles[lo] and twiddles[hi] for current iteration
        //   - Prefetch twiddles[k+2] and twiddles[P-2-k] for next iteration (k+1)
        // The prefetch ensures twiddles are in L1 cache before the next iteration starts.
        let tw_lo = unsafe { *twiddle_ptr.add(lo) };
        let tw_hi = unsafe { *twiddle_ptr.add(hi) };

        // Prefetch twiddles for next iteration when k < H-1
        if k < H - 1 {
            let next_lo = k + 2;
            let next_hi = P - 2 - k;
            // Use _mm_prefetch to bring data into L1 cache
            #[cfg(target_feature = "sse2")]
            {
                use std::arch::x86_64::_mm_prefetch;
                unsafe {
                    _mm_prefetch(twiddle_ptr.add(next_lo) as *const i8, 0);
                    _mm_prefetch(twiddle_ptr.add(next_hi) as *const i8, 0);
                }
            }
        }

        let even_lo =
            num_complex::Complex::new(even_base_re + even_delta_re, even_base_im + even_delta_im);
        let odd_lo =
            num_complex::Complex::new(odd_base_re + odd_delta_re, odd_base_im + odd_delta_im);
        let wb_lo = tw_lo * odd_lo;
        data[lo] = even_lo + wb_lo;
        data[lo + P] = even_lo - wb_lo;

        let even_hi =
            num_complex::Complex::new(even_base_re - even_delta_re, even_base_im - even_delta_im);
        let odd_hi =
            num_complex::Complex::new(odd_base_re - odd_delta_re, odd_base_im - odd_delta_im);
        let wb_hi = tw_hi * odd_hi;
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
