//! Runtime-dispatched pair-basis SIMD leaf for the 11-point DFT (f32).
//!
//! The same Winograd-pair basis [`super::odd_prime_pair::dft_pair_impl`]
//! uses — 11 inputs fold into one center value, five complex sums, and five
//! complex idiffs — but the five k-dots run in AVX2 lanes with FMA instead
//! of scalar code. The m-loop stays outer and ascending, so each lane keeps
//! the scalar summation order and only the multiply-add rounding changes
//! (one FMA rounding instead of two), well within f32 tolerance.
//!
//! The tables are the authoritative pair tables transposed once per process
//! into 8-wide columns (5 real k lanes plus 3 zero lanes), so the hot loop
//! issues contiguous loads with no per-call packing: 2 loads, 4 broadcasts,
//! and 4 FMAs per m, about 50 uops against the scalar form's ~400.
//!
//! Rationale: the pair form's short reductions are latency-bound at H = 5
//! (the scheduling-only wide form and a direct-11 matvec were both tried
//! and measured slower), while 121 costs exactly 22 leaf calls, so only a
//! lower-cost leaf moves N = 121. Bench builds carry no `target-cpu`, so
//! this stays behind the runtime AVX2+FMA probe in
//! [`dft11_dispatched`]; other targets run the scalar pair form.

use eunomia::Complex32;
use std::sync::LazyLock;

use super::odd_prime_pair::PrimePairTable;

/// Pair tables transposed for 8-wide k dots: `cos_t[m][k]` is
/// `cos_table[k][m]` for k < 5 with three zero lanes, likewise `sin_t`.
struct PairTablesT {
    cos_t: [[f32; 8]; 5],
    sin_t: [[f32; 8]; 5],
}

static PAIR11_T: LazyLock<PairTablesT> = LazyLock::new(|| {
    let cos = <f32 as PrimePairTable<11, 5>>::cos_table();
    let sin = <f32 as PrimePairTable<11, 5>>::sin_table();
    let mut t = PairTablesT {
        cos_t: [[0.0; 8]; 5],
        sin_t: [[0.0; 8]; 5],
    };
    for m in 0..5 {
        for k in 0..5 {
            t.cos_t[m][k] = cos[k][m];
            t.sin_t[m][k] = sin[k][m];
        }
    }
    t
});

/// 11-point f32 DFT through the fastest available leaf form.
#[inline]
pub(crate) fn dft11_dispatched<const INVERSE: bool>(
    data: &mut [Complex32; 11],
    cos: &[[f32; 5]; 5],
    sin: &[[f32; 5]; 5],
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA were just probed on this thread; the kernel
        // only reads/writes its typed array argument, all in bounds.
        unsafe {
            dft11_pair_simd::<INVERSE>(data);
        }
        return;
    }
    super::odd_prime_pair::dft_pair_impl::<f32, 11, 5, INVERSE>(data, cos, sin);
}

/// Pair-basis 11-point DFT with AVX2/FMA k dots.
///
/// # Safety
///
/// The caller must have established AVX2 and FMA support (see
/// [`dft11_dispatched`]). Memory traffic is scalar element reads of the
/// input, unaligned vector loads of the static transposed table columns
/// (each column is 8 in-bounds lanes), and unaligned vector stores into
/// local 8-element arrays. No over-reads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dft11_pair_simd<const INVERSE: bool>(data: &mut [Complex32; 11]) {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // Scalar prologue, identical to `dft_pair_impl` at N = 11, H = 5.
    let x0 = data[0];
    let mut sums = [Complex32::new(0.0, 0.0); 5];
    let mut idiffs = [Complex32::new(0.0, 0.0); 5];
    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    let sign = if INVERSE { 1.0f32 } else { -1.0f32 };
    for m in 0..5 {
        let a = data[m + 1];
        let b = data[10 - m];
        let sum_re = a.re + b.re;
        let sum_im = a.im + b.im;
        y0_re += sum_re;
        y0_im += sum_im;
        sums[m] = Complex32::new(sum_re, sum_im);
        let diff_re = a.re - b.re;
        let diff_im = a.im - b.im;
        idiffs[m] = Complex32::new(-diff_im * sign, diff_re * sign);
    }
    data[0] = Complex32::new(y0_re, y0_im);

    // SAFETY: AVX2+FMA are established by the caller; every load/store
    // below names typed in-bounds lanes, and the `u` intrinsics permit
    // unaligned addresses.
    unsafe {
        let t = &*PAIR11_T;
        let mut acc_bre = _mm256_setzero_ps();
        let mut acc_bim = _mm256_setzero_ps();
        let mut acc_dre = _mm256_setzero_ps();
        let mut acc_dim = _mm256_setzero_ps();
        for m in 0..5 {
            let tc = _mm256_loadu_ps(t.cos_t[m].as_ptr());
            let ts = _mm256_loadu_ps(t.sin_t[m].as_ptr());
            let s = sums[m];
            let d = idiffs[m];
            // Each dot is a pure multiply-accumulate, so one FMA per
            // lane does the whole update with no separate multiply.
            acc_bre = _mm256_fmadd_ps(_mm256_set1_ps(s.re), tc, acc_bre);
            acc_bim = _mm256_fmadd_ps(_mm256_set1_ps(s.im), tc, acc_bim);
            acc_dre = _mm256_fmadd_ps(_mm256_set1_ps(d.re), ts, acc_dre);
            acc_dim = _mm256_fmadd_ps(_mm256_set1_ps(d.im), ts, acc_dim);
        }
        let mut bre = [0.0f32; 8];
        let mut bim = [0.0f32; 8];
        let mut dre = [0.0f32; 8];
        let mut dim = [0.0f32; 8];
        _mm256_storeu_ps(bre.as_mut_ptr(), acc_bre);
        _mm256_storeu_ps(bim.as_mut_ptr(), acc_bim);
        _mm256_storeu_ps(dre.as_mut_ptr(), acc_dre);
        _mm256_storeu_ps(dim.as_mut_ptr(), acc_dim);

        for k in 0..5 {
            let base_re = x0.re + bre[k];
            let base_im = x0.im + bim[k];
            let delta_re = dre[k];
            let delta_im = dim[k];
            data[k + 1] = Complex32::new(base_re + delta_re, base_im + delta_im);
            data[10 - k] = Complex32::new(base_re - delta_re, base_im - delta_im);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::dft11_dispatched;
    use crate::application::execution::kernel::components::winograd::radix::odd_prime_pair::PrimePairTable;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
    use eunomia::{Complex32, Complex64};

    fn input(kind: usize) -> [Complex32; 11] {
        match kind {
            0 => [Complex32::new(0.0, 0.0); 11],
            1 => {
                let mut impulse = [Complex32::new(0.0, 0.0); 11];
                impulse[3] = Complex32::new(1.0, -1.0);
                impulse
            }
            2 => core::array::from_fn(|i| Complex32::new(i as f32 / 11.0, -(i as f32) / 11.0)),
            3 => core::array::from_fn(|i| {
                let x = i as f64 * 0.173;
                Complex32::new((x.sin() * 3.0 - 0.5) as f32, (x.cos() * 1.5) as f32)
            }),
            4 => core::array::from_fn(|i| {
                Complex32::new(
                    if i % 2 == 0 { 1.0 } else { -1.0 },
                    if i % 3 == 0 { 0.5 } else { -0.5 },
                )
            }),
            _ => core::array::from_fn(|i| {
                let x = i as f64 * 0.617;
                Complex32::new(x.cos() as f32, x.sin() as f32)
            }),
        }
    }

    fn as_f64(buf: &[Complex32; 11]) -> Vec<Complex64> {
        buf.iter()
            .map(|x| Complex64::new(f64::from(x.re), f64::from(x.im)))
            .collect()
    }

    fn max_err(got: &[Complex64], expected: &[Complex64]) -> f64 {
        got.iter().zip(expected).fold(0.0f64, |acc, (g, e)| {
            let d = ((g.re - e.re) * (g.re - e.re) + (g.im - e.im) * (g.im - e.im)).sqrt();
            acc.max(d)
        })
    }

    #[test]
    fn dispatched_dft11_matches_direct_within_tolerance() {
        for kind in 0..6 {
            let signal = input(kind);
            let signal64 = as_f64(&signal);
            let expected = dft_forward(&signal64);
            let mut candidate = signal;
            dft11_dispatched::<false>(
                &mut candidate,
                <f32 as PrimePairTable<11, 5>>::cos_table(),
                <f32 as PrimePairTable<11, 5>>::sin_table(),
            );
            let err = max_err(&as_f64(&candidate), &expected);
            assert!(err < 1e-5, "dft11 forward kind={kind} max_err={err:.2e}");

            let expected_inv: Vec<Complex64> = dft_inverse(&signal64)
                .into_iter()
                .map(|x| x * 11.0)
                .collect();
            let mut candidate_inv = signal;
            dft11_dispatched::<true>(
                &mut candidate_inv,
                <f32 as PrimePairTable<11, 5>>::cos_table(),
                <f32 as PrimePairTable<11, 5>>::sin_table(),
            );
            let err_inv = max_err(&as_f64(&candidate_inv), &expected_inv);
            assert!(
                err_inv < 1e-5,
                "dft11 inverse kind={kind} max_err={err_inv:.2e}"
            );
        }
    }
}
