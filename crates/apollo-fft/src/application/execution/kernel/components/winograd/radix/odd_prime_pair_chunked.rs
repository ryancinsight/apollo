//! Separated-stream 11-point DFTs and the chunked N = 121 driver (f32).
//!
//! [`super::odd_prime_pair_simd`] vectorizes one 11-point DFT across its
//! five k lanes. N = 121 needs 22 such calls, and each call reloads the
//! pair tables, re-broadcasts every sum, and pays a full scalar
//! prologue/epilogue. The separated-stream form instead runs 11 DFTs
//! together: one scalar prologue per stream gathers sums/idiffs into a
//! stream-major working set, the k dots vectorize across 8 streams plus a
//! 3-lane scalar tail, and one scalar epilogue per stream scatters the
//! results. Table traffic drops 11x and the broadcasts become contiguous
//! vector loads.
//!
//! [`dft121_chunked`] drives N = 121 as two chunked-11 calls (columns,
//! then rows) around an AVX2 complex twiddle stage, so the whole transform
//! issues 2 leaf dispatches instead of 22. Each stream keeps the scalar
//! m-ascending summation order; only multiply-add rounding changes (one
//! FMA rounding instead of two), within the same f32 tolerance the
//! single-call leaf documents.

use eunomia::Complex32;

use super::odd_prime_pair::PrimePairTable;

/// Test hook running 11 strided 11-point DFTs through one AVX2/FMA dispatch.
///
/// Stream `s` reads `in_starts[s] + i * in_step` and writes
/// `out_starts[s] + i * out_step` for `i` in `0..11`; all 121 input slots
/// must be in bounds and pairwise disjoint, and likewise the outputs.
/// Returns `false` without touching `base` when the host lacks AVX2+FMA
/// (or off x86-64). Production callers go through [`try_dft121_chunked`],
/// which probes once for the whole transform.
#[cfg(test)]
#[inline]
pub(crate) fn try_dft11_streams<const INVERSE: bool>(
    base: &mut [Complex32],
    in_starts: &[usize; 11],
    in_step: usize,
    out_starts: &[usize; 11],
    out_step: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA were just probed on this thread; the caller
        // guarantees the 121 input and 121 output slots as documented.
        unsafe {
            dft11_streams_simd::<INVERSE>(base, in_starts, in_step, out_starts, out_step);
        }
        return true;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (base, in_starts, in_step, out_starts, out_step);
    false
}

/// Run the N = 121 transform through the separated-stream formulation.
///
/// Returns `false` without touching `data` when the host lacks AVX2+FMA
/// (or off x86-64), so the caller falls back to the generated codelet.
#[inline]
pub(crate) fn try_dft121_chunked<const INVERSE: bool>(data: &mut [Complex32; 121]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA were just probed on this thread; the driver
        // only reads/writes its typed array argument, all in bounds.
        unsafe {
            dft121_chunked_simd::<INVERSE>(data);
        }
        return true;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = data;
    false
}

/// Pair tables in (m, k) orientation for stream-major dots: `cos_mk[m][k]`
/// is `cos_table[k][m]`, so the inner k loop walks one contiguous row.
struct PairTablesMk {
    cos_mk: [[f32; 5]; 5],
    sin_mk: [[f32; 5]; 5],
}

static PAIR11_MK: std::sync::LazyLock<PairTablesMk> = std::sync::LazyLock::new(|| {
    let cos = <f32 as PrimePairTable<11, 5>>::cos_table();
    let sin = <f32 as PrimePairTable<11, 5>>::sin_table();
    let mut t = PairTablesMk {
        cos_mk: [[0.0; 5]; 5],
        sin_mk: [[0.0; 5]; 5],
    };
    for m in 0..5 {
        for k in 0..5 {
            t.cos_mk[m][k] = cos[k][m];
            t.sin_mk[m][k] = sin[k][m];
        }
    }
    t
});

/// 11 stream-major pair-basis DFTs with AVX2/FMA k dots.
///
/// # Safety
///
/// The caller must have established AVX2 and FMA support and must
/// guarantee the 121 input and 121 output slots documented on
/// [`try_dft11_streams`]. All reads complete before any write, so
/// differing input/output layouts are safe in place.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dft11_streams_simd<const INVERSE: bool>(
    base: &mut [Complex32],
    in_starts: &[usize; 11],
    in_step: usize,
    out_starts: &[usize; 11],
    out_step: usize,
) {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let sign = if INVERSE { 1.0f32 } else { -1.0f32 };
    // Stream-major working set: sums_re[m][s] is stream s, pair m.
    let mut sums_re = [[0.0f32; 11]; 5];
    let mut sums_im = [[0.0f32; 11]; 5];
    let mut idiffs_re = [[0.0f32; 11]; 5];
    let mut idiffs_im = [[0.0f32; 11]; 5];
    let mut x0 = [Complex32::new(0.0, 0.0); 11];
    let mut y0 = [Complex32::new(0.0, 0.0); 11];

    // Scalar prologue per stream, identical to `dft_pair_impl` at N = 11.
    // Reads only; every write below lands in the working set.
    // SAFETY: the caller guarantees all 121 stream slots in bounds, and
    // the (m + 1) / (10 - m) indices stay below 11.
    unsafe {
        for s in 0..11 {
            let start = in_starts[s];
            let c0 = *base.get_unchecked(start);
            x0[s] = c0;
            let mut y0_re = c0.re;
            let mut y0_im = c0.im;
            for m in 0..5 {
                let a = *base.get_unchecked(start + (m + 1) * in_step);
                let b = *base.get_unchecked(start + (10 - m) * in_step);
                let sum_re = a.re + b.re;
                let sum_im = a.im + b.im;
                y0_re += sum_re;
                y0_im += sum_im;
                sums_re[m][s] = sum_re;
                sums_im[m][s] = sum_im;
                idiffs_re[m][s] = -(a.im - b.im) * sign;
                idiffs_im[m][s] = (a.re - b.re) * sign;
            }
            y0[s] = Complex32::new(y0_re, y0_im);
        }
    }

    // Scalar tail accumulators for streams 8..11.
    let mut tail_bre = [[0.0f32; 3]; 5];
    let mut tail_bim = [[0.0f32; 3]; 5];
    let mut tail_dre = [[0.0f32; 3]; 5];
    let mut tail_dim = [[0.0f32; 3]; 5];
    let mut bre = [[0.0f32; 8]; 5];
    let mut bim = [[0.0f32; 8]; 5];
    let mut dre = [[0.0f32; 8]; 5];
    let mut dim = [[0.0f32; 8]; 5];

    // SAFETY: AVX2+FMA are established by the caller; every load/store
    // below names typed in-bounds lanes of the working set, and the `u`
    // intrinsics permit unaligned addresses. The k-outer order keeps 10
    // live vector registers (4 accumulators, 4 stream loads, 2 table
    // broadcasts), spill-free on AVX2; each stream still sums m-ascending.
    unsafe {
        let t = &*PAIR11_MK;
        for k in 0..5 {
            let mut acc_bre = _mm256_setzero_ps();
            let mut acc_bim = _mm256_setzero_ps();
            let mut acc_dre = _mm256_setzero_ps();
            let mut acc_dim = _mm256_setzero_ps();
            for m in 0..5 {
                // Contiguous 8-wide stream loads; the tail stays scalar.
                let sre = _mm256_loadu_ps(sums_re[m].as_ptr());
                let sim = _mm256_loadu_ps(sums_im[m].as_ptr());
                let dre = _mm256_loadu_ps(idiffs_re[m].as_ptr());
                let dim = _mm256_loadu_ps(idiffs_im[m].as_ptr());
                let tc = _mm256_set1_ps(t.cos_mk[m][k]);
                let ts = _mm256_set1_ps(t.sin_mk[m][k]);
                acc_bre = _mm256_fmadd_ps(sre, tc, acc_bre);
                acc_bim = _mm256_fmadd_ps(sim, tc, acc_bim);
                acc_dre = _mm256_fmadd_ps(dre, ts, acc_dre);
                acc_dim = _mm256_fmadd_ps(dim, ts, acc_dim);
                let ck = t.cos_mk[m][k];
                let sk = t.sin_mk[m][k];
                for s in 0..3 {
                    tail_bre[k][s] += sums_re[m][8 + s] * ck;
                    tail_bim[k][s] += sums_im[m][8 + s] * ck;
                    tail_dre[k][s] += idiffs_re[m][8 + s] * sk;
                    tail_dim[k][s] += idiffs_im[m][8 + s] * sk;
                }
            }
            _mm256_storeu_ps(bre[k].as_mut_ptr(), acc_bre);
            _mm256_storeu_ps(bim[k].as_mut_ptr(), acc_bim);
            _mm256_storeu_ps(dre[k].as_mut_ptr(), acc_dre);
            _mm256_storeu_ps(dim[k].as_mut_ptr(), acc_dim);
        }
    }

    // Scalar epilogue per stream; the only writes to `base`.
    // SAFETY: the caller guarantees all 121 output slots in bounds, and
    // the (k + 1) / (10 - k) indices stay below 11.
    unsafe {
        for s in 0..11 {
            let start = out_starts[s];
            let c0 = x0[s];
            *base.get_unchecked_mut(start) = y0[s];
            for k in 0..5 {
                let (bre_k, bim_k, dre_k, dim_k) = if s < 8 {
                    (bre[k][s], bim[k][s], dre[k][s], dim[k][s])
                } else {
                    (
                        tail_bre[k][s - 8],
                        tail_bim[k][s - 8],
                        tail_dre[k][s - 8],
                        tail_dim[k][s - 8],
                    )
                };
                let base_re = c0.re + bre_k;
                let base_im = c0.im + bim_k;
                *base.get_unchecked_mut(start + (k + 1) * out_step) =
                    Complex32::new(base_re + dre_k, base_im + dim_k);
                *base.get_unchecked_mut(start + (10 - k) * out_step) =
                    Complex32::new(base_re - dre_k, base_im - dim_k);
            }
        }
    }
}

/// Forward Cooley-Tukey twiddles `W_121^{k1 * j}` in (k1, j) order.
///
/// Computed once from f64 sin/cos, matching the generated codelet's
/// `from_precise` constants; the inverse pass conjugates on load.
static TWIDDLES121: std::sync::LazyLock<[[Complex32; 11]; 11]> = std::sync::LazyLock::new(|| {
    core::array::from_fn(|k1| {
        core::array::from_fn(|j| {
            let angle = -2.0 * core::f64::consts::PI * ((k1 * j) as f64) / 121.0;
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        })
    })
});

/// N = 121 through two chunked-11 calls around a vector twiddle stage.
///
/// # Safety
///
/// The caller must have established AVX2 and FMA support. Only the typed
/// array argument is read/written, all in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dft121_chunked_simd<const INVERSE: bool>(data: &mut [Complex32; 121]) {
    use core::arch::x86_64::{
        _mm256_blend_ps, _mm256_fmadd_ps, _mm256_fmsub_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_shuffle_ps, _mm256_storeu_ps, _mm256_xor_ps,
    };

    // Columns: stream j reads data[j + 11 * i], in place.
    let col_starts: [usize; 11] = core::array::from_fn(|j| j);
    // SAFETY: AVX2+FMA are established by the caller; the 11 column
    // extents partition the 121 slots.
    unsafe {
        dft11_streams_simd::<INVERSE>(data, &col_starts, 11, &col_starts, 11);
    }

    // Twiddle stage: W_121^{k1 * j} over rows k1, vectorized across j.
    // The j = 0 column and k1 = 0 row are identities and stay untouched.
    // Each 4-complex block duplicates re/im lanes in place, evaluates the
    // same (re*tre - im*tim, re*tim + im*tre) formula as the scalar path
    // with one FMA rounding per lane, and blends back to interleaved.
    let tw = &*TWIDDLES121;
    // SAFETY: same establishment; every access below names typed in-bounds
    // lanes of `data` and the static table.
    unsafe {
        let sign_flip = _mm256_set1_ps(-0.0f32);
        for k1 in 1..11 {
            let row = &mut data[k1 * 11..k1 * 11 + 11];
            let twiddles = &tw[k1];
            for block in 0..2 {
                let base = 1 + block * 4;
                let values = _mm256_loadu_ps(row[base..].as_ptr().cast::<f32>());
                let factors = _mm256_loadu_ps(twiddles[base..].as_ptr().cast::<f32>());
                let vre = _mm256_shuffle_ps::<0xA0>(values, values);
                let vim = _mm256_shuffle_ps::<0xF5>(values, values);
                let tre = _mm256_shuffle_ps::<0xA0>(factors, factors);
                let mut tim = _mm256_shuffle_ps::<0xF5>(factors, factors);
                if INVERSE {
                    tim = _mm256_xor_ps(tim, sign_flip);
                }
                // real = re*tre - im*tim, imag = re*tim + im*tre.
                let real = _mm256_fmsub_ps(vre, tre, _mm256_mul_ps(vim, tim));
                let imag = _mm256_fmadd_ps(vre, tim, _mm256_mul_ps(vim, tre));
                let out = _mm256_blend_ps::<0xAA>(real, imag);
                _mm256_storeu_ps(row[base..].as_mut_ptr().cast::<f32>(), out);
            }
            // Scalar tail for j = 9, 10.
            for j in 9..11 {
                let (w_re, mut w_im) = (twiddles[j].re, twiddles[j].im);
                if INVERSE {
                    w_im = -w_im;
                }
                let value = row[j];
                row[j] = Complex32::new(
                    value.re * w_re - value.im * w_im,
                    value.re * w_im + value.im * w_re,
                );
            }
        }
    }

    // Rows: stream k1 reads data[k1 * 11 + j], writes the CT-transposed
    // output data[j * 11 + k1] directly.
    let row_in: [usize; 11] = core::array::from_fn(|k1| k1 * 11);
    let row_out: [usize; 11] = core::array::from_fn(|k1| k1);
    // SAFETY: same establishment; input rows partition the slots and the
    // transposed output writes are a permutation of the same 121 slots.
    unsafe {
        dft11_streams_simd::<INVERSE>(data, &row_in, 1, &row_out, 11);
    }
}

#[cfg(test)]
mod tests {
    use super::{try_dft11_streams, try_dft121_chunked};
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
    use eunomia::{Complex32, Complex64};

    fn input_11(kind: usize) -> [Complex32; 11] {
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

    fn as_f64(buf: &[Complex32]) -> Vec<Complex64> {
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
    fn chunked_streams_match_direct_within_tolerance() {
        for kind in 0..6 {
            // 11 distinct streams: contiguous layout, stride 11.
            let mut streams = [[Complex32::new(0.0, 0.0); 11]; 11];
            for (s, stream) in streams.iter_mut().enumerate() {
                *stream = input_11((kind + s) % 6);
            }
            let flat: Vec<Complex32> = streams.iter().flatten().copied().collect();
            let expected: Vec<Complex64> = flat
                .chunks_exact(11)
                .flat_map(|stream| dft_forward(&as_f64(stream)))
                .collect();
            let starts: [usize; 11] = core::array::from_fn(|s| s * 11);
            let mut candidate = flat.clone();
            assert!(
                try_dft11_streams::<false>(&mut candidate, &starts, 1, &starts, 1),
                "chunked streams require AVX2+FMA on this host"
            );
            let err = max_err(&as_f64(&candidate), &expected);
            assert!(err < 1e-5, "streams forward kind={kind} max_err={err:.2e}");

            let expected_inv: Vec<Complex64> = flat
                .chunks_exact(11)
                .flat_map(|stream| dft_inverse(&as_f64(stream)).into_iter().map(|x| x * 11.0))
                .collect();
            let mut candidate_inv = flat;
            assert!(
                try_dft11_streams::<true>(&mut candidate_inv, &starts, 1, &starts, 1),
                "chunked streams require AVX2+FMA on this host"
            );
            let err_inv = max_err(&as_f64(&candidate_inv), &expected_inv);
            assert!(
                err_inv < 1e-5,
                "streams inverse kind={kind} max_err={err_inv:.2e}"
            );
        }
    }

    #[test]
    fn chunked_121_matches_direct_within_tolerance() {
        for kind in 0..6 {
            let mut signal = [Complex32::new(0.0, 0.0); 121];
            for (s, chunk) in signal.chunks_exact_mut(11).enumerate() {
                chunk.copy_from_slice(&input_11((kind + s) % 6));
            }
            let signal64 = as_f64(&signal);
            let expected = dft_forward(&signal64);
            let mut candidate = signal;
            assert!(
                try_dft121_chunked::<false>(&mut candidate),
                "chunked 121 requires AVX2+FMA on this host"
            );
            let err = max_err(&as_f64(&candidate), &expected);
            assert!(err < 1e-4, "121 forward kind={kind} max_err={err:.2e}");

            let expected_inv: Vec<Complex64> = dft_inverse(&signal64)
                .into_iter()
                .map(|x| x * 121.0)
                .collect();
            let mut candidate_inv = signal;
            assert!(
                try_dft121_chunked::<true>(&mut candidate_inv),
                "chunked 121 requires AVX2+FMA on this host"
            );
            let err_inv = max_err(&as_f64(&candidate_inv), &expected_inv);
            assert!(
                err_inv < 1e-4,
                "121 inverse kind={kind} max_err={err_inv:.2e}"
            );
        }
    }
}
