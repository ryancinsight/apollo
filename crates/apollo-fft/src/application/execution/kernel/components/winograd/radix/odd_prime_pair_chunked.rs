//! The chunked N = 121 driver (f32).
//!
//! [`try_dft121_chunked`] drives N = 121 as two
//! [`super::odd_prime_pair_streams`] calls (columns, then rows) around an
//! AVX2 complex twiddle stage, so the whole transform issues 2 leaf
//! dispatches instead of 22. The rows call writes the CT-transposed
//! output directly.

use eunomia::Complex32;

#[cfg(target_arch = "x86_64")]
use super::odd_prime_pair_streams::dft11_streams_simd;

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
    use super::super::odd_prime_pair_streams::try_dft11_streams;
    use super::try_dft121_chunked;
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

    fn layout_bits<const INVERSE: bool>(
        streams: &[[Complex32; 11]; 11],
        in_starts: &[usize; 11],
        in_step: usize,
        out_starts: &[usize; 11],
        out_step: usize,
    ) -> Vec<(u32, u32)> {
        // Offsets, gaps, and guards must survive every in-place layout.
        let last_input = in_starts.iter().max().expect("eleven input starts") + 10 * in_step;
        let last_output = out_starts.iter().max().expect("eleven output starts") + 10 * out_step;
        let mut base = vec![Complex32::new(0.125, -0.375); last_input.max(last_output) + 2];
        for s in 0..11 {
            for i in 0..11 {
                base[in_starts[s] + i * in_step] = streams[s][i];
            }
        }
        let before = base.clone();
        assert!(
            try_dft11_streams::<INVERSE>(&mut base, in_starts, in_step, out_starts, out_step),
            "chunked streams require AVX2+FMA on this host"
        );
        let mut bits = Vec::with_capacity(121);
        let mut written = vec![false; base.len()];
        for s in 0..11 {
            for i in 0..11 {
                let slot = out_starts[s] + i * out_step;
                assert!(!written[slot], "output slots must be disjoint");
                written[slot] = true;
                let x = base[slot];
                bits.push((x.re.to_bits(), x.im.to_bits()));
            }
        }
        for (slot, (&actual, &original)) in base.iter().zip(&before).enumerate() {
            if !written[slot] {
                assert_eq!(
                    actual.re.to_bits(),
                    original.re.to_bits(),
                    "re guard {slot}"
                );
                assert_eq!(
                    actual.im.to_bits(),
                    original.im.to_bits(),
                    "im guard {slot}"
                );
            }
        }
        bits
    }

    #[test]
    fn chunked_streams_layouts_are_bitwise_identical() {
        // The reversed column layout forces the unchanged scalar prologue.
        // Rows vectorize within streams; columns vectorize across streams.
        // The padded layout exercises the scalar fallback at another stride.
        let layouts: [([usize; 11], usize); 5] = [
            (core::array::from_fn(|s| 3 + s * 11), 1),
            (core::array::from_fn(|s| 3 + s), 11),
            (core::array::from_fn(|s| 3 + 10 - s), 11),
            (core::array::from_fn(|s| 3 + ((s * 7) % 11) * 11), 1),
            (core::array::from_fn(|s| 3 + s * 23), 2),
        ];
        for kind in 0..6 {
            let mut streams = [[Complex32::new(0.0, 0.0); 11]; 11];
            for (s, stream) in streams.iter_mut().enumerate() {
                *stream = input_11((kind + s) % 6);
            }
            for inverse in [false, true] {
                let run = |i: &[usize; 11], is: usize, o: &[usize; 11], os: usize| {
                    if inverse {
                        layout_bits::<true>(&streams, i, is, o, os)
                    } else {
                        layout_bits::<false>(&streams, i, is, o, os)
                    }
                };
                let reference = run(&layouts[2].0, layouts[2].1, &layouts[0].0, layouts[0].1);
                for (input, (i, is)) in layouts.iter().enumerate() {
                    for (output, (o, os)) in layouts.iter().enumerate() {
                        assert_eq!(
                            run(i, *is, o, *os),
                            reference,
                            "kind={kind} inverse={inverse} input={input} output={output}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn chunked_dispatch_declines_without_mutation_when_unavailable() {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // The value tests above exercise the successful dispatch.
            return;
        }
        let original: [Complex32; 121] = core::array::from_fn(|i| {
            Complex32::new((i as f32 + 1.0) / 128.0, -(i as f32 + 1.0) / 256.0)
        });
        let mut candidate = original;
        assert!(!try_dft121_chunked::<false>(&mut candidate));
        assert_eq!(candidate, original);
        assert!(!try_dft121_chunked::<true>(&mut candidate));
        assert_eq!(candidate, original);
        let starts = core::array::from_fn(|s| s * 11);
        assert!(!try_dft11_streams::<false>(
            &mut candidate,
            &starts,
            1,
            &starts,
            1
        ));
        assert_eq!(candidate, original);
        assert!(!try_dft11_streams::<true>(
            &mut candidate,
            &starts,
            1,
            &starts,
            1
        ));
        assert_eq!(candidate, original);
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
