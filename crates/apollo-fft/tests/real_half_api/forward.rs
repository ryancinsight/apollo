//! The `n/2 + 1` half-spectrum forward.

use crate::{count_allocations, l1, signal, tolerance, FALLBACK_SIZES, SPLIT_SIZES};
use eunomia::Complex64;
use realfft::RealFftPlanner;

#[test]
fn half_spectrum_is_bitwise_the_full_spectrum_truncated() {
    for n in SPLIT_SIZES.into_iter().chain(FALLBACK_SIZES) {
        let src = signal(n);
        let full = apollo_fft::fft_1d_slice::<f64>(&src);
        let half = apollo_fft::fft_1d_slice_half::<f64>(&src);

        assert_eq!(half.len(), n / 2 + 1, "N={n}: half spectrum length");
        assert_eq!(full.len(), n, "N={n}: full spectrum length");

        for (bin, (h, f)) in half.iter().zip(full.iter()).enumerate() {
            // Bitwise: the two share a code path, so any difference is a
            // defect rather than rounding.
            assert!(
                h.re.to_bits() == f.re.to_bits() && h.im.to_bits() == f.im.to_bits(),
                "N={n} bin {bin}: half {h:?} differs from full {f:?}"
            );
        }
    }
}

#[test]
fn the_discarded_half_is_recoverable_by_conjugate_symmetry() {
    for n in SPLIT_SIZES {
        let src = signal(n);
        let half = apollo_fft::fft_1d_slice_half::<f64>(&src);
        let full = apollo_fft::fft_1d_slice::<f64>(&src);

        // X[n-k] = conj(X[k]) is a property of a real input, so reconstructing
        // the upper half from the retained bins must reproduce the full
        // spectrum exactly.
        for k in 1..n / 2 {
            let mirrored = full[n - k];
            let from_half = half[k];
            assert!(
                (mirrored.re - from_half.re).abs() <= f64::EPSILON * from_half.re.abs().max(1.0)
                    && (mirrored.im + from_half.im).abs()
                        <= f64::EPSILON * from_half.im.abs().max(1.0),
                "N={n} bin {k}: X[n-k] is not conj(X[k]) — {mirrored:?} against {from_half:?}"
            );
        }
    }
}

#[test]
fn matches_realfft_bin_for_bin() {
    let mut planner = RealFftPlanner::<f64>::new();
    for n in SPLIT_SIZES {
        let src = signal(n);
        let ours = apollo_fft::fft_1d_slice_half::<f64>(&src);

        let r2c = planner.plan_fft_forward(n);
        let mut input = r2c.make_input_vec();
        input.copy_from_slice(&src);
        let mut theirs = r2c.make_output_vec();
        r2c.process(&mut input, &mut theirs)
            .expect("realfft length agrees with the plan");

        assert_eq!(ours.len(), theirs.len(), "N={n}: both engines return n/2+1");
        let bound = tolerance(n, l1(&src), f64::EPSILON / 2.0);
        for (bin, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
            let err = (a.re - b.re).hypot(a.im - b.im);
            assert!(
                err <= bound,
                "N={n} bin {bin}: {err:.3e} exceeds {bound:.3e} against RealFFT"
            );
        }
    }
}

#[test]
fn the_into_form_allocates_nothing() {
    for n in SPLIT_SIZES {
        let src = signal(n);
        let mut out = vec![Complex64::default(); n / 2 + 1];

        // One warm call first: plan, twiddles and scratch are cached on the
        // first use of a length, and those are not per-call costs.
        apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut out);

        let ((), observed) =
            count_allocations(|| apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut out));
        assert_eq!(
            observed, 0,
            "N={n}: the _into form allocated {observed} times; the caller owns the output"
        );
    }
}

#[test]
#[should_panic(expected = "exactly n/2 + 1 bins")]
fn a_wrong_output_length_is_rejected() {
    let src = signal(64);
    let mut out = vec![Complex64::default(); 64];
    apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut out);
}
