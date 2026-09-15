//! The `n/2 + 1` half-spectrum inverse.

use crate::{count_allocations, l1, signal, tolerance, Sample, FALLBACK_SIZES, SPLIT_SIZES};
use apollo_fft::{PlanCacheProvider, PlanScratch, F16};
use eunomia::{Complex, Complex64};
use realfft::RealFftPlanner;

/// Forward then inverse over every size, for storage scalar `T`.
fn round_trip<T>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    for n in SPLIT_SIZES.into_iter().chain(FALLBACK_SIZES) {
        let x: Vec<T> = signal(n).into_iter().map(T::from_f64).collect();
        let stored: Vec<f64> = x.iter().map(|&v| v.to_f64()).collect();
        let mut half = vec![Complex::<T::PlanScalar>::default(); n / 2 + 1];
        apollo_fft::fft_1d_slice_half_into(&x, &mut half);
        let mut back = vec![T::from_f64(0.0); n];
        apollo_fft::ifft_1d_slice_half_into(&mut half, &mut back);

        // The forward and the normalized inverse each err by at most the
        // forward bound per sample — the inverse's `1/n` carries its
        // `‖X‖₁ <= n‖x‖₁` back to `‖x‖₁` — and narrowing to storage adds one
        // rounding of the result.
        let transforms = 2.0 * tolerance(n, l1(&stored), T::PLAN_UNIT);
        for (i, (got, want)) in back.iter().zip(&stored).enumerate() {
            let got = got.to_f64();
            let bound = transforms + T::STORAGE_UNIT * (want.abs() + transforms);
            assert!(
                (got - want).abs() <= bound,
                "{} N={n} sample {i}: {got} against {want} (bound {bound:.2e})",
                std::any::type_name::<T>()
            );
        }
    }
}

#[test]
fn the_half_inverse_round_trips_for_every_storage_scalar() {
    round_trip::<f64>();
    round_trip::<f32>();
    round_trip::<F16>();
}

/// Every length the split admits up to 1024, so each packed length's plan —
/// power of two, mixed radix, and the prime factors the Rader and Bluestein
/// routes serve — carries the retangle at least once.
#[test]
fn the_half_inverse_round_trips_every_admitted_length_to_1024() {
    for n in (4..=1024).step_by(4) {
        let x = signal(n);
        let mut half = vec![Complex64::default(); n / 2 + 1];
        apollo_fft::fft_1d_slice_half_into::<f64>(&x, &mut half);
        let mut back = vec![0.0_f64; n];
        apollo_fft::ifft_1d_slice_half_into::<f64>(&mut half, &mut back);
        // The forward and the normalized inverse, each within the forward bound.
        let bound = 2.0 * tolerance(n, l1(&x), f64::EPSILON / 2.0);
        for (i, (got, want)) in back.iter().zip(&x).enumerate() {
            assert!(
                (got - want).abs() <= bound,
                "N={n} sample {i}: {got} against {want} (bound {bound:.2e})"
            );
        }
    }
}

#[test]
fn matches_realfft_sample_for_sample() {
    let mut planner = RealFftPlanner::<f64>::new();
    for n in SPLIT_SIZES {
        let spectrum = apollo_fft::fft_1d_slice_half::<f64>(&signal(n));

        let c2r = planner.plan_fft_inverse(n);
        let mut input = c2r.make_input_vec();
        for (slot, bin) in input.iter_mut().zip(&spectrum) {
            slot.re = bin.re;
            slot.im = bin.im;
        }
        let mut theirs = c2r.make_output_vec();
        c2r.process(&mut input, &mut theirs)
            .expect("the spectrum of a real signal has real end bins");

        let mut scratch = spectrum.clone();
        let mut ours = vec![0.0_f64; n];
        apollo_fft::ifft_1d_slice_half_into::<f64>(&mut scratch, &mut ours);

        // RealFFT leaves the `1/n` to the caller, and dividing by a power of
        // two is exact. Each engine's normalized inverse errs by at most the
        // bound for the full spectrum's 1-norm over n, and that norm is at
        // most twice the half's.
        let half_l1: f64 = spectrum.iter().map(|z| z.re.abs() + z.im.abs()).sum();
        let bound = 2.0 * tolerance(n, 2.0 * half_l1 / n as f64, f64::EPSILON / 2.0);
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            let b = *b / n as f64;
            assert!(
                (a - b).abs() <= bound,
                "N={n} sample {i}: {a} against RealFFT's {b} (bound {bound:.2e})"
            );
        }
    }
}

#[test]
fn the_half_inverse_allocates_nothing() {
    for n in SPLIT_SIZES {
        let spectrum = apollo_fft::fft_1d_slice_half::<f64>(&signal(n));
        let mut scratch = spectrum.clone();
        let mut out = vec![0.0_f64; n];

        // One warm call: the plan and its inverse twiddles are built on the
        // first inverse of a length, and those are not per-call costs.
        apollo_fft::ifft_1d_slice_half_into::<f64>(&mut scratch, &mut out);
        scratch.copy_from_slice(&spectrum);

        let ((), observed) = count_allocations(|| {
            apollo_fft::ifft_1d_slice_half_into::<f64>(&mut scratch, &mut out)
        });
        assert_eq!(
            observed, 0,
            "N={n}: the inverse allocated {observed} times; the caller owns every buffer"
        );
    }
}

#[test]
#[should_panic(expected = "exactly n/2 + 1 bins")]
fn a_full_length_spectrum_is_rejected() {
    let mut spectrum = vec![Complex64::default(); 64];
    let mut out = vec![0.0_f64; 64];
    apollo_fft::ifft_1d_slice_half_into::<f64>(&mut spectrum, &mut out);
}

/// Every full-spectrum 1-D inverse entry point takes the half inverse where
/// the split admits the length: sample for sample identical to
/// `ifft_1d_slice_half_into` over the lower half, which is the same arithmetic.
fn full_spectrum_inverses_take_the_half_inverse<T>()
where
    T: Sample + Copy,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    for n in SPLIT_SIZES {
        let x: Vec<T> = signal(n).into_iter().map(T::from_f64).collect();
        let full = apollo_fft::fft_1d_slice::<T>(&x);
        let mut half = full[..=n / 2].to_vec();
        let mut want = vec![T::from_f64(0.0); n];
        apollo_fft::ifft_1d_slice_half_into(&mut half, &mut want);
        let want: Vec<u64> = want.iter().map(|v| v.to_f64().to_bits()).collect();
        let bits = |values: &[T]| {
            values
                .iter()
                .map(|v| v.to_f64().to_bits())
                .collect::<Vec<u64>>()
        };
        let label = std::any::type_name::<T>();

        assert_eq!(
            bits(&apollo_fft::ifft_1d_slice::<T>(&full)),
            want,
            "{label} N={n}: ifft_1d_slice"
        );

        let array = leto::Array1::from(full.clone());
        let got = apollo_fft::ifft_1d_array::<T>(&array);
        assert_eq!(
            bits(got.as_slice().expect("contiguous")),
            want,
            "{label} N={n}: ifft_1d_array"
        );

        let mut out = leto::Array1::from(vec![T::from_f64(0.0); n]);
        let mut scratch = leto::Array1::from(vec![Complex::<T::PlanScalar>::default(); n]);
        apollo_fft::ifft_1d_array_into::<T>(&array, &mut out, &mut scratch);
        assert_eq!(
            bits(out.as_slice().expect("contiguous")),
            want,
            "{label} N={n}: ifft_1d_array_into"
        );

        let mut spectrum = array.clone();
        apollo_fft::ifft_1d_array_into_spectrum_scratch::<T>(&mut spectrum, &mut out);
        assert_eq!(
            bits(out.as_slice().expect("contiguous")),
            want,
            "{label} N={n}: ifft_1d_array_into_spectrum_scratch"
        );
    }
}

#[test]
fn the_full_spectrum_inverses_take_the_half_inverse_for_every_storage_scalar() {
    full_spectrum_inverses_take_the_half_inverse::<f64>();
    full_spectrum_inverses_take_the_half_inverse::<f32>();
    full_spectrum_inverses_take_the_half_inverse::<F16>();
}

/// The caller-owned full-spectrum inverses allocate nothing once warm, and
/// the owned one allocates exactly its returned signal: the half copy rides
/// the rank-one staging role.
#[test]
fn the_full_spectrum_inverses_allocate_only_their_output_once_warm() {
    for n in SPLIT_SIZES {
        let full = apollo_fft::fft_1d_slice::<f64>(&signal(n));
        let array = leto::Array1::from(full.clone());
        let mut out = leto::Array1::from(vec![0.0_f64; n]);
        let mut scratch = leto::Array1::from(vec![Complex64::default(); n]);

        apollo_fft::ifft_1d_array_into::<f64>(&array, &mut out, &mut scratch);
        let ((), observed) = count_allocations(|| {
            apollo_fft::ifft_1d_array_into::<f64>(&array, &mut out, &mut scratch);
        });
        assert_eq!(
            observed, 0,
            "N={n}: ifft_1d_array_into allocated {observed} times"
        );

        let mut spectrum = array.clone();
        apollo_fft::ifft_1d_array_into_spectrum_scratch::<f64>(&mut spectrum, &mut out);
        let mut spectrum = array.clone();
        let ((), observed) = count_allocations(|| {
            apollo_fft::ifft_1d_array_into_spectrum_scratch::<f64>(&mut spectrum, &mut out);
        });
        assert_eq!(
            observed, 0,
            "N={n}: ifft_1d_array_into_spectrum_scratch allocated {observed} times"
        );

        let _ = apollo_fft::ifft_1d_slice::<f64>(&full);
        let (signal, observed) = count_allocations(|| apollo_fft::ifft_1d_slice::<f64>(&full));
        assert_eq!(signal.len(), n);
        assert_eq!(
            observed, 1,
            "N={n}: ifft_1d_slice allocated {observed} times, not its one signal"
        );
    }
}
