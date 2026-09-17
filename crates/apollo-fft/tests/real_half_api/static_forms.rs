//! The compile-time-length real forms, routed through the split (ADR 0063).
//!
//! Agreement with the dynamic forms is necessary but cannot tell the routes
//! apart: the widening path agrees within the same bound and allocates
//! nothing either. Two witnesses can, because each route leaves a mark the
//! other does not:
//!
//! - **Forward.** The split writes the upper half as the exact conjugate of
//!   the lower (`X[N - k] = conj(X[k])` bit for bit); the widening plan
//!   computes the upper half and rounds it independently. Checked where the
//!   forward takes the split; at short lengths the widening plan can land on
//!   the exact conjugate by chance, so no negative is asserted.
//! - **Inverse.** The split copies the lower `N/2 + 1` bins into the front of
//!   `scratch` and never touches the rest; the widening path overwrites all of
//!   it. Checked both ways, at every length.

use crate::{count_allocations, l1, tolerance, Sample};
use apollo_fft::{PlanCacheProvider, PlanScratch, F16};
use eunomia::Complex;
use leto::Array1;

/// The forward keeps the static plan's constant-length kernels on powers of
/// two below this length (`STATIC_FORWARD_SPLIT_FLOOR`, ADR 0063).
const FORWARD_SPLIT_FLOOR: usize = 128;

/// A value no transform of the test signal produces, left in the scratch the
/// split must not touch.
const SENTINEL: (f64, f64) = (7.25, -3.5);

/// The split's admission rule, stated here rather than read from the crate.
fn admitted(n: usize) -> bool {
    n >= 4 && n % 4 == 0
}

fn forward_takes_split(n: usize) -> bool {
    admitted(n) && (n >= FORWARD_SPLIT_FLOOR || !n.is_power_of_two())
}

fn signal<T: Sample>(n: usize) -> Vec<T> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            T::from_f64((0.017 * x).sin() + 0.4 * (0.083 * x).cos())
        })
        .collect()
}

fn parts<P: Copy + Into<f64>>(z: Complex<P>) -> (f64, f64) {
    (z.re.into(), z.im.into())
}

/// The static pair at `N` for storage scalar `T`: agreement with the dynamic
/// forms within the derived bound, and each route witness.
fn static_pair<T, const N: usize>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64> + From<f32>,
    Complex<T::PlanScalar>: PlanScratch,
{
    let name = std::any::type_name::<T>();
    let stored = signal::<T>(N);
    let array = Array1::from(stored.clone());
    let values: Vec<f64> = stored.iter().map(|&v| v.to_f64()).collect();
    // Two routes to one bin, each within one transform's bound of the exact
    // value; the inverse adds one rounding to storage.
    let one = tolerance(N, l1(&values), T::PLAN_UNIT);

    let dynamic = apollo_fft::fft_1d_array::<T>(&array);
    let mut spectrum = Array1::from(vec![Complex::<T::PlanScalar>::default(); N]);
    apollo_fft::fft_1d_array_static_into::<T, N>(&array, &mut spectrum);
    for (k, (&got, &want)) in spectrum.iter().zip(dynamic.iter()).enumerate() {
        let ((a_re, a_im), (b_re, b_im)) = (parts(got), parts(want));
        let error = (a_re - b_re).hypot(a_im - b_im);
        assert!(
            error <= 2.0 * one,
            "{name} N={N} bin {k}: static forward {error:.3e} from the dynamic split (bound {:.3e})",
            2.0 * one
        );
    }
    if forward_takes_split(N) {
        for k in 1..N / 2 {
            let ((lo_re, lo_im), (hi_re, hi_im)) = (parts(spectrum[k]), parts(spectrum[N - k]));
            assert!(
                hi_re.to_bits() == lo_re.to_bits() && hi_im.to_bits() == (-lo_im).to_bits(),
                "{name} N={N} bin {k}: the forward did not take the split \
                 (X[N-k] = ({hi_re}, {hi_im}) is not conj(X[k]) = ({lo_re}, {}))",
                -lo_im
            );
        }
    }

    let back = apollo_fft::ifft_1d_array::<T>(&dynamic);
    let mut recovered = Array1::from(vec![T::from_f64(0.0); N]);
    let sentinel = Complex::new(
        T::PlanScalar::from(SENTINEL.0 as f32),
        T::PlanScalar::from(SENTINEL.1 as f32),
    );
    let mut scratch = Array1::from(vec![sentinel; N]);
    apollo_fft::ifft_1d_array_static_into::<T, N>(&dynamic, &mut recovered, &mut scratch);
    for (k, (got, want)) in recovered.iter().zip(back.iter()).enumerate() {
        let (got, want) = (got.to_f64(), want.to_f64());
        let bound = 2.0 * one + T::STORAGE_UNIT * (want.abs() + 2.0 * one);
        assert!(
            (got - want).abs() <= bound,
            "{name} N={N} sample {k}: static inverse {got} against the dynamic split's {want} (bound {bound:.3e})"
        );
    }
    if admitted(N) {
        assert!(
            scratch
                .iter()
                .skip(N / 2 + 1)
                .all(|&z| parts(z) == SENTINEL),
            "{name} N={N}: the inverse wrote past the half it reads, so it did not take the split"
        );
    } else {
        // The widening path copies the whole spectrum into the scratch first.
        assert!(
            scratch.iter().all(|&z| parts(z) != SENTINEL),
            "{name} N={N}: a refused length left scratch untouched, so it did not widen"
        );
    }

    // Warm, neither route allocates.
    let ((), forward) =
        count_allocations(|| apollo_fft::fft_1d_array_static_into::<T, N>(&array, &mut spectrum));
    let ((), inverse) = count_allocations(|| {
        apollo_fft::ifft_1d_array_static_into::<T, N>(&dynamic, &mut recovered, &mut scratch);
    });
    assert_eq!(
        (forward, inverse),
        (0, 0),
        "{name} N={N}: the static forms allocated on a warm process"
    );
}

/// Every admitted length class the static forms route differently: powers of
/// two on each side of the forward floor, the floor itself, and lengths that
/// are multiples of four but not powers of two.
fn admitted_lengths<T>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64> + From<f32>,
    Complex<T::PlanScalar>: PlanScratch,
{
    static_pair::<T, 4>();
    static_pair::<T, 8>();
    static_pair::<T, 12>();
    static_pair::<T, 16>();
    static_pair::<T, 20>();
    static_pair::<T, 32>();
    static_pair::<T, 64>();
    static_pair::<T, 100>();
    static_pair::<T, 128>();
    static_pair::<T, 256>();
    static_pair::<T, 1024>();
    static_pair::<T, 4096>();
}

/// Lengths the split refuses: below four, even but not a multiple of four,
/// and odd.
fn refused_lengths<T>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64> + From<f32>,
    Complex<T::PlanScalar>: PlanScratch,
{
    static_pair::<T, 2>();
    static_pair::<T, 6>();
    static_pair::<T, 7>();
    static_pair::<T, 10>();
}

#[test]
fn the_static_forms_take_the_split() {
    admitted_lengths::<f64>();
    admitted_lengths::<f32>();
    admitted_lengths::<F16>();
}

#[test]
fn a_refused_length_keeps_the_static_plan() {
    refused_lengths::<f64>();
    refused_lengths::<f32>();
    refused_lengths::<F16>();
}
