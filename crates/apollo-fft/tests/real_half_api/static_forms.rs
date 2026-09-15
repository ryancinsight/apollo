//! The compile-time-length real forms, routed through the split (ADR 0063).

use crate::{count_allocations, l1, signal, tolerance, SPLIT_SIZES};
use eunomia::Complex64;
use leto::Array1;

/// Runs the static pair at `N` and checks it against the dynamic split forms
/// within the derived bound: the two take different half-length executors
/// (the runtime kernel against the cached plan), so agreement is bounded,
/// not bitwise. Returns the allocations of one warm forward and inverse.
fn static_pair<const N: usize>() -> (usize, usize) {
    let src = signal(N);
    let array = Array1::from(src.clone());
    let one = tolerance(N, l1(&src), f64::EPSILON / 2.0);

    let dynamic = apollo_fft::fft_1d_array::<f64>(&array);
    let mut spectrum = Array1::from(vec![Complex64::default(); N]);
    apollo_fft::fft_1d_array_static_into::<f64, N>(&array, &mut spectrum);
    for (k, (got, want)) in spectrum.iter().zip(dynamic.iter()).enumerate() {
        let error = (got - want).norm();
        assert!(
            error <= 2.0 * one,
            "N={N} bin {k}: static forward {error:.3e} from the dynamic split (bound {:.3e})",
            2.0 * one
        );
    }

    let back = apollo_fft::ifft_1d_array::<f64>(&dynamic);
    let mut recovered = Array1::from(vec![0.0_f64; N]);
    let mut scratch = Array1::from(vec![Complex64::default(); N]);
    apollo_fft::ifft_1d_array_static_into::<f64, N>(&dynamic, &mut recovered, &mut scratch);
    for (k, (got, want)) in recovered.iter().zip(back.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 2.0 * one,
            "N={N} sample {k}: static inverse {got} against the dynamic split's {want} (bound {:.3e})",
            2.0 * one
        );
    }

    let ((), forward) =
        count_allocations(|| apollo_fft::fft_1d_array_static_into::<f64, N>(&array, &mut spectrum));
    let ((), inverse) = count_allocations(|| {
        apollo_fft::ifft_1d_array_static_into::<f64, N>(&dynamic, &mut recovered, &mut scratch);
    });
    (forward, inverse)
}

/// The static forms agree with the dynamic split forms at every split size
/// and allocate nothing once warm — whichever route the forward takes (the
/// split from 128 or at a non-power-of-two length, the static plan below).
#[test]
fn the_static_forms_take_the_split() {
    // The const lengths mirror `SPLIT_SIZES`; the assertion keeps them in step.
    assert_eq!(SPLIT_SIZES, [4, 8, 16, 64, 256, 1024, 4096]);
    for (n, allocations) in [
        (4, static_pair::<4>()),
        (8, static_pair::<8>()),
        (16, static_pair::<16>()),
        (64, static_pair::<64>()),
        (256, static_pair::<256>()),
        (1024, static_pair::<1024>()),
        (4096, static_pair::<4096>()),
    ] {
        assert_eq!(
            allocations,
            (0, 0),
            "N={n}: the static forms allocated on a warm process"
        );
    }
}

/// A length the split refuses keeps the widening path, agreeing with the
/// dynamic forms within the bound.
#[test]
fn a_refused_length_keeps_the_static_plan() {
    const N: usize = 6;
    let src = signal(N);
    let array = Array1::from(src.clone());
    let one = tolerance(N, l1(&src), f64::EPSILON / 2.0);
    let dynamic = apollo_fft::fft_1d_array::<f64>(&array);
    let mut spectrum = Array1::from(vec![Complex64::default(); N]);
    apollo_fft::fft_1d_array_static_into::<f64, N>(&array, &mut spectrum);
    for (k, (got, want)) in spectrum.iter().zip(dynamic.iter()).enumerate() {
        let error = (got - want).norm();
        assert!(
            error <= 2.0 * one,
            "N={N} bin {k}: {error:.3e} (bound {:.3e})",
            2.0 * one
        );
    }
}
