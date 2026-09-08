//! Workspace bounds and borrowed execution against an exact impulse spectrum.

use super::super::{four_step_fft, scratch_len};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::{Complex, Complex64};

#[test]
fn workspace_covers_padded_planes_and_nested_gathers() {
    // Square planar rows carry eight padding elements. A fused odd split
    // holds two planes; an unfused odd split holds n gathered inputs plus
    // one reusable half-transform transpose buffer.
    for (n, expected) in [
        (4, 2 * (2 + 8)),
        (128, 128),
        (512, 2 * 16 * (16 + 8)),
        (4096, 64 * (64 + 8)),
        (32_768, 2 * 128 * (128 + 8)),
        (65_536, 65_536),
        (131_072, 131_072 + 65_536),
    ] {
        assert_eq!(scratch_len(n), Some(expected), "length {n}");
    }
    for invalid in [0, 1, 2, 3, 511, usize::MAX] {
        assert_eq!(scratch_len(invalid), None);
    }
}

fn check_impulse<F, const INVERSE: bool, const NORMALIZE: bool>(
    n: usize,
    unit_roundoff: f64,
    extra_workspace: usize,
) where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
{
    let zero = Complex::new(F::from(0.0), F::from(0.0));
    let mut data = vec![zero; n];
    data[n / 4] = Complex::new(F::from(0.5), F::from(-0.25));
    let required = scratch_len(n).expect("test length admits four-step workspace");
    // Nonzero scratch exposes read-before-write errors. The sentinel beyond
    // the required prefix also checks clipping when the caller lends it.
    let sentinel = Complex::new(F::from(7.0), F::from(-3.0));
    let mut scratch = vec![sentinel; required + 1];
    four_step_fft::<F, INVERSE, NORMALIZE>(&mut data, &mut scratch[..required + extra_workspace]);

    let scale = if INVERSE && NORMALIZE {
        f64::from(u32::try_from(n).expect("test length fits u32")).recip()
    } else {
        1.0
    };
    // The impulse has norm sqrt(5/16). Its exact spectrum consists of four
    // signed/permuted dyadic values. Sixteen rounded real operations per
    // stage yield Higham's gamma_k bound, matching the plan-routing oracle.
    let ku = 16.0 * f64::from(n.ilog2()) * unit_roundoff;
    let bound = ku / (1.0 - ku) * 0.3125_f64.sqrt() * scale;
    let phases = [
        Complex64::new(0.5, -0.25),
        Complex64::new(0.25, 0.5),
        Complex64::new(-0.5, 0.25),
        Complex64::new(-0.25, -0.5),
    ];
    for (bin, actual) in data.iter().enumerate() {
        let phase = if INVERSE { bin % 4 } else { (4 - bin % 4) % 4 };
        let expected = phases[phase] * scale;
        let observed = Complex64::new(actual.re.into(), actual.im.into());
        let error = (observed - expected).norm();
        assert!(error <= bound, "n={n}, bin={bin}: {error:e} > {bound:e}");
    }
    assert_eq!(scratch[required].re.into(), 7.0);
    assert_eq!(scratch[required].im.into(), -3.0);
}

fn check_scalar<F>(unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
{
    for n in [4, 128, 512, 4096, 65_536, 131_072] {
        for extra_workspace in [0, 1] {
            check_impulse::<F, false, false>(n, unit_roundoff, extra_workspace);
            check_impulse::<F, true, false>(n, unit_roundoff, extra_workspace);
            check_impulse::<F, true, true>(n, unit_roundoff, extra_workspace);
        }
    }
}

#[test]
fn workspace_extents_preserve_the_impulse_spectrum() {
    check_scalar::<f32>(f64::from(f32::EPSILON) / 2.0);
    check_scalar::<f64>(f64::EPSILON / 2.0);
}

#[test]
fn short_workspace_rejects_before_mutating_input() {
    for n in [4096, 131_072] {
        let mut source = vec![Complex64::new(0.5, -0.25); n];
        source[0] = Complex64::new(-0.0, f64::NAN);
        let mut data = source.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n).expect("valid length") - 1];
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            four_step_fft::<f64, false, false>(&mut data, &mut scratch);
        }));
        assert!(outcome.is_err(), "undersized workspace must be rejected");
        // Rejection preserves representation, including signed zero and NaN
        // payloads. Floating-point equality cannot express that contract.
        assert_eq!(
            eunomia::layout::cast_slice::<_, u8>(&data),
            eunomia::layout::cast_slice::<_, u8>(&source),
            "rejection must precede mutation"
        );
    }
}
