//! Unrolled small power-of-two codelets, split by SIMD lane density.

#[cfg(target_arch = "x86_64")]
mod n16;
#[cfg(target_arch = "x86_64")]
mod n32;
mod precise;
mod reduced;

// Test-gated deliberately: correct against the direct-DFT oracle in both
// directions, and measured slower than the scalar codelet it would replace
// (`small_pot_arms`). It stays as the instrument's subject, so the comparison
// that declined it can be re-run rather than becoming a note nobody can check.
#[cfg(test)]
mod n8;

pub(super) use precise::{
    small_pot_inplace_precise, small_pot_inplace_sized_precise,
    small_pot_inplace_sized_precise_framed,
};
pub(super) use reduced::{
    small_pot_inplace_reduced, small_pot_inplace_sized_reduced,
    small_pot_inplace_sized_reduced_framed,
};

// The probe entries the `small_pot_arms` instrument reaches; they exist only
// under `cfg(test)` and carry no production call site.
//
// The predicate matches that probe's own — it is `windows`-gated because it
// binds processors through Hermes — rather than just `test`: a narrower
// consumer than producer makes these dead code everywhere else, which is a
// `-D warnings` failure on every non-Windows target and invisible to a local
// gate run here.
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n16::framed_lane_pass as n16_framed_lane_pass;
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n16::fused_round_trip_unchecked as n16_fused_round_trip;
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n16::vector_arm_unchecked as n16_vector_arm_unchecked;
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n32::framed_lane_pass as n32_framed_lane_pass;
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n8::framed_lane_pass as n8_framed_lane_pass;
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n8::fused_round_trip_unchecked as n8_fused_round_trip;
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) use n8::vector_arm_unchecked as n8_vector_arm_unchecked;

#[cfg(test)]
mod tests {
    use super::super::simd::avx::vector_frame_available;
    use super::super::trait_def::MixedRadixScalar;
    use eunomia::Complex;

    /// The framed entry and the probing entry are one function of the input
    /// at every size the plan routes through them: the probing entry selects
    /// exactly the arm the frame runs, so the two agree bitwise. On a host
    /// without the frame neither the plan nor this check enters it.
    fn framed_matches_probing<
        F: MixedRadixScalar<Complex = Complex<F>>,
        const N: usize,
        const INVERSE: bool,
        const NORMALIZE: bool,
    >() {
        if !vector_frame_available() {
            return;
        }
        let input: Vec<Complex<F>> = (0..N)
            .map(|index| {
                let phase = index as f64;
                Complex::new(
                    F::from_precise((phase * 0.37).sin()),
                    F::from_precise((phase * 0.11).cos() - 0.5),
                )
            })
            .collect();
        let mut probing = input.clone();
        let mut framed = input;
        // SAFETY: both buffers hold exactly `N` samples, and the frame was
        // probed above.
        unsafe {
            F::small_pot_inplace_sized::<N, INVERSE, NORMALIZE>(&mut probing);
            F::small_pot_inplace_sized_framed::<N, INVERSE, NORMALIZE>(&mut framed);
        }
        assert_eq!(
            probing, framed,
            "N={N} INVERSE={INVERSE} NORMALIZE={NORMALIZE}: the two entries ran different arms"
        );
    }

    fn framed_matches_probing_at<F: MixedRadixScalar<Complex = Complex<F>>, const N: usize>() {
        framed_matches_probing::<F, N, false, false>();
        framed_matches_probing::<F, N, true, false>();
        framed_matches_probing::<F, N, true, true>();
    }

    #[test]
    fn framed_entry_matches_probing_entry() {
        framed_matches_probing_at::<f32, 8>();
        framed_matches_probing_at::<f32, 16>();
        framed_matches_probing_at::<f32, 32>();
        framed_matches_probing_at::<f32, 64>();
        framed_matches_probing_at::<f64, 8>();
        framed_matches_probing_at::<f64, 16>();
        framed_matches_probing_at::<f64, 32>();
        framed_matches_probing_at::<f64, 64>();
    }
}
