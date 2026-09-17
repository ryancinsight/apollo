//! The radix-3 kernel's folded quarter turn against the rotation it
//! replaces: `swap(d) * (s, -s)` is `rot(d) * s` for `rot` the lane swap
//! with one lane negated, because negation is exact, so the fused outputs
//! agree bit for bit. A sweep of lane values, both directions, both
//! register widths and both scalars.

use super::{radix3, rot90, Thirds};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{
    ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector,
};

/// The rotation form of `radix3`: the turned difference, then the scale.
fn rotation_form<T, A, const INVERSE: bool>(
    simd: Simd<T, A>,
    values: [ComplexReg<T, A>; 3],
) -> [ComplexReg<T, A>; 3]
where
    T: LaneScalar + MixedRadixScalar,
    A: SimdArch + SimdKernel<T>,
{
    let sine = simd.splat(T::from_precise(0.866_025_403_784_438_6));
    let sine_negative = simd.splat(T::from_precise(-0.866_025_403_784_438_6));
    let half_negative = simd.splat(T::from_precise(-0.5));
    let (sum, difference) = values[1].butterfly(values[2]);
    let m0 = sum
        .into_interleaved()
        .mul_add(half_negative, values[0].into_interleaved());
    let turned = rot90::<T, A, INVERSE>(difference).into_interleaved();
    [
        values[0] + sum,
        ComplexReg::from_interleaved(turned.mul_add(sine, m0)),
        ComplexReg::from_interleaved(turned.mul_add(sine_negative, m0)),
    ]
}

/// Both forms over `lanes` values a register (three registers), returning
/// the lanes of each, or none past the served widths.
struct Forms<'a, T, const INVERSE: bool> {
    lanes: &'a [T],
}

impl<T: LaneScalar + MixedRadixScalar, const INVERSE: bool> LaneKernel<T>
    for Forms<'_, T, INVERSE>
{
    type Output = (Vec<T>, Vec<T>);

    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> Self::Output {
        let width = <A as SimdStorage<T>>::LANE_COUNT;
        let register = |r: usize| {
            ComplexReg::from_interleaved(
                Vector::<T, A>::load_unaligned_from_slice(&self.lanes[r * width..(r + 1) * width])
                    .expect("invariant: the slice holds one register"),
            )
        };
        let values = [register(0), register(1), register(2)];
        let folded = radix3::<T, A, INVERSE>(values, &Thirds::new(simd));
        let rotated = rotation_form::<T, A, INVERSE>(simd, values);
        let flatten = |outputs: [ComplexReg<T, A>; 3]| {
            let mut lanes = vec![T::from_precise(0.0); 3 * width];
            for (o, out) in outputs.iter().zip(lanes.chunks_exact_mut(width)) {
                o.into_interleaved()
                    .store_unaligned_to_slice(out)
                    .expect("invariant: the chunk holds one register");
            }
            lanes
        };
        (flatten(folded), flatten(rotated))
    }
}

fn forms_agree<T, const LANES: usize>()
where
    T: LaneScalar + MixedRadixScalar + eunomia::layout::Pod,
{
    for step in 0..512_u32 {
        // Lanes spread across signs and magnitudes, including signed zeros.
        let lanes: Vec<T> = (0..3 * LANES as u32)
            .map(|i| {
                let x = f64::from(step * 37 + i * 11) * 0.618_033_988_749_895;
                let value = (x.fract() - 0.5) * 10f64.powi((step % 7) as i32 - 3);
                T::from_precise(if (step + i) % 29 == 0 { -0.0 } else { value })
            })
            .collect();
        for inverse in [false, true] {
            let outputs = if inverse {
                hermes_simd::vectorize_lanes::<LANES, T, _>(Forms::<T, true> { lanes: &lanes })
            } else {
                hermes_simd::vectorize_lanes::<LANES, T, _>(Forms::<T, false> { lanes: &lanes })
            };
            let Some((folded, rotated)) = outputs else {
                return;
            };
            assert_eq!(
                eunomia::layout::cast_slice::<T, u8>(&folded),
                eunomia::layout::cast_slice::<T, u8>(&rotated),
                "{LANES} lanes, step {step}, inverse {inverse}"
            );
        }
    }
}

#[test]
fn folded_quarter_turn_matches_the_rotation_bit_for_bit() {
    forms_agree::<f32, 4>();
    forms_agree::<f32, 8>();
    forms_agree::<f64, 4>();
    forms_agree::<f64, 8>();
}
