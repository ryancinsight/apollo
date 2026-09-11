//! The column route: a base transform held in registers under a radix-`r`
//! column pass (ADR 0062).
//!
//! `n = r × m` runs the twiddled radix-`r` across the `r` rows of `m` first,
//! then `r` register-resident `m`-point transforms one row each into a
//! scratch, and writes natural order through an `r`-way pair interleave:
//! RustFFT's mixed-radix shape at a composite length, the block-and-sink
//! construction of ADR 0061 with the transposes in registers. The first
//! instance is 180 = 5 × 36.
//!
//! With `x` as five rows of thirty-six, `x[36 c + j]`, and `k = k2 + 5 k1`:
//!
//! ```text
//!   y_{k2}[j] = W_180^{j k2} Σ_c x[36 c + j] W_5^{c k2}      the column pass
//!   Z_{k2}[k1] = Σ_j y_{k2}[j] W_36^{j k1}                  five 36-point transforms
//!   X[k2 + 5 k1] = Z_{k2}[k1]                               the interleave
//! ```

use crate::application::execution::kernel::components::aligned::CacheLineAligned;
use crate::application::execution::kernel::components::register_butterfly::{
    radix5, Fifths, FIFTH_C1, FIFTH_C2, FIFTH_S1, FIFTH_S2,
};
use crate::application::execution::kernel::components::winograd::composite::{
    dft36_kernel, load, store, twiddled, TwiddleRow, Twiddles36,
};
use crate::application::execution::kernel::mixed_radix::scalar::simd::avx::vector_frame_available;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use core::mem::MaybeUninit;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

#[cfg(test)]
mod tests;

/// The tables of one direction of the 180 route.
pub(crate) struct Tables180<F> {
    thirty_six: Twiddles36<F>,
    /// The radix-5 constants as lane rows: `c1` and `c2` broadcast, the
    /// direction-signed sines on alternate lanes.
    fifths: [[F; 8]; 4],
    /// `W_180^{j k2}` for the column chunk `q` (`j = 4 q .. 4 q + 4`) and
    /// `k2` in 1 to 4, at `4 q + k2 - 1`.
    columns: [TwiddleRow<F>; 36],
}

impl<F: MixedRadixScalar> Tables180<F> {
    fn new(inverse: bool) -> Self {
        let (s1, s2) = if inverse {
            (FIFTH_S1, FIFTH_S2)
        } else {
            (-FIFTH_S1, -FIFTH_S2)
        };
        let broadcast = |value: f64| [F::from_precise(value); 8];
        let turn = |value: f64| {
            let mut row = [F::from_precise(0.0); 8];
            for pair in row.chunks_exact_mut(2) {
                pair[0] = F::from_precise(-value);
                pair[1] = F::from_precise(value);
            }
            row
        };
        let mut columns = [TwiddleRow::of_roots(180, [0; 4], inverse); 36];
        for (index, row) in columns.iter_mut().enumerate() {
            let chunk = index / 4;
            let k2 = index % 4 + 1;
            let j = 4 * chunk;
            *row = TwiddleRow::of_roots(
                180,
                [j * k2, (j + 1) * k2, (j + 2) * k2, (j + 3) * k2],
                inverse,
            );
        }
        Self {
            thirty_six: Twiddles36::new(inverse),
            fifths: [broadcast(FIFTH_C1), broadcast(FIFTH_C2), turn(s1), turn(s2)],
            columns,
        }
    }
}

/// The plan state of the 180 route: both directions' tables, cache-line
/// aligned so a row never splits a line.
pub(crate) struct State180<F> {
    forward: Box<CacheLineAligned<Tables180<F>>>,
    inverse: Box<CacheLineAligned<Tables180<F>>>,
}

impl<F: MixedRadixScalar + LaneScalar> State180<F> {
    /// Builds the state where the route runs: the vector frame present and
    /// the scalar's frame register holding four complexes.
    pub(crate) fn new_if_supported() -> Option<Self> {
        (F::FRAME_LANES == 8 && vector_frame_available()).then(|| Self {
            forward: Box::new(CacheLineAligned(Tables180::new(false))),
            inverse: Box::new(CacheLineAligned(Tables180::new(true))),
        })
    }
}

/// The 180 route over `data`.
///
/// # Safety
///
/// `data.len()` is 180, and the caller has established the vector frame
/// (AVX2 and FMA on `x86_64`) — the state exists only where the plan found
/// it, and the plan enters this function through its framed executor.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
pub(crate) unsafe fn transform_180<F, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<F>],
    state: &State180<F>,
) where
    F: MixedRadixScalar + LaneScalar + eunomia::layout::Pod,
    Complex<F>: eunomia::layout::Pod,
{
    let data: &mut [Complex<F>; 180] = data
        .try_into()
        .expect("invariant: the 180 executor runs the plan length");
    let tables = if INVERSE {
        &state.inverse.0
    } else {
        &state.forward.0
    };
    // SAFETY: the caller carries the frame contract through.
    unsafe {
        hermes_simd::vectorize_in_frame::<F, _>(Route180::<F, INVERSE, NORMALIZE> { tables, data });
    }
}

struct Route180<'a, F, const INVERSE: bool, const NORMALIZE: bool> {
    tables: &'a Tables180<F>,
    data: &'a mut [Complex<F>; 180],
}

impl<F, const INVERSE: bool, const NORMALIZE: bool> LaneKernel<F>
    for Route180<'_, F, INVERSE, NORMALIZE>
where
    F: MixedRadixScalar + LaneScalar + eunomia::layout::Pod,
    Complex<F>: eunomia::layout::Pod,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "the route must remain in the selected target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<F>>(self, simd: Simd<F, A>) {
        let Self { tables, data } = self;
        debug_assert_eq!(
            <A as SimdStorage<F>>::LANE_COUNT,
            8,
            "invariant: the state exists only at the eight-lane frame"
        );
        let fifths = Fifths {
            c1: load_row(simd, &tables.fifths[0]),
            c2: load_row(simd, &tables.fifths[1]),
            s1_turn: load_row(simd, &tables.fifths[2]),
            s2_turn: load_row(simd, &tables.fifths[3]),
        };

        // The column pass: a radix-5 across the five rows at four columns a
        // register, the twiddled arms back in place.
        for chunk in 0..9 {
            let j = 4 * chunk;
            let rows = [
                load(simd, &data[j..j + 4]),
                load(simd, &data[36 + j..40 + j]),
                load(simd, &data[72 + j..76 + j]),
                load(simd, &data[108 + j..112 + j]),
                load(simd, &data[144 + j..148 + j]),
            ];
            let arms = radix5::<F, A>(rows, &fifths);
            store(arms[0], &mut data[j..j + 4]);
            store(
                twiddled(simd, arms[1], &tables.columns[4 * chunk]),
                &mut data[36 + j..40 + j],
            );
            store(
                twiddled(simd, arms[2], &tables.columns[4 * chunk + 1]),
                &mut data[72 + j..76 + j],
            );
            store(
                twiddled(simd, arms[3], &tables.columns[4 * chunk + 2]),
                &mut data[108 + j..112 + j],
            );
            store(
                twiddled(simd, arms[4], &tables.columns[4 * chunk + 3]),
                &mut data[144 + j..148 + j],
            );
        }

        // Five 36-point transforms, one row each, into the scratch.
        let mut scratch = [MaybeUninit::<Complex<F>>::uninit(); 180];
        for row in 0..5 {
            let input: &[Complex<F>; 36] = data[36 * row..36 * row + 36]
                .try_into()
                .expect("invariant: a row is thirty-six samples");
            let output: &mut [MaybeUninit<Complex<F>>; 36] = (&mut scratch
                [36 * row..36 * row + 36])
                .try_into()
                .expect("invariant: a row is thirty-six samples");
            dft36_kernel::<F, A, INVERSE>(simd, &tables.thirty_six, input, output);
        }
        // SAFETY: each of the five kernels wrote its thirty-six outputs as
        // nine registers of four, so every element of the scratch is
        // initialized.
        let spectra: &[Complex<F>; 180] = unsafe { &*scratch.as_ptr().cast::<[Complex<F>; 180]>() };

        // The five-way interleave: register `k1`-chunk `q` of row `k2` lands
        // at `5 (4 q + l) + k2`, natural order.
        let scale = simd.splat(F::from_precise(1.0 / 180.0));
        for chunk in 0..9 {
            let j = 4 * chunk;
            let packed = load(simd, &spectra[j..j + 4])
                .into_interleaved()
                .interleave_pairs5(
                    load(simd, &spectra[36 + j..40 + j]).into_interleaved(),
                    load(simd, &spectra[72 + j..76 + j]).into_interleaved(),
                    load(simd, &spectra[108 + j..112 + j]).into_interleaved(),
                    load(simd, &spectra[144 + j..148 + j]).into_interleaved(),
                );
            let base = 20 * chunk;
            for (index, register) in packed.into_iter().enumerate() {
                let value = if INVERSE && NORMALIZE {
                    register * scale
                } else {
                    register
                };
                let at = base + 4 * index;
                store(
                    hermes_simd::ComplexReg::from_interleaved(value),
                    &mut data[at..at + 4],
                );
            }
        }
    }
}

/// Loads eight lanes as one register.
#[expect(
    clippy::inline_always,
    reason = "the load must stay in the selected target-feature frame"
)]
#[inline(always)]
fn load_row<F, A>(simd: Simd<F, A>, lanes: &[F; 8]) -> Vector<F, A>
where
    F: LaneScalar,
    A: SimdArch + SimdKernel<F>,
{
    Vector::from_view_chunk(&simd.view(lanes), 0)
}
