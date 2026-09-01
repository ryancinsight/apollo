//! Four-complex-register variant of the instance-major base transform.
//!
//! AVX2 holds four interleaved `f32` complex samples per register. Keeping the
//! two-complex layout there executes the same stage network at half native
//! width. This leaf preserves the parent kernel's arithmetic while grouping
//! four rows and four columns per register.

use super::{radix4, root2_twiddle, BasePlan};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

/// Broadcasts one complex scalar across every interleaved sample.
#[expect(
    clippy::inline_always,
    reason = "must fold into the dispatcher's target-feature scope"
)]
#[inline(always)]
fn complex_splat<T, A>(simd: Simd<T, A>, sample: [T; 2]) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let (interleaved, _) = simd.splat(sample[0]).interleave(simd.splat(sample[1]));
    ComplexReg::from_interleaved(interleaved)
}

/// Transposes four registers of four complex values from column-major to
/// row-major grouping.
#[expect(
    clippy::inline_always,
    reason = "the transpose is part of the register-resident row stage"
)]
#[inline(always)]
fn transpose_complex_four<T, A>(
    values: [ComplexReg<T, A>; 4],
    zero: ComplexReg<T, A>,
) -> [ComplexReg<T, A>; 4]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let mut tile = [zero.into_interleaved(); 8];
    for (slot, value) in tile.iter_mut().zip(values) {
        *slot = value.into_interleaved();
    }
    hermes_simd::Vector::transpose_square(&mut tile);
    core::array::from_fn(|row| {
        let (interleaved, _) = tile[2 * row].interleave(tile[2 * row + 1]);
        ComplexReg::from_interleaved(interleaved)
    })
}

/// Eight-lane base transform: one register carries four complex samples.
pub(super) struct BaseTransform<
    'a,
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const ROWS: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
    S,
> {
    pub(super) data: &'a mut [T; LANES],
    pub(super) plan: &'a BasePlan<T, ROWS, TABLE_LANES>,
    /// Type-selected output strategy, shared with the four-lane kernel: the
    /// sinks index by view chunk, so the same fixed-size buffers serve both
    /// chunk geometries.
    pub(super) sink: S,
}

impl<
        T,
        const INVERSE: bool,
        const MEASURE_PHASES: bool,
        const ROWS: usize,
        const LANES: usize,
        const TABLE_LANES: usize,
        S,
    > LaneKernel<T> for BaseTransform<'_, T, INVERSE, MEASURE_PHASES, ROWS, LANES, TABLE_LANES, S>
where
    T: LaneScalar + MixedRadixScalar,
    S: super::store::StoreSink<T>,
{
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature frame"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one three-phase register-resident transform; splitting it moves live registers across call boundaries"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        if <A as SimdStorage<T>>::LANE_COUNT != 8 {
            return false;
        }
        debug_assert!(ROWS == 4 || ROWS == 8);
        debug_assert_eq!(LANES, 32 * ROWS);
        debug_assert_eq!(TABLE_LANES, super::table_lanes(ROWS));

        // The final 28 table lanes are the unchanged four-lane broadcast
        // constants. Loading their first real/imaginary scalar and splatting
        // it avoids enlarging every plan solely for the eight-lane route.
        let row_offset = (ROWS - 1) * 64;
        let row1 = complex_splat(
            simd,
            [self.plan.table[row_offset], self.plan.table[row_offset + 4]],
        );
        let row3 = complex_splat(
            simd,
            [
                self.plan.table[row_offset + 8],
                self.plan.table[row_offset + 12],
            ],
        );
        let neg_row1 = complex_splat(
            simd,
            [
                self.plan.table[row_offset + 16],
                self.plan.table[row_offset + 20],
            ],
        );
        let half_root2 = simd.splat(self.plan.table[row_offset + 24]);
        let zero_complex = ComplexReg::from_interleaved(simd.zero());

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t0 = if MEASURE_PHASES {
            super::phase_meter::stamp()
        } else {
            0
        };

        #[cfg(debug_assertions)]
        let mut staging_debug = [T::from_precise(f64::NAN); LANES];
        #[cfg(debug_assertions)]
        let staging_ptr = staging_debug.as_mut_ptr();
        #[cfg(not(debug_assertions))]
        let mut staging_uninit = core::mem::MaybeUninit::<[T; LANES]>::uninit();
        #[cfg(not(debug_assertions))]
        let staging_ptr = staging_uninit.as_mut_ptr().cast::<T>();

        {
            let data_view = simd.view(self.data.as_slice());
            let row_groups = ROWS / 4;

            for group in 0..row_groups {
                let mut zbuf = [T::from_precise(0.0); 128];
                {
                    let mut zv = simd.view_mut(&mut zbuf);
                    let load_r = |b: usize| {
                        ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
                            &data_view,
                            row_groups * b + group,
                        ))
                    };
                    for b0 in 0..4usize {
                        let y = radix4::<T, A, INVERSE>([
                            load_r(b0),
                            load_r(b0 + 4),
                            load_r(b0 + 8),
                            load_r(b0 + 12),
                        ]);
                        let z = match b0 {
                            0 => y,
                            1 => [
                                y[0],
                                y[1] * row1,
                                root2_twiddle::<T, A, INVERSE, false>(y[2], half_root2),
                                y[3] * row3,
                            ],
                            2 => [
                                y[0],
                                root2_twiddle::<T, A, INVERSE, false>(y[1], half_root2),
                                if INVERSE {
                                    y[2].mul_i()
                                } else {
                                    y[2].mul_neg_i()
                                },
                                root2_twiddle::<T, A, INVERSE, true>(y[3], half_root2),
                            ],
                            _ => [
                                y[0],
                                y[1] * row3,
                                root2_twiddle::<T, A, INVERSE, true>(y[2], half_root2),
                                y[3] * neg_row1,
                            ],
                        };
                        for (m, reg) in z.into_iter().enumerate() {
                            reg.into_interleaved()
                                .store_to_view_chunk(&mut zv, 4 * m + b0);
                        }
                    }
                }

                let zv = simd.view(&zbuf);
                let load_z = |m: usize, b0: usize| {
                    ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
                        &zv,
                        4 * m + b0,
                    ))
                };
                let outputs: [[ComplexReg<T, A>; 4]; 4] = core::array::from_fn(|m| {
                    radix4::<T, A, INVERSE>([
                        load_z(m, 0),
                        load_z(m, 1),
                        load_z(m, 2),
                        load_z(m, 3),
                    ])
                });
                for q in 0..4usize {
                    let rows = transpose_complex_four(
                        [outputs[0][q], outputs[1][q], outputs[2][q], outputs[3][q]],
                        zero_complex,
                    );
                    for (row, reg) in rows.into_iter().enumerate() {
                        let chunk = (4 * group + row) * 4 + q;
                        // SAFETY: `simd` proves host support. `group` covers
                        // `0..ROWS/4`, `row` covers `0..4`, and `q` covers
                        // `0..4`, so the chunks are exactly `0..4*ROWS`, each
                        // writing eight lanes into the `32*ROWS`-lane staging
                        // allocation without overlap.
                        unsafe {
                            reg.into_interleaved()
                                .store_unaligned(staging_ptr.add(8 * chunk));
                        }
                    }
                }
            }
        }

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t2m = if MEASURE_PHASES {
            let t = super::phase_meter::stamp();
            super::phase_meter::add(0, t - t0);
            t
        } else {
            0
        };

        #[cfg(debug_assertions)]
        let staging = &staging_debug;
        #[cfg(not(debug_assertions))]
        // SAFETY: the raw stores above initialize every lane exactly once
        // before this first typed read; their disjoint exhaustive ranges are
        // established at the store site.
        let staging = unsafe { staging_uninit.assume_init_ref() };

        // The shared lane-wise `ROWS`-point DIF column pass, four groups of
        // four interleaved complex samples at this width.
        super::column::column_pass::<T, A, S, INVERSE, ROWS, 4>(
            simd,
            staging.as_slice(),
            self.plan.table.as_slice(),
            &self.plan.col,
            self.data.as_mut_slice(),
            self.sink,
        );

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        if MEASURE_PHASES {
            let t = super::phase_meter::stamp();
            super::phase_meter::add(2, t - t2m);
            super::phase_meter::CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }
}
