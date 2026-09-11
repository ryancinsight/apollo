//! The 36-point transform held in registers.
//!
//! Rows of nine under a radix-4 across the four row registers, the 9x4 tile
//! transposed to 4x9 in registers, and a radix-9 (three by three, with the
//! ninth-turn twiddles between) across the nine column registers — every
//! lane a sample of the same transform, RustFFT's `Butterfly36Avx` shape
//! over hermes registers of four complexes. The base of the 180 column
//! route (ADR 0062). Each row of nine loads as three registers: the one at
//! the row start, of which only column 0 is used, and the full registers of
//! columns 1 to 4 and 5 to 8.

use crate::application::execution::kernel::components::register_butterfly::{
    radix3, radix4, Thirds,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use core::mem::MaybeUninit;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

/// One register of twiddles as its interleaved lanes, direct and with each
/// sample's real and imaginary lanes exchanged, the pair
/// [`ComplexReg::mul_with_swapped`] takes.
#[derive(Clone, Copy)]
pub(crate) struct TwiddleRow<F> {
    pub(crate) direct: [F; 8],
    pub(crate) swapped: [F; 8],
}

impl<F: MixedRadixScalar> TwiddleRow<F> {
    /// The four `n`-th roots `W_n^{e}` for the exponents, forward or inverse.
    pub(crate) fn of_roots(n: usize, exponents: [usize; 4], inverse: bool) -> Self {
        let mut direct = [F::from_precise(0.0); 8];
        let mut swapped = [F::from_precise(0.0); 8];
        for (sample, exponent) in exponents.into_iter().enumerate() {
            let angle = core::f64::consts::TAU * ((exponent % n) as f64) / (n as f64);
            let (sine, cosine) = angle.sin_cos();
            let imaginary = if inverse { sine } else { -sine };
            direct[2 * sample] = F::from_precise(cosine);
            direct[2 * sample + 1] = F::from_precise(imaginary);
            swapped[2 * sample] = F::from_precise(imaginary);
            swapped[2 * sample + 1] = F::from_precise(cosine);
        }
        Self { direct, swapped }
    }
}

/// The twiddle rows of one direction of the 36-point kernel.
pub(crate) struct Twiddles36<F> {
    /// `W_36^{c k1}` for `k1` in 1 to 3, the columns 1 to 4 then 5 to 8.
    rows: [TwiddleRow<F>; 6],
    /// `W_9^1`, `W_9^2`, `W_9^4` broadcast.
    ninths: [TwiddleRow<F>; 3],
}

impl<F: MixedRadixScalar> Twiddles36<F> {
    pub(crate) fn new(inverse: bool) -> Self {
        let row = |k1: usize, first_column: usize| {
            TwiddleRow::of_roots(
                36,
                [
                    first_column * k1,
                    (first_column + 1) * k1,
                    (first_column + 2) * k1,
                    (first_column + 3) * k1,
                ],
                inverse,
            )
        };
        Self {
            rows: [
                row(1, 1),
                row(1, 5),
                row(2, 1),
                row(2, 5),
                row(3, 1),
                row(3, 5),
            ],
            ninths: [
                TwiddleRow::of_roots(9, [1; 4], inverse),
                TwiddleRow::of_roots(9, [2; 4], inverse),
                TwiddleRow::of_roots(9, [4; 4], inverse),
            ],
        }
    }
}

/// Loads four complexes as one register.
#[expect(
    clippy::inline_always,
    reason = "the load must stay in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn load<F, A>(simd: Simd<F, A>, values: &[Complex<F>]) -> ComplexReg<F, A>
where
    F: LaneScalar + eunomia::layout::Pod,
    A: SimdArch + SimdKernel<F>,
    Complex<F>: eunomia::layout::Pod,
{
    let view = simd.view(eunomia::layout::cast_slice(values));
    ComplexReg::from_interleaved(Vector::from_view_chunk(&view, 0))
}

/// Loads eight interleaved lanes as one register.
#[expect(
    clippy::inline_always,
    reason = "the load must stay in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn load_lanes<F, A>(simd: Simd<F, A>, lanes: &[F; 8]) -> Vector<F, A>
where
    F: LaneScalar,
    A: SimdArch + SimdKernel<F>,
{
    let view = simd.view(lanes);
    Vector::from_view_chunk(&view, 0)
}

/// Multiplies a register by a twiddle row.
#[expect(
    clippy::inline_always,
    reason = "the loads must stay in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn twiddled<F, A>(
    simd: Simd<F, A>,
    value: ComplexReg<F, A>,
    row: &TwiddleRow<F>,
) -> ComplexReg<F, A>
where
    F: LaneScalar,
    A: SimdArch + SimdKernel<F>,
{
    value.mul_with_swapped(
        ComplexReg::from_interleaved(load_lanes(simd, &row.direct)),
        ComplexReg::from_interleaved(load_lanes(simd, &row.swapped)),
    )
}

/// Stores one register as four complexes.
#[expect(
    clippy::inline_always,
    reason = "the store must stay in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn store<F, A>(value: ComplexReg<F, A>, destination: &mut [Complex<F>])
where
    F: LaneScalar + eunomia::layout::Pod,
    A: SimdArch + SimdKernel<F>,
    Complex<F>: eunomia::layout::Pod,
{
    value
        .into_interleaved()
        .store_unaligned_to_slice(eunomia::layout::cast_slice_mut(destination))
        .expect("invariant: four complex samples fill eight lanes");
}

/// Stores one register as four complexes into uninitialized memory.
#[expect(
    clippy::inline_always,
    reason = "the store must stay in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn store_uninit<F, A>(
    value: ComplexReg<F, A>,
    destination: &mut [MaybeUninit<Complex<F>>; 4],
) where
    F: LaneScalar,
    A: SimdArch + SimdKernel<F>,
{
    // SAFETY: the destination is exactly four complexes, eight lanes of `F`,
    // the register width; writing them is the initialization the caller
    // reads back only after every register of the block has landed.
    unsafe {
        value
            .into_interleaved()
            .store_unaligned(destination.as_mut_ptr().cast::<F>());
    }
}

/// The 36-point transform of `input` into `output`, natural order both ways.
///
/// Runs on registers of four complexes: the route that owns the frame
/// selects it only where `LaneScalar::FRAME_LANES` is eight.
#[expect(
    clippy::inline_always,
    reason = "the codelet must remain in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn dft36_kernel<F, A, const INVERSE: bool>(
    simd: Simd<F, A>,
    twiddles: &Twiddles36<F>,
    input: &[Complex<F>; 36],
    output: &mut [MaybeUninit<Complex<F>>; 36],
) where
    F: LaneScalar + MixedRadixScalar + eunomia::layout::Pod,
    A: SimdArch + SimdKernel<F>,
    Complex<F>: eunomia::layout::Pod,
{
    let thirds = Thirds::new(simd);

    // The radix-4 across the row registers, per register group.
    let first = radix4::<F, A, INVERSE>([
        load(simd, &input[0..4]),
        load(simd, &input[9..13]),
        load(simd, &input[18..22]),
        load(simd, &input[27..31]),
    ]);
    let mut middle = radix4::<F, A, INVERSE>([
        load(simd, &input[1..5]),
        load(simd, &input[10..14]),
        load(simd, &input[19..23]),
        load(simd, &input[28..32]),
    ]);
    let mut last = radix4::<F, A, INVERSE>([
        load(simd, &input[5..9]),
        load(simd, &input[14..18]),
        load(simd, &input[23..27]),
        load(simd, &input[32..36]),
    ]);
    // `W_36^{c k1}` on row `k1`; column 0 carries `W^0`.
    middle[1] = twiddled(simd, middle[1], &twiddles.rows[0]);
    last[1] = twiddled(simd, last[1], &twiddles.rows[1]);
    middle[2] = twiddled(simd, middle[2], &twiddles.rows[2]);
    last[2] = twiddled(simd, last[2], &twiddles.rows[3]);
    middle[3] = twiddled(simd, middle[3], &twiddles.rows[4]);
    last[3] = twiddled(simd, last[3], &twiddles.rows[5]);

    // The 9x4 tile to 4x9: column 0 is lane 0 of the first group's four
    // registers, two pair decimations; the other columns are two square
    // transposes.
    let (even01, _) = first[0]
        .into_interleaved()
        .deinterleave_pairs(first[1].into_interleaved());
    let (even23, _) = first[2]
        .into_interleaved()
        .deinterleave_pairs(first[3].into_interleaved());
    let (column0, _) = even01.deinterleave_pairs(even23);
    ComplexReg::transpose_square(&mut middle);
    ComplexReg::transpose_square(&mut last);
    let columns = [
        ComplexReg::from_interleaved(column0),
        middle[0],
        middle[1],
        middle[2],
        middle[3],
        last[0],
        last[1],
        last[2],
        last[3],
    ];

    // The radix-9 as three by three: DFT-3s over the stride-3 columns, the
    // ninth-turn twiddles, DFT-3s across, each output register the four
    // consecutive outputs `4 k2 .. 4 k2 + 4`.
    let m0 = radix3::<F, A, INVERSE>([columns[0], columns[3], columns[6]], &thirds);
    let mut m1 = radix3::<F, A, INVERSE>([columns[1], columns[4], columns[7]], &thirds);
    let mut m2 = radix3::<F, A, INVERSE>([columns[2], columns[5], columns[8]], &thirds);
    m1[1] = twiddled(simd, m1[1], &twiddles.ninths[0]);
    m1[2] = twiddled(simd, m1[2], &twiddles.ninths[1]);
    m2[1] = twiddled(simd, m2[1], &twiddles.ninths[1]);
    m2[2] = twiddled(simd, m2[2], &twiddles.ninths[2]);
    let [x0, x3, x6] = radix3::<F, A, INVERSE>([m0[0], m1[0], m2[0]], &thirds);
    let [x1, x4, x7] = radix3::<F, A, INVERSE>([m0[1], m1[1], m2[1]], &thirds);
    let [x2, x5, x8] = radix3::<F, A, INVERSE>([m0[2], m1[2], m2[2]], &thirds);

    let (o0, rest) = output
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o1, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o2, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o3, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o4, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o5, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o6, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o7, rest) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    let (o8, _) = rest
        .split_first_chunk_mut::<4>()
        .expect("invariant: nine registers of four fill thirty-six");
    store_uninit(x0, o0);
    store_uninit(x1, o1);
    store_uninit(x2, o2);
    store_uninit(x3, o3);
    store_uninit(x4, o4);
    store_uninit(x5, o5);
    store_uninit(x6, o6);
    store_uninit(x7, o7);
    store_uninit(x8, o8);
}
