//! The 36-point transform held in registers, at either register width.
//!
//! Every lane a sample of the same transform — RustFFT's `Butterfly36Avx`
//! shape over hermes registers — the base of the 180 column route (ADR
//! 0062). At four complexes a register (`f32` on AVX2) the tile is four rows
//! of nine: a radix-4 across the four row registers, the 9x4 tile transposed
//! to 4x9, a radix-9 (three by three) across the nine column registers; each
//! row of nine loads as the register at the row start (only column 0 used)
//! and the full registers of columns 1 to 4 and 5 to 8. At two complexes a
//! register (`f64` on AVX2) the tile is six by six: a radix-6 across the six
//! row registers, nine 2x2 transposes, a radix-6 across the six column
//! registers, every row three registers exactly.

use crate::application::execution::kernel::components::register_butterfly::{
    radix3, radix4, radix6, Thirds,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use core::mem::MaybeUninit;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// One register of twiddles as its interleaved lanes, direct and with each
/// sample's real and imaginary lanes exchanged, the pair
/// [`ComplexReg::mul_with_swapped`] takes. A row holds eight lanes; a
/// narrower register reads its leading lanes.
#[derive(Clone, Copy)]
pub(crate) struct TwiddleRow<F> {
    pub(crate) direct: [F; 8],
    pub(crate) swapped: [F; 8],
}

impl<F: MixedRadixScalar> TwiddleRow<F> {
    /// The `n`-th roots `W_n^{e}` for up to four exponents, forward or
    /// inverse, the unused samples zero.
    pub(crate) fn of_roots(n: usize, exponents: &[usize], inverse: bool) -> Self {
        debug_assert!(exponents.len() <= 4, "a row holds four samples");
        let mut direct = [F::from_precise(0.0); 8];
        let mut swapped = [F::from_precise(0.0); 8];
        for (sample, &exponent) in exponents.iter().enumerate() {
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

/// The twiddle rows of one direction of the 36-point kernel, both layouts.
pub(crate) struct Twiddles36<F> {
    /// Four complexes a register: `W_36^{c k1}` for `k1` in 1 to 3, the
    /// columns 1 to 4 then 5 to 8.
    rows: [TwiddleRow<F>; 6],
    /// `W_9^1`, `W_9^2`, `W_9^4` broadcast.
    ninths: [TwiddleRow<F>; 3],
    /// Two complexes a register: `W_36^{c k1}` for `k1` in 1 to 5, the
    /// column pairs `(0, 1)`, `(2, 3)`, `(4, 5)`, at `3 (k1 - 1) + pair`.
    pairs: [TwiddleRow<F>; 15],
}

impl<F: MixedRadixScalar> Twiddles36<F> {
    pub(crate) fn new(inverse: bool) -> Self {
        let row = |k1: usize, first_column: usize| {
            TwiddleRow::of_roots(
                36,
                &[
                    first_column * k1,
                    (first_column + 1) * k1,
                    (first_column + 2) * k1,
                    (first_column + 3) * k1,
                ],
                inverse,
            )
        };
        let pair = |k1: usize, first_column: usize| {
            TwiddleRow::of_roots(36, &[first_column * k1, (first_column + 1) * k1], inverse)
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
                TwiddleRow::of_roots(9, &[1; 4], inverse),
                TwiddleRow::of_roots(9, &[2; 4], inverse),
                TwiddleRow::of_roots(9, &[4; 4], inverse),
            ],
            pairs: [
                pair(1, 0),
                pair(1, 2),
                pair(1, 4),
                pair(2, 0),
                pair(2, 2),
                pair(2, 4),
                pair(3, 0),
                pair(3, 2),
                pair(3, 4),
                pair(4, 0),
                pair(4, 2),
                pair(4, 4),
                pair(5, 0),
                pair(5, 2),
                pair(5, 4),
            ],
        }
    }
}

/// The complexes one register of `A` holds for `F`.
pub(crate) const fn complexes_per_register<F, A>() -> usize
where
    F: LaneScalar,
    A: SimdArch + SimdKernel<F>,
{
    <A as SimdStorage<F>>::LANE_COUNT / 2
}

/// Loads one register of complexes.
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

/// Loads a row's leading lanes as one register.
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

/// Stores one register of complexes.
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
        .expect("invariant: the destination holds one register of complexes");
}

/// Stores one register of complexes into uninitialized memory.
#[expect(
    clippy::inline_always,
    reason = "the store must stay in the selected target-feature frame"
)]
#[inline(always)]
pub(crate) fn store_uninit<F, A>(
    value: ComplexReg<F, A>,
    destination: &mut [MaybeUninit<Complex<F>>],
) where
    F: LaneScalar,
    A: SimdArch + SimdKernel<F>,
{
    debug_assert_eq!(
        destination.len(),
        complexes_per_register::<F, A>(),
        "invariant: the destination holds one register of complexes"
    );
    // SAFETY: the destination is one register of complexes, `LANE_COUNT`
    // lanes of `F`; writing them is the initialization the caller reads back
    // only after every register of the block has landed.
    unsafe {
        value
            .into_interleaved()
            .store_unaligned(destination.as_mut_ptr().cast::<F>());
    }
}

/// The 36-point transform of `input` into `output`, natural order both ways,
/// in the layout of the register width: four rows of nine at four complexes
/// a register, six by six at two.
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
    match complexes_per_register::<F, A>() {
        4 => four_by_nine::<F, A, INVERSE>(simd, twiddles, input, output),
        2 => six_by_six::<F, A, INVERSE>(simd, twiddles, input, output),
        _ => unreachable!(
            "invariant: the column route builds only at four or two complexes a register"
        ),
    }
}

/// Four rows of nine at four complexes a register.
#[expect(
    clippy::inline_always,
    reason = "the codelet must remain in the selected target-feature frame"
)]
#[inline(always)]
fn four_by_nine<F, A, const INVERSE: bool>(
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

    store_uninit(x0, &mut output[0..4]);
    store_uninit(x1, &mut output[4..8]);
    store_uninit(x2, &mut output[8..12]);
    store_uninit(x3, &mut output[12..16]);
    store_uninit(x4, &mut output[16..20]);
    store_uninit(x5, &mut output[20..24]);
    store_uninit(x6, &mut output[24..28]);
    store_uninit(x7, &mut output[28..32]);
    store_uninit(x8, &mut output[32..36]);
}

/// Six by six at two complexes a register.
///
/// `x[6 r + c]`: register `(r, pair)` holds columns `2 pair, 2 pair + 1` of
/// row `r`. The radix-6 across the six row registers of a pair indexes `k1`
/// with `W_36^{c k1}` after it; nine 2x2 transposes hand each column its
/// six rows as three registers of two; the radix-6 across those yields
/// `X[k1 + 6 k2]`, register `k2` of row-pair `p` at `6 k2 + 2 p`.
#[expect(
    clippy::inline_always,
    reason = "the codelet must remain in the selected target-feature frame"
)]
#[inline(always)]
fn six_by_six<F, A, const INVERSE: bool>(
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

    let mut groups = [
        radix6::<F, A, INVERSE>(
            [
                load(simd, &input[0..2]),
                load(simd, &input[6..8]),
                load(simd, &input[12..14]),
                load(simd, &input[18..20]),
                load(simd, &input[24..26]),
                load(simd, &input[30..32]),
            ],
            &thirds,
        ),
        radix6::<F, A, INVERSE>(
            [
                load(simd, &input[2..4]),
                load(simd, &input[8..10]),
                load(simd, &input[14..16]),
                load(simd, &input[20..22]),
                load(simd, &input[26..28]),
                load(simd, &input[32..34]),
            ],
            &thirds,
        ),
        radix6::<F, A, INVERSE>(
            [
                load(simd, &input[4..6]),
                load(simd, &input[10..12]),
                load(simd, &input[16..18]),
                load(simd, &input[22..24]),
                load(simd, &input[28..30]),
                load(simd, &input[34..36]),
            ],
            &thirds,
        ),
    ];
    // `W_36^{c k1}` on row `k1` of each column pair.
    for k1 in 1..6 {
        groups[0][k1] = twiddled(simd, groups[0][k1], &twiddles.pairs[3 * (k1 - 1)]);
        groups[1][k1] = twiddled(simd, groups[1][k1], &twiddles.pairs[3 * (k1 - 1) + 1]);
        groups[2][k1] = twiddled(simd, groups[2][k1], &twiddles.pairs[3 * (k1 - 1) + 2]);
    }

    // Nine 2x2 transposes: rows `(2 p, 2 p + 1)` of column pair `g` become
    // columns `2 g, 2 g + 1` at rows `(2 p, 2 p + 1)`.
    let mut tile = [ComplexReg::zero(); 2];
    let mut columns = [[ComplexReg::zero(); 3]; 6];
    for (pair, group) in groups.iter().enumerate() {
        for p in 0..3 {
            tile[0] = group[2 * p];
            tile[1] = group[2 * p + 1];
            ComplexReg::transpose_square(&mut tile);
            columns[2 * pair][p] = tile[0];
            columns[2 * pair + 1][p] = tile[1];
        }
    }

    // The radix-6 across the six columns, per row pair; register `k2` holds
    // `X[k1 + 6 k2]` for the pair's two `k1`.
    for p in 0..3 {
        let out = radix6::<F, A, INVERSE>(
            [
                columns[0][p],
                columns[1][p],
                columns[2][p],
                columns[3][p],
                columns[4][p],
                columns[5][p],
            ],
            &thirds,
        );
        for (k2, register) in out.into_iter().enumerate() {
            let at = 6 * k2 + 2 * p;
            store_uninit(register, &mut output[at..at + 2]);
        }
    }
}
