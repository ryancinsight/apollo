//! Exact eight-lane DFT-16 and DFT-32 codelets.
//!
//! Four interleaved complex samples occupy one eight-lane register. The
//! DFT-32 kernel views 32 samples as four rows of eight, computes eight
//! lane-wise DFT-4s, applies the six mixed-radix twiddle vectors, transposes
//! two 4×4 complex tiles, and finishes with four lane-wise DFT-8s. The DFT-16
//! kernel is the same construction over four rows of four: four lane-wise
//! DFT-4s, three twiddle vectors (the `W_16` powers are the even `W_32`
//! powers), one 4×4 transpose, and four lane-wise DFT-4s. Both stay register
//! resident and store natural order. Unsupported native widths decline
//! before either kernel observes the mutable operand.

use super::power::TWIDDLE32_FWD;
use crate::application::execution::kernel::components::register_butterfly::{radix4, radix8};
use eunomia::{Complex, Complex32};
use hermes_simd::{ComplexReg, LaneKernel, Simd, SimdArch, SimdKernel, Vector};

struct Dft16<'data, const INVERSE: bool> {
    data: &'data mut [Complex32; 16],
}

struct Dft32<'data, const INVERSE: bool> {
    data: &'data mut [Complex32; 32],
}

struct Dft32Rows<'data, const ROWS: usize, const INVERSE: bool> {
    data: &'data mut [[Complex32; 32]; ROWS],
}

#[expect(
    clippy::inline_always,
    reason = "the codelet must remain in the selected target-feature frame"
)]
#[inline(always)]
fn load<A>(simd: Simd<f32, A>, values: &[Complex32]) -> ComplexReg<f32, A>
where
    A: SimdArch + SimdKernel<f32>,
{
    let view = simd.view(eunomia::layout::cast_slice(values));
    ComplexReg::from_interleaved(Vector::from_view_chunk(&view, 0))
}

#[expect(
    clippy::inline_always,
    reason = "constant twiddles must fold into the selected target-feature frame"
)]
#[inline(always)]
fn twiddle<const INVERSE: bool, const INDEX: usize>() -> Complex32 {
    let value = TWIDDLE32_FWD[INDEX & 15];
    let sign = if INDEX >= 16 { -1.0 } else { 1.0 };
    let imaginary_sign = if INVERSE { -sign } else { sign };
    Complex::new((sign * value.re) as f32, (imaginary_sign * value.im) as f32)
}

#[expect(
    clippy::inline_always,
    reason = "constant twiddle loads must fold into the selected target-feature frame"
)]
#[inline(always)]
fn twiddles<
    A,
    const INVERSE: bool,
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
>(
    simd: Simd<f32, A>,
) -> ComplexReg<f32, A>
where
    A: SimdArch + SimdKernel<f32>,
{
    load::<A>(
        simd,
        &[
            twiddle::<INVERSE, I0>(),
            twiddle::<INVERSE, I1>(),
            twiddle::<INVERSE, I2>(),
            twiddle::<INVERSE, I3>(),
        ],
    )
}

#[expect(
    clippy::inline_always,
    reason = "stores must remain in the selected target-feature frame"
)]
#[inline(always)]
fn store<A>(value: ComplexReg<f32, A>, destination: &mut [Complex32])
where
    A: SimdArch + SimdKernel<f32>,
{
    value
        .into_interleaved()
        .store_unaligned_to_slice(eunomia::layout::cast_slice_mut(destination))
        .expect("invariant: four complex samples fill eight lanes");
}

#[expect(
    clippy::inline_always,
    reason = "the codelet must remain in the selected target-feature frame"
)]
#[inline(always)]
fn dft32_kernel<A, const INVERSE: bool>(simd: Simd<f32, A>, data: &mut [Complex32; 32])
where
    A: SimdArch + SimdKernel<f32>,
{
    let mut first = radix4::<f32, A, INVERSE>([
        load::<A>(simd, &data[0..4]),
        load::<A>(simd, &data[8..12]),
        load::<A>(simd, &data[16..20]),
        load::<A>(simd, &data[24..28]),
    ]);
    let mut second = radix4::<f32, A, INVERSE>([
        load::<A>(simd, &data[4..8]),
        load::<A>(simd, &data[12..16]),
        load::<A>(simd, &data[20..24]),
        load::<A>(simd, &data[28..32]),
    ]);
    first[1] = first[1] * twiddles::<A, INVERSE, 0, 1, 2, 3>(simd);
    second[1] = second[1] * twiddles::<A, INVERSE, 4, 5, 6, 7>(simd);
    first[2] = first[2] * twiddles::<A, INVERSE, 0, 2, 4, 6>(simd);
    second[2] = second[2] * twiddles::<A, INVERSE, 8, 10, 12, 14>(simd);
    first[3] = first[3] * twiddles::<A, INVERSE, 0, 3, 6, 9>(simd);
    second[3] = second[3] * twiddles::<A, INVERSE, 12, 15, 18, 21>(simd);
    ComplexReg::transpose_square(&mut first);
    ComplexReg::transpose_square(&mut second);
    let output = radix8::<f32, A, INVERSE>(
        [
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ],
        simd.splat(core::f32::consts::FRAC_1_SQRT_2),
    );

    store(output[0], &mut data[0..4]);
    store(output[1], &mut data[4..8]);
    store(output[2], &mut data[8..12]);
    store(output[3], &mut data[12..16]);
    store(output[4], &mut data[16..20]);
    store(output[5], &mut data[20..24]);
    store(output[6], &mut data[24..28]);
    store(output[7], &mut data[28..32]);
}

/// `x[4·b1 + b0]`: register `b1` holds lanes `b0`. A DFT-4 across the four
/// registers indexes `m`, the lane twiddle `W_16^{b0·m}` follows (row `m`
/// carries indices `2·b0·m` into the `W_32` table), the transpose hands each
/// register the four `b0` of one `m`, and the closing DFT-4 across registers
/// yields `X[4·m' + m]` with `m` in the lanes — natural order, contiguous
/// stores.
#[expect(
    clippy::inline_always,
    reason = "the codelet must remain in the selected target-feature frame"
)]
#[inline(always)]
fn dft16_kernel<A, const INVERSE: bool>(simd: Simd<f32, A>, data: &mut [Complex32; 16])
where
    A: SimdArch + SimdKernel<f32>,
{
    let mut rows = radix4::<f32, A, INVERSE>([
        load::<A>(simd, &data[0..4]),
        load::<A>(simd, &data[4..8]),
        load::<A>(simd, &data[8..12]),
        load::<A>(simd, &data[12..16]),
    ]);
    rows[1] = rows[1] * twiddles::<A, INVERSE, 0, 2, 4, 6>(simd);
    rows[2] = rows[2] * twiddles::<A, INVERSE, 0, 4, 8, 12>(simd);
    rows[3] = rows[3] * twiddles::<A, INVERSE, 0, 6, 12, 18>(simd);
    ComplexReg::transpose_square(&mut rows);
    let output = radix4::<f32, A, INVERSE>(rows);
    store(output[0], &mut data[0..4]);
    store(output[1], &mut data[4..8]);
    store(output[2], &mut data[8..12]);
    store(output[3], &mut data[12..16]);
}

impl<const INVERSE: bool> LaneKernel<f32> for Dft16<'_, INVERSE> {
    type Output = ();
    #[expect(
        clippy::inline_always,
        reason = "the codelet must remain in the selected target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) {
        dft16_kernel::<A, INVERSE>(simd, self.data);
    }
}

impl<const INVERSE: bool> LaneKernel<f32> for Dft32<'_, INVERSE> {
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "the codelet must remain in the selected target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) {
        dft32_kernel::<A, INVERSE>(simd, self.data);
    }
}

impl<const ROWS: usize, const INVERSE: bool> LaneKernel<f32> for Dft32Rows<'_, ROWS, INVERSE> {
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "the row batch must remain in one selected target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) {
        for row in self.data {
            dft32_kernel::<A, INVERSE>(simd, row);
        }
    }
}

/// Runs the DFT-16 codelet only when a native eight-lane backend exists.
pub(crate) fn try_dft16_hardware<const INVERSE: bool>(data: &mut [Complex32; 16]) -> bool {
    hermes_simd::vectorize_hardware_lanes::<8, f32, _>(Dft16::<INVERSE> { data }).is_some()
}

/// Runs the DFT-32 codelet only when a native eight-lane backend exists.
pub(crate) fn try_dft32_hardware<const INVERSE: bool>(data: &mut [Complex32; 32]) -> bool {
    hermes_simd::vectorize_hardware_lanes::<8, f32, _>(Dft32::<INVERSE> { data }).is_some()
}

/// Runs a homogeneous DFT-32 row batch inside one eight-lane target frame.
pub(crate) fn try_dft32_rows_hardware<const ROWS: usize, const INVERSE: bool>(
    data: &mut [[Complex32; 32]; ROWS],
) -> bool {
    hermes_simd::vectorize_hardware_lanes::<8, f32, _>(Dft32Rows::<ROWS, INVERSE> { data })
        .is_some()
}
