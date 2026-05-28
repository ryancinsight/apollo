//! Zero-copy real ↔ complex array fill helpers shared by every [`RealFftData`]
//! storage implementation.
//!
//! [`RealFftData`]: super::RealFftData

use half::f16;
use ndarray::{Array, Dimension, Zip};
use num_complex::{Complex32, Complex64};

#[inline]
pub(super) fn fill_complex64<D>(input: &Array<f64, D>, output: &mut Array<Complex64, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = Complex64::new(src, 0.0));
}

#[inline]
pub(super) fn fill_real64<D>(input: &Array<Complex64, D>, output: &mut Array<f64, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = src.re);
}

#[inline]
pub(super) fn fill_complex32<D>(input: &Array<f32, D>, output: &mut Array<Complex32, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = Complex32::new(src, 0.0));
}

#[inline]
pub(super) fn fill_real32<D>(input: &Array<Complex32, D>, output: &mut Array<f32, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = src.re);
}

#[inline]
pub(super) fn fill_complex32_from_f16<D>(input: &Array<f16, D>, output: &mut Array<Complex32, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "real-to-complex shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = Complex32::new(src.to_f32(), 0.0));
}

#[inline]
pub(super) fn fill_f16_from_complex32<D>(input: &Array<Complex32, D>, output: &mut Array<f16, D>)
where
    D: Dimension,
{
    debug_assert_eq!(
        input.shape(),
        output.shape(),
        "complex-to-real shape mismatch"
    );
    Zip::from(output.view_mut())
        .and(input.view())
        .for_each(|dst, &src| *dst = f16::from_f32(src.re));
}
