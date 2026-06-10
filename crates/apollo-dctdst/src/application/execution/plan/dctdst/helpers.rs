use crate::domain::contracts::error::{DctDstError, DctDstResult};
use apollo_fft::PrecisionProfile;
use ndarray::{Array2, Array3};
use std::borrow::Cow;

pub(crate) fn leto_view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    match view.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(
            (0..view.shape()[0])
                .map(|index| {
                    *view
                        .get([index])
                        .expect("Leto DCT/DST view index must be valid after shape validation")
                })
                .collect(),
        ),
    }
}

pub(crate) fn array2_from_leto_view(input: leto::ArrayView2<'_, f64>) -> Array2<f64> {
    let [rows, cols] = input.shape();
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        *input
            .get([row, col])
            .expect("Leto 2D DCT/DST view index must be valid after shape validation")
    })
}

pub(crate) fn array3_from_leto_view(input: leto::ArrayView3<'_, f64>) -> Array3<f64> {
    let [d0, d1, d2] = input.shape();
    Array3::from_shape_fn((d0, d1, d2), |(i, j, k)| {
        *input
            .get([i, j, k])
            .expect("Leto 3D DCT/DST view index must be valid after shape validation")
    })
}

pub(crate) fn leto_array1_from_slice<T: Copy>(
    output: &[T],
) -> leto::Array<T, leto::MnemosyneStorage<T>, 1> {
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice([output.len()], output)
        .expect("DCT/DST output length must match Leto output shape")
}

pub(crate) fn leto_array2_from_ndarray(
    output: &Array2<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 2> {
    let (rows, cols) = output.dim();
    leto::Array::<f64, leto::MnemosyneStorage<f64>, 2>::from_mnemosyne_slice(
        [rows, cols],
        output
            .as_slice()
            .expect("DCT/DST-owned 2D ndarray output must be contiguous"),
    )
    .expect("DCT/DST 2D output length must match Leto output shape")
}

pub(crate) fn leto_array3_from_ndarray(
    output: &Array3<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 3> {
    let (d0, d1, d2) = output.dim();
    leto::Array::<f64, leto::MnemosyneStorage<f64>, 3>::from_mnemosyne_slice(
        [d0, d1, d2],
        output
            .as_slice()
            .expect("DCT/DST-owned 3D ndarray output must be contiguous"),
    )
    .expect("DCT/DST 3D output length must match Leto output shape")
}

pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> DctDstResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(DctDstError::PrecisionMismatch)
    }
}
