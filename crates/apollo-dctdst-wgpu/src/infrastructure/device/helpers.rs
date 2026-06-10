use crate::domain::error::{WgpuError, WgpuResult};
use ndarray::{Array2, Array3};
use std::borrow::Cow;

pub(crate) fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let len = view.shape()[0];
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(*view.get([index]).map_err(|err| WgpuError::ShapeMismatch {
            message: format!("invalid Leto DCT/DST 1D view: {err:?}"),
        })?);
    }
    Ok(Cow::Owned(values))
}

pub(crate) fn array2_from_leto_view(view: leto::ArrayView2<'_, f32>) -> WgpuResult<Array2<f32>> {
    let shape = view.shape();
    let rows = shape[0];
    let cols = shape[1];
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(
                *view
                    .get([row, col])
                    .map_err(|err| WgpuError::ShapeMismatch {
                        message: format!("invalid Leto DCT/DST 2D view: {err:?}"),
                    })?,
            );
        }
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|err| WgpuError::ShapeMismatch {
        message: format!("failed to materialize Leto DCT/DST 2D view: {err}"),
    })
}

pub(crate) fn array3_from_leto_view(view: leto::ArrayView3<'_, f32>) -> WgpuResult<Array3<f32>> {
    let shape = view.shape();
    let d0 = shape[0];
    let d1 = shape[1];
    let d2 = shape[2];
    let mut values = Vec::with_capacity(d0 * d1 * d2);
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                values.push(
                    *view
                        .get([i, j, k])
                        .map_err(|err| WgpuError::ShapeMismatch {
                            message: format!("invalid Leto DCT/DST 3D view: {err:?}"),
                        })?,
                );
            }
        }
    }
    Array3::from_shape_vec((d0, d1, d2), values).map_err(|err| WgpuError::ShapeMismatch {
        message: format!("failed to materialize Leto DCT/DST 3D view: {err}"),
    })
}

pub(crate) fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto DCT/DST 1D output: {err:?}"),
        }
    })
}

pub(crate) fn leto_array2_from_ndarray(
    values: &Array2<f32>,
) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
    let (rows, cols) = values.dim();
    let flat: Vec<f32> = values.iter().copied().collect();
    leto::Array::from_mnemosyne_slice([rows, cols], &flat).map_err(|err| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto DCT/DST 2D output: {err:?}"),
    })
}

pub(crate) fn leto_array3_from_ndarray(
    values: &Array3<f32>,
) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
    let (d0, d1, d2) = values.dim();
    let flat: Vec<f32> = values.iter().copied().collect();
    leto::Array::from_mnemosyne_slice([d0, d1, d2], &flat).map_err(|err| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto DCT/DST 3D output: {err:?}"),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("input");
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0])
                .expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
