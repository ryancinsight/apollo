use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use apollo_fft::application::utilities::leto_interop;
use ndarray::{Array2, Array3};
use std::borrow::Cow;

pub(crate) fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    leto_interop::view1_cow(&view)
}
pub(crate) fn array2_from_leto_view(view: leto::ArrayView2<'_, f32>) -> Array2<f32> {
    leto_interop::array2_from_view(&view)
}
pub(crate) fn array3_from_leto_view(view: leto::ArrayView3<'_, f32>) -> Array3<f32> {
    leto_interop::array3_from_view(&view)
}
pub(crate) fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto_interop::try_array1_from_slice(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto DCT/DST 1D output".to_string(),
    })
}

pub(crate) fn leto_array2_from_ndarray(
    values: &Array2<f32>,
) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
    leto_interop::try_array2_from_ndarray(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto DCT/DST 2D output".to_string(),
    })
}

pub(crate) fn leto_array3_from_ndarray(
    values: &Array3<f32>,
) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
    leto_interop::try_array3_from_ndarray(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto DCT/DST 3D output".to_string(),
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
        let cow = leto_view1_cow(input.view());
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
        let cow = leto_view1_cow(view);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
