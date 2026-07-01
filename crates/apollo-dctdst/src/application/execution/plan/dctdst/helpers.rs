use crate::domain::contracts::error::{DctDstError, DctDstResult};
use apollo_fft::application::utilities::leto_interop;
use apollo_fft::PrecisionProfile;
use leto::{Array2, Array3};
use std::borrow::Cow;

pub(crate) fn leto_view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    leto_interop::view1_cow(view)
}

pub(crate) fn leto_array1_from_slice<T: Copy>(
    output: &[T],
) -> leto::Array<T, leto::MnemosyneStorage<T>, 1> {
    leto_interop::try_array1_from_slice(output)
        .expect("DCT/DST output length must match Leto output shape")
}

pub(crate) fn leto_array2_from_dense(
    output: &Array2<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 2> {
    leto_interop::try_dense_from_contiguous(output)
        .expect("DCT/DST-owned 2D dense output must be contiguous with matching Leto shape")
}

pub(crate) fn leto_array3_from_dense(
    output: &Array3<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 3> {
    leto_interop::try_dense_from_contiguous(output)
        .expect("DCT/DST-owned 3D dense output must be contiguous with matching Leto shape")
}

pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> DctDstResult<()> {
    if leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(DctDstError::PrecisionMismatch)
    }
}
