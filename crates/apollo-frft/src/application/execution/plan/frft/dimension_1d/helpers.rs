//! 1D Fractional Fourier Transform helpers.

use std::borrow::Cow;

#[must_use]
#[inline]
pub(crate) fn leto_view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    apollo_fft::application::utilities::leto_interop::view1_cow(view)
}
