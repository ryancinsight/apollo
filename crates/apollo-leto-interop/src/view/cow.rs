//! Logical-order slice access for Leto views.

use std::borrow::Cow;

/// Borrow a C-contiguous Leto view or materialize another layout once.
///
/// For a view with logical row-major sequence `v_i`, the returned slice has
/// `result[i] == v_i` at every valid index. Leto exposes a dense C-order slice
/// exactly when borrowing is sound; otherwise its element iterator visits that
/// logical sequence, so collection preserves the same value contract in owned
/// storage. This is a proof sketch over Leto's view and iterator contracts,
/// not a machine-checked proof.
#[must_use]
#[inline]
pub fn view_cow<'a, T: Copy, const N: usize>(view: &leto::ArrayView<'a, T, N>) -> Cow<'a, [T]> {
    match view.as_slice() {
        Some(values) => Cow::Borrowed(values),
        None => Cow::Owned(view.iter().copied().collect()),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::view_cow;

    #[test]
    fn borrows_a_contiguous_view() {
        let array = leto::Array1::from_shape_vec([4], vec![1_u32, 2, 3, 4])
            .expect("array shape must be valid");
        let values = view_cow(&array.view());

        assert!(matches!(values, Cow::Borrowed(_)));
        assert_eq!(values.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn materializes_a_strided_view_in_logical_order() {
        let array = leto::Array1::from_shape_vec([8], vec![1_u32, 99, 2, 99, 3, 99, 4, 99])
            .expect("array shape must be valid");
        let view = array
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("stride must be valid");
        let values = view_cow(&view);

        assert!(matches!(values, Cow::Owned(_)));
        assert_eq!(values.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn materializes_a_transposed_view_in_logical_order() {
        let array = leto::Array2::from_shape_vec([2, 3], vec![1_u32, 2, 3, 4, 5, 6])
            .expect("array shape must be valid");
        let view = array.view().transpose([1, 0]).expect("axes are valid");
        let values = view_cow(&view);

        assert!(matches!(values, Cow::Owned(_)));
        assert_eq!(values.as_ref(), &[1, 4, 2, 5, 3, 6]);
    }
}
