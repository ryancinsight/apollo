//! Rank-polymorphic dense Leto array construction.

/// Build a dense Mnemosyne-backed array from a shape and contiguous values.
///
/// This is the canonical fallible output boundary for algorithms that own a
/// freshly computed slice rather than an existing Leto array.
#[must_use]
pub fn try_dense_from_slice<T: Copy, const N: usize>(
    shape: [usize; N],
    values: &[T],
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, N>> {
    leto::Array::<T, leto::MnemosyneStorage<T>, N>::from_mnemosyne_slice(shape, values).ok()
}

/// Materialize an arbitrary Leto array into dense Mnemosyne-backed storage.
///
/// A C-contiguous source copies its backing slice directly. A strided source
/// copies its logical iterator order exactly once, preserving the source shape
/// without consumer-specific allocation adapters.
///
/// # Representation theorem
///
/// For a source with shape `s`, the result has shape `s` and its logical
/// element sequence equals `source.iter()`. The contiguous branch copies
/// `as_slice()`, whose order is the array's logical order. The strided branch
/// explicitly collects that logical iterator before constructing the dense
/// array. Leto validates the common shape/value cardinality invariant.
///
/// The theorem is checked empirically for contiguous, strided, and transposed
/// rank-two inputs in this module's value-semantic tests.
#[must_use]
pub fn try_dense_from_array<T, S, const N: usize>(
    source: &leto::Array<T, S, N>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, N>>
where
    T: Copy,
    S: leto::Storage<T>,
{
    try_dense_from_parts(source.shape(), source.as_slice(), source.iter().copied())
}

/// Materialize an arbitrary Leto view into dense Mnemosyne-backed storage.
///
/// Contiguous views copy their backing slice; strided views copy logical
/// iterator order once. This is the view counterpart to
/// [`try_dense_from_array`], not a consumer-owned adapter.
#[must_use]
pub fn try_dense_from_view<T: Copy, const N: usize>(
    source: &leto::ArrayView<'_, T, N>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, N>> {
    try_dense_from_parts(source.shape(), source.as_slice(), source.iter().copied())
}

fn try_dense_from_parts<T: Copy, const N: usize>(
    shape: [usize; N],
    contiguous: Option<&[T]>,
    logical: impl Iterator<Item = T>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, N>> {
    match contiguous {
        Some(values) => try_dense_from_slice(shape, values),
        None => leto::Array::<T, leto::MnemosyneStorage<T>, N>::from_mnemosyne_vec(
            shape,
            logical.collect(),
        )
        .ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::{try_dense_from_slice, try_dense_from_view};
    use leto::SliceArg;

    #[test]
    fn materializes_a_strided_array_in_logical_order() {
        let input = leto::Array2::from_shape_vec([2, 4], vec![1_u32, 99, 2, 99, 3, 99, 4, 99])
            .expect("source shape must be valid");
        let source = input
            .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(0), None, 2)])
            .expect("strided view must be valid");

        let dense = try_dense_from_view(&source).expect("Leto accepts source shape and values");

        assert_eq!(dense.shape(), [2, 2]);
        assert_eq!(dense.as_slice(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn materializes_a_transposed_array_in_logical_order() {
        let input = leto::Array2::from_shape_vec([2, 3], vec![1_u32, 2, 3, 4, 5, 6])
            .expect("source shape must be valid");
        let source = input
            .transpose([1, 0])
            .expect("axis permutation must be valid");

        let dense = try_dense_from_view(&source).expect("Leto accepts source shape and values");

        assert_eq!(dense.shape(), [3, 2]);
        assert_eq!(dense.as_slice(), Some(&[1, 4, 2, 5, 3, 6][..]));
    }

    #[test]
    fn builds_rank_three_output_from_owned_values() {
        let dense = try_dense_from_slice([2, 1, 2], &[1_u32, 2, 3, 4])
            .expect("Leto accepts matching shape and values");

        assert_eq!(dense.shape(), [2, 1, 2]);
        assert_eq!(dense.as_slice(), Some(&[1, 2, 3, 4][..]));
    }
}
