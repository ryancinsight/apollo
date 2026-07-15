//! One-dimensional Mnemosyne-backed Leto array construction.

/// Build a one-dimensional Mnemosyne-backed Leto array by copying `values`.
///
/// Returns [`None`] when Leto rejects the output shape.
#[must_use]
#[inline]
pub fn try_array1_from_slice<T: Copy>(
    values: &[T],
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice([values.len()], values)
        .ok()
}

/// Build a one-dimensional Mnemosyne-backed Leto array by moving `values`.
///
/// The vector elements are moved into Mnemosyne storage rather than cloned.
/// Returns [`None`] when Leto rejects the output shape.
#[must_use]
#[inline]
pub fn try_array1_from_vec<T>(
    values: Vec<T>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_vec([values.len()], values).ok()
}

#[cfg(test)]
mod tests {
    use super::{try_array1_from_slice, try_array1_from_vec};

    #[test]
    fn copied_values_round_trip() {
        let output = try_array1_from_slice(&[5_u32, 6, 7]).expect("length is valid");

        assert_eq!(output.shape(), [3]);
        assert_eq!(output.as_slice(), Some(&[5, 6, 7][..]));
    }

    #[test]
    fn moved_values_round_trip() {
        let output = try_array1_from_vec(vec![8_u32, 9, 10]).expect("length is valid");

        assert_eq!(output.shape(), [3]);
        assert_eq!(output.as_slice(), Some(&[8, 9, 10][..]));
    }
}
