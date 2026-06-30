//! Canonical Leto ↔ slice/ndarray interop helpers shared by every Apollo
//! transform crate.
//!
//! These helpers are error-agnostic: fallible conversions return [`Option`]
//! so each consumer maps `None` onto its own domain error type. Contiguous
//! views are always borrowed (`Cow::Borrowed`); only strided views are
//! materialized, preserving zero-copy on the common path.

use std::borrow::Cow;

use leto::{Array2, Array3};

/// Borrow a contiguous Leto 1D view or materialize a strided one.
///
/// Contiguous storage yields `Cow::Borrowed` with no copy; strided views are
/// gathered once into logical order.
#[must_use]
#[inline]
pub fn view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    match view.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => {
            let len = view.shape()[0];
            let mut values = Vec::with_capacity(len);
            values.extend((0..len).map(|index| {
                *view
                    .get([index])
                    .expect("Leto 1D view index must be valid after shape validation")
            }));
            Cow::Owned(values)
        }
    }
}

/// Copy a Leto 2D view of any stride into a dense `leto::Array2`.
#[must_use]
pub fn array2_from_view<T: Copy>(view: &leto::ArrayView2<'_, T>) -> Array2<T> {
    let [rows, cols] = view.shape();
    Array2::from_shape_fn([rows, cols], |[row, col]| {
        *view
            .get([row, col])
            .expect("Leto 2D view index must be valid after shape validation")
    })
}

/// Copy a Leto 3D view of any stride into a dense `leto::Array3`.
#[must_use]
pub fn array3_from_view<T: Copy>(view: &leto::ArrayView3<'_, T>) -> Array3<T> {
    let [d0, d1, d2] = view.shape();
    Array3::from_shape_fn([d0, d1, d2], |[i, j, k]| {
        *view
            .get([i, j, k])
            .expect("Leto 3D view index must be valid after shape validation")
    })
}

/// Build a Leto 1D array from a slice; `None` when the length is rejected.
#[must_use]
#[inline]
pub fn try_array1_from_slice<T: Copy>(
    output: &[T],
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice([output.len()], output)
        .ok()
}

/// Build a Leto 2D array from a standard-layout ndarray; `None` when the
/// source is non-contiguous or the shape is rejected.
#[must_use]
pub fn try_array2_from_ndarray<T: Copy>(
    output: &Array2<T>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
    let [rows, cols] = output.shape();
    leto::Array::<T, leto::MnemosyneStorage<T>, 2>::from_mnemosyne_slice(
        [rows, cols],
        output.as_slice()?,
    )
    .ok()
}

/// Build a Leto 3D array from a standard-layout ndarray; `None` when the
/// source is non-contiguous or the shape is rejected.
#[must_use]
pub fn try_array3_from_ndarray<T: Copy>(
    output: &Array3<T>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
    let [d0, d1, d2] = output.shape();
    leto::Array::<T, leto::MnemosyneStorage<T>, 3>::from_mnemosyne_slice(
        [d0, d1, d2],
        output.as_slice()?,
    )
    .ok()
}

/// Whether `actual` matches the `expected` precision profile in both storage
/// and compute dimensions. Consumers map `false` to their domain error.
#[must_use]
#[inline]
pub fn profile_matches(actual: crate::PrecisionProfile, expected: crate::PrecisionProfile) -> bool {
    actual.storage == expected.storage && actual.compute == expected.compute
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view1_cow_borrows_contiguous_views() {
        let array = leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::from_mnemosyne_slice(
            [4],
            &[1.0, 2.0, 3.0, 4.0],
        )
        .expect("contiguous source must build");
        let cow = view1_cow(&array.view());
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn try_array1_round_trips_values() {
        let restored = try_array1_from_slice(&[5.0_f64, 6.0, 7.0]).expect("valid length");
        assert_eq!(restored.shape(), [3]);
        assert_eq!(*restored.get([1]).expect("index valid"), 6.0);
    }
}
