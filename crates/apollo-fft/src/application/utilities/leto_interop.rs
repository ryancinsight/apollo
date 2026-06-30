//! Canonical Leto interop helpers shared by every Apollo transform crate.
//!
//! These helpers are error-agnostic: fallible conversions return [`Option`]
//! so each consumer maps `None` onto its own domain error type. Contiguous
//! views are always borrowed (`Cow::Borrowed`); only strided views are
//! materialized, preserving zero-copy on the common path.
//!
//! Rank-polymorphic by design: one [`try_dense_from_contiguous`] covers every
//! rank `N` (monomorphized per use), and view→owned materialization defers to
//! [`leto::ArrayView::to_contiguous`] rather than a per-rank copy helper.

use std::borrow::Cow;

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

/// Build a Leto 1D array from a slice; `None` when the length is rejected.
#[must_use]
#[inline]
pub fn try_array1_from_slice<T: Copy>(
    output: &[T],
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice([output.len()], output)
        .ok()
}

/// Build a dense, mnemosyne-backed Leto array from any **C-contiguous** Leto
/// array of the same rank — one rank-polymorphic entry point replacing the
/// former per-rank `try_array{2,3}_from_*` helpers. `None` when the source is
/// non-contiguous (no `as_slice`) or the shape is rejected.
///
/// Generic over rank `N` and source storage `S`; each instantiation
/// monomorphizes to the same code the hand-written per-rank helper would.
#[must_use]
pub fn try_dense_from_contiguous<T, S, const N: usize>(
    source: &leto::Array<T, S, N>,
) -> Option<leto::Array<T, leto::MnemosyneStorage<T>, N>>
where
    T: Copy,
    S: leto::Storage<T>,
{
    leto::Array::<T, leto::MnemosyneStorage<T>, N>::from_mnemosyne_slice(
        source.shape(),
        source.as_slice()?,
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
