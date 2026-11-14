//! Axis-set spectrum shifts over Leto arrays.
//!
//! For shape `[n_0, …, n_{N-1}]` and axis set `A`, `y[k] = x[σ(k)]` with
//! `σ(k)_a = (k_a + n_a − s_a) mod n_a` for `a ∈ A` and `k_a` otherwise: `numpy.roll` by
//! `s_a` per axis, the convention of `numpy.fft.fftshift(x, axes)`. `s_a` is `⌊n_a/2⌋` for
//! [`fftshift_axes`](crate::fftshift_axes) and `⌈n_a/2⌉` for
//! [`ifftshift_axes`](crate::ifftshift_axes); since `⌊n/2⌋ + ⌈n/2⌉ = n` the pair
//! is the identity at every extent, and the two differ at odd extents. Elements move without
//! arithmetic, so the shifts are exact for every element type.

use crate::domain::contracts::error::{ApolloError, ApolloResult};
use leto::{Array, ArrayView, MnemosyneStorage};

/// Move the zero-frequency component of every axis in `axes` to that axis's center.
///
/// The view may have any layout; it is read in logical row-major order and the result is
/// a C-contiguous Mnemosyne-backed array of the same shape. An empty `axes` copies the
/// field. Odd extents rotate by `⌊n/2⌋`, so zero lands at `(n − 1) / 2`.
///
/// # Errors
///
/// [`ApolloError::Validation`] with field `axes` when an axis is not below the rank `N` or
/// repeats.
///
/// # Examples
///
/// ```
/// use leto::{Array, MnemosyneStorage};
///
/// let field = Array::<u8, MnemosyneStorage<u8>, 2>::from_mnemosyne_vec([2, 3], (0..6).collect())?;
/// let shifted = apollo_fft::fftshift_axes(field.view(), &[1])?;
/// let values: Vec<u8> = shifted.view().iter().copied().collect();
/// assert_eq!(values, [2, 0, 1, 5, 3, 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn fftshift_axes<T: Copy, const N: usize>(
    field: ArrayView<'_, T, N>,
    axes: &[usize],
) -> ApolloResult<Array<T, MnemosyneStorage<T>, N>> {
    rotate_axes(field, axes, |extent| extent / 2)
}

/// Undo [`fftshift_axes`]: move the zero-frequency component of every axis in `axes` back to bin 0.
///
/// Odd extents rotate by `⌈n/2⌉`, so the pair is the identity at every extent. See
/// [`fftshift_axes`] for layout and the empty axis set.
///
/// # Errors
///
/// [`ApolloError::Validation`] with field `axes` when an axis is not below the rank `N` or
/// repeats.
pub fn ifftshift_axes<T: Copy, const N: usize>(
    field: ArrayView<'_, T, N>,
    axes: &[usize],
) -> ApolloResult<Array<T, MnemosyneStorage<T>, N>> {
    rotate_axes(field, axes, |extent| extent.div_ceil(2))
}

/// Rotate each axis of `axes` by `rotation(extent)` positions toward higher indices.
fn rotate_axes<T: Copy, const N: usize>(
    field: ArrayView<'_, T, N>,
    axes: &[usize],
    rotation: impl Fn(usize) -> usize,
) -> ApolloResult<Array<T, MnemosyneStorage<T>, N>> {
    let mut rotated = [false; N];
    for &axis in axes {
        let Some(flag) = rotated.get_mut(axis).filter(|flag| !**flag) else {
            return Err(ApolloError::validation(
                "axes",
                axis.to_string(),
                format!("each axis is below the rank {N} and appears once"),
            ));
        };
        *flag = true;
    }

    let shape = field.shape();
    let source_values: Vec<T> = field.iter().copied().collect();
    let mut strides = [0; N];
    let mut stride = 1;
    for (slot, &extent) in strides.iter_mut().zip(&shape).rev() {
        *slot = stride;
        stride *= extent;
    }
    // `source_offsets[a][k]` is the C-order offset along axis `a` of the element that
    // lands at position `k`.
    let source_offsets: [Vec<usize>; N] = std::array::from_fn(|axis| {
        let extent = shape[axis];
        let roll = if rotated[axis] { rotation(extent) } else { 0 };
        (0..extent)
            .map(|k| ((k + extent - roll) % extent) * strides[axis])
            .collect()
    });

    let mut values = Vec::with_capacity(source_values.len());
    let mut position = [0; N];
    for _ in 0..source_values.len() {
        let source: usize = source_offsets
            .iter()
            .zip(&position)
            .map(|(offsets, &k)| offsets[k])
            .sum();
        values.push(source_values[source]);
        for (k, &extent) in position.iter_mut().zip(&shape).rev() {
            *k += 1;
            if *k < extent {
                break;
            }
            *k = 0;
        }
    }
    Ok(
        Array::<T, MnemosyneStorage<T>, N>::from_mnemosyne_vec(shape, values)
            .expect("invariant: the rotated values fill the field's own shape"),
    )
}
