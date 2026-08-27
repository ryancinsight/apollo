use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_view_staging, PlanScratch,
};
use leto::{ArrayView2, ArrayViewMut, ArrayViewMut2, Layout};

/// Execute against a logical C-order view, staging only non-C-dense inputs.
///
/// A C-dense view, including one with a non-zero storage offset, is rewrapped
/// over its existing dense block and reaches `execute` without copying.
/// Fortran-dense and general strided views are assigned into a rank-disjoint
/// thread-local staging role so the nested axis-transpose scratch remains
/// available. Leto assignment preserves logical indices in both directions.
pub(super) fn with_c_order_view<T, const N: usize, R>(
    mut data: ArrayViewMut<'_, T, N>,
    execute: impl FnOnce(ArrayViewMut<'_, T, N>) -> R,
) -> R
where
    T: Copy + PlanScratch,
{
    let shape = data.shape();
    let layout = Layout::c_contiguous(shape)
        .expect("invariant: validated FFT view shape has a C-order layout");

    if let Some(slice) = data.as_mut_slice() {
        let view = ArrayViewMut::try_new(layout, slice)
            .expect("invariant: dense FFT view slice matches its logical shape");
        return execute(view);
    }

    with_view_staging::<T, N, _>(data.size(), |scratch| {
        let mut staged = ArrayViewMut::try_new(layout, scratch)
            .expect("invariant: FFT view staging matches its logical shape");
        staged.assign(&data.as_view());
        let result = execute(staged.reborrow());
        data.assign(&staged.as_view());
        result
    })
}

#[cfg(test)]
pub(crate) mod error_bound {
    const ROUNDED_OPS_PER_COMPLEX_TERM: usize = 8;

    fn gamma(operations: usize) -> f64 {
        let unit_roundoff = f64::EPSILON / 2.0;
        let scaled_roundoff = operations as f64 * unit_roundoff;
        assert!(
            scaled_roundoff < 1.0,
            "invariant: floating-point error model requires k * u < 1"
        );
        scaled_roundoff / (1.0 - scaled_roundoff)
    }

    fn transform_factor(dimensions: &[usize]) -> f64 {
        dimensions.iter().copied().fold(1.0, |factor, terms| {
            let operations = terms
                .checked_mul(ROUNDED_OPS_PER_COMPLEX_TERM)
                .expect("invariant: test transform operation count fits usize");
            factor * (1.0 + gamma(operations))
        }) - 1.0
    }

    fn element_count(dimensions: &[usize]) -> usize {
        dimensions.iter().copied().fold(1, |count, dimension| {
            count
                .checked_mul(dimension)
                .expect("invariant: test transform element count fits usize")
        })
    }

    /// Bound a separable transform against the direct discrete Fourier transform.
    ///
    /// The standard model `gamma(k) = k*u/(1-k*u)`, with unit roundoff
    /// `u = epsilon/2`, bounds a sequence of `k` rounded operations. One
    /// complex term contributes four real multiplications, two product sums,
    /// and two accumulator additions, so an `n`-term complex sum uses at most
    /// `8n` rounded operations. Separable axes compose multiplicatively. The
    /// direct reference is one sum over the complete input. The triangle
    /// inequality adds both routes' bounds, and `sqrt(2)` converts component
    /// error to complex magnitude.
    pub(crate) fn forward(input_l1: f64, dimensions: &[usize]) -> f64 {
        let transform = transform_factor(dimensions);
        let direct_operations = element_count(dimensions)
            .checked_mul(ROUNDED_OPS_PER_COMPLEX_TERM)
            .expect("invariant: direct-transform operation count fits usize");
        f64::sqrt(2.0) * (transform + gamma(direct_operations)) * input_l1
    }

    /// Bound a normalized forward/inverse separable-transform round trip.
    ///
    /// Two transform factors compose multiplicatively and normalization adds
    /// one rounded multiplication. Scaling by the input one-norm bounds every
    /// recovered output component by the triangle inequality.
    pub(crate) fn round_trip(input_l1: f64, dimensions: &[usize]) -> f64 {
        let transform = transform_factor(dimensions);
        let composed = (1.0 + transform).powi(2) * (1.0 + gamma(1)) - 1.0;
        f64::sqrt(2.0) * composed * input_l1
    }
}

/// Transpose one or more adjacent row-major matrices through Leto's assignment
/// kernel.
///
/// Each source matrix has shape `[rows, columns]`; each destination matrix has
/// shape `[columns, rows]`. Reinterpreting the row-major source as a
/// Fortran-contiguous view of the destination shape exposes the transpose
/// without an intermediate allocation.
pub(super) fn transpose_matrices<T: Copy>(
    source: &[T],
    destination: &mut [T],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) {
    let matrix_len = rows
        .checked_mul(columns)
        .expect("invariant: FFT matrix dimensions fit usize");
    let total_len = matrix_count
        .checked_mul(matrix_len)
        .expect("invariant: FFT matrix batch dimensions fit usize");
    assert_eq!(
        source.len(),
        total_len,
        "invariant: FFT transpose source length matches its shape"
    );
    assert_eq!(
        destination.len(),
        total_len,
        "invariant: FFT transpose destination length matches its shape"
    );
    if matrix_len == 0 || matrix_count == 0 {
        return;
    }

    let source_layout = Layout::f_contiguous([columns, rows])
        .expect("invariant: FFT transpose source layout fits isize");
    let destination_layout = Layout::c_contiguous([columns, rows])
        .expect("invariant: FFT transpose destination layout fits isize");

    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        let source_view = ArrayView2::try_new(source_layout, source_matrix)
            .expect("invariant: FFT transpose source storage matches its layout");
        let mut destination_view = ArrayViewMut2::try_new(destination_layout, destination_matrix)
            .expect("invariant: FFT transpose destination storage matches its layout");
        destination_view.assign(&source_view);
    }
}

#[cfg(test)]
mod tests {
    use super::{transpose_matrices, with_c_order_view};
    use eunomia::Complex64;
    use leto::{ArrayViewMut2, Layout};

    fn expected_transpose(
        source: &[usize],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Vec<usize> {
        let matrix_len = rows * columns;
        let mut expected = vec![0; source.len()];
        for matrix in 0..matrix_count {
            let base = matrix * matrix_len;
            for row in 0..rows {
                for column in 0..columns {
                    expected[base + column * rows + row] = source[base + row * columns + column];
                }
            }
        }
        expected
    }

    fn assert_transpose(matrix_count: usize, rows: usize, columns: usize) {
        let len = matrix_count * rows * columns;
        let source = (0..len).collect::<Vec<_>>();
        let expected = expected_transpose(&source, matrix_count, rows, columns);
        let mut destination = vec![usize::MAX; len];
        transpose_matrices(&source, &mut destination, matrix_count, rows, columns);
        assert_eq!(destination, expected);
    }

    #[test]
    fn transposes_rectangular_ragged_tiles() {
        assert_transpose(1, 35, 67);
        assert_transpose(1, 67, 35);
    }

    #[test]
    fn transposes_multiple_planes() {
        assert_transpose(3, 5, 7);
    }

    #[test]
    fn accepts_empty_and_singleton_matrices() {
        assert_transpose(0, 5, 7);
        assert_transpose(3, 0, 7);
        assert_transpose(3, 7, 0);
        assert_transpose(1, 1, 1);
    }

    #[test]
    fn c_order_offset_view_reuses_its_dense_block() {
        let layout = Layout::try_new([2, 3], [3, 1], 4).expect("valid offset layout");
        let mut storage = vec![Complex64::default(); 10];
        let expected = storage[4..].as_mut_ptr();
        let view = ArrayViewMut2::try_new(layout, &mut storage).expect("layout fits storage");

        with_c_order_view(view, |mut contiguous| {
            let actual = contiguous
                .as_mut_slice()
                .expect("helper supplies a C-order view")
                .as_mut_ptr();
            assert_eq!(actual, expected);
        });
    }
}
