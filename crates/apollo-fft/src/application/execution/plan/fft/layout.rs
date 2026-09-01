use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_view_staging, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use leto::{ArrayViewMut, Layout};
use leto_ops::transpose_complex_matrices;

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

/// Transpose one or more adjacent row-major complex matrices through Leto.
///
/// Each source matrix has shape `[rows, columns]`; each destination matrix has
/// shape `[columns, rows]`. Leto owns the allocation-free generic and
/// Hermes-backed register-tile selection for this layout operation.
pub(super) fn transpose_matrices<T>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    transpose_complex_matrices(source, destination, matrix_count, rows, columns)
        .expect("invariant: FFT matrix batch satisfies Leto's transpose contract");
}

#[cfg(test)]
mod tests {
    use super::{transpose_matrices, with_c_order_view};
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    use eunomia::{Complex, Complex64};
    use leto::{ArrayViewMut2, Layout};

    fn expected_transpose<T: Copy + Default>(
        source: &[Complex<T>],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Vec<Complex<T>> {
        let matrix_len = rows * columns;
        let mut expected = vec![Complex::default(); source.len()];
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

    fn assert_transpose<T>(matrix_count: usize, rows: usize, columns: usize)
    where
        T: MixedRadixScalar<Complex = Complex<T>> + std::fmt::Debug + PartialEq + Default,
    {
        let len = matrix_count * rows * columns;
        let source = (0..len)
            .map(|index| T::complex(index as f64, -(index as f64) - 0.25))
            .collect::<Vec<_>>();
        let expected = expected_transpose(&source, matrix_count, rows, columns);
        let mut destination = vec![Complex::default(); len];
        transpose_matrices(&source, &mut destination, matrix_count, rows, columns);
        assert_eq!(destination, expected);
    }

    #[test]
    fn transposes_rectangular_ragged_tiles() {
        assert_transpose::<f32>(1, 35, 67);
        assert_transpose::<f32>(1, 67, 35);
        assert_transpose::<f64>(1, 35, 67);
        assert_transpose::<f64>(1, 67, 35);
    }

    #[test]
    fn transposes_multiple_planes() {
        assert_transpose::<f32>(3, 5, 7);
        assert_transpose::<f64>(3, 5, 7);
    }

    #[test]
    fn transposes_provider_batch_with_ragged_edges() {
        assert_transpose::<f32>(256, 15, 13);
        assert_transpose::<f64>(256, 15, 13);
        assert_transpose::<f32>(256, 16, 16);
        assert_transpose::<f64>(256, 16, 16);
    }

    #[test]
    fn accepts_empty_and_singleton_matrices() {
        assert_transpose::<f64>(0, 5, 7);
        assert_transpose::<f64>(3, 0, 7);
        assert_transpose::<f64>(3, 7, 0);
        assert_transpose::<f64>(1, 1, 1);
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
