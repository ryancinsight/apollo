use leto::{ArrayView2, ArrayViewMut2, Layout};

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
    use super::transpose_matrices;

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
}
