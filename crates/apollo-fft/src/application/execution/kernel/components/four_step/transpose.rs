/// Tiled in-place square matrix transpose: swaps element (r, c) with (c, r) for r < c.
///
/// Avoids the out-of-place write to scratch followed by `copy_from_slice` that the
/// generic `transpose_matrix` + copy path requires. Cache behaviour: each 16×16 tile
/// pair is loaded into L1 before any writes, so non-sequential strides only appear at
/// the cache-line level, not at the element level.
pub(super) fn transpose_square_inplace<T: Copy>(data: &mut [T], n: usize) {
    const TILE: usize = 16;
    for i_base in (0..n).step_by(TILE) {
        for j_base in (i_base..n).step_by(TILE) {
            let i_end = (i_base + TILE).min(n);
            let j_end = (j_base + TILE).min(n);
            if i_base == j_base {
                // Diagonal tile: swap strictly upper triangle within the tile.
                for r in i_base..i_end {
                    for c in (r + 1)..j_end {
                        data.swap(r * n + c, c * n + r);
                    }
                }
            } else {
                // Off-diagonal tile: swap with its symmetric mirror tile.
                for r in i_base..i_end {
                    for c in j_base..j_end {
                        data.swap(r * n + c, c * n + r);
                    }
                }
            }
        }
    }
}
