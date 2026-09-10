//! The fold tables against the twiddle matrix.

use super::super::fold::FourStepFold;
use super::super::lane_order::LaneOrder;
use super::super::BatchedPlanCache;
use eunomia::Complex64;
use std::f64::consts::TAU;

/// The two-level fold tables must reproduce the twiddle matrix entry for
/// entry within the roundings the factoring adds.
///
/// Each factor is one direct evaluation within one ulp (`EPSILON`) of
/// exact, as is the matrix entry it is held against; the product's two
/// roundings per part add `2 EPSILON`, so the difference is within
/// `5 EPSILON` (measured worst `4.03`); the bound is `6 EPSILON`. Rows are
/// natural, columns in plane order, as the fold indexes them.
#[test]
fn fold_tables_reproduce_the_twiddle_matrix_within_their_roundings() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    let (n, m) = (256usize, 16usize);
    let fold = <f64 as BatchedPlanCache>::cached_four_step_fold::<false>(n, m, m);
    let interleaved = <f64 as MixedRadixScalar>::cached_four_step_twiddles::<false>(n, m, m);
    let order = LaneOrder::for_batch::<f64>(m);
    let lanes = fold.lanes;
    assert_eq!(
        lanes, m,
        "below the compact bound the fine row is the whole row"
    );
    let bound = 6.0 * f64::EPSILON;
    for row in 0..m {
        for col in 0..m {
            // Memory column `col` sits at plane column `plane(col)`.
            let k = order.plane(col);
            let (fine, coarse) = (row * lanes + k % lanes, row * (m / lanes) + k / lanes);
            let entry = Complex64::new(fold.fine_re[fine], fold.fine_im[fine])
                * Complex64::new(fold.coarse_re[coarse], fold.coarse_im[coarse]);
            let exact = interleaved[row * m + col];
            let err = (entry.re - exact.re).hypot(entry.im - exact.im);
            assert!(err <= bound, "({row},{col}): {err:.3e} > {bound:.3e}");
        }
    }
    let again = <f64 as BatchedPlanCache>::cached_four_step_fold::<false>(n, m, m);
    assert!(
        std::sync::Arc::ptr_eq(&fold, &again),
        "the fold tables must cache"
    );
}

/// Above the compact bound the fine table is one lane group wide and the
/// coarse table carries the rest; the same entry-for-entry agreement holds
/// there, checked on the smallest compact length.
#[test]
fn compact_fold_tables_reproduce_the_twiddle_matrix_within_their_roundings() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    let (n, m) = (1usize << 18, 512usize);
    let order = LaneOrder::for_batch::<f64>(m);
    let fold = FourStepFold::<f64>::new::<false>(n, m, m, order);
    let interleaved = <f64 as MixedRadixScalar>::cached_four_step_twiddles::<false>(n, m, m);
    let lanes = fold.lanes;
    assert_eq!(lanes, order.lanes());
    assert!(lanes < m, "the compact table has more than one lane group");
    let bound = 6.0 * f64::EPSILON;
    for row in 0..m {
        for col in 0..m {
            // Memory column `col` sits at plane column `plane(col)`.
            let k = order.plane(col);
            let (fine, coarse) = (row * lanes + k % lanes, row * (m / lanes) + k / lanes);
            let entry = Complex64::new(fold.fine_re[fine], fold.fine_im[fine])
                * Complex64::new(fold.coarse_re[coarse], fold.coarse_im[coarse]);
            let exact = interleaved[row * m + col];
            let err = (entry.re - exact.re).hypot(entry.im - exact.im);
            assert!(err <= bound, "({row},{col}): {err:.3e} > {bound:.3e}");
        }
    }
}
#[test]
fn rectangular_fold_table_reproduces_the_twiddle_matrix() {
    // 2048 = 32 × 64: the second set's planes are 64 rows of 32 columns,
    // below the compact bound, so the fine table is the whole row.
    let (n, rows, cols) = (2048usize, 64usize, 32usize);
    let order = LaneOrder::for_batch::<f64>(cols);
    let fold = FourStepFold::<f64>::new::<false>(n, rows, cols, order);
    assert_eq!(fold.lanes, cols);
    for p in 0..rows {
        for k in 0..cols {
            let angle = -TAU * ((p * order.column(k)) % n) as f64 / n as f64;
            let (sin, cos) = angle.sin_cos();
            let got = Complex64::new(fold.fine_re[p * cols + k], fold.fine_im[p * cols + k]);
            let error = (got - Complex64::new(cos, sin)).norm();
            assert!(error <= 6.0 * f64::EPSILON, "({p},{k}): {error:e}");
        }
    }
}
