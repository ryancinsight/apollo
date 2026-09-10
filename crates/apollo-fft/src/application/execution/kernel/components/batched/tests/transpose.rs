//! The in-place and rectangular plane transposes against their reference forms.

use super::super::lane_order::LaneOrder;
use super::super::plane::{transpose_planes, transpose_planes_into, PlaneView};
use super::super::{boundary, BatchedPlanCache};
use eunomia::Complex;

#[test]
fn transpose_is_its_own_inverse_and_never_touches_the_pad() {
    for m in [1usize, 2, 4, 8, 16, 33, 64] {
        // The dispatched order, and the AVX-512 orders whatever the host,
        // where the order is no involution; a group must fit the row.
        let mut orders = vec![
            (0usize, LaneOrder::IDENTITY),
            (8, LaneOrder::IDENTITY),
            (8, LaneOrder::for_batch::<f64>(m)),
        ];
        if m % 8 == 0 {
            orders.push((8, LaneOrder::from_geometry(8, 2)));
        }
        if m % 16 == 0 {
            orders.push((8, LaneOrder::from_geometry(16, 4)));
        }
        for (pad, order) in orders {
            let stride = m + pad;
            let sentinel = f64::NAN;
            let re0: Vec<f64> = (0..m * m).map(|i| 0.5 + i as f64).collect();
            let im0: Vec<f64> = (0..m * m).map(|i| 0.25 - i as f64 * 0.5).collect();
            let mut re = vec![sentinel; m * stride];
            let mut im = vec![sentinel; m * stride];
            for r in 0..m {
                re[r * stride..r * stride + m].copy_from_slice(&re0[r * m..(r + 1) * m]);
                im[r * stride..r * stride + m].copy_from_slice(&im0[r * m..(r + 1) * m]);
            }
            transpose_planes(&mut re, &mut im, m, stride, order);
            for r in 0..m {
                for c in 0..m {
                    // Plane cell (r, c) holds memory (r, order(c)), the
                    // transpose of memory (order(c), r), which sat at plane
                    // cell (order(c), plane(r)).
                    let from = order.column(c) * m + order.plane(r);
                    assert_eq!(
                        re[r * stride + c],
                        re0[from],
                        "m={m} pad={pad} re ({r},{c})"
                    );
                    assert_eq!(
                        im[r * stride + c],
                        im0[from],
                        "m={m} pad={pad} im ({r},{c})"
                    );
                }
                for c in m..stride {
                    assert!(
                        re[r * stride + c].is_nan() && im[r * stride + c].is_nan(),
                        "m={m} pad={pad}: pad column {c} of row {r} was written"
                    );
                }
            }
            transpose_planes(&mut re, &mut im, m, stride, order);
            for r in 0..m {
                assert_eq!(&re[r * stride..r * stride + m], &re0[r * m..(r + 1) * m]);
            }
        }
    }
}
/// The rectangular transpose lands every cell where the in-place rule
/// puts it, on the dispatched width and the scalar reference alike.
fn rectangular_transpose_matches_the_reference<T>()
where
    T: BatchedPlanCache<Complex = Complex<T>>
        + eunomia::FloatElement
        + PartialEq
        + core::fmt::Debug,
{
    // 24 rows: three eight-lane tiles, so the unpaired tile block runs.
    // The last shape pads by four, a stride no eight-lane width divides.
    for (rows, cols, pad) in [
        (16usize, 32usize, 8usize),
        (32, 64, 8),
        (24, 40, 8),
        (32, 64, 4),
    ] {
        let (src_stride, dst_stride) = (cols + pad, rows + pad);
        let order = LaneOrder::for_batch::<T>(rows);
        let src_re: Vec<T> = (0..rows * src_stride)
            .map(|i| T::from_f64(i as f64 + 0.25))
            .collect();
        let src_im: Vec<T> = (0..rows * src_stride)
            .map(|i| T::from_f64(-(i as f64) - 0.5))
            .collect();
        let sentinel = T::from_f64(-1.0);
        let mut expected_re = vec![sentinel; cols * dst_stride];
        let mut expected_im = vec![sentinel; cols * dst_stride];
        transpose_planes_into(
            PlaneView {
                re: &src_re,
                im: &src_im,
                rows,
                cols,
                stride: src_stride,
            },
            &mut expected_re,
            &mut expected_im,
            dst_stride,
            order,
        );
        for r in 0..rows {
            for c in 0..cols {
                let to = order.column(c) * dst_stride + order.plane(r);
                assert_eq!(expected_re[to], src_re[r * src_stride + c]);
            }
        }
        let mut actual_re = vec![sentinel; cols * dst_stride];
        let mut actual_im = vec![sentinel; cols * dst_stride];
        let handled = hermes_simd::vectorize(boundary::TransposePlanesInto {
            src_re: &src_re,
            src_im: &src_im,
            rows,
            cols,
            src_stride,
            dst_re: &mut actual_re,
            dst_im: &mut actual_im,
            dst_stride,
        });
        // The kernel takes a shape whose rows and columns are lane multiples,
        // whatever the padded stride; otherwise the scalar reference stands alone.
        let lanes = LaneOrder::for_batch::<T>(rows).lanes();
        let divides = [rows, cols].iter().all(|extent| extent % lanes == 0);
        assert_eq!(handled, divides, "{rows}x{cols} at {lanes} lanes");
        if handled {
            assert_eq!(actual_re, expected_re, "{rows}x{cols} re");
            assert_eq!(actual_im, expected_im, "{rows}x{cols} im");
        }
    }
}

#[test]
fn rectangular_transpose_matches_the_reference_in_both_precisions() {
    rectangular_transpose_matches_the_reference::<f32>();
    rectangular_transpose_matches_the_reference::<f64>();
}
