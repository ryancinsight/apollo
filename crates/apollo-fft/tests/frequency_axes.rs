//! Frequency-axis utilities: axis-set shifts against lane-wise 1-D shifts and NumPy, and
//! scalar-generic frequency grids against exact quotients.

use apollo_fft::{
    fftfreq, fftshift, fftshift_axes, ifftshift, ifftshift_axes, rfftfreq, ApolloError, F16,
};
use eunomia::FloatElement;
use leto::{Array, ArrayView, MnemosyneStorage};
use std::fmt::Debug;

type Field<T, const N: usize> = Array<T, MnemosyneStorage<T>, N>;

fn field<T: Copy, const N: usize>(shape: [usize; N], values: Vec<T>) -> Field<T, N> {
    Field::from_mnemosyne_vec(shape, values).expect("fixture values fill the shape")
}

fn logical<T: Copy, const N: usize>(view: ArrayView<'_, T, N>) -> Vec<T> {
    view.iter().copied().collect()
}

/// Rotate every lane of `axis` of a C-order field with the 1-D shift `lane_shift`.
fn shift_lanes<T: Copy>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    lane_shift: fn(&[T]) -> Vec<T>,
) -> Vec<T> {
    let inner: usize = shape[axis + 1..].iter().product();
    let extent = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let mut shifted = values.to_vec();
    for block in 0..outer {
        for offset in 0..inner {
            let at = |k: usize| (block * extent + k) * inner + offset;
            let lane: Vec<T> = (0..extent).map(|k| values[at(k)]).collect();
            for (k, value) in lane_shift(&lane).into_iter().enumerate() {
                shifted[at(k)] = value;
            }
        }
    }
    shifted
}

fn expected_by_lanes<T: Copy, const N: usize>(
    values: &[T],
    shape: [usize; N],
    axes: &[usize],
    lane_shift: fn(&[T]) -> Vec<T>,
) -> Vec<T> {
    axes.iter().fold(values.to_vec(), |current, &axis| {
        shift_lanes(&current, &shape, axis, lane_shift)
    })
}

/// Every subset of the three axes of a rank-3 field, as sorted axis lists.
fn axis_subsets() -> Vec<Vec<usize>> {
    (0..8_u8)
        .map(|mask| (0..3).filter(|axis| (mask >> axis) & 1 == 1).collect())
        .collect()
}

fn shifts_land_where_lane_shifts_put_them<T>()
where
    T: Copy + PartialEq + Debug + From<u8>,
{
    let shape = [3, 4, 5];
    let values: Vec<T> = (0..60_u8).map(T::from).collect();
    let input = field(shape, values.clone());
    for axes in axis_subsets() {
        let centered = fftshift_axes(input.view(), &axes).expect("axes are valid");
        let restored = ifftshift_axes(centered.view(), &axes).expect("axes are valid");
        assert_eq!(
            logical(centered.view()),
            expected_by_lanes(&values, shape, &axes, fftshift),
            "fftshift {axes:?}"
        );
        assert_eq!(logical(restored.view()), values, "round trip over {axes:?}");

        let uncentered = ifftshift_axes(input.view(), &axes).expect("axes are valid");
        let recovered = fftshift_axes(uncentered.view(), &axes).expect("axes are valid");
        assert_eq!(
            logical(uncentered.view()),
            expected_by_lanes(&values, shape, &axes, ifftshift),
            "ifftshift {axes:?}"
        );
        assert_eq!(logical(recovered.view()), values, "reverse {axes:?}");

        let odd_extent_axis = axes.iter().any(|&axis| shape[axis] % 2 == 1);
        assert_eq!(
            logical(centered.view()) != logical(uncentered.view()),
            odd_extent_axis,
            "odd extent {axes:?}"
        );
    }
}

#[test]
fn ifftshift_inverts_fftshift_on_every_axis_subset() {
    shifts_land_where_lane_shifts_put_them::<u8>();
    shifts_land_where_lane_shifts_put_them::<f32>();
    shifts_land_where_lane_shifts_put_them::<f64>();
}

#[test]
fn shifts_match_numpy_fixtures_at_even_and_odd_extents() {
    // numpy 2.5.2: np.fft.{fftshift,ifftshift}(np.arange(N).reshape(shape), axes=...)
    let matrix = field([3, 4], (0..12_i32).collect());
    let cases: [(&[usize], [i32; 12], [i32; 12]); 2] = [
        (
            &[0],
            [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3],
        ),
        (
            &[0, 1],
            [10, 11, 8, 9, 2, 3, 0, 1, 6, 7, 4, 5],
            [6, 7, 4, 5, 10, 11, 8, 9, 2, 3, 0, 1],
        ),
    ];
    for (axes, centered, uncentered) in cases {
        let forward = fftshift_axes(matrix.view(), axes).expect("axes are valid");
        let inverse = ifftshift_axes(matrix.view(), axes).expect("axes are valid");
        assert_eq!(logical(forward.view()), centered, "fftshift {axes:?}");
        assert_eq!(logical(inverse.view()), uncentered, "ifftshift {axes:?}");
    }
}

#[test]
fn a_transposed_view_shifts_in_its_logical_order() {
    let matrix = field([3, 4], (0..12_u8).collect());
    let transposed = matrix.transpose([1, 0]).expect("permutation of two axes");
    let rows_of_transposed = fftshift_axes(transposed, &[0]).expect("axes are valid");
    let columns_of_matrix = fftshift_axes(matrix.view(), &[1]).expect("axes are valid");
    let back = columns_of_matrix
        .transpose([1, 0])
        .expect("permutation of two axes");
    assert_eq!(logical(rows_of_transposed.view()), logical(back));
    assert_eq!(rows_of_transposed.shape(), [4, 3]);
}

#[test]
fn invalid_axis_sets_return_validation_errors() {
    let input = field([2, 3], (0..6_u8).collect());
    for (axes, value) in [(&[2][..], "2"), (&[0, 7][..], "7")] {
        let Err(error) = fftshift_axes(input.view(), axes) else {
            panic!("axes {axes:?} exceed the rank");
        };
        assert!(
            matches!(&error, ApolloError::Validation { field, value: got, .. }
                if field == "axes" && got == value),
            "{error:?}"
        );
    }
    let Err(error) = ifftshift_axes(input.view(), &[1, 0, 1]) else {
        panic!("a repeated axis is refused");
    };
    assert!(
        matches!(&error, ApolloError::Validation { field, value, constraint }
            if field == "axes" && value == "1" && constraint.contains("appears once")),
        "{error:?}"
    );
}

/// Power-of-two spacings keep `n * d` exact for `n <= 17` in an 11-bit significand.
const SPACINGS: [f64; 3] = [0.5, 2.0, 0.25];

fn exact(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("small count"))
}

/// Signed bin numerators of `fftfreq(n)`: `0..ceil(n/2)` then `-(n/2)..-1`.
fn signed_bins(n: usize) -> Vec<f64> {
    let positive = n.div_ceil(2);
    let negative = (positive..n).map(|k| -exact(n - k));
    (0..positive).map(exact).chain(negative).collect()
}

/// For a `p`-bit element `T` the span `n * d` is exact, so each bin is one correctly rounded
/// quotient of two exact values. Rounding that quotient through `f64` first reproduces it:
/// binary64 double rounding of a quotient is innocuous when `53 >= 2p + 2`, true for
/// `p <= 25`, which covers binary16, binary32 and binary64's own identity.
fn grids_equal_the_rounded_exact_quotient<T>()
where
    T: FloatElement + Debug,
{
    for n in 1..=17_usize {
        for d in SPACINGS {
            let span = exact(n) * d;
            let full: Vec<T> = signed_bins(n)
                .into_iter()
                .map(|numerator| T::from_f64(numerator / span))
                .collect();
            assert_eq!(fftfreq(n, T::from_f64(d)), full, "fftfreq n={n} d={d}");

            let half: Vec<T> = (0..=n / 2).map(|k| T::from_f64(exact(k) / span)).collect();
            assert_eq!(rfftfreq(n, T::from_f64(d)), half, "rfftfreq n={n} d={d}");
        }
    }
}

#[test]
fn frequency_grids_equal_the_rounded_exact_quotient_in_every_float_element() {
    grids_equal_the_rounded_exact_quotient::<f32>();
    grids_equal_the_rounded_exact_quotient::<f64>();
    grids_equal_the_rounded_exact_quotient::<F16>();
}

#[test]
fn the_empty_length_has_no_full_bins_and_one_dc_half_bin() {
    assert!(fftfreq(0, 1.0_f32).is_empty());
    assert_eq!(rfftfreq(0, 1.0_f32), [0.0]);
}
