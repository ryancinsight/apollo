//! The x-last forward against the C-order sweep and three-move chain it
//! replaces.

use eunomia::{Complex, F16};

use super::super::split;
use super::forward_half;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::lanes;
use crate::{PlanCacheProvider, Shape3D};

/// The forward as it ran before the x-last sweep: the z lanes in C order,
/// then the three-move chain.
fn c_order_forward<T>(source: &[T], [nx, ny, nz]: [usize; 3]) -> Vec<Complex<T::PlanScalar>>
where
    T: PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
{
    let plan = T::get_3d_plan(Shape3D::new(nx, ny, nz).expect("invariant: non-zero extents"));
    let depth = plan.nz_c();
    let half_lane = plan.half_z_lane::<true>();
    let mut want = vec![Complex::default(); nx * ny * depth];
    lanes::paired(&mut want, depth, source, nz, |bins_group, reals_group| {
        for (bins, reals) in bins_group
            .chunks_exact_mut(depth)
            .zip(reals_group.chunks_exact(nz))
        {
            split::forward(
                reals,
                bins,
                plan.split_twiddles().iter().copied(),
                &half_lane,
            );
        }
    });
    plan.xy_axes_inplace::<true>(&mut want, depth);
    want
}

/// Over volumes with both of `nx` and `ny` above one — odd, even, unequal,
/// and the 64³ the consumer runs, whose lanes go to the pool — the x-last
/// forward lands on the same bits: each lane's arithmetic is unchanged and the
/// skipped move only placed values.
fn x_last_forward_matches_the_c_order_chain<T>(
    make: impl Fn(f64) -> T,
    bits: impl Fn(Complex<T::PlanScalar>) -> (u64, u64),
) where
    T: PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
{
    for shape @ [nx, ny, nz] in [[3, 5, 8], [2, 2, 16], [7, 4, 32], [64, 64, 64]] {
        assert!(T::real_split_applies(nz), "{shape:?} must take the split");
        let source: Vec<T> = (0..nx * ny * nz)
            .map(|i| make((0.37 * i as f64).sin() + 0.5 * (0.011 * i as f64).cos()))
            .collect();
        let want = c_order_forward(&source, shape);
        let plan = T::get_3d_plan(Shape3D::new(nx, ny, nz).expect("invariant: non-zero extents"));
        let mut got = vec![Complex::default(); want.len()];
        forward_half(&plan, &source, &mut got);
        for (index, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(bits(g), bits(w), "{shape:?} bin {index}");
        }
    }
}

#[test]
fn x_last_forward_matches_the_c_order_chain_for_every_storage() {
    x_last_forward_matches_the_c_order_chain::<f64>(|v| v, |c| (c.re.to_bits(), c.im.to_bits()));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the test field is generated in f64 and stored at the scalar under test"
    )]
    x_last_forward_matches_the_c_order_chain::<f32>(
        |v| v as f32,
        |c| (u64::from(c.re.to_bits()), u64::from(c.im.to_bits())),
    );
    x_last_forward_matches_the_c_order_chain::<F16>(F16::from_f64, |c| {
        (u64::from(c.re.to_bits()), u64::from(c.im.to_bits()))
    });
}
