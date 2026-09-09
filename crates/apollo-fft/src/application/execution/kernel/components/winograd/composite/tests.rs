use super::schedule::{Fused, Split};
use super::{dft144_impl, dft50_impl};
use crate::application::execution::kernel::components::winograd::traits::WinogradScalar;
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
use eunomia::Complex;

fn compare_schedules<F, const N: usize, const INVERSE: bool>(
    fused: impl FnOnce(&mut [Complex<F>; N]),
    split: impl FnOnce(&mut [Complex<F>; N]),
) where
    F: WinogradScalar + ShortDft<N>,
{
    // Generate once and copy: independently evaluated transcendental inputs
    // need not agree under Miri. Conversion preserves the original fixture
    // values in each precision; transform arithmetic stays in F.
    let source = std::array::from_fn(|i| {
        let x = f64::from(u32::try_from(i).expect("invariant: test lengths fit u32"));
        Complex::new(
            F::from_precise((0.017 * x).sin()),
            F::from_precise((0.031 * x).cos()),
        )
    });
    let mut selected_output = source;
    <F as ShortDft<N>>::dft::<INVERSE>(&mut selected_output);
    let mut fused_output = source;
    fused(&mut fused_output);
    let mut split_output = source;
    split(&mut split_output);

    // Scheduling changes only the call boundaries, not evaluation order.
    assert_eq!(selected_output, fused_output, "production schedule N={N}");
    assert_eq!(split_output, fused_output, "split schedule N={N}");
}

fn compare_codelets<F, const INVERSE: bool>()
where
    F: WinogradScalar + ShortDft<2> + ShortDft<12> + ShortDft<25>,
{
    compare_schedules::<F, 50, INVERSE>(
        |output| dft50_impl::<F, INVERSE, Fused>(output),
        |output| dft50_impl::<F, INVERSE, Split>(output),
    );
    compare_schedules::<F, 144, INVERSE>(
        |output| dft144_impl::<F, INVERSE, Fused>(output),
        |output| dft144_impl::<F, INVERSE, Split>(output),
    );
}

#[test]
fn schedule_controls_preserve_composite_values() {
    compare_codelets::<f32, false>();
    compare_codelets::<f32, true>();
    compare_codelets::<f64, false>();
    compare_codelets::<f64, true>();
}
