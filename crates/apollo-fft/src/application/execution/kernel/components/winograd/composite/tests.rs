use core::mem::MaybeUninit;

use eunomia::Complex;

use super::super::traits::WinogradScalar;
use super::split::{dft144_cols, dft144_rows, dft50_cols, dft50_rows};
use super::{dft144_impl, dft50_impl};
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;

fn compare_phases<F: WinogradScalar, const N: usize>(
    fused: impl FnOnce(&mut [Complex<F>; N]),
    first_phase: impl for<'scratch> FnOnce(
        &[Complex<F>; N],
        &'scratch mut MaybeUninit<[Complex<F>; N]>,
    ) -> &'scratch mut [Complex<F>; N],
    second_phase: impl FnOnce(&mut [Complex<F>; N], &mut [Complex<F>; N]),
) {
    // Preserve the original probe's precise sine/cosine values and scalar
    // conversion. Both routes copy one source, including under Miri's libm.
    let source = core::array::from_fn(|i| {
        let x = f64::from(u32::try_from(i).expect("invariant: test lengths fit u32"));
        Complex::new(
            F::from_precise((0.017 * x).sin()),
            F::from_precise((0.031 * x).cos()),
        )
    });
    let mut fused_output = source;
    fused(&mut fused_output);
    let mut split_output = source;
    let mut scratch = MaybeUninit::uninit();
    let scratch = first_phase(&split_output, &mut scratch);
    second_phase(scratch, &mut split_output);

    // Both routes emit the same arithmetic blocks in the same order, so no
    // tolerance is needed. This asserts exact values for the finite fixture.
    assert_eq!(fused_output, split_output, "split phases diverge at N={N}");
}

fn compare_codelets<F, const INVERSE: bool>()
where
    F: WinogradScalar + ShortDft<2> + ShortDft<12> + ShortDft<25>,
{
    compare_phases(
        |output| dft50_impl::<F, INVERSE>(output),
        |input, scratch| {
            dft50_rows::<F, INVERSE>(input, scratch);
            // SAFETY: the 2 × 25 gather writes all 50 slots before its row
            // DFTs read them, and all slots remain initialized on return.
            unsafe { scratch.assume_init_mut() }
        },
        |scratch, output| dft50_cols::<F, INVERSE>(scratch, output),
    );
    compare_phases(
        |output| dft144_impl::<F, INVERSE>(output),
        |input, scratch| {
            dft144_cols::<F, INVERSE>(input, scratch);
            // SAFETY: columns j in 0..12 write scratch[k * 12 + j] for
            // every k in 0..12, initializing each of the 144 slots once.
            unsafe { scratch.assume_init_mut() }
        },
        |scratch, output| dft144_rows::<F, INVERSE>(scratch, output),
    );
}

#[test]
fn split_phases_equal_fused_codelets() {
    compare_codelets::<f32, false>();
    compare_codelets::<f32, true>();
    compare_codelets::<f64, false>();
    compare_codelets::<f64, true>();
}
