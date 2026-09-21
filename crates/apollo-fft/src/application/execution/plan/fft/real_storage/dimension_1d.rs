//! One-dimensional real-storage transform implementations.

use super::{fill_real, fill_spectrum, split, RealFftData};
use crate::application::execution::kernel::mixed_radix::{forward_inplace, inverse_inplace};
use crate::application::execution::kernel::real_fft::{
    mirror_half_spectrum_in_place, split_twiddles,
};
use crate::application::execution::plan::fft::dimension_1d::{FftPlan1D, StaticFftPlan1D};
use eunomia::Complex;
use leto::Array1;

/// The power-of-two length from which the static forward takes the split;
/// below it the zero-sized plan's constant-length kernels win or tie, and at
/// it the two measure at parity (ADR 0063).
const STATIC_FORWARD_SPLIT_FLOOR: usize = 128;

pub(super) fn forward_half_into<T: RealFftData>(
    half_plan: &FftPlan1D<T::PlanScalar>,
    input: &[T],
    out: &mut [Complex<T::PlanScalar>],
) {
    split::forward(input, out, split_twiddles(input.len()), |packed| {
        half_plan.forward_complex_slice_inplace(packed);
    });
}

pub(super) fn inverse_half_into<T: RealFftData>(
    half_plan: &FftPlan1D<T::PlanScalar>,
    input: &mut [Complex<T::PlanScalar>],
    out: &mut [T],
) {
    let n = out.len();
    split::inverse_packed::<T>(input, n, split_twiddles(n), |packed| {
        half_plan.inverse_complex_slice_inplace(packed);
    });
    split::unpack(&input[..n / 2], out);
}

pub(super) fn forward_into_via_split<T: RealFftData>(
    half_plan: &FftPlan1D<T::PlanScalar>,
    input: &Array1<T>,
    output: &mut Array1<Complex<T::PlanScalar>>,
) -> bool {
    let n = input.size();
    if !T::real_split_applies(n) || output.size() != n {
        return false;
    }
    let (Some(src), Some(dst)) = (input.as_slice(), output.as_slice_mut()) else {
        return false;
    };
    T::forward_1d_half_into(half_plan, src, dst);
    mirror_half_spectrum_in_place(dst);
    true
}

pub(super) fn forward_slice_owned_via_split<T: RealFftData>(
    half_plan: &FftPlan1D<T::PlanScalar>,
    input: &[T],
) -> Vec<Complex<T::PlanScalar>> {
    let n = input.len();
    let mut output = vec![Complex::<T::PlanScalar>::default(); n];
    T::forward_1d_half_into(half_plan, input, &mut output);
    mirror_half_spectrum_in_place(&mut output);
    output
}

pub(super) fn forward_static_into<T: RealFftData, const N: usize>(
    input: &Array1<T>,
    output: &mut Array1<Complex<T::PlanScalar>>,
) {
    // Where the split admits `N`, the half transform runs through the
    // plan-free runtime kernel (ADR 0063): `N / 2` is not a const-generic
    // argument on the stable toolchain and the bound names no plan cache.
    // On the forward the static plan's constant-length power-of-two kernels
    // win at 4 and 8 by more than half (0.44x, 0.45x of the split's runtime
    // half on the census host, two runs) and at 32 by 14% in one run; at 16
    // and 64 the two sit inside that host's run-to-run drift (1.04-1.06x,
    // 0.95-1.25x) and at 128 at parity (1.02x, one run). The forward keeps
    // the static plan on powers of two below 128 and takes the split from
    // there, where it wins from 256 (1.35x).
    if T::real_split_applies(N) && (N >= STATIC_FORWARD_SPLIT_FLOOR || !N.is_power_of_two()) {
        if let (Some(src), Some(dst)) = (input.as_slice(), output.as_slice_mut()) {
            if src.len() == N && dst.len() == N {
                split::forward(src, dst, split_twiddles(N), |packed| {
                    forward_inplace::<T::PlanScalar>(packed);
                });
                mirror_half_spectrum_in_place(dst);
                return;
            }
        }
    }
    fill_spectrum(input, output);
    StaticFftPlan1D::<T::PlanScalar, N>::new().forward_complex_inplace(output);
}

pub(super) fn inverse_static_into<T: RealFftData, const N: usize>(
    input: &Array1<Complex<T::PlanScalar>>,
    output: &mut Array1<T>,
    scratch: &mut Array1<Complex<T::PlanScalar>>,
) {
    // The split reads only the lower `N / 2 + 1` bins, copied into the front
    // of `scratch`, and runs the half-length inverse through the runtime
    // kernel (ADR 0063).
    if T::real_split_applies(N) {
        if let (Some(bins), Some(values), Some(work)) = (
            input.as_slice(),
            output.as_slice_mut(),
            scratch.as_slice_mut(),
        ) {
            if bins.len() == N && values.len() == N && work.len() == N {
                let depth = N / 2 + 1;
                let half = &mut work[..depth];
                half.copy_from_slice(&bins[..depth]);
                split::inverse_packed::<T>(half, N, split_twiddles(N), |packed| {
                    inverse_inplace::<T::PlanScalar>(packed);
                });
                split::unpack(&half[..N / 2], values);
                return;
            }
        }
    }
    scratch.assign(&input.view());
    StaticFftPlan1D::<T::PlanScalar, N>::new().inverse_complex_inplace(scratch);
    fill_real(scratch, output);
}
