//! Specialized executor functions for the 1D FFT plan.
//!
//! These are the fn-pointer targets assigned by `FftPlan1D::new()` during plan
//! construction, plus the static dispatch helpers used by `StaticFftPlan1D`.
//! Extracted from `dimension_1d.rs` to honour SRP and keep the plan module
//! focused on data structures and construction logic.

use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::pot::StockhamAutosort;
use crate::with_pot_zst;
use eunomia::Complex;

use super::FftPlan1D;

// ── Static dispatch (used by StaticFftPlan1D) ────────────────────────────────

#[inline]
pub(super) fn static_fft_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    slice: &mut [F::Complex],
) {
    assert_eq!(slice.len(), N, "static FFT plan length mismatch");

    if N <= 1 {
        return;
    }

    if tiny_direct_dispatch::<F, N, INVERSE, NORMALIZE>(slice) {
        return;
    }

    if static_small_pot_dispatch::<F, N, INVERSE, NORMALIZE>(slice) {
        return;
    }

    if N.is_power_of_two() {
        static_pot_dispatch::<F, N, INVERSE, NORMALIZE>(slice);
    } else if N == 385 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[11, 5, 7]);
    } else if N == 180 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[5, 3, 3, 4]);
    } else if N == 144 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[4, 4, 3, 3]);
    } else if N == 176 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[11, 4, 4]);
    } else if N == 200 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, F::COMPOSITE_RADICES_200);
    } else if F::use_generated_codelet_plan(N) {
        F::short_winograd::<INVERSE, NORMALIZE>(slice);
    } else if N == 72 && !F::FORCE_COMPOSITE_72 {
        static_good_thomas_dispatch::<F, N, INVERSE, NORMALIZE>(slice, 9, 8);
    } else if N == 511 {
        static_good_thomas_dispatch::<F, N, INVERSE, NORMALIZE>(slice, 73, 7);
    } else if N == 36 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[4, 3, 3]);
    } else if N == 48 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[4, 4, 3]);
    } else if N == 63 && F::FORCE_COMPOSITE_63 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[3, 3, 7]);
    } else if N == 72 && F::FORCE_COMPOSITE_72 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[4, 2, 3, 3]);
    } else if N == 72 {
        static_good_thomas_dispatch::<F, N, INVERSE, NORMALIZE>(slice, 9, 8);
    } else if N == 90 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[2, 3, 3, 5]);
    } else if N == 198 {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, &[2, 3, 3, 11]);
    } else if crate::application::execution::kernel::mixed_radix::traits::is_short_winograd_size(N)
        && (N <= 64 || F::use_generated_codelet_plan(N))
    {
        F::short_winograd::<INVERSE, NORMALIZE>(slice);
    } else if let Some(radices) =
        crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(N)
    {
        static_composite_dispatch::<F, INVERSE, NORMALIZE>(slice, radices.as_ref());
    } else if let Some((n1, n2)) =
        crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(N)
            .filter(|&(n1, n2)| {
                crate::application::execution::kernel::components::good_thomas::has_static_coprime_codelet(n1, n2)
            })
    {
        static_good_thomas_dispatch::<F, N, INVERSE, NORMALIZE>(slice, n1, n2);
    } else if let Some((n1, n2)) =
        crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(N)
    {
        static_good_thomas_dispatch::<F, N, INVERSE, NORMALIZE>(slice, n1, n2);
    } else {
        crate::application::execution::kernel::components::rader::rader_fft::<F, INVERSE>(slice);
        if INVERSE && NORMALIZE {
            F::normalize(slice, N);
        }
    }
}

#[inline]
pub(super) fn runtime_tiny_direct_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    len: usize,
    slice: &mut [F::Complex],
) -> bool {
    match len {
        2 => unsafe { F::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(slice) },
        3 => crate::application::execution::kernel::components::butterflies::dft3_impl::<
            F,
            INVERSE,
            NORMALIZE,
        >(slice),
        4 => unsafe { F::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(slice) },
        _ => return false,
    }
    true
}

#[inline]
fn tiny_direct_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    slice: &mut [F::Complex],
) -> bool {
    match N {
        2 => unsafe { F::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(slice) },
        3 => crate::application::execution::kernel::components::butterflies::dft3_impl::<
            F,
            INVERSE,
            NORMALIZE,
        >(slice),
        4 => unsafe { F::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(slice) },
        _ => return false,
    }
    true
}

#[inline]
fn static_small_pot_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    slice: &mut [F::Complex],
) -> bool {
    match N {
        8 => unsafe { F::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(slice) },
        16 => unsafe { F::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(slice) },
        32 => unsafe { F::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(slice) },
        64 => unsafe { F::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(slice) },
        _ => return false,
    }
    true
}

#[inline]
fn static_pot_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    slice: &mut [F::Complex],
) {
    let twiddles = if INVERSE {
        F::cached_twiddle_inv(N)
    } else {
        F::cached_twiddle_fwd(N)
    };

    match N {
        128 => with_pot_zst!(7, s, {
            F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, 7>(
                slice,
                twiddles.as_ref(),
                s,
            );
        }),
        256 => with_pot_zst!(8, s, {
            F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, 8>(
                slice,
                twiddles.as_ref(),
                s,
            );
        }),
        512 => with_pot_zst!(9, s, {
            F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, 9>(
                slice,
                twiddles.as_ref(),
                s,
            );
        }),
        1024 => with_pot_zst!(10, s, {
            F::pot_inplace_sized::<INVERSE, NORMALIZE, StockhamAutosort, 10>(
                slice,
                twiddles.as_ref(),
                s,
            );
        }),
        _ => F::pot_inplace::<INVERSE, NORMALIZE>(slice, twiddles.as_ref()),
    }
}

#[inline]
fn static_good_thomas_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const N: usize,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    slice: &mut [F::Complex],
    n1: usize,
    n2: usize,
) {
    crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, INVERSE>(
        slice, n1, n2,
    );
    if INVERSE && NORMALIZE {
        F::normalize(slice, N);
    }
}

#[inline]
fn static_composite_dispatch<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    slice: &mut [F::Complex],
    radices: &[usize],
) {
    if INVERSE {
        if NORMALIZE {
            F::composite_inverse(slice, radices);
        } else {
            F::composite_inverse_unnorm(slice, radices);
        }
    } else {
        F::composite_forward(slice, radices);
    }
}

// ── Runtime executors (fn-pointer targets for FftPlan1D) ────────────────────

// 1. Identity
pub(super) fn exec_identity<F: MixedRadixScalar>(_: &FftPlan1D<F>, _: &mut [F::Complex]) {}

// 2. ShortWinograd
pub(super) fn exec_winograd_forward<F, const N: usize>(_: &FftPlan1D<F>, slice: &mut [F::Complex])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<N>,
{
    if let Ok(data) = slice.try_into() {
        F::dft::<false>(data);
    }
}

pub(super) fn exec_winograd_inverse<F, const N: usize>(_: &FftPlan1D<F>, slice: &mut [F::Complex])
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<N>,
{
    if let Ok(data) = slice.try_into() {
        F::dft::<true>(data);
        F::normalize(slice, N);
    }
}

pub(super) fn exec_winograd_inverse_unnorm<F, const N: usize>(
    _: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<N>,
{
    if let Ok(data) = slice.try_into() {
        F::dft::<true>(data);
    }
}

// 3. PowerOfTwo small sizes
macro_rules! define_pot_executors {
    ($size:expr, $fwd:ident, $inv:ident, $inv_un:ident) => {
        pub(super) fn $fwd<F: MixedRadixScalar<Complex = Complex<F>>>(
            _: &FftPlan1D<F>,
            slice: &mut [F::Complex],
        ) {
            unsafe {
                F::small_pot_inplace_sized::<$size, false, false>(slice);
            }
        }
        pub(super) fn $inv<F: MixedRadixScalar<Complex = Complex<F>>>(
            _: &FftPlan1D<F>,
            slice: &mut [F::Complex],
        ) {
            unsafe {
                F::small_pot_inplace_sized::<$size, true, true>(slice);
            }
        }
        pub(super) fn $inv_un<F: MixedRadixScalar<Complex = Complex<F>>>(
            _: &FftPlan1D<F>,
            slice: &mut [F::Complex],
        ) {
            unsafe {
                F::small_pot_inplace_sized::<$size, true, false>(slice);
            }
        }
    };
}
define_pot_executors!(
    2,
    exec_pot_forward_2,
    exec_pot_inverse_2,
    exec_pot_inverse_unnorm_2
);
define_pot_executors!(
    4,
    exec_pot_forward_4,
    exec_pot_inverse_4,
    exec_pot_inverse_unnorm_4
);
define_pot_executors!(
    8,
    exec_pot_forward_8,
    exec_pot_inverse_8,
    exec_pot_inverse_unnorm_8
);
define_pot_executors!(
    16,
    exec_pot_forward_16,
    exec_pot_inverse_16,
    exec_pot_inverse_unnorm_16
);
define_pot_executors!(
    32,
    exec_pot_forward_32,
    exec_pot_inverse_32,
    exec_pot_inverse_unnorm_32
);
define_pot_executors!(
    64,
    exec_pot_forward_64,
    exec_pot_inverse_64,
    exec_pot_inverse_unnorm_64
);

/// ZST-wired sized PoT executor helper (monomorphizes the exact LOG2).
/// The SizedPoT<StockhamAutosort, LOG2> is constructed here for zero-cost
/// strategy selection through the Stockham autosort path.
#[inline]
pub(super) fn exec_pot_forward_sized<F: MixedRadixScalar<Complex = Complex<F>>, const LOG2: u32>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    with_pot_zst!(LOG2, _s, {
        if let Some(tw) = &plan.twiddle_fwd {
            F::pot_inplace_sized::<false, false, StockhamAutosort, LOG2>(slice, tw, _s);
        }
    });
}

pub(super) fn exec_pot_inverse_sized<F: MixedRadixScalar<Complex = Complex<F>>, const LOG2: u32>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    with_pot_zst!(LOG2, _s, {
        if let Some(tw) = &plan.twiddle_inv {
            F::pot_inplace_sized::<true, true, StockhamAutosort, LOG2>(slice, tw, _s);
        }
    });
}

pub(super) fn exec_pot_inverse_unnorm_sized<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const LOG2: u32,
>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    with_pot_zst!(LOG2, _s, {
        if let Some(tw) = &plan.twiddle_inv {
            F::pot_inplace_sized::<true, false, StockhamAutosort, LOG2>(slice, tw, _s);
        }
    });
}

// 512 explicit (2^9) using ZST wiring.
pub(super) fn exec_pot_forward_512<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    exec_pot_forward_sized::<F, 9>(plan, slice);
}
pub(super) fn exec_pot_inverse_512<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    exec_pot_inverse_sized::<F, 9>(plan, slice);
}
pub(super) fn exec_pot_inverse_unnorm_512<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    exec_pot_inverse_unnorm_sized::<F, 9>(plan, slice);
}

// 4. PowerOfTwo generic sizes (using cached twiddles)
pub(super) fn exec_pot_forward_generic<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    if let Some(tw) = &plan.twiddle_fwd {
        F::pot_inplace::<false, false>(slice, tw);
    }
}
pub(super) fn exec_pot_inverse_generic<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    if let Some(tw) = &plan.twiddle_inv {
        F::pot_inplace::<true, true>(slice, tw);
    }
}
pub(super) fn exec_pot_inverse_unnorm_generic<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    if let Some(tw) = &plan.twiddle_inv {
        F::pot_inplace::<true, false>(slice, tw);
    }
}

// 5. Good-Thomas (Static or Generic)
pub(super) fn exec_good_thomas_forward<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, false>(
        slice, plan.n1, plan.n2,
    );
}

pub(super) fn exec_good_thomas_inverse<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, true>(
        slice, plan.n1, plan.n2,
    );
    F::normalize(slice, plan.n);
}

pub(super) fn exec_good_thomas_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, true>(
        slice, plan.n1, plan.n2,
    );
}

// 6. Composite
pub(super) fn exec_composite_forward<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    if let Some(radices) = &plan.radices {
        F::composite_forward(slice, radices);
    }
}
pub(super) fn exec_composite_inverse<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    if let Some(radices) = &plan.radices {
        F::composite_inverse(slice, radices);
    }
}
pub(super) fn exec_composite_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    if let Some(radices) = &plan.radices {
        F::composite_inverse_unnorm(slice, radices);
    }
}

// 7. Rader
pub(super) fn exec_rader_forward<F: MixedRadixScalar<Complex = Complex<F>>>(
    _: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    crate::application::execution::kernel::components::rader::rader_fft::<F, false>(slice);
}

pub(super) fn exec_rader_inverse<F: MixedRadixScalar<Complex = Complex<F>>>(
    plan: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    crate::application::execution::kernel::components::rader::rader_fft::<F, true>(slice);
    F::normalize(slice, plan.n);
}

pub(super) fn exec_rader_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>>(
    _: &FftPlan1D<F>,
    slice: &mut [F::Complex],
) {
    crate::application::execution::kernel::components::rader::rader_fft::<F, true>(slice);
}
