//! The real split of one lane: `n` reals as `n/2` packed complex samples.
//!
//! A size-`n/2` complex transform of the packed pairs, untangled, gives the
//! size-`n` real transform's `n/2 + 1` independent bins; retangled, those bins
//! give the packed pairs back to a size-`n/2` inverse. The half-length
//! transform is the caller's — a 1-D plan, or a 3-D plan's z lanes — and the
//! split around it is this one.

use super::RealFftData;
use crate::application::execution::kernel::real_fft::{retangle_real_half, untangle_real_half};
use eunomia::Complex;

/// Packs `input`, runs `transform` on the `n/2` packed samples, and untangles
/// them into the `n/2 + 1` bins of `out`, allocating nothing.
///
/// # Panics
///
/// If the split does not admit `input.len()`, or `out` has fewer than
/// `input.len() / 2 + 1` slots.
pub(super) fn forward<T: RealFftData>(
    input: &[T],
    out: &mut [Complex<T::PlanScalar>],
    transform: impl FnOnce(&mut [Complex<T::PlanScalar>]),
) {
    let n = input.len();
    assert!(
        T::real_split_applies(n),
        "real split does not apply to length {n}"
    );
    let m = n / 2;
    assert!(out.len() > m, "real spectrum needs n/2 + 1 slots");
    T::pack_real_pairs(input, &mut out[..m]);
    transform(&mut out[..m]);
    untangle_real_half(out, n);
}

/// Retangles one lane's `n/2 + 1` bins in place and runs `transform`, a
/// normalized half-length inverse, on the packed samples: the `n` reals are
/// then pairs in `bins[..n/2]`, which [`unpack`] reads out.
///
/// # Panics
///
/// If the split does not admit `n`, or `bins` has fewer than `n/2 + 1` slots.
pub(super) fn inverse_packed<T: RealFftData>(
    bins: &mut [Complex<T::PlanScalar>],
    n: usize,
    transform: impl FnOnce(&mut [Complex<T::PlanScalar>]),
) {
    assert!(
        T::real_split_applies(n),
        "real split does not apply to length {n}"
    );
    let m = n / 2;
    assert!(bins.len() > m, "real spectrum needs n/2 + 1 slots");
    retangle_real_half(bins, n);
    transform(&mut bins[..m]);
}

/// Unpacks `n/2` complex samples into `n` consecutive reals, the inverse of
/// [`RealFftData::pack_real_pairs`], narrowing each through
/// [`RealFftData::from_spectrum`].
pub(super) fn unpack<T: RealFftData>(packed: &[Complex<T::PlanScalar>], out: &mut [T]) {
    for (pair, value) in out.chunks_exact_mut(2).zip(packed) {
        pair[0] = T::from_spectrum(*value);
        // `from_spectrum` reads the real part, and `value · (-i)` carries the
        // imaginary part there.
        pair[1] = T::from_spectrum(Complex::new(value.im, -value.re));
    }
}
