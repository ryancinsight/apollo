//! One-dimensional inverse real FFT operations.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_view_staging, PlanScratch,
};
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::Shape1D;
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::Array1;

/// Runs the half-spectrum inverse over the first `n/2 + 1` bins of a full
/// spectrum when the real split admits `n = out.len()`.
///
/// A real signal's spectrum above the Nyquist bin is the conjugate mirror of
/// the bins below it, so the inverse reads only the lower half and pays half
/// the arithmetic; `bins` are consumed as scratch. Returns whether it ran, so
/// a caller takes the full inverse without probing admissibility itself.
fn inverse_via_split<T>(bins: &mut [Complex<T::PlanScalar>], out: &mut [T]) -> bool
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = out.len();
    if !T::real_split_applies(n) || bins.len() <= n / 2 {
        return false;
    }
    let half_plan = T::get_1d_plan(
        Shape1D::new(n / 2).expect("half length is non-zero when the split applies"),
    );
    T::inverse_1d_half_into(half_plan.as_ref(), bins, out);
    true
}

/// [`inverse_via_split`] over a borrowed spectrum: the lower half is copied
/// into the rank-one staging role, so the caller pays no allocation for it.
fn inverse_via_split_staged<T>(spectrum: &[Complex<T::PlanScalar>], out: &mut [T]) -> bool
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = out.len();
    if !T::real_split_applies(n) || spectrum.len() <= n / 2 {
        return false;
    }
    with_view_staging::<Complex<T::PlanScalar>, 1, _>(n / 2 + 1, |half| {
        half.copy_from_slice(&spectrum[..=n / 2]);
        inverse_via_split::<T>(half, out)
    })
}

/// A zero-filled real signal of length `n`.
fn zero_signal<T: RealFftData>(n: usize) -> Vec<T> {
    (0..n)
        .map(|_| T::from_spectrum(Complex::default()))
        .collect()
}

/// Inverse 1D FFT of a complex spectrum using generic storage dispatch.
///
/// Where the real split admits the length, only the lower half of the
/// spectrum is read (the upper half of a real signal's spectrum is its
/// conjugate mirror) and the transform runs at half length.
#[must_use]
pub fn ifft_1d_array<T>(field_hat: &Array1<Complex<T::PlanScalar>>) -> Array1<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = field_hat.size();
    if let Some(bins) = field_hat.as_slice().filter(|_| T::real_split_applies(n)) {
        let mut out = zero_signal::<T>(n);
        if inverse_via_split_staged::<T>(bins, &mut out) {
            return Array1::from(out);
        }
    }
    T::inverse_1d(
        T::get_1d_plan(
            Shape1D::new(field_hat.size()).expect("ifft_1d_array requires non-zero length"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 1D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_1d_array_into_spectrum_scratch<T>(
    field_hat: &mut Array1<Complex<T::PlanScalar>>,
    out: &mut Array1<T>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    debug_assert_eq!(
        field_hat.size(),
        out.size(),
        "ifft_1d_array_into_spectrum_scratch: length mismatch"
    );
    if let (Some(bins), Some(signal)) = (field_hat.as_slice_mut(), out.as_slice_mut()) {
        if inverse_via_split::<T>(bins, signal) {
            return;
        }
    }
    T::inverse_1d_spectrum_into(
        T::get_1d_plan(
            Shape1D::new(field_hat.size())
                .expect("ifft_1d_array_into_spectrum_scratch requires non-zero length"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 1D FFT into caller-owned typed real storage and typed scratch spectrum.
pub fn ifft_1d_array_into<T>(
    field_hat: &Array1<Complex<T::PlanScalar>>,
    out: &mut Array1<T>,
    scratch: &mut Array1<Complex<T::PlanScalar>>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    if let (Some(bins), Some(signal), Some(half)) = (
        field_hat.as_slice(),
        out.as_slice_mut(),
        scratch.as_slice_mut(),
    ) {
        let n = signal.len();
        if T::real_split_applies(n) && bins.len() > n / 2 && half.len() > n / 2 {
            half[..=n / 2].copy_from_slice(&bins[..=n / 2]);
            if inverse_via_split::<T>(&mut half[..=n / 2], signal) {
                return;
            }
        }
    }
    T::inverse_1d_into(
        T::get_1d_plan(
            Shape1D::new(field_hat.size()).expect("ifft_1d_array_into requires non-zero length"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    );
}

/// Inverse 1D FFT into caller-owned typed real storage and typed scratch
/// spectrum for a compile-time-known length.
///
/// Where the real split admits `N` and the three arrays are contiguous, only
/// the lower `N/2 + 1` bins are read, into the front of `scratch`, and the
/// half-length inverse runs through the runtime kernel (ADR 0063); other
/// lengths copy the whole spectrum and run the zero-sized static plan.
pub fn ifft_1d_array_static_into<T, const N: usize>(
    field_hat: &Array1<Complex<T::PlanScalar>>,
    out: &mut Array1<T>,
    scratch: &mut Array1<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field_hat.size(),
        N,
        "ifft_1d_array_static_into: input length mismatch"
    );
    debug_assert_eq!(
        out.size(),
        N,
        "ifft_1d_array_static_into: output length mismatch"
    );
    debug_assert_eq!(
        scratch.size(),
        N,
        "ifft_1d_array_static_into: scratch length mismatch"
    );
    T::inverse_1d_static_into::<N>(field_hat, out, scratch);
}

/// Inverse 1D FFT of a complex spectrum slice, returning an owned `Vec` signal.
///
/// Slice/`Vec`-based wrapper for callers that prefer raw slices over Leto `Array` types.
#[must_use]
pub fn ifft_1d_slice<T>(spectrum: &[Complex<T::PlanScalar>]) -> Vec<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = spectrum.len();
    if T::real_split_applies(n) {
        let mut out = zero_signal::<T>(n);
        if inverse_via_split_staged::<T>(spectrum, &mut out) {
            return out;
        }
    }
    T::inverse_1d_slice_owned(
        T::get_1d_plan(Shape1D::new(n).expect("ifft_1d_slice requires non-zero length")).as_ref(),
        spectrum,
    )
}

/// Inverse 1D FFT of the `n/2 + 1` bins of a real signal's spectrum into `n`
/// reals.
///
/// The inverse of [`fft_1d_slice_half_into`](crate::fft_1d_slice_half_into):
/// a caller holding only the non-redundant half of a spectrum transforms it
/// back without mirroring it to full length first. `spectrum` is consumed as
/// scratch. Lengths the real split does not admit mirror the bins to a full
/// spectrum and take the full inverse, allocating it, so every length is
/// served. The imaginary parts of the zero and Nyquist bins are ignored.
///
/// # Panics
///
/// If `out` is empty or `spectrum` is not exactly `out.len() / 2 + 1` long.
pub fn ifft_1d_slice_half_into<T>(spectrum: &mut [Complex<T::PlanScalar>], out: &mut [T])
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = out.len();
    assert!(n > 0, "ifft_1d_slice_half_into requires a non-empty output");
    assert_eq!(
        spectrum.len(),
        n / 2 + 1,
        "ifft_1d_slice_half_into: spectrum must hold exactly n/2 + 1 bins"
    );
    if T::real_split_applies(n) {
        let half_plan = T::get_1d_plan(
            Shape1D::new(n / 2).expect("half length is non-zero when the split applies"),
        );
        T::inverse_1d_half_into(half_plan.as_ref(), spectrum, out);
        return;
    }
    let mut full = Vec::with_capacity(n);
    full.extend_from_slice(spectrum);
    full.extend(
        spectrum[1..=n - spectrum.len()]
            .iter()
            .rev()
            .map(|bin| Complex::new(bin.re, -bin.im)),
    );
    let values = T::inverse_1d_slice_owned(
        T::get_1d_plan(Shape1D::new(n).expect("a non-empty output has a non-zero length")).as_ref(),
        &full,
    );
    out.copy_from_slice(&values);
}

/// Inverse 1D FFT of a Leto spectrum view using generic storage dispatch.
///
/// C-contiguous Leto views are consumed through a borrowed slice. Strided views
/// are copied once in logical row-major order before entering the existing IFFT
/// slice boundary. The returned Leto array is backed by Mnemosyne allocation.
#[must_use]
pub fn ifft_1d_leto<T>(
    field_hat: leto::ArrayView1<'_, Complex<T::PlanScalar>>,
) -> leto::Array<T, leto::MnemosyneStorage<T>, 1>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let spectrum = view_cow(&field_hat);
    let signal = ifft_1d_slice::<T>(&spectrum);
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_vec([signal.len()], signal)
        .expect("IFFT signal length must match Leto output shape")
}
