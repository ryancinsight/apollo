//! Forward real FFT API functions.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::{Array1, Array2, Array3};

/// Forward 1D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_1d_array<T>(field: &Array1<T>) -> Array1<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = field.size();
    if T::real_split_applies(n) {
        let mut out =
            Array1::<Complex<T::PlanScalar>>::from(vec![Complex::<T::PlanScalar>::default(); n]);
        let half_plan = T::get_1d_plan(
            Shape1D::new(n / 2).expect("half length is non-zero when the split applies"),
        );
        if T::forward_1d_into_via_split(half_plan.as_ref(), field, &mut out) {
            return out;
        }
    }
    T::forward_1d(
        T::get_1d_plan(Shape1D::new(n).expect("fft_1d_array requires non-zero length")).as_ref(),
        field,
    )
}

/// Forward 1D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_1d_array_into<T>(field: &Array1<T>, out: &mut Array1<Complex<T::PlanScalar>>)
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = field.size();
    if T::real_split_applies(n) {
        let half_plan = T::get_1d_plan(
            Shape1D::new(n / 2).expect("half length is non-zero when the split applies"),
        );
        if T::forward_1d_into_via_split(half_plan.as_ref(), field, out) {
            return;
        }
    }
    T::forward_1d_into(
        T::get_1d_plan(Shape1D::new(n).expect("fft_1d_array_into requires non-zero length"))
            .as_ref(),
        field,
        out,
    );
}

/// Forward 1D FFT of a real array into caller-owned typed spectrum storage for
/// a compile-time-known length.
pub fn fft_1d_array_static_into<T, const N: usize>(
    field: &Array1<T>,
    out: &mut Array1<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field.size(),
        N,
        "fft_1d_array_static_into: input length mismatch"
    );
    debug_assert_eq!(
        out.size(),
        N,
        "fft_1d_array_static_into: output length mismatch"
    );
    T::forward_1d_static_into::<N>(field, out);
}

/// Forward 1D FFT of a real signal slice, returning an owned `Vec` spectrum.
///
/// Slice/`Vec`-based wrapper for callers that prefer raw slices over Leto `Array` types.
#[must_use]
pub fn fft_1d_slice<T>(signal: &[T]) -> Vec<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = signal.len();
    if T::real_split_applies(n) {
        // One size-n/2 complex transform plus an untangle, against one size-n
        // transform on zero-imaginary input: about half the arithmetic for the
        // same spectrum.
        let half_plan = T::get_1d_plan(
            Shape1D::new(n / 2).expect("half length is non-zero when the split applies"),
        );
        return T::forward_1d_slice_owned_via_split(half_plan.as_ref(), signal);
    }
    T::forward_1d_slice_owned(
        T::get_1d_plan(Shape1D::new(n).expect("fft_1d_slice requires non-zero length")).as_ref(),
        signal,
    )
}

/// Forward 1D FFT of a real signal slice into caller storage, writing only the
/// `n / 2 + 1` independent bins.
///
/// A real signal's spectrum satisfies `X[n-k] = conj(X[k])`, so the upper half
/// carries no information. [`fft_1d_slice`] materializes it anyway, which costs
/// a mirror pass and twice the output storage. Callers that consume the
/// half-spectrum directly — power spectra, filtering, and anything that
/// round-trips through [`irfft`](crate::ifft_1d_slice) — should use this.
///
/// This is the form other real-FFT implementations expose, so it is also the
/// shape in which Apollo is comparable to them.
///
/// # Panics
///
/// If `out` is not exactly `signal.len() / 2 + 1` long.
pub fn fft_1d_slice_half_into<T>(signal: &[T], out: &mut [Complex<T::PlanScalar>])
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let n = signal.len();
    assert_eq!(
        out.len(),
        n / 2 + 1,
        "fft_1d_slice_half_into: output must hold exactly n/2 + 1 bins"
    );

    if T::real_split_applies(n) {
        let half_plan = T::get_1d_plan(
            Shape1D::new(n / 2).expect("half length is non-zero when the split applies"),
        );
        T::forward_1d_half_into(half_plan.as_ref(), signal, out);
        return;
    }

    // Lengths the split does not admit still owe the caller the same contract,
    // so the full transform runs and its redundant half is dropped rather than
    // copied out.
    let full = T::forward_1d_slice_owned(
        T::get_1d_plan(Shape1D::new(n).expect("fft_1d_slice_half_into requires non-zero length"))
            .as_ref(),
        signal,
    );
    out.copy_from_slice(&full[..=n / 2]);
}

/// Forward 1D FFT of a real signal slice, returning the `n / 2 + 1` independent
/// bins as an owned `Vec`.
///
/// One allocation, half the size [`fft_1d_slice`] returns. See
/// [`fft_1d_slice_half_into`] for the allocation-free form and for why the
/// upper half is redundant.
#[must_use]
pub fn fft_1d_slice_half<T>(signal: &[T]) -> Vec<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let mut out = vec![Complex::<T::PlanScalar>::default(); signal.len() / 2 + 1];
    fft_1d_slice_half_into::<T>(signal, &mut out);
    out
}

/// Forward 1D FFT of a Leto real view using generic storage dispatch.
///
/// C-contiguous Leto views are consumed through a borrowed slice. Strided views
/// are copied once in logical row-major order before entering the existing FFT
/// slice boundary. The returned Leto array is backed by Mnemosyne allocation.
#[must_use]
pub fn fft_1d_leto<T>(
    field: leto::ArrayView1<'_, T>,
) -> leto::Array<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 1>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let signal = view_cow(&field);
    let spectrum = fft_1d_slice::<T>(&signal);
    leto::Array::<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 1>::from_mnemosyne_vec(
        [spectrum.len()],
        spectrum,
    )
    .expect("FFT spectrum length must match Leto output shape")
}

/// Forward 2D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_2d_array<T>(field: &Array2<T>) -> Array2<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field.shape();
    T::forward_2d(
        T::get_2d_plan(Shape2D::new(nx, ny).expect("fft_2d_array requires non-zero dimensions"))
            .as_ref(),
        field,
    )
}

/// Forward 2D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_2d_array_into<T>(field: &Array2<T>, out: &mut Array2<Complex<T::PlanScalar>>)
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field.shape();
    T::forward_2d_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("fft_2d_array_into requires non-zero dimensions"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 2D FFT of a real array into caller-owned typed spectrum storage for
/// a compile-time-known shape.
pub fn fft_2d_array_static_into<T, const NX: usize, const NY: usize>(
    field: &Array2<T>,
    out: &mut Array2<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        [NX, NY],
        "fft_2d_array_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY],
        "fft_2d_array_static_into: output shape mismatch"
    );
    T::forward_2d_static_into::<NX, NY>(field, out);
}

/// Forward 3D FFT of a real array using generic storage dispatch.
#[must_use]
pub fn fft_3d_array<T>(field: &Array3<T>) -> Array3<Complex<T::PlanScalar>>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field.shape();
    T::forward_3d(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array requires non-zero dimensions"),
        )
        .as_ref(),
        field,
    )
}

/// Forward 3D FFT of a real array into caller-owned typed spectrum storage.
pub fn fft_3d_array_into<T>(field: &Array3<T>, out: &mut Array3<Complex<T::PlanScalar>>)
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field.shape();
    T::forward_3d_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("fft_3d_array_into requires non-zero dimensions"),
        )
        .as_ref(),
        field,
        out,
    );
}

/// Forward 3D FFT of a real array into caller-owned typed spectrum storage for
/// a compile-time-known shape.
pub fn fft_3d_array_static_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field: &Array3<T>,
    out: &mut Array3<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field.shape(),
        [NX, NY, NZ],
        "fft_3d_array_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY, NZ],
        "fft_3d_array_static_into: output shape mismatch"
    );
    T::forward_3d_static_into::<NX, NY, NZ>(field, out);
}

/// Forward 2D FFT of a Leto real view using generic storage dispatch.
#[must_use]
pub fn fft_2d_leto<T>(
    field: leto::ArrayView2<'_, T>,
) -> leto::Array<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 2>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let dense_array = field.to_contiguous();
    let output = fft_2d_array::<T>(&dense_array);
    leto::Array::<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 2>::from_mnemosyne_slice(
        output.shape(),
        output
            .as_slice()
            .expect("FFT output from Leto is contiguous"),
    )
    .expect("FFT spectrum shape must match Leto output shape")
}

/// Forward 3D FFT of a Leto real view using generic storage dispatch.
#[must_use]
pub fn fft_3d_leto<T>(
    field: leto::ArrayView3<'_, T>,
) -> leto::Array<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 3>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let dense_array = field.to_contiguous();
    let output = fft_3d_array::<T>(&dense_array);
    leto::Array::<Complex<T::PlanScalar>, leto::MnemosyneStorage<Complex<T::PlanScalar>>, 3>::from_mnemosyne_slice(
        output.shape(),
        output
            .as_slice()
            .expect("FFT output from Leto is contiguous"),
    )
    .expect("FFT spectrum shape must match Leto output shape")
}
