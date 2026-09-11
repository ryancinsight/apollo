//! Inverse real FFT API functions.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::{Array1, Array2, Array3};

/// Inverse 1D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_1d_array<T>(field_hat: &Array1<Complex<T::PlanScalar>>) -> Array1<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
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
    T::inverse_1d_slice_owned(
        T::get_1d_plan(
            Shape1D::new(spectrum.len()).expect("ifft_1d_slice requires non-zero length"),
        )
        .as_ref(),
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

/// Inverse 2D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_2d_array<T>(field_hat: &Array2<Complex<T::PlanScalar>>) -> Array2<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field_hat.shape();
    T::inverse_2d(
        T::get_2d_plan(Shape2D::new(nx, ny).expect("ifft_2d_array requires non-zero dimensions"))
            .as_ref(),
        field_hat,
    )
}

/// Inverse 2D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_2d_array_into_spectrum_scratch<T>(
    field_hat: &mut Array2<Complex<T::PlanScalar>>,
    out: &mut Array2<T>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field_hat.shape();
    debug_assert_eq!(
        out.shape(),
        [nx, ny],
        "ifft_2d_array_into_spectrum_scratch: shape mismatch"
    );
    T::inverse_2d_spectrum_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny)
                .expect("ifft_2d_array_into_spectrum_scratch requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 2D FFT into caller-owned typed real storage and typed scratch spectrum.
pub fn ifft_2d_array_into<T>(
    field_hat: &Array2<Complex<T::PlanScalar>>,
    out: &mut Array2<T>,
    scratch: &mut Array2<Complex<T::PlanScalar>>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field_hat.shape();
    T::inverse_2d_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("ifft_2d_array_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    );
}

/// Inverse 2D FFT into caller-owned typed real storage and typed scratch
/// spectrum for a compile-time-known shape.
pub fn ifft_2d_array_static_into<T, const NX: usize, const NY: usize>(
    field_hat: &Array2<Complex<T::PlanScalar>>,
    out: &mut Array2<T>,
    scratch: &mut Array2<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        [NX, NY],
        "ifft_2d_array_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY],
        "ifft_2d_array_static_into: output shape mismatch"
    );
    debug_assert_eq!(
        scratch.shape(),
        [NX, NY],
        "ifft_2d_array_static_into: scratch shape mismatch"
    );
    T::inverse_2d_static_into::<NX, NY>(field_hat, out, scratch);
}

/// Inverse 3D FFT of a complex spectrum using generic storage dispatch.
#[must_use]
pub fn ifft_3d_array<T>(field_hat: &Array3<Complex<T::PlanScalar>>) -> Array3<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field_hat.shape();
    T::inverse_3d(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
    )
}

/// Inverse 3D FFT into caller-owned typed real storage and typed scratch spectrum.
pub fn ifft_3d_array_into<T>(
    field_hat: &Array3<Complex<T::PlanScalar>>,
    out: &mut Array3<T>,
    scratch: &mut Array3<Complex<T::PlanScalar>>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field_hat.shape();
    T::inverse_3d_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
        scratch,
    )
}

/// Inverse 3D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`.
pub fn ifft_3d_array_into_spectrum_scratch<T>(
    field_hat: &mut Array3<Complex<T::PlanScalar>>,
    out: &mut Array3<T>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field_hat.shape();
    debug_assert_eq!(
        out.shape(),
        [nx, ny, nz],
        "ifft_3d_array_into_spectrum_scratch: shape mismatch"
    );
    T::inverse_3d_spectrum_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz)
                .expect("ifft_3d_array_into_spectrum_scratch requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 3D FFT of a `(nx, ny, nz/2 + 1)` half spectrum into real storage.
///
/// The inverse of [`fft_3d_array_half_into`](crate::fft_3d_array_half_into),
/// normalized like [`ifft_3d_array_into`]. `field_hat` is consumed as scratch.
/// The result is the real part of the full inverse of the half spectrum's
/// Hermitian completion, so the imaginary parts a real field's spectrum cannot
/// carry are ignored.
///
/// # Panics
///
/// If `field_hat` is not a C-contiguous `(nx, ny, nz/2 + 1)` array for `out`'s
/// shape, or `out` is not C-contiguous.
pub fn ifft_3d_array_half_into<T>(
    field_hat: &mut Array3<Complex<T::PlanScalar>>,
    out: &mut Array3<T>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = out.shape();
    T::inverse_3d_half_into(
        T::get_3d_plan(
            Shape3D::new(nx, ny, nz).expect("ifft_3d_array_half_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
    );
}

/// Inverse 3D FFT into caller-owned typed real storage and typed scratch
/// spectrum for a compile-time-known shape.
pub fn ifft_3d_array_static_into<T, const NX: usize, const NY: usize, const NZ: usize>(
    field_hat: &Array3<Complex<T::PlanScalar>>,
    out: &mut Array3<T>,
    scratch: &mut Array3<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    debug_assert_eq!(
        field_hat.shape(),
        [NX, NY, NZ],
        "ifft_3d_array_static_into: input shape mismatch"
    );
    debug_assert_eq!(
        out.shape(),
        [NX, NY, NZ],
        "ifft_3d_array_static_into: output shape mismatch"
    );
    debug_assert_eq!(
        scratch.shape(),
        [NX, NY, NZ],
        "ifft_3d_array_static_into: scratch shape mismatch"
    );
    T::inverse_3d_static_into::<NX, NY, NZ>(field_hat, out, scratch);
}

/// Inverse 2D FFT of a Leto spectrum view using generic storage dispatch.
#[must_use]
pub fn ifft_2d_leto<T>(
    field_hat: leto::ArrayView2<'_, Complex<T::PlanScalar>>,
) -> leto::Array<T, leto::MnemosyneStorage<T>, 2>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let dense_array = field_hat.to_contiguous();
    let output = ifft_2d_array::<T>(&dense_array);
    leto::Array::<T, leto::MnemosyneStorage<T>, 2>::from_mnemosyne_slice(
        output.shape(),
        output
            .as_slice()
            .expect("IFFT output from Leto is contiguous"),
    )
    .expect("IFFT signal shape must match Leto output shape")
}

/// Inverse 3D FFT of a Leto spectrum view using generic storage dispatch.
#[must_use]
pub fn ifft_3d_leto<T>(
    field_hat: leto::ArrayView3<'_, Complex<T::PlanScalar>>,
) -> leto::Array<T, leto::MnemosyneStorage<T>, 3>
where
    T: RealFftData + PlanCacheProvider + Copy,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let dense_array = field_hat.to_contiguous();
    let output = ifft_3d_array::<T>(&dense_array);
    leto::Array::<T, leto::MnemosyneStorage<T>, 3>::from_mnemosyne_slice(
        output.shape(),
        output
            .as_slice()
            .expect("IFFT output from Leto is contiguous"),
    )
    .expect("IFFT signal shape must match Leto output shape")
}
