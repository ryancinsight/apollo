//! Two-dimensional inverse real FFT operations.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::real_storage::{half_plane, RealFftData};
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::Shape2D;
use eunomia::Complex;
use leto::Array2;

/// Inverse 2D FFT of a complex spectrum using generic storage dispatch.
///
/// Where the real split admits `ny`, only the lower `ny/2 + 1` bins of each
/// row are read (the rest of a real field's spectrum is their conjugate
/// mirror) and the rows run at half length through the half-spectrum pair
/// ([`ifft_2d_array_half_into`]); other lengths take the full inverse. One
/// allocation, the returned plane.
#[must_use]
pub fn ifft_2d_array<T>(field_hat: &Array2<Complex<T::PlanScalar>>) -> Array2<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = field_hat.shape();
    let plan =
        T::get_2d_plan(Shape2D::new(nx, ny).expect("ifft_2d_array requires non-zero dimensions"));
    if let Some(out) = half_plane::inverse_owned_via_split::<T>(plan.as_ref(), field_hat) {
        return out;
    }
    T::inverse_2d(plan.as_ref(), field_hat)
}

/// Inverse 2D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`. Routed like [`ifft_2d_array`] where the split
/// admits `ny` and both arrays are C-contiguous: the lower bins of each row
/// are packed in place and nothing else is read. Allocates nothing on a warm
/// plan.
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
    let plan = T::get_2d_plan(
        Shape2D::new(nx, ny)
            .expect("ifft_2d_array_into_spectrum_scratch requires non-zero dimensions"),
    );
    if half_plane::inverse_spectrum_via_split::<T>(plan.as_ref(), field_hat, out) {
        return;
    }
    T::inverse_2d_spectrum_into(plan.as_ref(), field_hat, out);
}

/// Inverse 2D FFT into caller-owned typed real storage and typed scratch spectrum.
///
/// Routed like [`ifft_2d_array`] where the split admits `ny` and the three
/// arrays are C-contiguous, packing the lower bins into the front of
/// `scratch`; allocates nothing on a warm plan.
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
    let plan = T::get_2d_plan(
        Shape2D::new(nx, ny).expect("ifft_2d_array_into requires non-zero dimensions"),
    );
    if half_plane::inverse_into_via_split::<T>(plan.as_ref(), field_hat, out, scratch) {
        return;
    }
    T::inverse_2d_into(plan.as_ref(), field_hat, out, scratch);
}

/// Inverse 2D FFT of an `(nx, ny/2 + 1)` half spectrum into caller-owned real
/// storage, consuming the spectrum as scratch.
///
/// Inverts [`fft_2d_array_half_into`](crate::fft_2d_array_half_into) with the
/// `1 / (nx * ny)` normalization of the full inverse: the result is the real
/// part of the full inverse of the half spectrum's Hermitian completion, so
/// imaginary parts a real field's spectrum cannot carry are ignored. The x
/// pass runs on the half plane and each row through the half-length inverse
/// where the split admits `ny`; allocates nothing on a warm plan.
///
/// # Panics
///
/// Panics on a zero dimension, or if `field_hat` is not a C-contiguous
/// `(nx, ny/2 + 1)` array for `out`'s shape.
pub fn ifft_2d_array_half_into<T>(
    field_hat: &mut Array2<Complex<T::PlanScalar>>,
    out: &mut Array2<T>,
) where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny] = out.shape();
    T::inverse_2d_half_into(
        T::get_2d_plan(
            Shape2D::new(nx, ny).expect("ifft_2d_array_half_into requires non-zero dimensions"),
        )
        .as_ref(),
        field_hat,
        out,
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
