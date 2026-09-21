//! Three-dimensional inverse real FFT operations.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::real_storage::{RealFftData, half_volume};
use crate::application::orchestration::cache::plans::PlanCacheProvider;
use crate::domain::metadata::shape::Shape3D;
use eunomia::Complex;
use leto::Array3;

/// Inverse 3D FFT of a complex spectrum using generic storage dispatch.
///
/// Where the real split admits `nz`, only the lower `nz/2 + 1` bins of each z
/// lane are read (the rest of a real field's spectrum is their conjugate
/// mirror) and the lanes run at half length through the half-spectrum pair
/// ([`ifft_3d_array_half_into`]); other lengths take the full inverse. One
/// allocation, the returned volume.
#[must_use]
pub fn ifft_3d_array<T>(field_hat: &Array3<Complex<T::PlanScalar>>) -> Array3<T>
where
    T: RealFftData + PlanCacheProvider,
    Complex<T::PlanScalar>: PlanScratch,
    <T as RealFftData>::PlanScalar: PlanCacheProvider,
{
    let [nx, ny, nz] = field_hat.shape();
    let plan = T::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("ifft_3d_array requires non-zero dimensions"),
    );
    if let Some(out) = half_volume::routed::inverse_owned_via_split::<T>(plan.as_ref(), field_hat) {
        return out;
    }
    T::inverse_3d(plan.as_ref(), field_hat)
}

/// Inverse 3D FFT into caller-owned typed real storage and typed scratch spectrum.
///
/// Routed like [`ifft_3d_array`] where the split admits `nz` and the three
/// arrays are C-contiguous, packing the lower bins into the front of
/// `scratch`; allocates nothing on a warm plan.
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
    let plan = T::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("ifft_3d_array_into requires non-zero dimensions"),
    );
    if half_volume::routed::inverse_into_via_split::<T>(plan.as_ref(), field_hat, out, scratch) {
        return;
    }
    T::inverse_3d_into(plan.as_ref(), field_hat, out, scratch);
}

/// Inverse 3D FFT into caller-owned typed real storage, reusing the mutable
/// typed spectrum as scratch.
///
/// This mutates `field_hat`. Routed like [`ifft_3d_array`] where the split
/// admits `nz` and both arrays are C-contiguous: the lower bins of each lane
/// are packed in place and nothing else is read. Allocates nothing on a warm
/// plan.
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
    let plan = T::get_3d_plan(
        Shape3D::new(nx, ny, nz)
            .expect("ifft_3d_array_into_spectrum_scratch requires non-zero dimensions"),
    );
    if half_volume::routed::inverse_spectrum_via_split::<T>(plan.as_ref(), field_hat, out) {
        return;
    }
    T::inverse_3d_spectrum_into(plan.as_ref(), field_hat, out);
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
