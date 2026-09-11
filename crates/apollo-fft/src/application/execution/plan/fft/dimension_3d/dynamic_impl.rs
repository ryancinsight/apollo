use super::super::twiddles::cached_power_of_two_twiddle;
use super::passes::{self, AxisLanes};
use super::rotated::RotatedSpectrum;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::{
    dispatch_inplace, forward_inplace, inverse_inplace, MixedRadixScalar,
};
use crate::application::execution::kernel::real_fft::split_twiddle_table;
use crate::application::execution::plan::fft::layout::with_c_order_view;
use crate::domain::metadata::shape::Shape3D;
use eunomia::Complex;
use leto::Array3;
use leto::ArrayViewMut3;
use std::sync::Arc;

/// Reusable separable 3D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan3D<F: MixedRadixScalar> {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
    pub(crate) nz_c: usize,
    pub(crate) twiddle_z_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_z_inv: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_y_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_y_inv: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_x_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_x_inv: Option<Arc<[F::Complex]>>,
    /// Tables for the packed z lanes of the real split: a real lane of `nz`
    /// samples is `nz/2` complex ones.
    pub(crate) twiddle_half_z_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_half_z_inv: Option<Arc<[F::Complex]>>,
    /// The real split's `W_nz^k`, evaluated once for every z lane of the
    /// half-spectrum pair rather than once per lane.
    pub(crate) split_twiddles: Box<[F::Complex]>,
}

impl<F: MixedRadixScalar> std::fmt::Debug for FftPlan3D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan3D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nz_c", &self.nz_c)
            .finish()
    }
}

impl<F> FftPlan3D<F>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    /// Create a new 3D plan.
    #[must_use]
    pub fn new(shape: Shape3D) -> Self {
        crate::ensure_thread_local_scratch_hook_registered();
        let (nx, ny, nz) = (shape.nx(), shape.ny(), shape.nz());
        let m = nz / 2;
        let nz_c_val = m + 1;
        Self {
            nx,
            ny,
            nz,
            nz_c: nz_c_val,
            twiddle_z_fwd: cached_power_of_two_twiddle::<F, true>(nz),
            twiddle_z_inv: cached_power_of_two_twiddle::<F, false>(nz),
            twiddle_y_fwd: cached_power_of_two_twiddle::<F, true>(ny),
            twiddle_y_inv: cached_power_of_two_twiddle::<F, false>(ny),
            twiddle_x_fwd: cached_power_of_two_twiddle::<F, true>(nx),
            twiddle_x_inv: cached_power_of_two_twiddle::<F, false>(nx),
            twiddle_half_z_fwd: cached_power_of_two_twiddle::<F, true>(m),
            twiddle_half_z_inv: cached_power_of_two_twiddle::<F, false>(m),
            split_twiddles: split_twiddle_table::<F>(nz),
        }
    }

    /// Return the half-spectrum bookkeeping value `nz / 2 + 1`.
    #[must_use]
    pub fn nz_c(&self) -> usize {
        self.nz_c
    }

    /// Return the full real-domain shape owned by this plan.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape3D {
        Shape3D::new(self.nx, self.ny, self.nz)
            .expect("invariant: the plan was built from a validated shape")
    }

    /// Forward transform of a complex field in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex forward shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex field in-place with FFTW-compatible normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex inverse shape mismatch"
        );
        let view = ArrayViewMut3::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward complex FFT along a single `axis` (0, 1, or 2) in-place.
    ///
    /// This is the batched, cache-tiled, parallel per-axis building block of
    /// [`Self::forward_complex_inplace`] — it transforms all pencils along `axis`
    /// at once (32×32 tiled gather/scatter for non-contiguous axes, Moirai
    /// parallelism over pencils, cached power-of-two twiddles). Exposing it lets
    /// callers that need only one axis (e.g. spectral derivatives `∂/∂xₐ`) avoid
    /// the cost of a full 3-D transform. Unnormalized, matching the 1-D forward
    /// convention; an `axis` whose extent is 1 is a no-op.
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or `axis >= 3`.
    pub fn forward_axis_complex_inplace(&self, data: &mut Array3<F::Complex>, axis: usize) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex::<true>(ArrayViewMut3::from(data.view_mut()), axis);
    }

    /// Inverse complex FFT along a single `axis` in-place, normalized by that
    /// axis's length, so `forward_axis` followed by `inverse_axis` along the same
    /// axis is the identity. See [`Self::forward_axis_complex_inplace`].
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or `axis >= 3`.
    pub fn inverse_axis_complex_inplace(&self, data: &mut Array3<F::Complex>, axis: usize) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "axis FFT shape mismatch"
        );
        assert!(axis < 3, "axis must be 0, 1, or 2");
        self.axis_pass_complex::<false>(ArrayViewMut3::from(data.view_mut()), axis);
    }

    /// Forward transform of a complex Leto view in-place.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex forward shape mismatch"
        );
        with_c_order_view(data, |contiguous| self.all_axes::<true>(contiguous));
    }

    /// Inverse transform of a complex Leto view in-place with FFTW-compatible normalization.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut3<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "complex inverse shape mismatch"
        );
        with_c_order_view(data, |contiguous| self.all_axes::<false>(contiguous));
    }

    /// Forward transform leaving the spectrum in `(z, x, y)` order.
    ///
    /// Every axis is transformed, in two full-volume moves instead of the
    /// three [`Self::forward_complex_inplace`] pays, because the move that
    /// only restores the caller's C order is not made. The order rides in the
    /// returned [`RotatedSpectrum`], which borrows `data` until the matching
    /// [`Self::inverse_complex_rotated`] takes it back.
    ///
    /// # Panics
    /// - Shape mismatch with the plan, or a view that is not C-order dense.
    pub fn forward_complex_rotated<'data>(
        &self,
        data: &'data mut Array3<F::Complex>,
    ) -> RotatedSpectrum<'data, F> {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny, self.nz],
            "rotated forward shape mismatch"
        );
        let shape = [self.nx, self.ny, self.nz];
        let slice = data
            .as_slice_mut()
            .expect("invariant: 3D rotated execution receives C-order data");
        passes::all_axes_leaving_rotated::<F, true, _, _, _>(
            slice,
            shape,
            AxisLanes {
                x: self.lane::<true>(0),
                y: self.lane::<true>(1),
                z: self.lane::<true>(2),
            },
        );
        RotatedSpectrum::new(slice, shape)
    }

    /// Inverse transform of a `(z, x, y)` spectrum, leaving C order.
    ///
    /// Two full-volume moves, so a round trip through this pair costs four
    /// where the C-order pair costs six. Normalized like
    /// [`Self::inverse_complex_inplace`].
    pub fn inverse_complex_rotated(&self, spectrum: RotatedSpectrum<'_, F>) {
        let (slice, shape) = spectrum.into_parts();
        assert_eq!(
            shape,
            [self.nx, self.ny, self.nz],
            "rotated inverse shape mismatch"
        );
        passes::all_axes_from_rotated::<F, false, _, _, _>(
            slice,
            shape,
            AxisLanes {
                x: self.lane::<false>(0),
                y: self.lane::<false>(1),
                z: self.lane::<false>(2),
            },
        );
    }

    /// One direction's transform of a packed z lane: the `nz/2` complex
    /// samples that carry a real lane's `nz` values in the real split.
    pub(crate) fn half_z_lane<const FORWARD: bool>(
        &self,
    ) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_ {
        let twiddles = if FORWARD {
            &self.twiddle_half_z_fwd
        } else {
            &self.twiddle_half_z_inv
        };
        lane_over::<F, FORWARD>(twiddles.as_deref())
    }

    /// The real split's twiddles for the z lanes, `W_nz^k` for `k = 1..⌈nz/4⌉`.
    pub(crate) fn split_twiddles(&self) -> &[F::Complex] {
        &self.split_twiddles
    }

    /// One direction's transform of a full z lane, for real lanes the split
    /// does not admit.
    pub(crate) fn z_lane<const FORWARD: bool>(
        &self,
    ) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_ {
        self.lane::<FORWARD>(2)
    }

    /// Transforms axes 0 and 1 of a C-order `[nx, ny, depth]` volume in place.
    ///
    /// The half-spectrum pair runs x and y on the `(nx, ny, nz/2 + 1)` volume
    /// its z lanes leave, so `depth` is the volume's, not the plan's.
    pub(crate) fn xy_axes_inplace<const FORWARD: bool>(
        &self,
        data: &mut [F::Complex],
        depth: usize,
    ) {
        passes::xy_axes::<F, FORWARD>(
            data,
            [self.nx, self.ny, depth],
            self.lane::<FORWARD>(0),
            self.lane::<FORWARD>(1),
        );
    }

    fn axis_pass_complex<const FORWARD: bool>(
        &self,
        mut data: ArrayViewMut3<'_, F::Complex>,
        axis: usize,
    ) {
        let shape = [self.nx, self.ny, self.nz];
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 3D axis execution receives C-order data");
        match axis {
            0 => passes::axis0::<F, FORWARD>(data_slice, shape, self.lane::<FORWARD>(0)),
            1 => passes::axis1::<F, FORWARD>(data_slice, shape, self.lane::<FORWARD>(1)),
            2 => passes::axis2::<F, FORWARD>(data_slice, self.nz, self.lane::<FORWARD>(2)),
            _ => unreachable!("invariant: the entry points validate the axis"),
        }
    }

    fn all_axes<const FORWARD: bool>(&self, mut data: ArrayViewMut3<'_, F::Complex>) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 3D axis execution receives C-order data");
        passes::all_axes::<F, FORWARD, _, _, _>(
            data_slice,
            [self.nx, self.ny, self.nz],
            AxisLanes {
                x: self.lane::<FORWARD>(0),
                y: self.lane::<FORWARD>(1),
                z: self.lane::<FORWARD>(2),
            },
        );
    }

    /// One direction's lane transform along `axis`: the cached power-of-two
    /// twiddles where the length has them, the generic mixed radix otherwise.
    fn lane<const FORWARD: bool>(
        &self,
        axis: usize,
    ) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_ {
        let twiddles = match (axis, FORWARD) {
            (0, true) => &self.twiddle_x_fwd,
            (0, false) => &self.twiddle_x_inv,
            (1, true) => &self.twiddle_y_fwd,
            (1, false) => &self.twiddle_y_inv,
            (2, true) => &self.twiddle_z_fwd,
            (2, false) => &self.twiddle_z_inv,
            _ => unreachable!("invariant: the entry points validate the axis"),
        }
        .as_deref();
        lane_over::<F, FORWARD>(twiddles)
    }
}

/// One direction's transform of a lane of the table's length: the cached
/// power-of-two twiddles where the length has them, the generic mixed radix
/// otherwise.
fn lane_over<F, const FORWARD: bool>(
    twiddles: Option<&[F::Complex]>,
) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    move |lane: &mut [F::Complex]| match (FORWARD, twiddles) {
        (true, Some(twiddles)) => dispatch_inplace::<F, false, false>(lane, Some(twiddles)),
        (false, Some(twiddles)) => dispatch_inplace::<F, true, true>(lane, Some(twiddles)),
        (true, None) => forward_inplace::<F>(lane),
        (false, None) => inverse_inplace::<F>(lane),
    }
}
