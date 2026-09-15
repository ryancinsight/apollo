use super::super::lanes;
use super::super::twiddles::cached_power_of_two_twiddle;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_2d_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::real_fft::split_twiddle_table;
use crate::application::execution::plan::fft::layout::{transpose_matrices, with_c_order_view};
use crate::domain::metadata::shape::Shape2D;
use eunomia::Complex;
use leto::{Array2, ArrayViewMut2};
use std::sync::Arc;

/// Reusable 2D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan2D<F: MixedRadixScalar> {
    nx: usize,
    ny: usize,
    ny_c: usize,
    twiddle_row_fwd: Option<Arc<[F::Complex]>>,
    twiddle_row_inv: Option<Arc<[F::Complex]>>,
    twiddle_col_fwd: Option<Arc<[F::Complex]>>,
    twiddle_col_inv: Option<Arc<[F::Complex]>>,
    /// Tables for the packed rows of the real split: a real row of `ny`
    /// samples is `ny/2` complex ones.
    twiddle_half_row_fwd: Option<Arc<[F::Complex]>>,
    twiddle_half_row_inv: Option<Arc<[F::Complex]>>,
    /// The real split's `W_ny^k`, evaluated once for every row of the
    /// half-spectrum pair rather than once per row.
    split_twiddles: Box<[F::Complex]>,
}

impl<F: MixedRadixScalar> std::fmt::Debug for FftPlan2D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan2D")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("ny_c", &self.ny_c)
            .finish()
    }
}

impl<F> FftPlan2D<F>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    /// Create a new 2D plan.
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        crate::application::execution::kernel::scratch_hook::ensure_registered();
        let (nx, ny) = (shape.nx(), shape.ny());
        let m = ny / 2;
        Self {
            nx,
            ny,
            ny_c: m + 1,
            twiddle_row_fwd: cached_power_of_two_twiddle::<F, true>(ny),
            twiddle_row_inv: cached_power_of_two_twiddle::<F, false>(ny),
            twiddle_col_fwd: cached_power_of_two_twiddle::<F, true>(nx),
            twiddle_col_inv: cached_power_of_two_twiddle::<F, false>(nx),
            twiddle_half_row_fwd: cached_power_of_two_twiddle::<F, true>(m),
            twiddle_half_row_inv: cached_power_of_two_twiddle::<F, false>(m),
            split_twiddles: split_twiddle_table::<F>(ny),
        }
    }

    /// Return the half-spectrum bookkeeping value `ny / 2 + 1`.
    #[must_use]
    pub fn ny_c(&self) -> usize {
        self.ny_c
    }

    /// Return the full real-domain shape owned by this plan.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape2D {
        Shape2D::new(self.nx, self.ny)
            .expect("invariant: the plan was built from a validated shape")
    }

    /// Forward transform of a complex array in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex forward shape mismatch"
        );
        let view = ArrayViewMut2::from(data.view_mut());
        self.forward_complex_leto_inplace(view);
    }

    /// Inverse transform of a complex array in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array2<F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex inverse shape mismatch"
        );
        let view = ArrayViewMut2::from(data.view_mut());
        self.inverse_complex_leto_inplace(view);
    }

    /// Forward transform of a complex Leto view in-place.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex forward shape mismatch"
        );
        with_c_order_view(data, |mut contiguous| {
            self.axis_pass_complex::<true>(contiguous.reborrow(), 1);
            self.axis_pass_complex::<true>(contiguous, 0);
        });
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    ///
    /// C-dense views execute directly. Other valid layouts use reusable
    /// thread-local staging and preserve the view's logical row-major order.
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut2<'_, F::Complex>) {
        assert_eq!(
            data.shape(),
            [self.nx, self.ny],
            "complex inverse shape mismatch"
        );
        with_c_order_view(data, |mut contiguous| {
            self.axis_pass_complex::<false>(contiguous.reborrow(), 0);
            self.axis_pass_complex::<false>(contiguous, 1);
        });
    }

    /// One direction's transform of a packed row: the `ny/2` complex samples
    /// that carry a real row's `ny` values in the real split.
    pub(crate) fn half_y_lane<const FORWARD: bool>(
        &self,
    ) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_ {
        let twiddles = if FORWARD {
            &self.twiddle_half_row_fwd
        } else {
            &self.twiddle_half_row_inv
        };
        lanes::lane_over::<F, FORWARD>(twiddles.as_deref())
    }

    /// The real split's twiddles for the rows, `W_ny^k` for `k = 1..⌈ny/4⌉`.
    pub(crate) fn split_twiddles(&self) -> &[F::Complex] {
        &self.split_twiddles
    }

    /// One direction's transform of a full row, for real rows the split does
    /// not admit.
    pub(crate) fn y_lane<const FORWARD: bool>(
        &self,
    ) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_ {
        self.lane::<FORWARD>(1)
    }

    /// Transforms axis 0 of a C-order `[nx, row_len]` plane in place.
    ///
    /// The half-spectrum pair runs x on the `(nx, ny/2 + 1)` plane its rows
    /// leave, so `row_len` is the plane's, not the plan's. The columns are
    /// transposed into the plan's full-plane scratch, transformed there as
    /// contiguous lanes, and transposed back.
    pub(crate) fn x_axis_inplace<const FORWARD: bool>(
        &self,
        data: &mut [F::Complex],
        row_len: usize,
    ) {
        with_2d_scratch::<F::Complex, _>(self.nx * row_len, |scratch| {
            transpose_matrices(data, scratch, 1, self.nx, row_len);
            lanes::execute::<F, FORWARD>(scratch, data, self.nx, self.lane::<FORWARD>(0));
            transpose_matrices(scratch, data, 1, row_len, self.nx);
        });
    }

    fn axis_pass_complex<const FORWARD: bool>(
        &self,
        mut data: ArrayViewMut2<'_, F::Complex>,
        axis: usize,
    ) {
        let data_slice = data
            .as_mut_slice()
            .expect("invariant: 2D axis execution receives C-order data");
        match axis {
            0 => self.x_axis_inplace::<FORWARD>(data_slice, self.ny),
            1 => lanes::contiguous::<F, FORWARD, 2>(data_slice, self.ny, self.lane::<FORWARD>(1)),
            _ => unreachable!("invariant: the entry points validate the axis"),
        }
    }

    /// One direction's lane transform along `axis`: the cached power-of-two
    /// twiddles where the length has them, the generic mixed radix otherwise.
    fn lane<const FORWARD: bool>(
        &self,
        axis: usize,
    ) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_ {
        let twiddles = match (axis, FORWARD) {
            (0, true) => &self.twiddle_col_fwd,
            (0, false) => &self.twiddle_col_inv,
            (1, true) => &self.twiddle_row_fwd,
            (1, false) => &self.twiddle_row_inv,
            _ => unreachable!("invariant: the entry points validate the axis"),
        }
        .as_deref();
        lanes::lane_over::<F, FORWARD>(twiddles)
    }
}
