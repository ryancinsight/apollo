//! WGPU device acquisition and FrFT execution backend.

use std::{borrow::Cow, sync::Arc};

use num_complex::{Complex32, Complex64};

use apollo_fft::PrecisionProfile;
use apollo_frft::FrftStorage;

use crate::application::plan::{FrftWgpuPlan, UnitaryFrftWgpuPlan};
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::FrftGpuKernel;
use crate::infrastructure::unitary_kernel::UnitaryFrftGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    FrftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for FrFT execution.
#[derive(Debug, Clone)]
pub struct FrftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<FrftGpuKernel>,
    unitary_kernel: Arc<UnitaryFrftGpuKernel>,
}

impl FrftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(FrftGpuKernel::new(device.inner()));
        let unitary_kernel = Arc::new(UnitaryFrftGpuKernel::new(device.inner()));
        Ok(Self {
            device,
            kernel,
            unitary_kernel,
        })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-frft-wgpu")?)
    }

    /// Return truthful current capabilities (forward and inverse both implemented).
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        self.device.device()
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        self.device.queue()
    }

    /// Create a metadata plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, order: f32) -> FrftWgpuPlan {
        FrftWgpuPlan::new(len, order)
    }

    /// Create a unitary-FrFT metadata plan descriptor.
    #[must_use]
    pub const fn plan_unitary(&self, len: usize, order: f32) -> UnitaryFrftWgpuPlan {
        UnitaryFrftWgpuPlan::new(len, order)
    }

    /// Execute the forward FrFT for a complex-valued f32 signal.
    ///
    /// Validates input length, determines the dispatch mode from the plan order,
    /// precomputes cot/csc/scale for non-integer orders, then calls the kernel.
    pub fn execute_forward(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate(plan, input)?;
        let (mode, cot, csc, scale_re, scale_im) = mode_params(plan);
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            plan.len(),
            mode,
            cot,
            csc,
            scale_re,
            scale_im,
        )
    }

    /// Execute the forward FrFT from a Leto complex host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &FrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse FrFT, equivalent to the forward FrFT of order -a.
    pub fn execute_inverse(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let inv_plan = FrftWgpuPlan::new(plan.len(), -plan.order());
        Self::validate(&inv_plan, input)?;
        let (mode, cot, csc, scale_re, scale_im) = mode_params(&inv_plan);
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            inv_plan.len(),
            mode,
            cot,
            csc,
            scale_re,
            scale_im,
        )
    }

    /// Execute the inverse FrFT from a Leto complex host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &FrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_inverse(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward unitary DFrFT for a complex-valued f32 signal.
    ///
    /// Computes DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x using the
    /// Grünbaum eigenbasis (Candan 2000). Provably norm-preserving for all real orders.
    ///
    /// V is computed CPU-side (O(N³)) and uploaded to GPU as a column-major f32 buffer.
    /// Three GPU passes enforce cross-workgroup storage ordering.
    pub fn execute_unitary_forward(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_unitary(plan, input)?;
        self.unitary_kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            plan.order(),
        )
    }

    /// Execute the forward unitary DFrFT from a Leto complex host view.
    pub fn execute_unitary_forward_leto(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_unitary_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse unitary DFrFT, equivalent to the forward DFrFT of order −a.
    ///
    /// The inverse of DFrFT_a is DFrFT_{−a}: negate the order. The result satisfies
    /// DFrFT_{−a}(DFrFT_a(x)) = x for all real a and all inputs x.
    pub fn execute_unitary_inverse(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let inv_plan = UnitaryFrftWgpuPlan::new(plan.len(), -plan.order());
        Self::validate_unitary(&inv_plan, input)?;
        self.unitary_kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            inv_plan.order(),
        )
    }

    /// Execute the inverse unitary DFrFT from a Leto complex host view.
    pub fn execute_unitary_inverse_leto(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_unitary_inverse(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward FrFT with typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// Promotes represented input once to `Complex32`, dispatches the GPU kernel,
    /// and quantizes output back to the requested storage type.
    pub fn execute_forward_typed_into<T: FrftStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_frft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<Complex32> = input
            .iter()
            .map(|v| {
                let c = v.to_complex64();
                Complex32::new(c.re as f32, c.im as f32)
            })
            .collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    /// Execute typed forward FrFT from a Leto host view.
    pub fn execute_forward_leto_typed<T: FrftStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse FrFT with typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn execute_inverse_typed_into<T: FrftStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_frft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented: Vec<Complex32> = input
            .iter()
            .map(|v| {
                let c = v.to_complex64();
                Complex32::new(c.re as f32, c.im as f32)
            })
            .collect();
        let computed = self.execute_inverse(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    /// Execute typed inverse FrFT from a Leto host view.
    pub fn execute_inverse_leto_typed<T: FrftStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_frft_typed_precision<T: FrftStorage>(
        precision: PrecisionProfile,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate(plan: &FrftWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "invalid plan: length must be greater than zero".to_owned(),
            });
        }
        if !plan.order().is_finite() {
            return Err(WgpuError::NonFiniteOrder);
        }
        if input.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        Ok(())
    }

    fn validate_unitary(plan: &UnitaryFrftWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "invalid plan: length must be greater than zero".to_owned(),
            });
        }
        if !plan.order().is_finite() {
            return Err(WgpuError::NonFiniteOrder);
        }
        if input.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        Ok(())
    }
}

/// Determine the dispatch mode and trigonometric parameters from a plan.
///
/// Returns (mode, cot, csc, scale_re, scale_im) where:
/// - Integer rotations (frac < 1e-5): mode in {0,1,2,3}, chirp params zeroed.
/// - Non-integer order: mode=4, cot/csc/scale computed from alpha=reduced*pi/2.
fn mode_params(plan: &FrftWgpuPlan) -> (u32, f32, f32, f32, f32) {
    let order = plan.order();
    let reduced = ((order % 4.0_f32) + 4.0_f32) % 4.0_f32;
    let rounded = reduced.round();
    let frac = (reduced - rounded).abs();
    if frac < 1.0e-5_f32 {
        let mode = if reduced < 0.5_f32 || reduced > 3.5_f32 {
            0_u32
        } else if reduced < 1.5_f32 {
            1_u32
        } else if reduced < 2.5_f32 {
            2_u32
        } else {
            3_u32
        };
        (mode, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32)
    } else {
        let alpha = reduced * std::f32::consts::FRAC_PI_2;
        let sin_a = alpha.sin();
        let cos_a = alpha.cos();
        let cot = cos_a / sin_a;
        let csc = 1.0_f32 / sin_a;
        let n_f = plan.len() as f32;
        // scale = sqrt(1 - i*cot) / sqrt(n)
        // = sqrt((1-i*cot)/n) via polar form
        let z_norm = (1.0_f32 + cot * cot).sqrt();
        let z_arg = (-cot).atan2(1.0_f32);
        let sr = z_norm.sqrt() / n_f.sqrt();
        let sa = z_arg * 0.5_f32;
        let scale_re = sr * sa.cos();
        let scale_im = sr * sa.sin();
        (4_u32, cot, csc, scale_re, scale_im)
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let len = view.shape()[0];
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(*view.get([index]).map_err(|err| WgpuError::ShapeMismatch {
            message: format!("invalid Leto FrFT 1D view: {err:?}"),
        })?);
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto FrFT output: {err:?}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1_u32, 2, 3, 4]).expect("input");
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1_u32, 99, 2, 99, 3, 99, 4, 99]).expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }
}
