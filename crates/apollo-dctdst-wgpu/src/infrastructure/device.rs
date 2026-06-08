//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_dctdst::{RealTransformKind, RealTransformStorage};
use apollo_fft::PrecisionProfile;
use ndarray::{Array2, Array3};

use crate::application::plan::DctDstWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{DctGpuKernel, DctMode};
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    DctDstWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct DctDstWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<DctGpuKernel>,
}

impl DctDstWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(DctGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-dctdst-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::full(true)
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

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, kind: RealTransformKind) -> DctDstWgpuPlan {
        DctDstWgpuPlan::new(len, kind)
    }

    /// Execute the unnormalized configured real-to-real transform for a real-valued `f32` signal.
    ///
    /// Supported kinds: DCT-I, DCT-II, DCT-III, DCT-IV, DST-I, DST-II, DST-III, and DST-IV.
    /// DCT-I requires length >= 2 and returns [`WgpuError::InvalidPlan`] otherwise.
    pub fn execute_forward(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        let mode = match plan.kind() {
            RealTransformKind::DctII => DctMode::Dct2,
            RealTransformKind::DctIII => DctMode::Dct3,
            RealTransformKind::DstII => DctMode::Dst2,
            RealTransformKind::DstIII => DctMode::Dst3,
            RealTransformKind::DctI => {
                if plan.len() < 2 {
                    return Err(WgpuError::InvalidPlan {
                        message: format!(
                            "invalid length {}: DCT-I requires length >= 2",
                            plan.len()
                        ),
                    });
                }
                DctMode::Dct1
            }
            RealTransformKind::DctIV => DctMode::Dct4,
            RealTransformKind::DstI => DctMode::Dst1,
            RealTransformKind::DstIV => DctMode::Dst4,
        };
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            mode,
            1.0,
        )
    }

    /// Execute the normalized inverse of the configured real-to-real transform for a real-valued `f32` signal.
    ///
    /// Supported kinds: DCT-I, DCT-II, DCT-III, DCT-IV, DST-I, DST-II, DST-III, and DST-IV.
    /// DCT-I requires length >= 2 and returns [`WgpuError::InvalidPlan`] otherwise.
    /// Inverse scales: DCT-I = 1/(2*(N-1)), DCT-IV = 2/N, DST-I = 1/(2*(N+1)), DST-IV = 2/N.
    pub fn execute_inverse(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        let (mode, scale) = match plan.kind() {
            RealTransformKind::DctII => (DctMode::Dct3, 2.0 / plan.len() as f32),
            RealTransformKind::DctIII => (DctMode::Dct2, 2.0 / plan.len() as f32),
            RealTransformKind::DstII => (DctMode::Dst3, 2.0 / plan.len() as f32),
            RealTransformKind::DstIII => (DctMode::Dst2, 2.0 / plan.len() as f32),
            RealTransformKind::DctI => {
                if plan.len() < 2 {
                    return Err(WgpuError::InvalidPlan {
                        message: format!(
                            "invalid length {}: DCT-I requires length >= 2",
                            plan.len()
                        ),
                    });
                }
                (DctMode::Dct1, 1.0 / (2.0 * (plan.len() - 1) as f32))
            }
            RealTransformKind::DctIV => (DctMode::Dct4, 2.0 / plan.len() as f32),
            RealTransformKind::DstI => (DctMode::Dst1, 1.0 / (2.0 * (plan.len() + 1) as f32)),
            RealTransformKind::DstIV => (DctMode::Dst4, 2.0 / plan.len() as f32),
        };
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            mode,
            scale,
        )
    }

    /// Execute the forward real-to-real transform with typed storage.
    ///
    /// WGPU arithmetic is `f32`; mixed `f16` storage is promoted once to `f32` at
    /// the dispatch boundary and quantized at the output boundary.
    pub fn execute_forward_typed_into<T: RealTransformStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_dct_typed_precision::<T>(precision)?;
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute the normalized inverse real-to-real transform with typed storage.
    pub fn execute_inverse_typed_into<T: RealTransformStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_dct_typed_precision::<T>(precision)?;
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_inverse(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    fn validate_dct_typed_precision<T: RealTransformStorage>(
        precision: PrecisionProfile,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate_plan_input(plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        Ok(())
    }

    /// Execute the unnormalized 2D separable forward real-to-real transform.
    ///
    /// Applies the 1D forward transform along each row then each column.
    /// Requires a square `n × n` input where `n == plan.len()`.
    pub fn execute_forward_2d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array2<f32>,
    ) -> WgpuResult<Array2<f32>> {
        let n = plan.len();
        let (rows, cols) = input.dim();
        if rows != n || cols != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("2D input expected {n}x{n}, got {rows}x{cols}"),
            });
        }
        // Row pass.
        let mut tmp = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row: Vec<f32> = input.row(r).iter().copied().collect();
            let out = self.execute_forward(plan, &row)?;
            for c in 0..n {
                tmp[[r, c]] = out[c];
            }
        }
        // Column pass.
        let mut result = Array2::<f32>::zeros((n, n));
        for c in 0..n {
            let col: Vec<f32> = tmp.column(c).iter().copied().collect();
            let out = self.execute_forward(plan, &col)?;
            for r in 0..n {
                result[[r, c]] = out[r];
            }
        }
        Ok(result)
    }

    /// Execute the normalized 2D separable inverse real-to-real transform.
    ///
    /// Applies the 1D inverse transform along each column then each row.
    /// Requires a square `n × n` input where `n == plan.len()`.
    pub fn execute_inverse_2d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array2<f32>,
    ) -> WgpuResult<Array2<f32>> {
        let n = plan.len();
        let (rows, cols) = input.dim();
        if rows != n || cols != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("2D input expected {n}x{n}, got {rows}x{cols}"),
            });
        }
        // Column pass (inverse).
        let mut tmp = Array2::<f32>::zeros((n, n));
        for c in 0..n {
            let col: Vec<f32> = input.column(c).iter().copied().collect();
            let out = self.execute_inverse(plan, &col)?;
            for r in 0..n {
                tmp[[r, c]] = out[r];
            }
        }
        // Row pass (inverse).
        let mut result = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row: Vec<f32> = tmp.row(r).iter().copied().collect();
            let out = self.execute_inverse(plan, &row)?;
            for c in 0..n {
                result[[r, c]] = out[c];
            }
        }
        Ok(result)
    }

    /// Execute the unnormalized 3D separable forward real-to-real transform.
    ///
    /// Applies the 1D forward transform along axes 0, 1, and 2 in sequence.
    /// Requires a cubic `n × n × n` input where `n == plan.len()`.
    pub fn execute_forward_3d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array3<f32>,
    ) -> WgpuResult<Array3<f32>> {
        let n = plan.len();
        let (d0, d1, d2) = input.dim();
        if d0 != n || d1 != n || d2 != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("3D input expected {n}x{n}x{n}, got {d0}x{d1}x{d2}"),
            });
        }
        // Axis-0 pass.
        let mut tmp0 = Array3::<f32>::zeros((n, n, n));
        for j in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|i| input[[i, j, k]]).collect();
                let out = self.execute_forward(plan, &fiber)?;
                for i in 0..n {
                    tmp0[[i, j, k]] = out[i];
                }
            }
        }
        // Axis-1 pass.
        let mut tmp1 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|j| tmp0[[i, j, k]]).collect();
                let out = self.execute_forward(plan, &fiber)?;
                for j in 0..n {
                    tmp1[[i, j, k]] = out[j];
                }
            }
        }
        // Axis-2 pass.
        let mut result = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                let fiber: Vec<f32> = (0..n).map(|k| tmp1[[i, j, k]]).collect();
                let out = self.execute_forward(plan, &fiber)?;
                for k in 0..n {
                    result[[i, j, k]] = out[k];
                }
            }
        }
        Ok(result)
    }

    /// Execute the normalized 3D separable inverse real-to-real transform.
    ///
    /// Applies the 1D inverse transform along axes 2, 1, and 0 in sequence.
    /// Requires a cubic `n × n × n` input where `n == plan.len()`.
    pub fn execute_inverse_3d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array3<f32>,
    ) -> WgpuResult<Array3<f32>> {
        let n = plan.len();
        let (d0, d1, d2) = input.dim();
        if d0 != n || d1 != n || d2 != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("3D input expected {n}x{n}x{n}, got {d0}x{d1}x{d2}"),
            });
        }
        // Axis-2 pass (inverse).
        let mut tmp0 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                let fiber: Vec<f32> = (0..n).map(|k| input[[i, j, k]]).collect();
                let out = self.execute_inverse(plan, &fiber)?;
                for k in 0..n {
                    tmp0[[i, j, k]] = out[k];
                }
            }
        }
        // Axis-1 pass (inverse).
        let mut tmp1 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|j| tmp0[[i, j, k]]).collect();
                let out = self.execute_inverse(plan, &fiber)?;
                for j in 0..n {
                    tmp1[[i, j, k]] = out[j];
                }
            }
        }
        // Axis-0 pass (inverse).
        let mut result = Array3::<f32>::zeros((n, n, n));
        for j in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|i| tmp1[[i, j, k]]).collect();
                let out = self.execute_inverse(plan, &fiber)?;
                for i in 0..n {
                    result[[i, j, k]] = out[i];
                }
            }
        }
        Ok(result)
    }
}
