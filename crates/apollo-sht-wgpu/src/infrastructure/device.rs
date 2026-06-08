//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_sht::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
use apollo_sht::{ShtComplexStorage, SphericalGridSpec, SphericalHarmonicCoefficients};
use ndarray::Array2;
use num_complex::{Complex32, Complex64};

use crate::application::plan::ShtWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{GridPod, ShtGpuKernel};
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    ShtWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct ShtWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<ShtGpuKernel>,
}

impl ShtWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(ShtGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-sht-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_complex(true)
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

    /// Create a WGPU plan descriptor.
    #[must_use]
    pub const fn plan(
        &self,
        latitudes: usize,
        longitudes: usize,
        max_degree: usize,
    ) -> ShtWgpuPlan {
        ShtWgpuPlan::new(latitudes, longitudes, max_degree)
    }

    /// Execute forward complex SHT by direct quadrature matrix summation.
    pub fn execute_forward(
        &self,
        plan: &ShtWgpuPlan,
        samples: &Array2<Complex32>,
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        validate_plan(plan)?;
        if samples.dim() != (plan.latitudes(), plan.longitudes()) {
            let (actual_latitudes, actual_longitudes) = samples.dim();
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "samples expected {}x{}, got {}x{}",
                    plan.latitudes(),
                    plan.longitudes(),
                    actual_latitudes,
                    actual_longitudes
                ),
            });
        }
        let grid = grid_samples(plan);
        let input: Vec<Complex32> = samples.iter().copied().collect();
        let raw = self.kernel.execute_forward(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan.mode_count(),
            plan.sample_count(),
            &input,
            &grid,
        )?;
        Ok(coefficients_from_modes(plan.max_degree(), &raw))
    }

    /// Execute inverse complex SHT by direct synthesis matrix summation.
    pub fn execute_inverse(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<Array2<Complex64>> {
        validate_plan(plan)?;
        if coefficients.max_degree() != plan.max_degree() {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "coefficient shape mismatch: expected max_degree {}, got {}",
                    plan.max_degree(),
                    coefficients.max_degree()
                ),
            });
        }
        let grid = grid_samples(plan);
        let input = modes_from_coefficients(coefficients);
        let raw = self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan.sample_count(),
            plan.mode_count(),
            &input,
            &grid,
        )?;
        Array2::from_shape_vec(
            (plan.latitudes(), plan.longitudes()),
            raw.into_iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        )
        .map_err(|_| WgpuError::InvalidPlan {
                message: format!("invalid plan latitudes={}, longitudes={}, max_degree={}: inverse output shape does not match plan", plan.latitudes(), plan.longitudes(), plan.max_degree()),
            })
    }

    /// Execute forward complex SHT from a flat typed sample slice.
    ///
    /// `flat_samples` must have exactly `plan.latitudes() * plan.longitudes()` elements
    /// in row-major (latitude × longitude) order.
    /// Promotes represented input once to `Complex32` and returns `SphericalHarmonicCoefficients`.
    pub fn execute_forward_flat_typed<T: ShtComplexStorage>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: &[T],
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_len = plan.latitudes() * plan.longitudes();
        if flat_samples.len() != expected_len {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "sample shape mismatch: expected ({}, {}), got ({}, 1)",
                    plan.latitudes(),
                    plan.longitudes(),
                    flat_samples.len()
                ),
            });
        }
        let promoted: Vec<Complex32> = flat_samples
            .iter()
            .map(|v| {
                let c = v.to_complex64();
                Complex32::new(c.re as f32, c.im as f32)
            })
            .collect();
        let samples_2d = Array2::from_shape_vec((plan.latitudes(), plan.longitudes()), promoted)
            .map_err(|_| WgpuError::InvalidPlan {
                    message: format!("invalid plan latitudes={}, longitudes={}, max_degree={}: flat sample reshape failed", plan.latitudes(), plan.longitudes(), plan.max_degree()),
                })?;
        self.execute_forward(plan, &samples_2d)
    }

    /// Execute inverse complex SHT and write the flat output to a typed slice.
    ///
    /// The output slice must have exactly `plan.latitudes() * plan.longitudes()` elements.
    pub fn execute_inverse_flat_typed_into<T: ShtComplexStorage>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: &SphericalHarmonicCoefficients,
        output: &mut [T],
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_len = plan.latitudes() * plan.longitudes();
        if output.len() != expected_len {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "sample shape mismatch: expected ({}, {}), got ({}, 1)",
                    plan.latitudes(),
                    plan.longitudes(),
                    output.len()
                ),
            });
        }
        let result = self.execute_inverse(plan, coefficients)?;
        for (slot, value) in output.iter_mut().zip(result.iter()) {
            *slot = T::from_complex64(*value);
        }
        Ok(())
    }
}

fn validate_plan(plan: &ShtWgpuPlan) -> WgpuResult<()> {
    if SphericalGridSpec::new(plan.latitudes(), plan.longitudes(), plan.max_degree()).is_err() {
        return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan latitudes={}, longitudes={}, max_degree={}: grid must be non-empty and satisfy max_degree < latitudes and 2*max_degree+1 <= longitudes", plan.latitudes(), plan.longitudes(), plan.max_degree()),
            });
    }
    if plan.sample_count() > u32::MAX as usize || plan.mode_count() > u32::MAX as usize {
        return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan latitudes={}, longitudes={}, max_degree={}: WGPU dispatch dimensions must fit in u32", plan.latitudes(), plan.longitudes(), plan.max_degree()),
            });
    }
    Ok(())
}

fn mode_pairs(max_degree: usize) -> impl Iterator<Item = (usize, isize)> {
    (0..=max_degree).flat_map(|degree| {
        (-(degree as isize)..=(degree as isize)).map(move |order| (degree, order))
    })
}

fn grid_samples(plan: &ShtWgpuPlan) -> Vec<GridPod> {
    let (cos_theta_nodes, theta_weights) = gauss_legendre_nodes_weights(plan.latitudes());
    let longitude_weight = std::f64::consts::TAU / plan.longitudes() as f64;
    (0..plan.latitudes())
        .flat_map(|lat| {
            let cos_theta = cos_theta_nodes[lat];
            let weight = theta_weights[lat] * longitude_weight;
            (0..plan.longitudes()).map(move |lon| GridPod {
                cos_theta: cos_theta as f32,
                phi: (std::f64::consts::TAU * lon as f64 / plan.longitudes() as f64) as f32,
                weight: weight as f32,
                _padding: 0.0,
            })
        })
        .collect()
}

fn coefficients_from_modes(max_degree: usize, raw: &[Complex32]) -> SphericalHarmonicCoefficients {
    let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
    for ((degree, order), value) in mode_pairs(max_degree).zip(raw.iter().copied()) {
        coefficients.set(
            degree,
            order,
            Complex64::new(value.re as f64, value.im as f64),
        );
    }
    coefficients
}

fn modes_from_coefficients(coefficients: &SphericalHarmonicCoefficients) -> Vec<Complex32> {
    mode_pairs(coefficients.max_degree())
        .map(|(degree, order)| {
            let value = coefficients.get(degree, order);
            Complex32::new(value.re as f32, value.im as f32)
        })
        .collect()
}
