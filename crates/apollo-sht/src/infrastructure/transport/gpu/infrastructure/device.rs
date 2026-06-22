//! WGPU device acquisition for this transform backend.

use apollo_fft::application::utilities::leto_interop;
use std::{borrow::Cow, sync::Arc};

use apollo_fft::PrecisionProfile;
use crate::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
use crate::{ShtComplexStorage, SphericalGridSpec, SphericalHarmonicCoefficients};
use ndarray::Array2;
use num_complex::{Complex32, Complex64};

use crate::infrastructure::transport::gpu::application::plan::ShtWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{GridPod, ShtGpuKernel};
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
            &self.device,
            plan.mode_count(),
            plan.sample_count(),
            &input,
            &grid,
        )?;
        Ok(coefficients_from_modes(plan.max_degree(), &raw))
    }

    /// Execute forward complex SHT from a Leto sample grid.
    ///
    /// The returned dense coefficient matrix has shape
    /// `(max_degree + 1, 2 * max_degree + 1)` and uses Mnemosyne-backed Leto
    /// storage. Strided sample views are materialized once into logical order
    /// before the existing WGPU ndarray execution path.
    pub fn execute_forward_leto(
        &self,
        plan: &ShtWgpuPlan,
        samples: leto::ArrayView2<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let samples = array2_from_leto_view(samples);
        let coefficients = self.execute_forward(plan, &samples)?;
        leto_array2_from_ndarray(coefficients.values(), "SHT coefficients")
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
            &self.device,
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

    /// Execute inverse complex SHT from a dense Leto coefficient matrix.
    pub fn execute_inverse_leto(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let coefficients = coefficients_from_leto_view(plan, coefficients)?;
        let samples = self.execute_inverse(plan, &coefficients)?;
        leto_array2_from_ndarray(&samples, "SHT inverse samples")
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

    /// Execute forward complex SHT from a flat typed Leto sample view.
    pub fn execute_forward_flat_leto_typed<T: ShtComplexStorage>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let flat_samples = leto_view1_cow(flat_samples);
        let coefficients = self.execute_forward_flat_typed(plan, precision, &flat_samples)?;
        leto_array2_from_ndarray(coefficients.values(), "SHT typed coefficients")
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

    /// Execute inverse complex SHT from a Leto coefficient matrix and return a
    /// flat typed Leto sample buffer in row-major latitude × longitude order.
    pub fn execute_inverse_flat_leto_typed<T: ShtComplexStorage>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let coefficients = coefficients_from_leto_view(plan, coefficients)?;
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.sample_count()];
        self.execute_inverse_flat_typed_into(plan, precision, &coefficients, &mut output)?;
        leto_array1_from_slice(&output, "SHT typed inverse samples")
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

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    leto_interop::view1_cow(&view)
}
fn array2_from_leto_view<T: Copy>(view: leto::ArrayView2<'_, T>) -> Array2<T> {
    leto_interop::array2_from_view(&view)
}
fn coefficients_from_leto_view(
    plan: &ShtWgpuPlan,
    coefficients: leto::ArrayView2<'_, Complex64>,
) -> WgpuResult<SphericalHarmonicCoefficients> {
    let values = array2_from_leto_view(coefficients);
    let expected = (plan.max_degree() + 1, 2 * plan.max_degree() + 1);
    if values.dim() != expected {
        let (rows, cols) = values.dim();
        return Err(WgpuError::ShapeMismatch {
            message: format!(
                "coefficient shape mismatch: expected {}x{}, got {}x{}",
                expected.0, expected.1, rows, cols
            ),
        });
    }
    Ok(SphericalHarmonicCoefficients::from_values(
        plan.max_degree(),
        values,
    ))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
    label: &str,
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto_interop::try_array1_from_slice(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto {label}"),
    })
}

fn leto_array2_from_ndarray<T: Copy>(
    values: &Array2<T>,
    label: &str,
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
    leto_interop::try_array2_from_ndarray(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto {label}"),
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
        let cow = leto_view1_cow(input.view());
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
        let cow = leto_view1_cow(view);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }
}
