//! Native half-precision FFT plan over typed Hephaestus transport.
//!
//! # Architecture
//!
//! The plan stores IEEE-754 half bit patterns in `WgpuBuffer<u16>` while the
//! selected WGSL sources declare `array<f16>`. The storage-generic dense FFT
//! core owns typed allocation, binding validation, command ordering,
//! submission, and readback. Apollo retains only host conversion and the FFT
//! equations; it does not construct WGPU buffers, pipelines, bind groups, or
//! command encoders.
//!
//! # Mathematical contract
//!
//! For dimensions `N_x`, `N_y`, and `N_z`, forward execution computes
//!
//! `F[k_x,k_y,k_z] = Σ f[x,y,z] exp(-2πi(k_xx/N_x + k_yy/N_y + k_zz/N_z))`.
//!
//! The inverse uses the positive exponent and applies `1/(N_x N_y N_z)`. Root
//! of unity orthogonality therefore gives `F⁻¹(F(f)) = f` in exact arithmetic.
//! Native half arithmetic gives finite-precision evidence through the derived
//! differential and Bluestein roundtrip tests below; this is not a
//! machine-checked proof.

use crate::f16 as HalfF16;
use hephaestus_core::{DeviceFeature, DevicePreference};
use hephaestus_wgpu::WgpuDevice;
use leto::Array3;

use super::pipeline::GpuFft3d;

/// Validate the native-half plan's documented minimum axis length.
///
/// Radix axes and Bluestein axes are both supported; this public plan requires
/// every axis to contain at least two samples.
fn validate_dimensions(nx: usize, ny: usize, nz: usize) -> Result<(), String> {
    for (name, value) in [("nx", nx), ("ny", ny), ("nz", nz)] {
        if value < 2 {
            return Err(format!(
                "{name}={value} is invalid; native-half axes require N >= 2"
            ));
        }
    }
    Ok(())
}

/// GPU-backed 3D FFT plan executing all shader arithmetic in native f16.
///
/// The plan requires a Hephaestus device with [`DeviceFeature::ShaderF16`].
/// Host data crosses the boundary as IEEE-754 half bit patterns and is decoded
/// back to f32 only after provider-owned readback completes.
pub struct GpuFft3dF16Native {
    /// X dimension.
    nx: usize,
    /// Y dimension.
    ny: usize,
    /// Z dimension.
    nz: usize,
    /// Storage-generic provider plan for native half bit patterns.
    plan: GpuFft3d<u16>,
}

impl GpuFft3dF16Native {
    /// Return true when the device was acquired with `ShaderF16` enabled.
    #[must_use]
    pub fn device_supports_f16(device: &WgpuDevice) -> bool {
        device.supports_device_feature(DeviceFeature::ShaderF16)
    }

    /// Create a plan by requesting a Hephaestus device with `ShaderF16`.
    ///
    /// # Errors
    ///
    /// Returns an error if no available adapter can satisfy `ShaderF16`, or if
    /// any axis has fewer than two samples.
    pub fn try_new(nx: usize, ny: usize, nz: usize) -> Result<Self, String> {
        validate_dimensions(nx, ny, nz)?;
        let device = WgpuDevice::try_with_device_preference_and_required_device_features(
            "apollo-fft-native-f16",
            DevicePreference::HighPerformance,
            &[DeviceFeature::ShaderF16],
        )
        .map_err(|error| error.to_string())?;
        Self::try_from_device(device, nx, ny, nz)
    }

    /// Create a plan from a caller-supplied Hephaestus device.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` lacks `ShaderF16` or if any axis has fewer
    /// than two samples.
    pub fn try_from_device(
        device: WgpuDevice,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self, String> {
        validate_dimensions(nx, ny, nz)?;
        if !Self::device_supports_f16(&device) {
            return Err(
                "device does not have ShaderF16 enabled; acquire it with DeviceFeature::ShaderF16"
                    .to_owned(),
            );
        }
        let plan = GpuFft3d::<u16>::new_typed(device, nx, ny, nz)?;
        Ok(Self { nx, ny, nz, plan })
    }

    /// Forward 3D FFT of a real f32 field.
    ///
    /// Returns interleaved complex f32 values ordered as
    /// `[re₀, im₀, re₁, im₁, …]`; shader arithmetic remains f16.
    ///
    /// # Errors
    ///
    /// Returns an error when `field` does not match this plan's dimensions or
    /// when the provider rejects transfer or dispatch.
    pub fn forward_native_f16(&self, field: &Array3<f32>) -> Result<Vec<f32>, String> {
        self.validate_field_shape(field.shape())?;
        let mut real = field
            .iter()
            .copied()
            .map(|value| HalfF16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let mut imaginary = vec![0_u16; self.plan.element_count()];
        self.plan
            .execute_forward_in_place(&mut real, &mut imaginary)
            .map_err(|error| error.to_string())?;
        Ok(interleave_components(&real, &imaginary))
    }

    /// Inverse 3D FFT from an interleaved complex f32 spectrum.
    ///
    /// # Errors
    ///
    /// Returns an error when the spectrum length does not equal twice the plan
    /// element count or when the provider rejects transfer or dispatch.
    pub fn inverse_native_f16(&self, spectrum: &[f32]) -> Result<Vec<f32>, String> {
        let expected = self
            .plan
            .element_count()
            .checked_mul(2)
            .expect("invariant: validated plan element count fits interleaved length");
        if spectrum.len() != expected {
            return Err(format!(
                "native-half spectrum length {} does not match expected {expected}",
                spectrum.len()
            ));
        }
        let mut real = Vec::with_capacity(self.plan.element_count());
        let mut imaginary = Vec::with_capacity(self.plan.element_count());
        for pair in spectrum.chunks_exact(2) {
            real.push(HalfF16::from_f32(pair[0]).to_bits());
            imaginary.push(HalfF16::from_f32(pair[1]).to_bits());
        }
        self.plan
            .execute_inverse_in_place(&mut real, &mut imaginary)
            .map_err(|error| error.to_string())?;
        Ok(real
            .into_iter()
            .map(|value| HalfF16::from_bits(value).to_f32())
            .collect())
    }

    fn validate_field_shape(&self, actual: [usize; 3]) -> Result<(), String> {
        let expected = [self.nx, self.ny, self.nz];
        if actual == expected {
            Ok(())
        } else {
            Err(format!(
                "native-half field shape {actual:?} does not match plan shape {expected:?}"
            ))
        }
    }
}

fn interleave_components(real: &[u16], imaginary: &[u16]) -> Vec<f32> {
    real.iter()
        .zip(imaginary)
        .flat_map(|(&real, &imaginary)| {
            [
                HalfF16::from_bits(real).to_f32(),
                HalfF16::from_bits(imaginary).to_f32(),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify native-half forward FFT against the typed f32 provider plan.
    ///
    /// A radix-2 path has two multiplications, one product addition, one
    /// butterfly addition, and one twiddle quantization per component per
    /// stage. Including input quantization, a 4×4×4 transform has
    /// `k = 1 + 5·(log₂4 + log₂4 + log₂4) = 31` rounding sites. With half unit
    /// roundoff `u = 2⁻¹¹`, `γₖ = ku/(1-ku)` bounds relative accumulation
    /// error; each DFT component is a weighted input sum, giving the asserted
    /// absolute bound `γ₃₁·‖input‖₁`.
    #[test]
    fn native_f16_forward_matches_f32_within_f16_tolerance_when_device_exists() {
        let Ok(plan_f16) = GpuFft3dF16Native::try_new(4, 4, 4) else {
            return;
        };
        let plan_f32 = GpuFft3d::new(plan_f16.plan.device.clone(), 4, 4, 4)
            .expect("f32 provider plan must share the acquired native-half device");

        let field_f64 = leto::Array3::from_shape_fn([4, 4, 4], |[i, j, k]| {
            let value = (i + j * 3 + k * 7) as f64;
            (0.3 * value).sin() + 0.5 * (0.7 * value).cos()
        });
        let field_f32 = leto::Array3::from_shape_fn([4, 4, 4], |[i, j, k]| {
            let value = (i + j * 3 + k * 7) as f64;
            ((0.3 * value).sin() + 0.5 * (0.7 * value).cos()) as f32
        });
        let input_l1 = field_f32.iter().map(|value| value.abs()).sum::<f32>();
        let unit_roundoff = HalfF16::EPSILON.to_f32() / 2.0;
        let rounding_sites = 31.0_f32;
        let gamma = rounding_sites * unit_roundoff / (1.0 - rounding_sites * unit_roundoff);
        let error_bound = gamma * input_l1;

        let f32_reference = plan_f32
            .forward(&field_f64)
            .expect("typed f32 provider readback must succeed");
        let native = plan_f16
            .forward_native_f16(&field_f32)
            .expect("typed native-half provider readback must succeed");

        assert_eq!(f32_reference.len(), native.len(), "output length mismatch");
        for (index, (reference, actual)) in f32_reference.iter().zip(native.iter()).enumerate() {
            let error = (reference - actual).abs();
            assert!(
                error <= error_bound,
                "native-half error {error:.2e} exceeds derived bound {error_bound:.2e} \
                 at index {index} (f32_ref={reference:.6}, native={actual:.6})"
            );
        }
    }

    /// Verify forward→inverse reconstruction for an all-Bluestein 3×3×3 plan.
    ///
    /// Each length-three Bluestein axis uses an eight-point radix-two
    /// convolution. A forward axis has 43 half-rounding sites: four in the
    /// premultiply, fifteen in the forward radix stages, four in pointwise
    /// multiplication, fifteen in the inverse radix stages, one in the
    /// power-of-two normalization, and four in the postmultiply. The inverse
    /// adds two sites for its `1/N` scale. Three forward and three inverse
    /// axes, plus input quantization, give `k = 1 + 3·(43 + 45) = 265`.
    /// With half unit roundoff `u = 2⁻¹¹`, the asserted bound is
    /// `γₖ·‖input‖₁`, where `γₖ = ku/(1-ku)`.
    #[test]
    fn non_pow2_f16_forward_inverse_roundtrip_when_device_exists() {
        let Ok(plan) = GpuFft3dF16Native::try_new(3, 3, 3) else {
            return;
        };
        let field = leto::Array3::from_shape_fn([3, 3, 3], |[i, j, k]| {
            let value = (i * 9 + j * 3 + k) as f32;
            (0.21 * value).sin() + 0.2 * (0.37 * value).cos()
        });
        let spectrum = plan
            .forward_native_f16(&field)
            .expect("native-half Bluestein forward transform must succeed");
        let reconstructed = plan
            .inverse_native_f16(&spectrum)
            .expect("native-half Bluestein inverse transform must succeed");

        let input_l1 = field.iter().map(|value| value.abs()).sum::<f32>();
        let unit_roundoff = HalfF16::EPSILON.to_f32() / 2.0;
        let rounding_sites = 265.0_f32;
        let gamma = rounding_sites * unit_roundoff / (1.0 - rounding_sites * unit_roundoff);
        let error_bound = gamma * input_l1;
        let max_error = field
            .iter()
            .zip(reconstructed)
            .map(|(&expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= error_bound,
            "native-half Bluestein reconstruction error {max_error:.3e} exceeds derived \
             bound {error_bound:.3e}"
        );
    }

    #[test]
    fn native_f16_rejects_singleton_axis() {
        let Err(error) = GpuFft3dF16Native::try_new(1, 2, 2) else {
            panic!("singleton native-half axis must be rejected before device acquisition")
        };
        assert_eq!(error, "nx=1 is invalid; native-half axes require N >= 2");
    }

    #[test]
    fn interleaving_preserves_half_bit_patterns() {
        let real = [
            HalfF16::from_f32(0.5).to_bits(),
            HalfF16::from_f32(-1.25).to_bits(),
        ];
        let imaginary = [
            HalfF16::from_f32(-0.25).to_bits(),
            HalfF16::from_f32(2.0).to_bits(),
        ];
        assert_eq!(
            interleave_components(&real, &imaginary),
            [0.5, -0.25, -1.25, 2.0]
        );
    }
}
