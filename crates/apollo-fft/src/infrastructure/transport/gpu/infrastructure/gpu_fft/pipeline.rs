//! Typed Hephaestus planning and storage for dense three-dimensional FFTs.

use crate::{fft_1d_complex_inplace, Complex64};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};
use leto::Array1;

use super::{
    kernel::{ChirpParams, PackParams},
    strategy::{Axis, AxisStrategy, ChirpData, RadixStages},
};

/// Returns true when this crate is linked with its Hephaestus device backend.
#[must_use]
pub fn gpu_fft_available() -> bool {
    true
}

/// GPU-backed three-dimensional dense FFT plan.
///
/// The plan owns provider-native scratch and volume storage.  External typed
/// buffers can be transformed through the command-stream methods without
/// exposing WGPU handles or encoders.
pub struct GpuFft3d {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
    pub(crate) device: WgpuDevice,
    pub(crate) workspace_real: WgpuBuffer<f32>,
    pub(crate) workspace_imaginary: WgpuBuffer<f32>,
    pub(crate) volume_real: WgpuBuffer<f32>,
    pub(crate) volume_imaginary: WgpuBuffer<f32>,
    pub(crate) strategy_x: AxisStrategy,
    pub(crate) strategy_y: AxisStrategy,
    pub(crate) strategy_z: AxisStrategy,
    pub(crate) pack_x: PackParams,
    pub(crate) pack_y: PackParams,
    pub(crate) pack_z: PackParams,
    pub(crate) chirp_x: Option<ChirpData>,
    pub(crate) chirp_y: Option<ChirpData>,
    pub(crate) chirp_z: Option<ChirpData>,
    pub(crate) axis_forward_x: RadixStages,
    pub(crate) axis_inverse_x: RadixStages,
    pub(crate) axis_forward_y: RadixStages,
    pub(crate) axis_inverse_y: RadixStages,
    pub(crate) axis_forward_z: RadixStages,
    pub(crate) axis_inverse_z: RadixStages,
}

fn next_power_of_two(n: usize) -> Result<usize, String> {
    let mut power = 1usize;
    while power < n {
        power = power
            .checked_mul(2)
            .ok_or_else(|| format!("FFT length {n} cannot be rounded to a power of two"))?;
    }
    Ok(power)
}

#[inline]
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn axis_strategy_for(n: usize) -> Result<AxisStrategy, String> {
    if is_power_of_two(n) {
        return Ok(AxisStrategy::Radix2);
    }
    let convolution_len = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| format!("Bluestein convolution length overflows for axis length {n}"))?;
    Ok(AxisStrategy::ChirpZ {
        n,
        m: next_power_of_two(convolution_len)?,
    })
}

fn axis_workspace_elements(nx: usize, ny: usize, nz: usize, axis: Axis) -> Result<usize, String> {
    let axis_len = axis.len(nx, ny, nz);
    let transform_len = match axis_strategy_for(axis_len)? {
        AxisStrategy::Radix2 => axis_len,
        AxisStrategy::ChirpZ { m, .. } => m,
    };
    transform_len
        .checked_mul(axis.batch_count(nx, ny, nz))
        .ok_or_else(|| format!("FFT workspace element count overflows for {axis:?} axis"))
}

fn validate_dimensions(
    max_buffer_size: u64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<(), String> {
    for (name, value) in [("nx", nx), ("ny", ny), ("nz", nz)] {
        if value == 0 {
            return Err(format!("GpuFft3d: {name} must be greater than zero"));
        }
        u32::try_from(value)
            .map_err(|_| format!("GpuFft3d: {name}={value} exceeds the shader u32 domain"))?;
    }

    let required_elements = [
        axis_workspace_elements(nx, ny, nz, Axis::X)?,
        axis_workspace_elements(nx, ny, nz, Axis::Y)?,
        axis_workspace_elements(nx, ny, nz, Axis::Z)?,
        nx.checked_mul(ny)
            .and_then(|value| value.checked_mul(nz))
            .ok_or_else(|| "GpuFft3d: volume element count overflows".to_owned())?,
    ]
    .into_iter()
    .max()
    .expect("invariant: the fixed workspace requirement set is non-empty");
    let required_bytes = u64::try_from(required_elements)
        .map_err(|_| "GpuFft3d: workspace element count exceeds u64".to_owned())?
        .checked_mul(
            u64::try_from(core::mem::size_of::<f32>()).expect("invariant: f32 size fits u64"),
        )
        .ok_or_else(|| "GpuFft3d: workspace byte count overflows".to_owned())?;
    if required_bytes > max_buffer_size {
        return Err(format!(
            "GpuFft3d: workspace requires {required_bytes} bytes, exceeds device max_buffer_size={max_buffer_size}"
        ));
    }
    Ok(())
}

fn dimension(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("GpuFft3d: {name}={value} exceeds u32"))
}

fn provider_error(error: impl core::fmt::Display) -> String {
    error.to_string()
}

impl GpuFft3d {
    /// Create a plan over a Hephaestus WGPU device.
    pub fn new(device: WgpuDevice, nx: usize, ny: usize, nz: usize) -> Result<Self, String> {
        validate_dimensions(device.device_limits().max_buffer_size, nx, ny, nz)?;
        let strategy_x = axis_strategy_for(nx)?;
        let strategy_y = axis_strategy_for(ny)?;
        let strategy_z = axis_strategy_for(nz)?;
        let batch_x = dimension(
            ny.checked_mul(nz)
                .ok_or_else(|| "GpuFft3d: X-axis batch count overflows".to_owned())?,
            "x batch count",
        )?;
        let batch_y = dimension(
            nx.checked_mul(nz)
                .ok_or_else(|| "GpuFft3d: Y-axis batch count overflows".to_owned())?,
            "y batch count",
        )?;
        let batch_z = dimension(
            nx.checked_mul(ny)
                .ok_or_else(|| "GpuFft3d: Z-axis batch count overflows".to_owned())?,
            "z batch count",
        )?;
        let volume_elements = nx
            .checked_mul(ny)
            .and_then(|value| value.checked_mul(nz))
            .ok_or_else(|| "GpuFft3d: volume element count overflows".to_owned())?;
        let workspace_elements = [
            axis_workspace_elements(nx, ny, nz, Axis::X)?,
            axis_workspace_elements(nx, ny, nz, Axis::Y)?,
            axis_workspace_elements(nx, ny, nz, Axis::Z)?,
        ]
        .into_iter()
        .max()
        .expect("invariant: the fixed workspace requirement set is non-empty");
        let workspace_real = device
            .alloc_zeroed(workspace_elements)
            .map_err(provider_error)?;
        let workspace_imaginary = device
            .alloc_zeroed(workspace_elements)
            .map_err(provider_error)?;
        let volume_real = device
            .alloc_zeroed(volume_elements)
            .map_err(provider_error)?;
        let volume_imaginary = device
            .alloc_zeroed(volume_elements)
            .map_err(provider_error)?;
        let pack_x = Self::pack_params(Axis::X, nx, ny, nz, strategy_x, batch_x)?;
        let pack_y = Self::pack_params(Axis::Y, nx, ny, nz, strategy_y, batch_y)?;
        let pack_z = Self::pack_params(Axis::Z, nx, ny, nz, strategy_z, batch_z)?;

        Ok(Self {
            nx,
            ny,
            nz,
            workspace_real,
            workspace_imaginary,
            volume_real,
            volume_imaginary,
            chirp_x: Self::build_chirp_data(&device, strategy_x, batch_x)?,
            chirp_y: Self::build_chirp_data(&device, strategy_y, batch_y)?,
            chirp_z: Self::build_chirp_data(&device, strategy_z, batch_z)?,
            axis_forward_x: Self::radix_stages(nx, strategy_x, batch_x, false)?,
            axis_inverse_x: Self::radix_stages(nx, strategy_x, batch_x, true)?,
            axis_forward_y: Self::radix_stages(ny, strategy_y, batch_y, false)?,
            axis_inverse_y: Self::radix_stages(ny, strategy_y, batch_y, true)?,
            axis_forward_z: Self::radix_stages(nz, strategy_z, batch_z, false)?,
            axis_inverse_z: Self::radix_stages(nz, strategy_z, batch_z, true)?,
            strategy_x,
            strategy_y,
            strategy_z,
            pack_x,
            pack_y,
            pack_z,
            device,
        })
    }

    fn pack_params(
        axis: Axis,
        nx: usize,
        ny: usize,
        nz: usize,
        strategy: AxisStrategy,
        batch_count: u32,
    ) -> Result<PackParams, String> {
        let axis_len = dimension(axis.len(nx, ny, nz), "axis length")?;
        let fft_len = match strategy {
            AxisStrategy::Radix2 => axis_len,
            AxisStrategy::ChirpZ { m, .. } => dimension(m, "Bluestein workspace length")?,
        };
        Ok(PackParams {
            n: axis_len,
            stage: 0,
            inverse: 0,
            batch_count,
            nx: dimension(nx, "nx")?,
            ny: dimension(ny, "ny")?,
            nz: dimension(nz, "nz")?,
            axis: match axis {
                Axis::X => 0,
                Axis::Y => 1,
                Axis::Z => 2,
            },
            fft_len,
            padding: [0; 3],
        })
    }

    fn radix_stages(
        axis_len: usize,
        strategy: AxisStrategy,
        batch_count: u32,
        inverse: bool,
    ) -> Result<RadixStages, String> {
        if !matches!(strategy, AxisStrategy::Radix2) {
            return Ok(RadixStages::empty());
        }
        let fft_len = dimension(axis_len, "radix axis length")?;
        if fft_len.trailing_zeros() % 2 == 0 {
            Ok(RadixStages::radix_four(fft_len, batch_count, inverse))
        } else {
            Ok(RadixStages::radix_two(fft_len, batch_count, inverse))
        }
    }

    fn build_chirp_data(
        device: &WgpuDevice,
        strategy: AxisStrategy,
        batch_count: u32,
    ) -> Result<Option<ChirpData>, String> {
        let AxisStrategy::ChirpZ { n, m } = strategy else {
            return Ok(None);
        };
        let mut chirp = Array1::<Complex64>::zeros([m]);
        for index in 0..n {
            let angle = core::f64::consts::PI * (index * index) as f64 / n as f64;
            let value = Complex64::new(angle.cos(), angle.sin());
            chirp[[index]] = value;
            if index > 0 {
                chirp[[m - index]] = value;
            }
        }
        fft_1d_complex_inplace(&mut chirp);
        // The WGSL kernel contract is native f32 storage; this is the explicit
        // host-to-device precision boundary for precomputed chirp coefficients.
        let real: Vec<f32> = chirp.iter().map(|value| value.re as f32).collect();
        let imaginary: Vec<f32> = chirp.iter().map(|value| value.im as f32).collect();
        let n = dimension(n, "Bluestein axis length")?;
        let m = dimension(m, "Bluestein workspace length")?;
        Ok(Some(ChirpData {
            real_kernel: device.upload(&real).map_err(provider_error)?,
            imaginary_kernel: device.upload(&imaginary).map_err(provider_error)?,
            params: ChirpParams {
                n,
                m,
                batch_count,
                padding: 0,
            },
            forward_radix: RadixStages::radix_two(m, batch_count, false),
            inverse_radix: RadixStages::radix_two(m, batch_count, true),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        axis_strategy_for, axis_workspace_elements, next_power_of_two, validate_dimensions,
    };
    use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::strategy::{
        Axis, AxisStrategy,
    };

    #[test]
    fn next_power_of_two_rounds_up_non_power_of_two_lengths() {
        assert_eq!(next_power_of_two(1), Ok(1));
        assert_eq!(next_power_of_two(12), Ok(16));
        assert_eq!(next_power_of_two(127), Ok(128));
    }

    #[test]
    fn axis_strategy_uses_bluestein_only_for_non_power_of_two_lengths() {
        assert_eq!(axis_strategy_for(64), Ok(AxisStrategy::Radix2));
        assert_eq!(
            axis_strategy_for(12),
            Ok(AxisStrategy::ChirpZ { n: 12, m: 32 })
        );
    }

    #[test]
    fn axis_workspace_matches_axis_batch_geometry() {
        assert_eq!(axis_workspace_elements(2, 3, 4, Axis::X), Ok(32));
        assert_eq!(axis_workspace_elements(2, 3, 4, Axis::Y), Ok(32));
        assert_eq!(axis_workspace_elements(2, 3, 4, Axis::Z), Ok(24));
    }

    #[test]
    fn validate_dimensions_rejects_zero_and_oversized_shapes() {
        assert!(validate_dimensions(1024, 0, 2, 2).is_err());
        assert!(validate_dimensions(4, 2, 2, 2).is_err());
    }

    #[test]
    fn validate_dimensions_accepts_small_valid_shapes() {
        assert_eq!(validate_dimensions(1024, 2, 2, 2), Ok(()));
    }
}
