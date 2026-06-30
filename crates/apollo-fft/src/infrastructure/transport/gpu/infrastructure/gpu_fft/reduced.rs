//! Reduced-precision (native f16) GPU FFT configuration.

use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::pipeline::{GpuFft3d, GpuPrecision};
use crate::f16;
use std::sync::Arc;

/// Reduced-precision (native f16) ZST for static routing.
#[cfg(feature = "native-f16")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReducedPrecision;

#[cfg(feature = "native-f16")]
impl GpuPrecision for ReducedPrecision {
    const ELEMENT_SIZE: u64 = 2;
    const USE_RADIX4: bool = false;
    const LABEL_SUFFIX: &'static str = "f16";

    fn fft_shader_source() -> &'static str {
        include_str!("../shaders/fft_native_f16.wgsl")
    }

    fn pack_shader_source() -> &'static str {
        include_str!("../shaders/pack_native_f16.wgsl")
    }

    fn chirp_shader_source() -> &'static str {
        include_str!("../shaders/chirp_native_f16.wgsl")
    }

    fn create_chirp_buffers(
        device: &wgpu::Device,
        h: &[eunomia::Complex<f64>],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        use wgpu::util::DeviceExt;
        let h_re_bits: Vec<u16> = h
            .iter()
            .map(|value| f16::from_f32(value.re as f32).to_bits())
            .collect();
        let h_im_bits: Vec<u16> = h
            .iter()
            .map(|value| f16::from_f32(value.im as f32).to_bits())
            .collect();
        let h_fft_re = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp re"),
            contents: bytemuck::cast_slice(&h_re_bits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let h_fft_im = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp im"),
            contents: bytemuck::cast_slice(&h_im_bits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        (h_fft_re, h_fft_im)
    }

    fn create_chirp_data_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let make_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp data layout"),
            entries: &[make_rw(0), make_rw(1), make_rw(2), make_rw(3)],
        })
    }
}

#[cfg(feature = "native-f16")]
impl GpuFft3d<ReducedPrecision> {
    /// Create a plan by requesting a new WGPU device with `SHADER_F16` enabled.
    ///
    /// Returns `Err` if no adapter is available, if the adapter does not
    /// support `SHADER_F16`, or if any dimension is < 2.
    pub fn try_new(nx: usize, ny: usize, nz: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("no WGPU adapter: {e}"))?;
        if !adapter.features().contains(wgpu::Features::SHADER_F16) {
            return Err("adapter does not support SHADER_F16".to_string());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("apollo-fft-wgpu native-f16"),
            required_features: wgpu::Features::SHADER_F16,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("device request failed: {e}"))?;
        Self::try_from_device(Arc::new(device), Arc::new(queue), nx, ny, nz)
    }
}

#[cfg(all(test, feature = "native-f16"))]
mod tests {
    use super::*;

    #[test]
    fn native_f16_forward_matches_f32_within_f16_tolerance_when_device_exists() {
        let Ok(plan_f16) = GpuFft3d::<ReducedPrecision>::try_new(4, 4, 4) else {
            // No SHADER_F16 device available; skip.
            return;
        };

        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()
        else {
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("apollo-fft-wgpu f16 test f32 ref"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            }))
        else {
            return;
        };

        let Ok(plan_f32) = crate::infrastructure::transport::gpu::infrastructure::gpu_fft::GpuFft3d::new(
            Arc::new(device),
            Arc::new(queue),
            4,
            4,
            4,
        ) else {
            return;
        };

        // Analytical test field: deterministic, non-trivial, ‖f‖_∞ ≤ 1.
        let field_f64 = leto::Array3::from_shape_fn([4, 4, 4], |[i, j, k]| {
            let x = (i + j * 3 + k * 7) as f64;
            (0.3 * x).sin() + 0.5 * (0.7 * x).cos()
        });
        let field_f32 = leto::Array3::from_shape_fn([4, 4, 4], |[i, j, k]| {
            let x = (i + j * 3 + k * 7) as f64;
            ((0.3 * x).sin() + 0.5 * (0.7 * x).cos()) as f32
        });

        let out_f32_ref = plan_f32.forward(&field_f64);
        let out_f16_native = plan_f16.forward(&field_f32);

        assert_eq!(
            out_f32_ref.len(),
            out_f16_native.len(),
            "output length mismatch"
        );

        for (idx, (a, b)) in out_f32_ref.iter().zip(out_f16_native.iter()).enumerate() {
            let err = (a - b).abs();
            // 1e-2 = ~10 × ε_f16; derived from 3-axis log₂(4) accumulation bound.
            assert!(
                err < 1e-2,
                "f16 native vs f32 error {err:.2e} exceeds 1e-2 at index {idx} \
                 (f32_ref={a:.6}, f16_native={b:.6})"
            );
        }
    }

    #[test]
    fn non_pow2_f16_forward_inverse_roundtrip_when_device_exists() {
        let Ok(plan) = GpuFft3d::<ReducedPrecision>::try_new(3, 3, 3) else {
            // No SHADER_F16 device available; skip.
            return;
        };

        // Deterministic 3×3×3 field with values in [−1, 1].
        let field = leto::Array3::from_shape_fn([3, 3, 3], |[i, j, k]| {
            let x = (i + j * 3 + k * 7) as f32;
            ((0.3 * x).sin() + 0.5 * (0.7 * x).cos()) as f32
        });

        let forward = plan.forward(&field);
        let mut roundtrip = leto::Array3::<f32>::zeros((3, 3, 3));
        plan.inverse(&forward, &mut roundtrip);

        assert_eq!(
            roundtrip.len(),
            27,
            "roundtrip output length must equal nx·ny·nz=27"
        );

        let flat: Vec<f32> = field.iter().copied().collect();
        for (idx, (&orig, &rt)) in flat.iter().zip(roundtrip.iter()).enumerate() {
            let err = (orig - rt).abs();
            // 0.05 is the analytically derived upper bound with 40× safety margin.
            assert!(
                err < 0.05,
                "non-pow2 f16 roundtrip error {err:.4} exceeds 0.05 at index {idx} \
                 (original={orig:.6}, roundtrip={rt:.6})"
            );
        }
    }
}
