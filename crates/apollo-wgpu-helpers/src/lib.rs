//! Shared WGPU device acquisition and error contracts for the Apollo WGPU
//! crate family.
//!
//! ## Motivation
//!
//! 16 WGPU backend crates (`apollo-*-wgpu`) each duplicated the same ~40-line
//! `try_default()` adapter/device acquisition boilerplate, the same
//! `device()`/`queue()` getters, the same `AdapterUnavailable` /
//! `DeviceUnavailable` error variants, and a per-crate `WgpuError` enum with
//! largely identical variants.  This crate factors that common surface into
//! a single [`WgpuDevice`] struct and a single [`WgpuError`] enum so crate
//! authors only carry the kernel pipeline and domain-specific execution logic.
//!
//! ## Usage
//!
//! ```ignore
//! use apollo_wgpu_helpers::WgpuDevice;
//!
//! pub struct MyWgpuBackend {
//!     device: WgpuDevice,
//!     kernel: Arc<MyGpuKernel>,
//! }
//!
//! impl MyWgpuBackend {
//!     pub fn try_default() -> Result<Self, MyError> {
//!         let device = WgpuDevice::try_default("apollo-mine-wgpu")?;
//!         Ok(Self { kernel: Arc::new(MyGpuKernel::new(device.inner())), device })
//!     }
//! }
//! ```

use std::sync::Arc;

pub use error::{WgpuDeviceError, WgpuDeviceResult, WgpuError, WgpuResult};

mod error;

// ── WgpuDevice ──────────────────────────────────────────────────────────────

/// An acquired WGPU device + queue pair.
///
/// Created via [`WgpuDevice::new`] (with caller-owned `Arc`s) or
/// [`WgpuDevice::try_default`] (auto-acquire).
///
/// `Clone` is cheap (two `Arc` clones); `Debug` shows the label.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WgpuDevice {
    /// Wrap an existing device and queue.
    #[must_use]
    #[inline]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    /// Acquire a default adapter and device.
    ///
    /// `label` is used as the WGPU device label (e.g. `"apollo-fft-wgpu"`).
    /// Uses [`wgpu::Limits::downlevel_defaults`]; for custom limits use
    /// [`try_default_with_limits`](Self::try_default_with_limits).
    ///
    /// # Errors
    ///
    /// Returns [`WgpuDeviceError::AdapterUnavailable`] or
    /// [`WgpuDeviceError::DeviceUnavailable`] on failure.
    #[inline]
    pub fn try_default(label: &str) -> WgpuDeviceResult<Self> {
        Self::try_default_with_limits(label, wgpu::Limits::downlevel_defaults())
    }

    /// Acquire a default adapter and device with custom limits.
    ///
    /// Use this when the kernel requires non-default buffer counts (e.g.
    /// `max_storage_buffers_per_shader_stage`).
    ///
    /// Acquisition is delegated to the shared Atlas GPU substrate
    /// (`hephaestus-wgpu`, atlas ADR 0001); this crate keeps Apollo's error
    /// contracts and `Arc`-pair surface stable for the `-wgpu` crate family.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuDeviceError::AdapterUnavailable`] or
    /// [`WgpuDeviceError::DeviceUnavailable`] on failure.
    pub fn try_default_with_limits(
        label: &str,
        required_limits: wgpu::Limits,
    ) -> WgpuDeviceResult<Self> {
        let acquired = hephaestus_wgpu::WgpuDevice::try_default_with_limits(label, required_limits)
            .map_err(|e| match e {
                hephaestus_wgpu::HephaestusError::AdapterUnavailable { message } => {
                    WgpuDeviceError::AdapterUnavailable { message }
                }
                other => WgpuDeviceError::DeviceUnavailable {
                    message: other.to_string(),
                },
            })?;
        Ok(Self::new(
            Arc::clone(acquired.device()),
            Arc::clone(acquired.queue()),
        ))
    }

    /// Return a reference to the inner WGPU device, for kernel construction.
    /// Kernel constructors typically take `&wgpu::Device`; use this method
    /// when you need the raw reference rather than the `Arc` returned by
    /// [`device()`](Self::device).
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &wgpu::Device {
        &self.device
    }

    /// Return a reference to the WGPU device `Arc`.
    #[must_use]
    #[inline]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Return a reference to the WGPU queue `Arc`.
    #[must_use]
    #[inline]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}
