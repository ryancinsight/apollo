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
/// `Clone` is cheap (three `Arc` clones inside inner_device); `Debug` shows the label.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    inner_device: hephaestus_wgpu::WgpuDevice,
}

impl WgpuDevice {
    /// Wrap an existing device and queue.
    #[must_use]
    #[inline]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            inner_device: hephaestus_wgpu::WgpuDevice::new(device, queue),
        }
    }

    /// Wrap an existing hephaestus-wgpu WgpuDevice.
    #[must_use]
    #[inline]
    pub fn from_hephaestus(inner: hephaestus_wgpu::WgpuDevice) -> Self {
        Self {
            inner_device: inner,
        }
    }

    /// Access the underlying hephaestus-wgpu WgpuDevice.
    #[must_use]
    #[inline]
    pub fn hephaestus(&self) -> &hephaestus_wgpu::WgpuDevice {
        &self.inner_device
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
        Self::try_default_with_features_and_limits(label, wgpu::Features::empty(), required_limits)
    }

    /// Acquire a default adapter and device with custom features and limits.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuDeviceError::AdapterUnavailable`] or
    /// [`WgpuDeviceError::DeviceUnavailable`] on failure.
    pub fn try_default_with_features_and_limits(
        label: &str,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> WgpuDeviceResult<Self> {
        let acquired = hephaestus_wgpu::WgpuDevice::try_default_with_features_and_limits(
            label,
            required_features,
            required_limits,
        )
        .map_err(|e| match e {
            hephaestus_wgpu::HephaestusError::AdapterUnavailable { message } => {
                WgpuDeviceError::AdapterUnavailable { message }
            }
            other => WgpuDeviceError::DeviceUnavailable {
                message: other.to_string(),
            },
        })?;
        Ok(Self::from_hephaestus(acquired))
    }

    /// Return a reference to the inner WGPU device, for kernel construction.
    /// Kernel constructors typically take `&wgpu::Device`; use this method
    /// when you need the raw reference rather than the `Arc` returned by
    /// [`device()`](Self::device).
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &wgpu::Device {
        self.inner_device.inner()
    }

    /// Return a reference to the WGPU device `Arc`.
    #[must_use]
    #[inline]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        self.inner_device.device()
    }

    /// Return a reference to the WGPU queue `Arc`.
    #[must_use]
    #[inline]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        self.inner_device.queue()
    }

    /// Retrieve a staging buffer of size >= size from the pool, or create a new one.
    /// The size is automatically aligned to `wgpu::MAP_ALIGNMENT` (8 bytes).
    #[must_use]
    #[inline]
    pub fn get_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.inner_device
            .get_staging_buffer(size)
            .expect("Failed to allocate or acquire staging buffer from Hephaestus device")
    }

    /// Return a staging buffer back to the pool for reuse.
    #[inline]
    pub fn recycle_staging_buffer(&self, buffer: wgpu::Buffer) {
        self.inner_device.recycle_staging_buffer(buffer);
    }
}
