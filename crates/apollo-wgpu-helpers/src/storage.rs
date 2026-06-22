use std::sync::{Arc, OnceLock};
use hephaestus_wgpu::{ComputeDevice, DeviceBuffer, WgpuBuffer};
use coeus_core::{Scalar, Storage, StorageMut};
use crate::WgpuDevice;

static GLOBAL_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

/// Get or initialize the global WGPU device.
#[must_use]
pub fn get_global_device() -> &'static WgpuDevice {
    GLOBAL_DEVICE.get_or_init(|| {
        WgpuDevice::try_default("apollo-global-wgpu")
            .expect("Failed to initialize global WgpuDevice")
    })
}

/// GPU-allocated buffer managed by hephaestus-wgpu.
pub struct WgpuStorage<T> {
    /// Underlying hephaestus-wgpu buffer wrapped in an Arc.
    pub buffer: Arc<WgpuBuffer<T>>,
}

impl<T> coeus_core::storage::private::Sealed for WgpuStorage<T> {}

impl<T> Clone for WgpuStorage<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
        }
    }
}

unsafe impl<T: Send> Send for WgpuStorage<T> {}
unsafe impl<T: Sync> Sync for WgpuStorage<T> {}

impl<T: Scalar> WgpuStorage<T> {
    /// Allocate a new GPU buffer for `len` elements on the global device.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::new_on(get_global_device(), len)
    }

    /// Allocate a new GPU buffer for `len` elements on a specific device.
    #[must_use]
    pub fn new_on(device: &WgpuDevice, len: usize) -> Self {
        let buffer = device
            .hephaestus()
            .alloc_zeroed::<T>(len)
            .expect("Failed to allocate GPU buffer");
        Self {
            buffer: Arc::new(buffer),
        }
    }
}

impl<T: Scalar> Storage<T> for WgpuStorage<T> {
    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<T: Scalar> StorageMut<T> for WgpuStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn make_unique(&mut self) {
        if Arc::strong_count(&self.buffer) > 1 {
            let device = get_global_device();
            let len = self.buffer.len();
            let new_buffer = device
                .hephaestus()
                .alloc_zeroed::<T>(len)
                .expect("Failed to allocate CoW buffer");

            let size_in_bytes = (len * std::mem::size_of::<T>()).max(4) as u64;

            let mut encoder = device
                .inner()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apollo-wgpu-cow-copy"),
                });
            encoder.copy_buffer_to_buffer(self.buffer.raw(), 0, new_buffer.raw(), 0, size_in_bytes);
            device.queue().submit(std::iter::once(encoder.finish()));

            self.buffer = Arc::new(new_buffer);
        }
    }
}
