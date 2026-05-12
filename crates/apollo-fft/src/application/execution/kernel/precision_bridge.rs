//! Generic storage-precision bridge for kernels executed in `Complex32`.

#![allow(clippy::uninit_vec)]

use half::f16;
use num_complex::{Complex, Complex32};
use std::cell::RefCell;

thread_local! {
    static COMPLEX32_BRIDGE_SCRATCH: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
}

/// Storage element that can be transformed through a `Complex32` execution buffer.
pub(crate) trait Complex32Bridge: Copy {
    /// Load the storage element into `Complex32` compute representation.
    fn to_complex32(self) -> Complex32;

    /// Store a `Complex32` compute result back into the storage representation.
    fn from_complex32(value: Complex32) -> Self;
}

impl Complex32Bridge for Complex<f16> {
    #[inline]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self.re.to_f32(), self.im.to_f32())
    }

    #[inline]
    fn from_complex32(value: Complex32) -> Self {
        Self::new(f16::from_f32(value.re), f16::from_f32(value.im))
    }
}

/// Execute `kernel` over a reused `Complex32` scratch view and store results back.
#[inline]
pub(crate) fn run_via_complex32<S, F>(data: &mut [S], kernel: F)
where
    S: Complex32Bridge,
    F: FnOnce(&mut [Complex32]),
{
    COMPLEX32_BRIDGE_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let n = data.len();
        if scratch.capacity() < n {
            let cap = scratch.capacity();
            scratch.reserve(n.saturating_sub(cap));
        }
        // SAFETY: The kernel will immediately overwrite the entire vector in the loop below.
        unsafe {
            scratch.set_len(n);
        }
        for (i, v) in data.iter().enumerate() {
            scratch[i] = v.to_complex32();
        }
        kernel(scratch.as_mut_slice());
        data.iter_mut()
            .zip(scratch.iter().copied())
            .for_each(|(dst, src)| *dst = S::from_complex32(src));
    });
}
