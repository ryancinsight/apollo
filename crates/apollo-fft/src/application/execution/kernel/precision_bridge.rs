//! Generic storage-precision bridge for kernels executed in `Complex32`.

#![allow(clippy::uninit_vec)]

use half::f16;
use mnemosyne::scratch::ScratchPool;
use num_complex::{Complex, Complex32};

thread_local! {
    static COMPLEX32_BRIDGE_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
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
    let n = data.len();
    COMPLEX32_BRIDGE_SCRATCH.with(|pool| {
        pool.with_scratch(n, |scratch| {
            for (i, v) in data.iter().enumerate() {
                scratch[i] = v.to_complex32();
            }
            kernel(scratch);
            data.iter_mut()
                .zip(scratch.iter().copied())
                .for_each(|(dst, src)| *dst = S::from_complex32(src));
        });
    });
}
