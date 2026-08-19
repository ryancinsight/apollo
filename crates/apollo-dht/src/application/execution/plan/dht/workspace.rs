//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use mnemosyne::scratch::ScratchPool;

thread_local! {
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}
