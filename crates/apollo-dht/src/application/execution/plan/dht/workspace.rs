//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use mnemosyne::scratch::ScratchPool;

thread_local! {
    #[cfg_attr(
        windows,
        expect(
            clippy::missing_const_for_thread_local,
            reason = "false positive: the initializer is already a const block"
        )
    )]
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    #[cfg_attr(
        windows,
        expect(
            clippy::missing_const_for_thread_local,
            reason = "false positive: the initializer is already a const block"
        )
    )]
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}
