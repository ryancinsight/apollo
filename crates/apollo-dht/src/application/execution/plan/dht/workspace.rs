//! Thread-local scratch pools and interop/validation helpers for Discrete Hartley Transform.

use mnemosyne::scratch::ScratchPool;

thread_local! {
    #[cfg_attr(
        all(windows, target_env = "gnu"),
        expect(
            clippy::missing_const_for_thread_local,
            reason = "clippy 1.97 false positive on the windows-gnu thread_local expansion: the initializer is already a const block"
        )
    )]
    pub(crate) static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    #[cfg_attr(
        all(windows, target_env = "gnu"),
        expect(
            clippy::missing_const_for_thread_local,
            reason = "clippy 1.97 false positive on the windows-gnu thread_local expansion: the initializer is already a const block"
        )
    )]
    pub(crate) static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}
