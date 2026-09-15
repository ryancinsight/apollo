//! The worker idle hook that releases idle FFT scratch, registered once per
//! process with Moirai.
//!
//! Moirai runs registered hooks on each worker before it parks
//! (`moirai::register_idle_hook`), from a fixed registry of
//! `moirai::MAX_IDLE_HOOKS` slots that no consumer ever releases. Apollo takes
//! one slot the first time a dynamic plan is built or a lane pass runs, and
//! records the outcome: a refused registration is not fatal, since every
//! transform still runs, but idle scratch then waits for explicit
//! [`release_thread_local_scratch`] calls at quiescent boundaries, which
//! [`thread_local_scratch_hook_registered`] lets a runtime owner detect.

use std::sync::OnceLock;

/// Whether the release holds a registry slot, decided on the first attempt.
static HOOK_REGISTERED: OnceLock<bool> = OnceLock::new();
#[cfg(test)]
static RELEASES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Registers the release with Moirai once per process and reports whether it
/// holds a slot. A full registry is recorded and never retried, since Moirai
/// never frees a slot.
pub(crate) fn ensure_registered() -> bool {
    *HOOK_REGISTERED
        .get_or_init(|| moirai::register_idle_hook(release_thread_local_scratch).is_ok())
}

/// Whether the idle-scratch release runs on Moirai's workers.
///
/// Registers the release on the first call. `false` means Moirai refused the
/// registration because its `MAX_IDLE_HOOKS` slots were already taken by other
/// consumers in the process; transforms are unaffected, and idle scratch is
/// then reclaimed only by explicit [`release_thread_local_scratch`] calls. The
/// answer is fixed for the life of the process.
#[must_use]
pub fn thread_local_scratch_hook_registered() -> bool {
    ensure_registered()
}

/// Releases idle FFT scratch capacity held by the current thread.
///
/// Call this at a quiescent boundary on each long-lived worker after its FFT
/// workload has completed. The function affects only the current thread's
/// thread-local banks; calling it from a coordinator does not reach worker
/// banks. It does not invalidate a live scratch borrow, and it is not intended
/// for the per-transform hot path. Dynamic FFT plan construction registers this
/// release with Moirai automatically. For const-constructed static plans or
/// direct kernel entry points, call this once at a runtime boundary before work
/// is submitted so the worker-idle hook is installed; when Moirai's registry
/// was already full at that point ([`thread_local_scratch_hook_registered`]
/// reads `false`), this call is the only release.
pub fn release_thread_local_scratch() {
    #[cfg(test)]
    RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ensure_registered();
    super::mixed_radix::release_thread_local_scratch();
}

/// How many times the release ran in this process.
#[cfg(test)]
pub(crate) fn release_count() -> usize {
    RELEASES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::{
        release_count, release_thread_local_scratch, thread_local_scratch_hook_registered,
    };
    use crate::{FftPlan1D, Shape1D};
    use eunomia::Complex;

    /// Under nextest every test is its own process, so the registry this test
    /// fills is the one Apollo then finds full.
    #[test]
    fn a_full_registry_refuses_the_hook_and_transforms_still_run() {
        fn occupant() {}
        let mut taken = 0;
        while moirai::register_idle_hook(occupant).is_ok() {
            taken += 1;
        }
        assert!(
            taken <= moirai::MAX_IDLE_HOOKS,
            "the registry admitted {taken} occupants"
        );
        assert!(!thread_local_scratch_hook_registered());

        let plan = FftPlan1D::<f64>::new(Shape1D::new(8).expect("shape"));
        let mut data: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        plan.forward_complex_slice_inplace(&mut data);
        // The DC bin is the sum 0 + 1 + ... + 7.
        assert!((data[0].re - 28.0).abs() < 1.0e-12 && data[0].im.abs() < 1.0e-12);

        release_thread_local_scratch();
        assert_eq!(release_count(), 1);
        assert!(!thread_local_scratch_hook_registered());
    }
}
