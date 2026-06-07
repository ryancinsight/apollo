//! Lightweight cache counters for identifying poor locality.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Per-cache statistics
#[derive(Default, Debug)]
pub struct CacheStats {
    /// Thread-local hits
    pub tl_hits: AtomicU64,
    /// Global cache hits (after TL miss)
    pub global_hits: AtomicU64,
    /// Cache misses (required build)
    pub misses: AtomicU64,
}

impl CacheStats {
    #[inline]
    pub fn tl_hit(&self) {
        self.tl_hits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn global_hit(&self) {
        self.global_hits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
}

/// Global profiling state
static PROFILER: OnceLock<Profiler> = OnceLock::new();

/// All tracked cache statistics
#[derive(Default)]
pub struct Profiler {
    // Twiddle caches
    pub twiddle_fwd_precise: CacheStats,
    pub twiddle_inv_precise: CacheStats,

    // Rader caches
    pub rader_negacyclic_precise: CacheStats,
    pub rader_order: CacheStats,

    // Bluestein caches
    pub bluestein_precise: CacheStats,
    pub bluestein_reduced: CacheStats,

    // Factorization caches
    pub prime23_radix: CacheStats,
    pub coprime_factors: CacheStats,
    pub is_prime: CacheStats,
    pub pfa_perm: CacheStats,
}

/// Get the global profiler instance (auto-initializes on first call)
#[inline]
pub fn get() -> &'static Profiler {
    PROFILER.get_or_init(Profiler::default)
}
