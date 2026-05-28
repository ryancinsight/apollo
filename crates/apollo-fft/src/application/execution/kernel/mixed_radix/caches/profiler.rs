//! Lightweight cache hit rate profiler for identifying poor locality.
//!
//! # Usage
//!
//! Enable with `cache-profiling` feature at compile time.
//! Statistics are printed when `cache_profiler::report()` is called.
//!
//! ```ignore
//! // At start of program
//! cache_profiler::init();
//!
//! // At end or periodically
//! cache_profiler::report();
//! ```

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[allow(dead_code)]
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

    /// Total accesses
    pub fn total(&self) -> u64 {
        self.tl_hits.load(Ordering::Relaxed)
            + self.global_hits.load(Ordering::Relaxed)
            + self.misses.load(Ordering::Relaxed)
    }

    /// TL hit rate
    pub fn tl_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        self.tl_hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Global hit rate (of remaining after TL miss)
    pub fn global_rate(&self) -> f64 {
        let after_tl = self.global_hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed);
        if after_tl == 0 {
            return 0.0;
        }
        self.global_hits.load(Ordering::Relaxed) as f64 / after_tl as f64
    }

    /// Combined hit rate (TL + global hits / total)
    pub fn hit_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.tl_hits.load(Ordering::Relaxed) + self.global_hits.load(Ordering::Relaxed)) as f64 / total as f64
    }

    /// Format a report line
    pub fn report_line(&self, name: &str) -> String {
        let tl = self.tl_hits.load(Ordering::Relaxed);
        let gbl = self.global_hits.load(Ordering::Relaxed);
        let miss = self.misses.load(Ordering::Relaxed);
        let total = tl + gbl + miss;
        format!(
            concat!(
                "{:30} TL:{TL:>8}  GBL:{GBL:>8}  MISS:{MISS:>8}  TOT:{TOT:>8}  ",
                "TL%:{TL_RATE:>6.2}%  GBL%:{GBL_RATE:>6.2}%  HIT%:{HIT_RATE:>6.2}%"
            ),
            name,
            TL = tl,
            GBL = gbl,
            MISS = miss,
            TOT = total,
            TL_RATE = self.tl_rate() * 100.0,
            GBL_RATE = self.global_rate() * 100.0,
            HIT_RATE = self.hit_rate() * 100.0
        )
    }
}

/// Global profiling state
static PROFILER: OnceLock<Profiler> = OnceLock::new();

/// All tracked cache statistics
pub struct Profiler {
    // Twiddle caches
    pub twiddle_fwd_precise: CacheStats,
    pub twiddle_fwd_reduced: CacheStats,
    pub twiddle_inv_precise: CacheStats,
    pub twiddle_inv_reduced: CacheStats,

    // Rader caches
    pub rader_spectrum_precise: CacheStats,
    pub rader_spectrum_reduced: CacheStats,
    pub rader_negacyclic_precise: CacheStats,
    pub rader_negacyclic_reduced: CacheStats,
    pub rader_neg_twiddles_precise: CacheStats,
    pub rader_neg_twiddles_reduced: CacheStats,
    pub rader_order: CacheStats,

    // Bluestein caches
    pub bluestein_precise: CacheStats,
    pub bluestein_reduced: CacheStats,

    // Four-step cache
    pub four_step_precise: CacheStats,
    pub four_step_reduced: CacheStats,

    // Factorization caches
    pub prime23_radix: CacheStats,
    pub coprime_factors: CacheStats,
    pub is_prime: CacheStats,
    pub pfa_perm: CacheStats,
}

impl Default for Profiler {
    fn default() -> Self {
        Self {
            twiddle_fwd_precise: CacheStats::default(),
            twiddle_fwd_reduced: CacheStats::default(),
            twiddle_inv_precise: CacheStats::default(),
            twiddle_inv_reduced: CacheStats::default(),
            rader_spectrum_precise: CacheStats::default(),
            rader_spectrum_reduced: CacheStats::default(),
            rader_negacyclic_precise: CacheStats::default(),
            rader_negacyclic_reduced: CacheStats::default(),
            rader_neg_twiddles_precise: CacheStats::default(),
            rader_neg_twiddles_reduced: CacheStats::default(),
            rader_order: CacheStats::default(),
            bluestein_precise: CacheStats::default(),
            bluestein_reduced: CacheStats::default(),
            four_step_precise: CacheStats::default(),
            four_step_reduced: CacheStats::default(),
            prime23_radix: CacheStats::default(),
            coprime_factors: CacheStats::default(),
            is_prime: CacheStats::default(),
            pfa_perm: CacheStats::default(),
        }
    }
}

/// Initialize the profiler (call at program start)
#[inline]
pub fn init() {
    let _ = PROFILER.get_or_init(|| Profiler::default());
}

/// Get the global profiler instance (auto-initializes on first call)
#[inline(always)]
pub fn get() -> &'static Profiler {
    PROFILER.get_or_init(|| Profiler::default())
}

/// Report all cache statistics
pub fn report() {
    let p = get();

    println!("\n========== Cache Hit Rate Profile ==========");
    println!("{:30} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Cache", "TL Hits", "GBL Hits", "Misses", "Total", "TL Rate", "GBL Rate", "Hit Rate");
    println!("----------------------------------------------------------------------------------------------------");

    // Twiddle caches
    println!("{}", p.twiddle_fwd_precise.report_line("twiddle_fwd_precise"));
    println!("{}", p.twiddle_fwd_reduced.report_line("twiddle_fwd_reduced"));
    println!("{}", p.twiddle_inv_precise.report_line("twiddle_inv_precise"));
    println!("{}", p.twiddle_inv_reduced.report_line("twiddle_inv_reduced"));

    // Rader caches
    println!("{}", p.rader_spectrum_precise.report_line("rader_spectrum_precise"));
    println!("{}", p.rader_spectrum_reduced.report_line("rader_spectrum_reduced"));
    println!("{}", p.rader_negacyclic_precise.report_line("rader_negacyclic_precise"));
    println!("{}", p.rader_negacyclic_reduced.report_line("rader_negacyclic_reduced"));
    println!("{}", p.rader_neg_twiddles_precise.report_line("rader_neg_twiddles_precise"));
    println!("{}", p.rader_neg_twiddles_reduced.report_line("rader_neg_twiddles_reduced"));
    println!("{}", p.rader_order.report_line("rader_order"));

    // Bluestein caches
    println!("{}", p.bluestein_precise.report_line("bluestein_precise"));
    println!("{}", p.bluestein_reduced.report_line("bluestein_reduced"));

    // Four-step cache
    println!("{}", p.four_step_precise.report_line("four_step_precise"));
    println!("{}", p.four_step_reduced.report_line("four_step_reduced"));

    // Factorization caches
    println!("{}", p.prime23_radix.report_line("prime23_radix"));
    println!("{}", p.coprime_factors.report_line("coprime_factors"));
    println!("{}", p.is_prime.report_line("is_prime"));
    println!("{}", p.pfa_perm.report_line("pfa_perm"));

    println!("----------------------------------------------------------------------------------------------------");

    // Summary: identify caches with poor locality
    let poor_locality: Vec<(&str, f64)> = vec![
        ("twiddle_fwd_precise", p.twiddle_fwd_precise.hit_rate()),
        ("twiddle_fwd_reduced", p.twiddle_fwd_reduced.hit_rate()),
        ("twiddle_inv_precise", p.twiddle_inv_precise.hit_rate()),
        ("twiddle_inv_reduced", p.twiddle_inv_reduced.hit_rate()),
        ("rader_spectrum_precise", p.rader_spectrum_precise.hit_rate()),
        ("rader_spectrum_reduced", p.rader_spectrum_reduced.hit_rate()),
        ("rader_negacyclic_precise", p.rader_negacyclic_precise.hit_rate()),
        ("rader_negacyclic_reduced", p.rader_negacyclic_reduced.hit_rate()),
        ("rader_neg_twiddles_precise", p.rader_neg_twiddles_precise.hit_rate()),
        ("rader_neg_twiddles_reduced", p.rader_neg_twiddles_reduced.hit_rate()),
        ("rader_order", p.rader_order.hit_rate()),
        ("bluestein_precise", p.bluestein_precise.hit_rate()),
        ("bluestein_reduced", p.bluestein_reduced.hit_rate()),
        ("four_step_precise", p.four_step_precise.hit_rate()),
        ("four_step_reduced", p.four_step_reduced.hit_rate()),
        ("prime23_radix", p.prime23_radix.hit_rate()),
        ("coprime_factors", p.coprime_factors.hit_rate()),
        ("is_prime", p.is_prime.hit_rate()),
        ("pfa_perm", p.pfa_perm.hit_rate()),
    ];

    let poor: Vec<_> = poor_locality.iter()
        .filter(|(_, rate)| *rate < 0.80)
        .collect();

    if !poor.is_empty() {
        println!("\nWARNING: Caches with poor locality (hit rate < 80%):");
        for (name, rate) in poor {
            println!("  - {}: {:.1}%", name, rate * 100.0);
        }
    } else {
        println!("\nAll caches have good locality (hit rate >= 80%)");
    }
    println!("==========================================\n");
}

/// Accessor for profiler stats by cache name (for integration into existing cache functions)
#[derive(Clone, Copy)]
pub enum CacheId {
    TwiddleFwdPrecise,
    TwiddleFwdReduced,
    TwiddleInvPrecise,
    TwiddleInvReduced,
    RaderSpectrumPrecise,
    RaderSpectrumReduced,
    RaderNegacyclicPrecise,
    RaderNegacyclicReduced,
    RaderNegTwiddlesPrecise,
    RaderNegTwiddlesReduced,
    RaderOrder,
    BluesteinPrecise,
    BluesteinReduced,
    FourStepPrecise,
    FourStepReduced,
    Prime23Radix,
    CoprimeFactors,
    IsPrime,
    PfaPerm,
}

impl Profiler {
    pub fn get_stats(&self, id: CacheId) -> &CacheStats {
        match id {
            CacheId::TwiddleFwdPrecise => &self.twiddle_fwd_precise,
            CacheId::TwiddleFwdReduced => &self.twiddle_fwd_reduced,
            CacheId::TwiddleInvPrecise => &self.twiddle_inv_precise,
            CacheId::TwiddleInvReduced => &self.twiddle_inv_reduced,
            CacheId::RaderSpectrumPrecise => &self.rader_spectrum_precise,
            CacheId::RaderSpectrumReduced => &self.rader_spectrum_reduced,
            CacheId::RaderNegacyclicPrecise => &self.rader_negacyclic_precise,
            CacheId::RaderNegacyclicReduced => &self.rader_negacyclic_reduced,
            CacheId::RaderNegTwiddlesPrecise => &self.rader_neg_twiddles_precise,
            CacheId::RaderNegTwiddlesReduced => &self.rader_neg_twiddles_reduced,
            CacheId::RaderOrder => &self.rader_order,
            CacheId::BluesteinPrecise => &self.bluestein_precise,
            CacheId::BluesteinReduced => &self.bluestein_reduced,
            CacheId::FourStepPrecise => &self.four_step_precise,
            CacheId::FourStepReduced => &self.four_step_reduced,
            CacheId::Prime23Radix => &self.prime23_radix,
            CacheId::CoprimeFactors => &self.coprime_factors,
            CacheId::IsPrime => &self.is_prime,
            CacheId::PfaPerm => &self.pfa_perm,
        }
    }
}