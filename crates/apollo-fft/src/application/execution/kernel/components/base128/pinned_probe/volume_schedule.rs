//! The 64³ volume pass under process-wide core schedules.
//!
//! [`super::small_pot_arms`] measured the N = 64 codelet itself and found the
//! shipped arm optimal in its own regime (`#apollo-n64-lane-pass`), but its
//! closing question was about the *pass*: a 64³ transform hands moirai
//! 3 × 4,096 length-64 lanes, and the efficiency core runs those lanes 2.1×
//! slower than the performance core. Whether moirai's indexed dispatch plus
//! work stealing hides that imbalance, or the pass finishes when its last
//! efficiency-core lane finishes, is what this instrument answers. It needs no
//! lane-assignment API: the schedule is imposed from outside the scheduler.
//!
//! # Mechanism: process affinity, applied before the pool exists
//!
//! Moirai's global executor is built once per process at machine width
//! (`ExecutorBuilder::new` reads the themis topology), and its workers are
//! plain threads that inherit the *process* affinity mask on Windows. Each
//! regime therefore runs in its own test process (nextest's execution model)
//! and applies `SetProcessAffinityMask` before the first scheduler call:
//!
//! - `mixed` applies no restriction — the production schedule.
//! - `all-performance` restricts the process to the queried performance set.
//! - `all-efficiency` restricts it to the queried efficiency set.
//!
//! The regime is then *verified from inside the scheduler*: one throwaway
//! indexed pass over wide chunks records the processor every executing worker
//! landed on, and the probe refuses to report numbers from a process whose
//! observed worker set escapes the requested class. A machine-width pool under
//! a class mask oversubscribes the class by design — that is the honest
//! production-shaped reading of "the stack ran this pass on one core type".
//!
//! # The measured operation
//!
//! One production forward plus one production inverse on a fixed 64³ field
//! through [`crate::FftPlan3D`] — the transform pair kwavers' PSTD executes
//! per timestep. The inverse returns the field to its input, so one buffer is
//! reused with no reseeding inside the reading (the same rationale
//! `super::small_pot_arms` documents). The identity is asserted once per
//! regime before timing, so a schedule that corrupted the transform could not
//! win by doing less work.
//!
//! The value oracles own correctness; this probe reports, and its verdicts
//! are the comparisons between regimes, not an absolute threshold.

use crate::application::execution::kernel::measurement_cores;
use crate::FftPlan3D;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::ProcessorIndex;
use leto::Array3;
use std::collections::BTreeSet;
use std::sync::Mutex;

/// The volume edge this stack's dominant workload runs. 64³ complex f64 is
/// 4 MiB — not L1-resident, which is exactly the production shape: the lanes
/// are L1-resident, the volume they sweep is not.
const N: usize = 64;

/// A process schedule and the worker placement it actually produced.
///
/// Reported beside every table so a misapplied mask cannot pass silently; the
/// measurement refuses to run from an unverified regime.
struct Regime {
    allowed: Option<BTreeSet<u32>>,
    observed: BTreeSet<u32>,
}

impl Regime {
    /// A regime is verified when every observed worker landed inside the
    /// requested set. `allowed = None` (uniform host, platform refusal) means
    /// "the mixed schedule with nothing to verify".
    fn verified(&self) -> bool {
        match &self.allowed {
            None => true,
            Some(allowed) => self.observed.is_subset(allowed),
        }
    }
}

#[cfg(target_os = "windows")]
mod windows_affinity {
    // Instrument-local FFI: themis owns the affinity *representation*
    // (`ProcessorAffinityGroups`), not an apply primitive, and this probe is
    // its only consumer so far. Promotion into themis is a later round if a
    // second consumer appears.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessAffinityMask(
            process: *mut core::ffi::c_void,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
        fn SetProcessAffinityMask(process: *mut core::ffi::c_void, mask: usize) -> i32;
    }

    /// The processors this process may run on right now.
    pub(super) fn current_mask() -> Option<usize> {
        let mut process_mask = 0usize;
        let mut system_mask = 0usize;
        // SAFETY: both pointers name local `usize`s; the query has no other
        // obligations.
        let ok = unsafe {
            GetProcessAffinityMask(
                GetCurrentProcess(),
                core::ptr::addr_of_mut!(process_mask),
                core::ptr::addr_of_mut!(system_mask),
            )
        };
        (ok != 0).then_some(process_mask)
    }

    /// Restrict the process to `mask`. Returns `None` when the platform
    /// refused, so the caller skips rather than reporting from a process whose
    /// schedule it does not know.
    pub(super) fn set_process_mask(mask: usize) -> Option<()> {
        // SAFETY: the handle is the pseudo-handle of the calling process, and
        // `mask` is a nonzero processor subset of the current mask the caller
        // just read.
        let ok = unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask) };
        (ok != 0).then_some(())
    }
}

/// One production round trip over the fixed volume, buffer access included.
#[inline(never)]
fn round_trip(plan: &FftPlan3D<f64>, field: &mut Array3<Complex64>) {
    std::hint::black_box(&mut *field);
    plan.forward_complex_inplace(field);
    plan.inverse_complex_inplace(field);
}

fn field() -> Array3<Complex64> {
    Array3::from_shape_fn([N, N, N], |[i, j, k]| {
        let x = ((i * N + j) * N + k) as f64;
        Complex64::new(
            (0.17 * x).sin() + 0.11 * (0.07 * x).cos(),
            0.23 * (0.31 * x).cos(),
        )
    })
}

/// Restrict the process to `processors`.
///
/// The requested set is intersected with the process's current mask — a CI
/// runner or a parent shell may already restrict the process, and a mask bit
/// outside it would be refused wholesale by the platform. Returns the allowed
/// set actually applied, or `None` when the platform cannot apply any
/// schedule — the caller then runs the unrestricted reading and says so.
fn apply_schedule(processors: impl IntoIterator<Item = u32>) -> Option<BTreeSet<u32>> {
    #[cfg(target_os = "windows")]
    {
        let current = windows_affinity::current_mask()?;
        let mut mask = 0usize;
        let mut allowed = BTreeSet::new();
        for processor in processors {
            let bit = 1usize
                .checked_shl(processor % 64)
                .expect("processor index below 64 fits one group mask");
            if current & bit != 0 {
                mask |= bit;
                allowed.insert(processor);
            }
        }
        if mask == 0 {
            return None;
        }
        windows_affinity::set_process_mask(mask)?;
        Some(allowed)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = processors;
        None
    }
}

/// Run one wide indexed pass through moirai and record which processors its
/// workers executed on. This is both the regime verification and the pool's
/// first touch — the workers it spawns here are the workers the measurement
/// uses, all created after the schedule's mask was applied.
fn observe_worker_placement() -> BTreeSet<u32> {
    let observed = Mutex::new(BTreeSet::new());
    let mut scratch: Vec<u64> = vec![0; 96 * 4096];
    moirai::for_each_chunk_mut_with::<moirai::AdaptiveWithThreshold<4096>, u64, _>(
        &mut scratch,
        4096,
        |chunk| {
            chunk[0] = chunk.len() as u64;
            if let Ok(mut seen) = observed.lock() {
                if let Ok(processor) = ProcessorIndex::current() {
                    seen.insert(processor.get());
                }
            }
        },
    );
    let _ = std::hint::black_box(&scratch);
    observed
        .into_inner()
        .expect("census lock cannot be poisoned")
}

/// The per-regime measurement: verify the schedule, assert the transform's
/// identity, then time the production round trip.
fn measure_volume_pass(label: &'static str, allowed: Option<BTreeSet<u32>>) {
    let observed = observe_worker_placement();
    let regime = Regime { allowed, observed };
    let verified = regime.verified();
    println!(
        "regime {label}: workers on {:?}{}",
        regime.observed,
        if verified {
            match regime.allowed {
                Some(ref allowed) => format!(" (allowed {allowed:?})"),
                None => String::new(),
            }
        } else {
            " — SCHEDULE NOT VERIFIED, NUMBERS BELOW ARE NOT A MEASUREMENT".to_string()
        }
    );
    assert!(
        verified,
        "worker placement escaped the requested schedule {label}"
    );

    let plan =
        FftPlan3D::<f64>::new(crate::Shape3D::new(N, N, N).expect("invariant: 64 is non-zero"));
    let mut work = field();
    let reference = work.clone();

    // Identity gate: a schedule that corrupted the transform must not be
    // timed. The inverse is normalized, so one round trip returns the input.
    plan.forward_complex_inplace(&mut work);
    plan.inverse_complex_inplace(&mut work);
    let max_error = work
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 1e-8,
        "round trip must reproduce the input before timing (max error {max_error})"
    );

    let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
    suite.run(
        BenchmarkCase::new("volume-64-schedule", "round-trip-transform", label),
        || round_trip(&plan, &mut work),
    );
    println!("VOLUME-64 regime={label}");
    print!("{}", suite.report());
}

#[test]
#[ignore = "measurement instrument: the 64³ pass under the production (mixed) schedule"]
fn volume_pass_under_the_production_schedule() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());

    // No restriction: the production schedule, and the baseline every other
    // regime is read against.
    measure_volume_pass("mixed", None);
}

#[test]
#[ignore = "measurement instrument: the 64³ pass with the process pinned to the performance set"]
fn volume_pass_performance_only() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());

    let Some(class) = selection.performance_class() else {
        eprintln!("no performance-class processor selected; skipping");
        return;
    };
    let performance_set: Vec<u32> = selection.processors_in_class(class).collect();
    assert!(
        performance_set.len() > 1,
        "a class with one member is a uniform host mislabelled as hybrid"
    );
    measure_volume_pass("all-performance", apply_schedule(performance_set));
}

#[test]
#[ignore = "measurement instrument: the 64³ pass with the process pinned to the efficiency set"]
fn volume_pass_efficiency_only() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());

    let Some(class) = selection.efficiency_class() else {
        eprintln!("host reports a single class; there is no efficiency set to pin");
        return;
    };
    let efficiency_set: Vec<u32> = selection.processors_in_class(class).collect();
    assert!(
        !efficiency_set.is_empty(),
        "the census must show the rank-0 class it reported"
    );
    measure_volume_pass("all-efficiency", apply_schedule(efficiency_set));
}
