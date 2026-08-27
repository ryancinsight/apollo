//! Retained-footprint attribution across the power-of-two ladder
//! (`ATLAS-APOLLO-RETAINED-FOOTPRINT-2026-08-27`).
//!
//! The peak census (`engine_census::peak_working_set_census`) established that
//! Apollo retains 3.8–10.4x the signal size while the references hold ~1.0x,
//! but it measures from outside the crate and can only report totals. This
//! probe attributes the total: a windowed counting allocator opens one window
//! per acquisition stage — forward twiddle table, plan construction, first
//! transform, warm transform — and a ledger records every allocation of at
//! least `n` bytes with its size, so each retained block maps to an owner by
//! its byte signature (`16n` interleaved scratch or twiddle table, `8n`
//! planar half, `2 x FUSE_THRESHOLD x 16` compose arena, and so on).
//!
//! Asserts nothing; it is a named measurement instrument, run with
//! `--ignored --nocapture`. Allocation accounting is exact under host load —
//! no pinning needed. The ledger matches frees to entries by size, so two
//! live blocks of one size are indistinguishable; the report therefore prints
//! size x count, which is what the attribution needs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex64;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// Live-bytes balance inside the window; signed for frees of pre-window blocks.
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
/// Ledger floor: allocations below this many bytes are counted but not listed.
static LEDGER_FLOOR: AtomicUsize = AtomicUsize::new(usize::MAX);

const LEDGER_SLOTS: usize = 256;
/// Block sizes observed inside the window; slot is dead once freed.
static LEDGER_SIZE: [AtomicUsize; LEDGER_SLOTS] = [const { AtomicUsize::new(0) }; LEDGER_SLOTS];
static LEDGER_LIVE: [AtomicBool; LEDGER_SLOTS] = [const { AtomicBool::new(false) }; LEDGER_SLOTS];
static LEDGER_NEXT: AtomicUsize = AtomicUsize::new(0);
static LEDGER_DROPPED: AtomicUsize = AtomicUsize::new(0);

fn ledger_push(size: usize) {
    let slot = LEDGER_NEXT.fetch_add(1, Ordering::Relaxed);
    if slot < LEDGER_SLOTS {
        LEDGER_SIZE[slot].store(size, Ordering::Relaxed);
        LEDGER_LIVE[slot].store(true, Ordering::Relaxed);
    } else {
        LEDGER_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

fn ledger_free(size: usize) {
    let filled = LEDGER_NEXT.load(Ordering::Relaxed).min(LEDGER_SLOTS);
    for slot in 0..filled {
        if LEDGER_SIZE[slot].load(Ordering::Relaxed) == size
            && LEDGER_LIVE[slot]
                .compare_exchange(true, false, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            return;
        }
    }
}

fn track(delta: isize) {
    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

struct Attributing;

// SAFETY: every method forwards to `System` unchanged; the counters and the
// ledger observe sizes and never touch the returned pointer.
unsafe impl GlobalAlloc for Attributing {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            track(isize::try_from(l.size()).expect("layout size fits isize"));
            if l.size() >= LEDGER_FLOOR.load(Ordering::Relaxed) {
                ledger_push(l.size());
            }
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            track(-isize::try_from(l.size()).expect("layout size fits isize"));
            if l.size() >= LEDGER_FLOOR.load(Ordering::Relaxed) {
                ledger_free(l.size());
            }
        }
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            track(
                isize::try_from(new).expect("layout size fits isize")
                    - isize::try_from(l.size()).expect("layout size fits isize"),
            );
            if l.size() >= LEDGER_FLOOR.load(Ordering::Relaxed) {
                ledger_free(l.size());
            }
            if new >= LEDGER_FLOOR.load(Ordering::Relaxed) {
                ledger_push(new);
            }
        }
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOCATOR: Attributing = Attributing;

/// Runs `f` in a fresh window and prints its allocation summary and the
/// surviving ledger blocks as `size x count`.
fn window<R>(label: &str, floor: usize, f: impl FnOnce() -> R) -> R {
    ALLOCS.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    LEDGER_NEXT.store(0, Ordering::Relaxed);
    LEDGER_DROPPED.store(0, Ordering::Relaxed);
    LEDGER_FLOOR.store(floor, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let result = f();
    COUNTING.store(false, Ordering::Relaxed);

    let filled = LEDGER_NEXT.load(Ordering::Relaxed).min(LEDGER_SLOTS);
    let mut survivors: Vec<(usize, usize)> = Vec::new();
    for slot in 0..filled {
        if LEDGER_LIVE[slot].load(Ordering::Relaxed) {
            let size = LEDGER_SIZE[slot].load(Ordering::Relaxed);
            match survivors.iter_mut().find(|(s, _)| *s == size) {
                Some((_, count)) => *count += 1,
                None => survivors.push((size, 1)),
            }
        }
    }
    survivors.sort_unstable_by(|a, b| b.cmp(a));
    let blocks: String = survivors
        .iter()
        .map(|(size, count)| format!("{size}x{count}"))
        .collect::<Vec<_>>()
        .join(" ");
    let dropped = LEDGER_DROPPED.load(Ordering::Relaxed);
    println!(
        "  {label:<14} allocs {:>5}  peak {:>10}  retained {:>10}  blocks: {blocks}{}",
        ALLOCS.load(Ordering::Relaxed),
        PEAK.load(Ordering::Relaxed),
        LIVE.load(Ordering::Relaxed),
        if dropped > 0 {
            format!(" (+{dropped} past ledger capacity)")
        } else {
            String::new()
        },
    );
    result
}

#[test]
#[ignore = "measurement probe for retained-footprint attribution"]
fn retained_footprint_attribution() {
    for n in [1024usize, 4096, 16384, 65536, 262144] {
        println!("n = {n} (signal 16n = {} bytes; ledger floor n)", n * 16);
        let mut signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect();

        let table = window("twiddle table", n, || {
            <f64 as MixedRadixScalar>::cached_twiddle_fwd(n)
        });
        let plan = window("plan build", n, || {
            crate::FftPlan1D::<f64>::new(crate::Shape1D { n })
        });
        window("first forward", n, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });
        window("warm forward", n, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });
        drop(plan);
        drop(table);
    }
}
