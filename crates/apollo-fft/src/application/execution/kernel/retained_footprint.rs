//! Retained-footprint attribution across the power-of-two ladder
//! (`ATLAS-APOLLO-RETAINED-FOOTPRINT-2026-08-27`).
//!
//! The outer peak census (`engine_census::peak_working_set_census`) reported
//! 3.8–10.4x the signal size through Apollo's process while the references held
//! ~1.0x, but that aggregate combined transform and runtime state. This probe
//! attributes the global-allocator portion: a windowed counting allocator
//! opens one window per acquisition stage — forward twiddle table, plan
//! construction, first transform, warm transform — and a ledger records every
//! allocation with its size, so each retained block maps to an owner by its byte
//! signature (`16n` interleaved scratch or twiddle table, `8n` planar half,
//! `2 x FUSE_THRESHOLD x 16` compose arena, and so on). A second ledger receives
//! Mnemosyne's process-wide allocation hooks, covering local-deque payload arrays
//! that bypass the installed global allocator.
//!
//! It is a named measurement instrument, run through Nextest with
//! `--run-ignored ignored-only --no-capture`. Run the probe as the only selected
//! test in its process. Both fixed ledgers
//! record successful allocations by pointer, so frees of blocks that predate a
//! window cannot consume same-sized window entries. Generation-tagged windows
//! drain admitted events before reporting. The instrument asserts only that its
//! fixed ledgers did not overflow, never that a measured footprint meets a
//! target.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering};

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex64;

/// Even values are closed; odd values identify one open measurement window.
static WINDOW_EPOCH: AtomicUsize = AtomicUsize::new(0);
/// Global-allocator calls admitted before a window closes.
static ACTIVE_EVENTS: AtomicUsize = AtomicUsize::new(0);

const LEDGER_SLOTS: usize = 2_048;
struct AllocationCounter {
    allocations: AtomicUsize,
    live: AtomicIsize,
    peak: AtomicIsize,
    sizes: [AtomicUsize; LEDGER_SLOTS],
    pointers: [AtomicPtr<u8>; LEDGER_SLOTS],
    next: AtomicUsize,
    dropped: AtomicUsize,
}

impl AllocationCounter {
    const fn new() -> Self {
        Self {
            allocations: AtomicUsize::new(0),
            live: AtomicIsize::new(0),
            peak: AtomicIsize::new(0),
            sizes: [const { AtomicUsize::new(0) }; LEDGER_SLOTS],
            pointers: [const { AtomicPtr::new(std::ptr::null_mut()) }; LEDGER_SLOTS],
            next: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        for pointer in &self.pointers {
            pointer.store(std::ptr::null_mut(), Ordering::Relaxed);
        }
        self.allocations.store(0, Ordering::Relaxed);
        self.live.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.next.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
    }

    fn push(&self, pointer: *mut u8, size: usize) {
        let slot = self.next.fetch_add(1, Ordering::Relaxed);
        if slot < LEDGER_SLOTS {
            self.sizes[slot].store(size, Ordering::Relaxed);
            self.pointers[slot].store(pointer, Ordering::Release);
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn claim(&self, pointer: *mut u8) -> Option<(usize, usize)> {
        let filled = self.next.load(Ordering::Relaxed).min(LEDGER_SLOTS);
        for slot in 0..filled {
            if self.pointers[slot]
                .compare_exchange(
                    pointer,
                    std::ptr::null_mut(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Some((slot, self.sizes[slot].load(Ordering::Relaxed)));
            }
        }
        None
    }

    fn restore(&self, slot: usize, pointer: *mut u8) {
        self.pointers[slot].store(pointer, Ordering::Release);
    }

    fn track(&self, delta: isize) {
        let live = self.live.fetch_add(delta, Ordering::Relaxed) + delta;
        self.peak.fetch_max(live, Ordering::Relaxed);
    }

    fn record_alloc(&self, pointer: *mut u8, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let Ok(delta) = isize::try_from(size) else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        self.track(delta);
        self.push(pointer, size);
    }

    fn record_free(&self, pointer: *mut u8) {
        if let Some((_, size)) = self.claim(pointer) {
            let Ok(delta) = isize::try_from(size) else {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            };
            self.track(-delta);
        }
    }

    fn record_realloc_claimed(
        &self,
        old: Option<(usize, usize)>,
        new_pointer: *mut u8,
        new_size: usize,
    ) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let Ok(new_delta) = isize::try_from(new_size) else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if let Some((_, old_size)) = old {
            let Ok(old_delta) = isize::try_from(old_size) else {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            };
            self.track(new_delta - old_delta);
        } else {
            self.track(new_delta);
        }
        self.push(new_pointer, new_size);
    }
}

static GLOBAL_ALLOCATIONS: AllocationCounter = AllocationCounter::new();
static MNEMOSYNE_ALLOCATIONS: AllocationCounter = AllocationCounter::new();

fn enter_window() -> bool {
    let epoch = WINDOW_EPOCH.load(Ordering::Acquire);
    if epoch & 1 == 0 {
        return false;
    }
    ACTIVE_EVENTS.fetch_add(1, Ordering::AcqRel);
    if WINDOW_EPOCH.load(Ordering::Acquire) == epoch {
        true
    } else {
        ACTIVE_EVENTS.fetch_sub(1, Ordering::AcqRel);
        false
    }
}

fn leave_window() {
    ACTIVE_EVENTS.fetch_sub(1, Ordering::AcqRel);
}

struct Attributing;

// SAFETY: every method forwards to `System` unchanged; the counters and the
// ledger observe sizes and never touch the returned pointer.
unsafe impl GlobalAlloc for Attributing {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let count = enter_window();
        // SAFETY: `GlobalAlloc::alloc` gives the caller responsibility for
        // supplying a valid layout; this wrapper forwards it unchanged.
        let pointer = unsafe { System.alloc(l) };
        if count && !pointer.is_null() {
            GLOBAL_ALLOCATIONS.record_alloc(pointer, l.size());
        }
        if count {
            leave_window();
        }
        pointer
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        let count = enter_window();
        if count {
            GLOBAL_ALLOCATIONS.record_free(p);
        }
        // SAFETY: `GlobalAlloc::dealloc` requires `p` to denote a live block
        // allocated for `l`; this wrapper preserves both values unchanged.
        unsafe { System.dealloc(p, l) };
        if count {
            leave_window();
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let count = enter_window();
        let old = if count {
            GLOBAL_ALLOCATIONS.claim(p)
        } else {
            None
        };
        // SAFETY: `GlobalAlloc::realloc` gives the caller responsibility for
        // the live source block and valid nonzero replacement size.
        let pointer = unsafe { System.realloc(p, l, new) };
        if count && !pointer.is_null() {
            GLOBAL_ALLOCATIONS.record_realloc_claimed(old, pointer, new);
        } else if let Some((slot, _)) = old {
            // A failed `realloc` leaves the original allocation live.
            GLOBAL_ALLOCATIONS.restore(slot, p);
        }
        if count {
            leave_window();
        }
        pointer
    }
}

unsafe extern "C" fn mnemosyne_alloc_hook(pointer: *mut core::ffi::c_void, size: usize) {
    let count = enter_window();
    if count && !pointer.is_null() && size != 0 {
        MNEMOSYNE_ALLOCATIONS.record_alloc(pointer.cast(), size);
    }
    if count {
        leave_window();
    }
}

unsafe extern "C" fn mnemosyne_free_hook(pointer: *mut core::ffi::c_void, _size: usize) {
    let count = enter_window();
    if count && !pointer.is_null() {
        MNEMOSYNE_ALLOCATIONS.record_free(pointer.cast());
    }
    if count {
        leave_window();
    }
}

struct MnemosyneHooks;

impl MnemosyneHooks {
    fn install() -> Self {
        mnemosyne::register_alloc_hook(Some(mnemosyne_alloc_hook));
        mnemosyne::register_free_hook(Some(mnemosyne_free_hook));
        Self
    }
}

impl Drop for MnemosyneHooks {
    fn drop(&mut self) {
        mnemosyne::register_alloc_hook(None);
        mnemosyne::register_free_hook(None);
    }
}

#[global_allocator]
static ALLOCATOR: Attributing = Attributing;

/// Runs `f` in a fresh window and prints its allocation summary and the
/// surviving ledger blocks as `size x count`.
fn window<R>(label: &str, f: impl FnOnce() -> R) -> R {
    assert_eq!(
        ACTIVE_EVENTS.load(Ordering::Acquire),
        0,
        "retained-footprint window opened with allocator events still active"
    );
    GLOBAL_ALLOCATIONS.reset();
    MNEMOSYNE_ALLOCATIONS.reset();
    let closed_epoch = WINDOW_EPOCH.fetch_add(1, Ordering::AcqRel);
    assert_eq!(
        closed_epoch & 1,
        0,
        "retained-footprint window opened while another window was active"
    );
    let result = f();
    let open_epoch = WINDOW_EPOCH.fetch_add(1, Ordering::AcqRel);
    assert_eq!(
        open_epoch & 1,
        1,
        "retained-footprint window closed from an inactive state"
    );
    while ACTIVE_EVENTS.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
    }

    print_counter(label, "global", &GLOBAL_ALLOCATIONS);
    print_counter(label, "Mnemosyne direct", &MNEMOSYNE_ALLOCATIONS);
    result
}

fn print_counter(label: &str, source: &str, counter: &AllocationCounter) {
    let filled = counter.next.load(Ordering::Relaxed).min(LEDGER_SLOTS);
    let mut survivors: Vec<(usize, usize)> = Vec::new();
    for slot in 0..filled {
        if !counter.pointers[slot].load(Ordering::Acquire).is_null() {
            let size = counter.sizes[slot].load(Ordering::Relaxed);
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
    let listed_bytes = survivors
        .iter()
        .map(|(size, count)| size * count)
        .sum::<usize>();
    let dropped = counter.dropped.load(Ordering::Relaxed);
    let retained = counter.live.load(Ordering::Relaxed);
    println!(
        "  {label:<14} {source:<17} allocs {:>5}  peak {:>10}  retained {:>10}  \
         blocks: {blocks}{}",
        counter.allocations.load(Ordering::Relaxed),
        counter.peak.load(Ordering::Relaxed),
        retained,
        if dropped > 0 {
            format!(" (+{dropped} past ledger capacity)")
        } else {
            String::new()
        },
    );
    println!("    listed total: {listed_bytes} bytes");
    assert_eq!(
        retained,
        isize::try_from(listed_bytes).expect("listed bytes fit isize"),
        "{source} live-byte balance must equal the pointer-ledger total"
    );
    assert_eq!(
        dropped, 0,
        "{source} retained-footprint ledger overflowed; increase LEDGER_SLOTS before using the report"
    );
}

fn print_mnemosyne_delta(before: &mnemosyne::MemoryStats, after: &mnemosyne::MemoryStats) {
    let count_delta = |before: usize, after: usize| {
        isize::try_from(after).expect("allocator count fits isize")
            - isize::try_from(before).expect("allocator count fits isize")
    };
    let class_deltas = before
        .size_class_occupancy
        .iter()
        .zip(&after.size_class_occupancy)
        .enumerate()
        .filter_map(|(class, (before, after))| {
            let delta = count_delta(before.live_allocations, after.live_allocations);
            (delta != 0).then(|| format!("{class}:{delta:+}"))
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "    Mnemosyne process mapped bytes {:+}; caller-thread live allocations {:+}; \
         caller-thread class deltas {class_deltas}",
        count_delta(before.current_mapped_bytes, after.current_mapped_bytes),
        count_delta(
            before.current_thread_live_allocations,
            after.current_thread_live_allocations
        )
    );
}

#[test]
#[ignore = "measurement probe for retained-footprint attribution"]
fn retained_footprint_attribution() {
    let _mnemosyne_hooks = MnemosyneHooks::install();
    // A trivial parallel operation ahead of the FFT ladder isolates Moirai's
    // process-global pool startup from transform-owned acquisition.
    let mut warmup = vec![Complex64::new(0.0, 0.0); 65536];
    let mnemosyne_before = mnemosyne::memory_stats();
    window("pool warmup", || {
        moirai::for_each_chunk_mut_with::<moirai::Parallel, _, _>(&mut warmup, 256, |row| {
            std::hint::black_box(row);
        });
    });
    let mnemosyne_after = mnemosyne::memory_stats();
    print_mnemosyne_delta(&mnemosyne_before, &mnemosyne_after);
    drop(warmup);

    for n in [1024usize, 4096, 16384, 65536, 262144] {
        println!("n = {n} (signal 16n = {} bytes)", n * 16);
        let mut signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect();

        let table = window("twiddle table", || {
            <f64 as MixedRadixScalar>::cached_twiddle_fwd(n)
        });
        let plan = window("plan build", || {
            crate::FftPlan1D::<f64>::new(
                crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
            )
        });
        window("first forward", || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });
        window("warm forward", || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });
        drop(plan);
        drop(table);
    }
}

#[test]
#[ignore = "allocation probe must run as the only selected test in its process"]
fn native_width_short_and_base_warm_execution_is_allocation_free() {
    let _mnemosyne_hooks = MnemosyneHooks::install();

    for n in [64usize, 96, 128, 256, 512] {
        let plan = crate::FftPlan1D::<f32>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut signal = (0..n)
            .map(|index| {
                let x = index as f32;
                eunomia::Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect::<Vec<_>>();

        plan.forward_complex_slice_inplace(&mut signal);
        let label = format!("warm f32 n={n}");
        window(&label, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });

        assert_eq!(
            GLOBAL_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f32 n={n} execution allocated through the global allocator"
        );
        assert_eq!(
            MNEMOSYNE_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f32 n={n} execution allocated directly through Mnemosyne"
        );
    }

    for n in [64usize, 96, 128, 256, 512] {
        let plan = crate::FftPlan1D::<f64>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut signal = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect::<Vec<_>>();

        plan.forward_complex_slice_inplace(&mut signal);
        let label = format!("warm f64 n={n}");
        window(&label, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });

        assert_eq!(
            GLOBAL_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f64 n={n} execution allocated through the global allocator"
        );
        assert_eq!(
            MNEMOSYNE_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f64 n={n} execution allocated directly through Mnemosyne"
        );
    }
}

#[test]
#[ignore = "allocation probe must run as the only selected test in its process"]
fn wide_planar_transpose_warm_execution_is_allocation_free() {
    let _mnemosyne_hooks = MnemosyneHooks::install();

    for n in [16_384usize, 32_768] {
        let plan = crate::FftPlan1D::<f32>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut signal = (0..n)
            .map(|index| {
                let x = index as f32;
                eunomia::Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect::<Vec<_>>();

        plan.forward_complex_slice_inplace(&mut signal);
        let label = format!("warm planar f32 n={n}");
        window(&label, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });

        assert_eq!(
            GLOBAL_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed planar f32 n={n} execution allocated through the global allocator"
        );
        assert_eq!(
            MNEMOSYNE_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed planar f32 n={n} execution allocated directly through Mnemosyne"
        );
    }
}

#[test]
#[ignore = "allocation probe must run as the only selected test in its process"]
fn small_nonsmooth_rader_warm_execution_is_allocation_free() {
    let _mnemosyne_hooks = MnemosyneHooks::install();

    for n in [59usize, 83, 107] {
        let plan = crate::FftPlan1D::<f64>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut signal = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect::<Vec<_>>();

        plan.forward_complex_slice_inplace(&mut signal);
        let label = format!("warm f64 n={n}");
        window(&label, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });

        assert_eq!(
            GLOBAL_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f64 n={n} execution allocated through the global allocator"
        );
        assert_eq!(
            MNEMOSYNE_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f64 n={n} execution allocated directly through Mnemosyne"
        );
    }

    for n in [59usize, 83, 107] {
        let plan = crate::FftPlan1D::<f32>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut signal = (0..n)
            .map(|index| {
                let x = index as f32;
                eunomia::Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect::<Vec<_>>();

        plan.forward_complex_slice_inplace(&mut signal);
        let label = format!("warm f32 n={n}");
        window(&label, || {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
        });

        assert_eq!(
            GLOBAL_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f32 n={n} execution allocated through the global allocator"
        );
        assert_eq!(
            MNEMOSYNE_ALLOCATIONS.allocations.load(Ordering::Relaxed),
            0,
            "warmed f32 n={n} execution allocated directly through Mnemosyne"
        );
    }
}

#[test]
fn reset_hides_stale_slots_from_an_unpublished_current_slot() {
    static COUNTER: AllocationCounter = AllocationCounter::new();
    let mut byte = 0u8;
    let pointer = std::ptr::from_mut(&mut byte);

    COUNTER.reset();
    COUNTER.record_alloc(pointer, 64);
    COUNTER.reset();
    // Model a concurrent allocator that reserved slot zero but has not yet
    // published its pointer. A pre-window free must not consume the stale
    // pointer that occupied this slot before reset.
    COUNTER.next.store(1, Ordering::Relaxed);
    COUNTER.record_free(pointer);

    assert_eq!(COUNTER.live.load(Ordering::Relaxed), 0);
    assert!(COUNTER.pointers[0].load(Ordering::Acquire).is_null());
}

#[test]
fn claimed_realloc_identity_cannot_consume_a_reused_address() {
    static COUNTER: AllocationCounter = AllocationCounter::new();
    let mut old_byte = 0u8;
    let mut replacement_byte = 0u8;
    let old_pointer = std::ptr::from_mut(&mut old_byte);
    let replacement_pointer = std::ptr::from_mut(&mut replacement_byte);

    COUNTER.reset();
    COUNTER.record_alloc(old_pointer, 64);
    let old = COUNTER
        .claim(old_pointer)
        .expect("the realloc source must be tracked");
    // Model another allocation receiving the released address while the
    // underlying realloc is in progress.
    COUNTER.record_alloc(old_pointer, 32);
    COUNTER.record_realloc_claimed(Some(old), replacement_pointer, 128);

    let (_, reused_size) = COUNTER
        .claim(old_pointer)
        .expect("the reused address must retain its own identity");
    let (_, replacement_size) = COUNTER
        .claim(replacement_pointer)
        .expect("the realloc result must be tracked independently");
    assert_eq!(reused_size, 32);
    assert_eq!(replacement_size, 128);
}

#[test]
fn failed_realloc_restores_the_claimed_source() {
    static COUNTER: AllocationCounter = AllocationCounter::new();
    let mut byte = 0u8;
    let pointer = std::ptr::from_mut(&mut byte);

    COUNTER.reset();
    COUNTER.record_alloc(pointer, 64);
    let (slot, size) = COUNTER
        .claim(pointer)
        .expect("the realloc source must be tracked");
    // `GlobalAlloc::realloc` takes this branch when the system allocator
    // returns null: ownership of the original allocation is unchanged.
    COUNTER.restore(slot, pointer);

    let (_, restored_size) = COUNTER
        .claim(pointer)
        .expect("a failed realloc must leave its source tracked");
    assert_eq!(restored_size, size);
    assert_eq!(COUNTER.live.load(Ordering::Relaxed), 64);
}

#[test]
#[ignore = "measurement probe for cross-thread plan retention"]
fn cross_thread_plan_retention() {
    // `BatchedPlan`, `FourStepPlanes`, and `ResidentPlan` are cached per
    // thread. What the caches do on a miss decides whether the O(16n) tables
    // exist once or once per thread, and nothing evicts them, so a thread pool
    // sized to the machine multiplies the retained figure this file reports for
    // one thread. Read the block listing: the 16n-sized rows carry the count.
    let _mnemosyne_hooks = MnemosyneHooks::install();
    const N: usize = 65_536;

    for threads in [1usize, 2, 8] {
        println!("threads = {threads}, n = {N} (16n = {} bytes)", N * 16);
        let label = format!("first forward on {threads} thread(s)");
        window(&label, || {
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        let mut signal: Vec<Complex64> = (0..N)
                            .map(|index| {
                                let x = index as f64;
                                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                            })
                            .collect();
                        let plan = crate::FftPlan1D::<f64>::new(
                            crate::Shape1D::new(N).expect("invariant: shape lengths are non-zero"),
                        );
                        plan.forward_complex_slice_inplace(std::hint::black_box(&mut signal));
                    });
                }
            });
        });
    }
}
