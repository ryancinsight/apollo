# 0068 — Bounded, clearable plan and table caches

- Status: Accepted
- Date: 2026-09-18
- Items: `backlog.md#apollo-mem-cache-bounds`
- Evidence: `output/apollo-memory-audit-2026-09-18.md#f4` (the finding and
  its growth measurements), `output/apollo-mem-cache-bounds-2026-09-18/`
  (slice 1's hit-path timing)

## Context

Every distinct length or shape a process transforms leaves state behind for
the life of the process, at two levels.

- **Plans.** `PlanCacheProvider` kept a process-wide map and a per-thread map
  per scalar and dimension, plus a one-entry per-thread slot for 1-D. Neither
  was bounded, and no API released them. The per-thread maps pinned an
  `Arc` of every plan the thread had used, so evicting the global map alone
  would free nothing.
- **Kernel tables.** The twiddle, four-step, Rader, Bluestein and batched
  caches are append-only. The `_RAW` thread-local pointer caches in
  `mixed_radix/caches/twiddle.rs` are sound only because those global entries
  are immortal.

The audit measured 2.3 times the data per distinct smooth length, and 8.8 to
11.8 times for Rader primes. A service cycling through many lengths grows
without bound.

## Decision

Two slices, the plan level first.

1. **Plans (slice 1).** One generic cache replaces the six hand-copied ones:
   - **Shared table:** at most `SHARED_CAPACITY = 64` plans per scalar and
     dimension. On a miss past that bound, the least recently used plan is
     evicted, judged by a per-entry atomic stamp, so lookups stay under the
     read lock.
   - **Per-thread ring:** the `LOCAL_CAPACITY = 4` most recently used plans,
     replacing the unbounded per-thread map and the one-entry slot. A
     repeated or alternating shape takes no lock and allocates nothing.
   - **`apollo_fft::clear_plan_caches()`:** empties every shared table and
     the calling thread's rings, and bumps a process epoch. Any other
     thread's ring empties on that thread's next lookup.
   - Retained plan memory per scalar and dimension is then at most 64 shared
     plans plus 4 in each thread's ring, times the largest plan. A ring
     outlives a clear only until its thread's next lookup or exit.
2. **Tables (slice 2).** Plans own their tables through `Arc`, and the
   global kernel caches hold `Weak`, so a table dies with the last plan
   using it. The `_RAW` pointer fast path goes first: it is replaced by
   slices borrowed from plan-owned `Arc`s, under Miri. After slice 2,
   `clear_plan_caches()` returns a process to its baseline once no caller
   holds a plan.

## Alternatives

- **A byte budget instead of an entry count.** This bounds memory directly,
  but it needs a per-plan footprint that no plan reports today, and the
  audit measured footprints from 1.1 to 11.8 times the data across route
  families. Rejected for slice 1;
  it can replace the count once plans report their footprint, which
  slice 2's owned tables make possible.
- **`Weak` plans only, no bound.** A plan would die when the last caller
  drops it, and a caller transforming one length in a loop through the free
  functions would rebuild the plan on every call. Rejected: the free
  functions promise an allocation-free warm path, and the ring is what keeps
  that promise.
- **A per-thread map with its own bound.** This would cost a bound per
  thread times the thread count. A ring of four holds a repeated shape, a
  transform's forward and inverse lengths, or the three axes of a volume,
  at a fixed small cost.

## Verification

- **Slice 1 tests (`orchestration/cache/plans/tests.rs`):**
  - repeated shapes share a plan through the ring and the shared table;
  - cycling 256 lengths leaves exactly `SHARED_CAPACITY` plans;
  - the least recently used entry is the one evicted;
  - a clear releases unheld plans, and the calling thread's ring with them;
  - another thread's ring empties on its next lookup while that thread is
    still alive. A mutation that skips the epoch check fails this test.
- **Slice 1 timing:** a pinned A/B of the public-API hit path, same shape
  and alternating shapes.
- **Slice 2:** the audit's oracle. Retained bytes return to baseline after
  a clear, the allocation-free probes still pass, and Miri runs over the
  replaced raw path.
