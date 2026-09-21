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
     evicted. The table and every ring holding a plan share one slot with
     its stamp, so a use served by any thread's ring counts. Recency is kept
     to miss granularity: the table's tick advances once per plan built, a
     use stores the current tick only when the stamp moves, and plans used
     since the last build tie. Lookups stay under the read lock, and a hit
     does no atomic read-modify-write.

     Ties are the common case, so how they break is part of the policy, not
     an implementation detail. The table alone holding a slot is the only
     evidence of non-use left between ticks -- a ring holds the slot of
     every plan its thread touched recently -- so the victim among tied
     entries is one whose slot has no ring holder. Breaking the tie by
     position instead, as the first draft did, is not arbitrary but biased:
     new entries append, so the victim is the longest-resident tied entry,
     which is a plan kept hot through a ring as readily as a one-shot built
     at the current tick. Ties that remain, between entries equally held,
     carry no information to order them and break by position.
   - **Per-thread ring:** the `LOCAL_CAPACITY = 4` most recently used plans,
     replacing the unbounded per-thread map and the one-entry slot. A
     repeated or alternating shape takes no lock and allocates nothing.
   - **`apollo_fft::clear_plan_caches()`:** empties every shared table and
     the calling thread's rings, and bumps a process epoch. Each of another
     thread's rings (one per scalar and dimension) empties on that thread's
     next lookup through that ring. Called from a thread-local destructor,
     it skips the caller's rings already destroyed.
   - Retained plan memory per scalar and dimension is then at most 64 shared
     plans plus 4 in each thread's ring, times the largest plan. A ring
     outlives a clear only until its thread's next lookup or exit.
2. **Tables (slice 2).** Plans own their tables through `Arc`, and the
   global kernel caches hold `Weak`, so a table dies with the last plan
   using it. The `_RAW` pointer fast path goes first: it is replaced by
   slices borrowed from plan-owned `Arc`s, under Miri. After slice 2,
   `clear_plan_caches()` returns a process to its baseline once no caller
   holds a plan. Plan-less kernel paths also read these caches: dispatch
   with no table, Bluestein's padded transforms, and the four-step
   sub-transforms. Under `Weak` they would rebuild on every call, so slice 2
   starts with a spike that enumerates them and assigns each an owner.

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
  - the least recently used entry is the one evicted when the stamps
    differ, and a plan kept hot only through a thread's ring survives
    eviction for another thread;
  - when every stamp ties, the entry evicted is one no ring holds, not the
    ring-held entry that happens to sit first;
  - a ring hit consults no table and moves to the front;
  - concurrent misses build a shape once;
  - a clear releases unheld plans and the calling thread's rings;
  - another thread's ring empties on its next lookup while that thread is
    still alive;
  - a clear from a thread-local destructor, after the rings are destroyed,
    does not abort.
  Each of eight mutations (no epoch check, no ring hit, no move to front,
  no write-lock re-check, no ring stamp, most-recent eviction, position
  tie-breaking, borrowing a destroyed ring) fails at least one of these
  tests.
- **Slice 1 timing:** a pinned A/B of two release builds, minimum of 21
  samples per case, four rounds on a P core and an E core. The ring pays
  where it was meant to: two alternating shapes run 6.4% (E) and 9.1% (P)
  faster than the map it replaces. One shape and six cycling shapes, which
  overflow the ring on every call, are within the drift band. The lookup
  itself, timed alone, is 8.6 to 8.7 ns (E) and 10.4 ns unchanged (P).
  The F16 automatic path at 96 -- where const thread-local initialization
  once regressed (commit 691fddcc) -- reads 2.9% slower on the E core, but
  the same transform driven from a plan held outside the cache, whose code
  is identical in both arms and consults no cache, reads 3.9% slower in the
  same runs. The difference is code layout, not lookup cost; the P core
  shows neither.
- **Slice 2:** the audit's oracle. Retained bytes return to baseline after
  a clear, the allocation-free probes still pass, and Miri runs over the
  replaced raw path.
